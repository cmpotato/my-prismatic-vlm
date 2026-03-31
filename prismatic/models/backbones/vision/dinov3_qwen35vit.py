"""
dinov3_qwen35vit.py

Dual-tower vision backbone: DINOv3-ViT-L (TIMM) + Qwen3.5-ViT (transformers).
Returns concatenated features [B, num_patches, 1024+1152=2176] for FusedMLPProjector.

Reference: DinoSigLIP dual-tower pattern in dinosiglip_vit.py.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import timm
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.base_vision import ImageTransform, LetterboxPad, VisionBackbone, unpack_tuple
from prismatic.models.backbones.vision.dinov3_vit import DINOv3_VISION_BACKBONES
from prismatic.models.backbones.vision.qwen35_vit import QWEN35_VIT_BACKBONES, Qwen35ViTBackbone

# Registry of supported DINOv3 + Qwen3.5-ViT pairs
DINOV3_QWEN35VIT_BACKBONES = {
    "dinov3qwen35vit-224px": {
        "dinov3": "dinov3-vit-l",        # key into DINOv3_VISION_BACKBONES
        "qwen35vit": "qwen35-vit",       # key into QWEN35_VIT_BACKBONES
    },
}


@dataclass
class DINOv3Qwen35ViTImageTransform:
    """Dual-path image transform: each backbone gets its own normalization."""
    dinov3_image_transform: ImageTransform
    qwen35vit_image_transform: ImageTransform
    is_prismatic: bool = True

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        return {
            "dinov3": self.dinov3_image_transform(img, **kwargs),
            "qwen35vit": self.qwen35vit_image_transform(img, **kwargs),
        }


class DINOv3Qwen35ViTBackbone(VisionBackbone):
    """Dual-tower backbone concatenating DINOv3 (1024d) and Qwen3.5-ViT (1152d) features."""

    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)

        pair_cfg = DINOV3_QWEN35VIT_BACKBONES[vision_backbone_id]

        # === DINOv3 side: TIMM ViT ===
        dinov3_timm_id = DINOv3_VISION_BACKBONES[pair_cfg["dinov3"]]
        self.dinov3_featurizer: VisionTransformer = timm.create_model(
            dinov3_timm_id, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.dinov3_featurizer.eval()

        # Monkey-patch forward to return second-to-last layer features
        if hasattr(self.dinov3_featurizer, "get_intermediate_layers"):
            self.dinov3_featurizer.forward = unpack_tuple(
                partial(
                    self.dinov3_featurizer.get_intermediate_layers,
                    n={len(self.dinov3_featurizer.blocks) - 2},
                )
            )
        elif hasattr(self.dinov3_featurizer, "forward_features"):
            self.dinov3_featurizer.forward = self.dinov3_featurizer.forward_features
        else:
            raise ValueError("DINOv3 model exposes neither `get_intermediate_layers` nor `forward_features`.")

        # === Qwen3.5-ViT side ===
        self.qwen35vit = Qwen35ViTBackbone(
            pair_cfg["qwen35vit"],
            image_resize_strategy,
            default_image_size=self.default_image_size,
        )

        # === Build dual-path transforms ===
        # DINOv3 transform from TIMM data config
        self.dinov3_data_cfg = timm.data.resolve_model_data_config(self.dinov3_featurizer)
        self.dinov3_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)
        default_dinov3_transform = timm.data.create_transform(**self.dinov3_data_cfg, is_training=False)

        # Qwen3.5-ViT transform (already built in the backbone)
        default_qwen35_transform = self.qwen35vit.get_image_transform()

        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_dinov3_transform, Compose), "Unexpected `default_dinov3_transform`!"
            assert isinstance(default_dinov3_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            dinov3_transform = Compose(
                [
                    Resize(target_size, interpolation=default_dinov3_transform.transforms[0].interpolation),
                    *default_dinov3_transform.transforms[1:],
                ]
            )
            self.image_transform = DINOv3Qwen35ViTImageTransform(dinov3_transform, default_qwen35_transform)

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = DINOv3Qwen35ViTImageTransform(default_dinov3_transform, default_qwen35_transform)

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_dinov3_transform, Compose), "Unexpected `default_dinov3_transform`!"
            assert "mean" in self.dinov3_data_cfg, "DINOv3 `data_cfg` missing `mean`!"

            dinov3_fill = tuple([int(x * 255) for x in self.dinov3_data_cfg["mean"]])
            dinov3_lb_transform = Compose([LetterboxPad(dinov3_fill), *default_dinov3_transform.transforms])
            self.image_transform = DINOv3Qwen35ViTImageTransform(dinov3_lb_transform, default_qwen35_transform)

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP policy wrapping both ViT backbones and their transformer blocks."""
        # DINOv3 side (TIMM)
        dinov3_block_cls = type(self.dinov3_featurizer.blocks[0]) if len(self.dinov3_featurizer.blocks) > 0 else None
        # Qwen3.5-ViT side
        qwen35_block_cls = type(self.qwen35vit.featurizer.blocks[0])

        module_classes = {type(self.dinov3_featurizer), type(self.qwen35vit.featurizer)}
        block_classes = {qwen35_block_cls}
        if dinov3_block_cls is not None:
            block_classes.add(dinov3_block_cls)

        vit_wrap_policy = partial(_module_wrap_policy, module_classes=module_classes)
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=block_classes)
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward through both towers and concatenate along feature dim.

        Args:
            pixel_values: {"dinov3": [B, 3, H, W], "qwen35vit": [B, 3, H, W]}

        Returns:
            [B, num_patches, 1024 + 1152 = 2176]
        """
        dinov3_patches = self.dinov3_featurizer(pixel_values["dinov3"])     # [B, N, 1024]
        qwen35_patches = self.qwen35vit(pixel_values["qwen35vit"])         # [B, 196, 1152]

        # Strip CLS + register prefix tokens from DINOv3 if present (e.g. 1 CLS + 4 regs = 5)
        n_prefix = getattr(self.dinov3_featurizer, "num_prefix_tokens", 0)
        if n_prefix > 0:
            dinov3_patches = dinov3_patches[:, n_prefix:]                   # [B, 196, 1024]

        return torch.cat([dinov3_patches, qwen35_patches], dim=2)          # [B, 196, 2176]

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.dinov3_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.dinov3_featurizer.embed_dim + self.qwen35vit.embed_dim  # 1024 + 1152 = 2176

    @property
    def num_patches(self) -> int:
        return self.qwen35vit.num_patches  # 196

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
