"""
qwen35_vit.py

Wraps the Qwen3.5 VisionModel (from transformers) as a prismatic VisionBackbone.
Only the raw ViT output (last_hidden_state) is used — the merger is excluded.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL.Image import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from prismatic.models.backbones.vision.base_vision import ImageTransform, VisionBackbone

logger = logging.getLogger(__name__)


# === Qwen3.5 ViT Registry ===
QWEN35_VIT_BACKBONES = {
    "qwen35-vit": {
        "model_path": "/home/max/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B-Base",
        "embed_dim": 1152,
        "depth": 27,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "in_channels": 3,
    },
}


def _get_qwen35_vision_classes():
    """Lazily import Qwen3.5 vision model and config from transformers."""
    module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    config_module = importlib.import_module("transformers.models.qwen3_5.configuration_qwen3_5")
    return module.Qwen3_5VisionModel, config_module.Qwen3_5VisionConfig


class Qwen35ViTBackbone(VisionBackbone):
    """Wraps Qwen3_5VisionModel as a VisionBackbone with [B, 3, H, W] input interface."""

    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        cfg = QWEN35_VIT_BACKBONES[vision_backbone_id]
        self._embed_dim = cfg["embed_dim"]
        self._patch_size = cfg["patch_size"]
        self._temporal_patch_size = cfg["temporal_patch_size"]
        self._in_channels = cfg["in_channels"]
        self._num_patches = (default_image_size // self._patch_size) ** 2  # 224 // 16 = 14 -> 196
        self.dtype = torch.bfloat16

        # --- Instantiate Qwen3.5 VisionModel (without merger weights) ---
        VisionModelCls, VisionConfigCls = _get_qwen35_vision_classes()

        # Build config from the checkpoint's vision_config
        from transformers import AutoConfig

        full_config = AutoConfig.from_pretrained(cfg["model_path"], trust_remote_code=True)
        vision_config = full_config.vision_config
        if not isinstance(vision_config, VisionConfigCls):
            vision_config = VisionConfigCls(**vision_config)

        self.featurizer = VisionModelCls(vision_config)
        self.featurizer.eval()

        # Load only visual.* weights (excluding merger) from safetensors
        self._load_vision_weights(cfg["model_path"])

        # --- Image Transform: Normalize(mean=0.5, std=0.5) as used by Qwen VL models ---
        target_size = (self.default_image_size, self.default_image_size)
        self.image_transform = Compose([
            Resize(target_size),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_vision_weights(self, model_path: str) -> None:
        """Load only vision encoder weights (skip merger) from safetensors checkpoint."""
        import glob
        from safetensors import safe_open

        state_dict = {}
        safetensor_files = glob.glob(f"{model_path}/*.safetensors")
        for f in safetensor_files:
            with safe_open(f, framework="pt", device="cpu") as st:
                for key in st.keys():
                    if key.startswith("model.visual.") and "merger" not in key:
                        # Strip "model.visual." prefix to match featurizer's state_dict
                        new_key = key.replace("model.visual.", "")
                        state_dict[new_key] = st.get_tensor(key)

        missing, unexpected = self.featurizer.load_state_dict(state_dict, strict=False)
        # Merger keys are expected to be missing since we intentionally skip them
        merger_missing = [k for k in missing if "merger" in k]
        real_missing = [k for k in missing if "merger" not in k]
        if real_missing:
            logger.warning(f"Qwen3.5 ViT: missing non-merger keys: {real_missing}")
        if unexpected:
            logger.warning(f"Qwen3.5 ViT: unexpected keys: {unexpected}")
        logger.info(
            f"Qwen3.5 ViT weights loaded: {len(state_dict)} keys, "
            f"{len(merger_missing)} merger keys skipped, {len(real_missing)} missing"
        )

    def _prepare_pixels(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert [B, 3, H, W] to Qwen3.5-ViT expected format: (flattened_patches, grid_thw).

        Qwen3.5 PatchEmbed expects input shaped as:
            [num_patches_total, in_channels * temporal_patch_size * patch_size * patch_size]
        where temporal_patch_size=2, so we duplicate the frame.

        grid_thw: [B, 3] with (temporal_patches=1, h_patches, w_patches) per image.
        """
        B, C, H, W = pixel_values.shape
        h_patches = H // self._patch_size  # 14
        w_patches = W // self._patch_size  # 14
        tp = self._temporal_patch_size  # 2
        ps = self._patch_size  # 16

        # Duplicate frame to satisfy temporal_patch_size=2: [B, 3, H, W] -> [B, 3, 2, H, W]
        frames = pixel_values.unsqueeze(2).expand(-1, -1, tp, -1, -1)  # [B, C, 2, H, W]

        # Reshape into patch tokens:
        # [B, C, tp, H, W] -> [B, C, tp, h_patches, ps, w_patches, ps]
        frames = frames.reshape(B, C, tp, h_patches, ps, w_patches, ps)
        # -> [B, h_patches, w_patches, C, tp, ps, ps]
        frames = frames.permute(0, 3, 5, 1, 2, 4, 6)
        # -> [B * h_patches * w_patches, C * tp * ps * ps]
        patches = frames.reshape(B * h_patches * w_patches, C * tp * ps * ps)

        # grid_thw: temporal=1 (one "temporal group" of tp=2 frames), h_patches, w_patches
        grid_thw = torch.tensor(
            [[1, h_patches, w_patches]] * B,
            dtype=torch.long,
            device=pixel_values.device,
        )

        return patches, grid_thw

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for Qwen3.5 ViT blocks."""
        # Get the block class from the featurizer
        block_cls = type(self.featurizer.blocks[0])
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={type(self.featurizer)})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={block_cls})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, num_patches, 1152]"""
        B = pixel_values.shape[0]
        prepared_pixels, grid_thw = self._prepare_pixels(pixel_values)

        # Forward through Qwen3.5 ViT (returns BaseModelOutputWithPooling)
        outputs = self.featurizer(hidden_states=prepared_pixels, grid_thw=grid_thw)

        # Use last_hidden_state (raw ViT output, before merger)
        # Shape: [total_patches, embed_dim] -> [B, num_patches, embed_dim]
        hidden = outputs.last_hidden_state
        return hidden.reshape(B, -1, self._embed_dim)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
