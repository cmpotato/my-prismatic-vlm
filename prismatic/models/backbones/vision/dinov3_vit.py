"""
dinov3_vit.py
"""

from prismatic.models.backbones.vision.base_vision import TimmViTBackbone

# Registry =>> Supported DINOv3 Vision Backbones (from TIMM)
#   => Reference: https://arxiv.org/abs/2508.10104
DINOv3_VISION_BACKBONES = {
    "dinov3-vit-s": "vit_small_patch16_dinov3",
    "dinov3-vit-b": "vit_base_patch16_dinov3",
    "dinov3-vit-l": "vit_large_patch16_dinov3",
    "dinov3-vit-h+": "vit_huge_plus_patch16_dinov3",
    "dinov3-vit-7b": "vit_7b_patch16_dinov3",
}


class DinoV3ViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            DINOv3_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )
