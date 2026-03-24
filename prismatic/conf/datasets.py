"""
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Tuple

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: Tuple[Path, Path]       # Path to annotation file and images directory for `align` stage
    align_val_stage_components: Optional[Tuple[Path, Path]]  # Optional annotation/images for align validation
    finetune_stage_components: Tuple[Path, Path]    # Path to annotation file and images directory for `finetune` stage
    dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
    finetune_val_stage_components: Optional[Tuple[Path, Path]] = None  # Optional annotation/images for finetune val
    # fmt: on


# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat_train.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    align_val_stage_components: Optional[Tuple[Path, Path]] = (
        Path("download/llava-laion-cc-sbu-558k/chat_val.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_mix665k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")


# [Multimodal-Only] LLava-v15 WITHOUT the Language-Only ShareGPT Data (No Co-Training)
@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    dataset_id: str = "llava-multimodal"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    align_val_stage_components: Optional[Tuple[Path, Path]] = None
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_stripped625k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V
@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    align_val_stage_components: Optional[Tuple[Path, Path]] = None
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LRV-Instruct
@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    align_val_stage_components: Optional[Tuple[Path, Path]] = None
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    align_val_stage_components: Optional[Tuple[Path, Path]] = None
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# Car Paint Binary Finetune Dataset (OK/NG -> yes/no)
@dataclass
class CarPaint_Binary_Config(DatasetConfig):
    dataset_id: str = "carpaint-binary"

    # Keep align fields valid to preserve current framework contract.
    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat_train.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    align_val_stage_components: Optional[Tuple[Path, Path]] = (
        Path("download/llava-laion-cc-sbu-558k/chat_val.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("labeled_jpg/carpaint_finetune_chat_train.json"),
        Path("labeled_jpg/"),
    )
    finetune_val_stage_components: Optional[Tuple[Path, Path]] = (
        Path("labeled_jpg/carpaint_finetune_chat_val.json"),
        Path("labeled_jpg/"),
    )
    dataset_root_dir: Path = Path("data")


@dataclass
class CarPaint_Binary_Balanced_Config(DatasetConfig):
    dataset_id: str = "carpaint-binary-balanced"

    # Keep align fields valid to preserve current framework contract.
    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat_train.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    align_val_stage_components: Optional[Tuple[Path, Path]] = (
        Path("download/llava-laion-cc-sbu-558k/chat_val.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("labeled_jpg_1-1/carpaint_finetune_chat_train.json"),
        Path("labeled_jpg_1-1/"),
    )
    finetune_val_stage_components: Optional[Tuple[Path, Path]] = (
        Path("labeled_jpg_1-1/carpaint_finetune_chat_val.json"),
        Path("labeled_jpg_1-1/"),
    )
    dataset_root_dir: Path = Path("data")


# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config

    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config

    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config

    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config
    CARPAINT_BINARY = CarPaint_Binary_Config
    CARPAINT_BINARY_BALANCED = CarPaint_Binary_Balanced_Config

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
