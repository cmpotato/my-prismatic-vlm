"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""

from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.conf import DatasetConfig
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.preprocessing.datasets import AlignDataset, FinetuneDataset
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    split: str = "train",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        if split == "train":
            annotation_json, image_dir = dataset_cfg.align_stage_components
        elif split == "val":
            if dataset_cfg.align_val_stage_components is None:
                raise ValueError("`align_val_stage_components` is not configured for this dataset!")
            annotation_json, image_dir = dataset_cfg.align_val_stage_components
        else:
            raise ValueError(f"Split `{split}` is not supported for `align` stage!")

        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer
        )
        return dataset, collator

    elif stage in {"finetune", "full-finetune"}:
        if split == "train":
            annotation_json, image_dir = dataset_cfg.finetune_stage_components
        elif split == "val":
            if dataset_cfg.finetune_val_stage_components is None:
                raise ValueError("`finetune_val_stage_components` is not configured for this dataset!")
            annotation_json, image_dir = dataset_cfg.finetune_val_stage_components
        else:
            raise ValueError(f"Split `{split}` is not supported for `{stage}` stage!")

        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")
