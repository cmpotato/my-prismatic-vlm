"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def _collate_tensor_pixel_values(
        self, pixel_values: Sequence[torch.Tensor | None], multimodal_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        multimodal_index_set = set(multimodal_indices.tolist())
        n_images = [int(pixel_values[idx].shape[0]) if pixel_values[idx] is not None else 0 for idx in range(len(pixel_values))]
        max_n_images = max(max(n_images), 1)
        batch_size = len(pixel_values)

        stacked_pixel_values = torch.zeros(
            (batch_size, max_n_images, *self.default_image_resolution), dtype=self.pixel_values_dtype
        )
        image_attention_mask = torch.zeros((batch_size, max_n_images), dtype=torch.bool)

        for idx in range(batch_size):
            if idx not in multimodal_index_set:
                continue
            sample_pixel_values = pixel_values[idx]
            assert isinstance(sample_pixel_values, torch.Tensor), "Expected tensor pixel values for multimodal sample."
            n_sample_images = sample_pixel_values.shape[0]
            stacked_pixel_values[idx, :n_sample_images] = sample_pixel_values
            image_attention_mask[idx, :n_sample_images] = True

        return stacked_pixel_values, image_attention_mask

    def _collate_dict_pixel_values(
        self, pixel_values: Sequence[Dict[str, torch.Tensor] | None], multimodal_indices: torch.Tensor
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        multimodal_index_set = set(multimodal_indices.tolist())
        n_images = [int(pixel_values[idx][next(iter(pixel_values[idx]))].shape[0]) if pixel_values[idx] is not None else 0 for idx in range(len(pixel_values))]
        max_n_images = max(max(n_images), 1)
        batch_size = len(pixel_values)
        example_idx = int(multimodal_indices[0].item())
        example = pixel_values[example_idx]
        assert isinstance(example, dict), "Expected dict pixel values for multimodal sample."

        stacked_pixel_values = {
            key: torch.zeros((batch_size, max_n_images, *value.shape[1:]), dtype=value.dtype) for key, value in example.items()
        }
        image_attention_mask = torch.zeros((batch_size, max_n_images), dtype=torch.bool)

        for idx in range(batch_size):
            if idx not in multimodal_index_set:
                continue
            sample_pixel_values = pixel_values[idx]
            assert isinstance(sample_pixel_values, dict), "Expected dict pixel values for multimodal sample."
            n_sample_images = sample_pixel_values[next(iter(sample_pixel_values))].shape[0]
            for key in stacked_pixel_values:
                stacked_pixel_values[key][idx, :n_sample_images] = sample_pixel_values[key]
            image_attention_mask[idx, :n_sample_images] = True

        return stacked_pixel_values, image_attention_mask

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        image_attention_mask = torch.zeros((len(input_ids), 1), dtype=torch.bool)

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))]).unsqueeze(1)
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values, image_attention_mask = self._collate_tensor_pixel_values(pixel_values, multimodal_indices)
        elif isinstance(pv_example, dict):
            pixel_values, image_attention_mask = self._collate_dict_pixel_values(pixel_values, multimodal_indices)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            image_attention_mask=image_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )
