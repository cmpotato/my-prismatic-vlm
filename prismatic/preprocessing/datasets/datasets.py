"""
datasets.py

PyTorch Dataset Definitions for Prismatic models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for Vicuña v1.5 Chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Type

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import LlamaTokenizerFast, PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"
        self.image_token = "<image>"
        self.image_token_id = self._resolve_image_token_id()

        # Preserve a real image placeholder token in the text sequence so `PrismaticVLM.forward()` can
        # replace it with projected patch embeddings during align training.
        self.prompt_template = f"{self.image_token}\n{{caption}}{self.tokenizer.eos_token}"

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)

    def _resolve_image_token_id(self) -> int:
        image_token_ids = self.tokenizer(self.image_token, add_special_tokens=False).input_ids
        if len(image_token_ids) != 1:
            raise ValueError(
                "AlignDataset requires `<image>` to be preserved as a single tokenizer token, "
                f"but got ids={image_token_ids} for tokenizer `{type(self.tokenizer)}`."
            )
        return int(image_token_ids[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations"]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"

        # Format Align Prompt --> keep the placeholder token in-text and supervise only the caption tokens.
        caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = input_ids.clone()

        # Ignore supervision on the placeholder token itself; `PrismaticVLM.forward()` will replace it
        # with projected patch embeddings and ignore the corresponding patch block labels.
        labels[input_ids == self.image_token_id] = IGNORE_INDEX

        # Process Image and normalize single-image samples to the same per-sample shape contract used by
        # `FinetuneDataset`: tensors become [1, C, H, W], dict values become [1, ...]. The collator interprets
        # the leading dimension as "number of images", so returning raw [C, H, W] here would be misread as 3 images.
        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.unsqueeze(0)
        elif isinstance(pixel_values, dict):
            pixel_values = {key: value.unsqueeze(0) for key, value in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported image transform output type `{type(pixel_values)}` in align dataset.")

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].replace("<image>", "").split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    @staticmethod
    def _extract_image_paths(example: Dict[str, object]) -> List[Path]:
        if "images" in example:
            image_field = example["images"]
        elif "image" in example:
            image_field = example["image"]
        else:
            return []

        if isinstance(image_field, str):
            return [Path(image_field)]
        if isinstance(image_field, list):
            return [Path(image_path) for image_path in image_field]

        raise ValueError(f"Unsupported image field type `{type(image_field)}` in finetune example.")

    @staticmethod
    def _count_image_placeholders(conversation: List[Dict[str, str]]) -> int:
        return sum(turn["value"].count("<image>") for turn in conversation)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        conversation = self.examples[idx]["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
            if isinstance(self.tokenizer, LlamaTokenizerFast):
                msg = msg.rstrip()
            # Other tokenizer families (e.g., Qwen) follow generic behavior.

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        image_paths = self._extract_image_paths(self.examples[idx])
        if image_paths:
            n_placeholders = self._count_image_placeholders(conversation)
            if n_placeholders != len(image_paths):
                raise ValueError(
                    f"Example `{self.examples[idx].get('id', idx)}` has {len(image_paths)} image(s) but "
                    f"{n_placeholders} `<image>` placeholder(s)."
                )

            processed_images = [
                self.image_transform(Image.open(self.image_dir / image_path).convert("RGB")) for image_path in image_paths
            ]
            if isinstance(processed_images[0], torch.Tensor):
                pixel_values = torch.stack(processed_images)
            elif isinstance(processed_images[0], dict):
                pixel_values = {
                    key: torch.stack([image[key] for image in processed_images]) for key in processed_images[0]
                }
            else:
                raise ValueError(f"Unsupported image transform output type `{type(processed_images[0])}`.")

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = bool(self._extract_image_paths(example))
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)
