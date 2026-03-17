"""
prismatic.py

PyTorch Module defining a PrismaticVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class PrismaticVLM(VLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            raise ValueError(f"PrismaticVLM with `{arch_specifier = }` is not supported!")

        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
        )

        # Load from Checkpoint:
        #   1) Full checkpoint format: `projector` + `llm_backbone`
        #   2) LoRA adapter format:    `projector` + `llm_backbone_lora` + top-level `lora_config`
        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
        model_state_dict = checkpoint["model"]
        assert "projector" in model_state_dict, (
            "PrismaticVLM `from_pretrained` expects checkpoint key `projector` in `checkpoint['model']`!"
        )

        vlm.projector.load_state_dict(model_state_dict["projector"])
        if "llm_backbone" in model_state_dict:
            vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        elif "llm_backbone_lora" in model_state_dict:
            if not hasattr(vlm.llm_backbone, "enable_lora"):
                raise ValueError(
                    f"LLM backbone `{vlm.llm_backbone.identifier}` does not implement `enable_lora(...)` "
                    "required to load LoRA adapter checkpoints."
                )
            if getattr(vlm.llm_backbone, "inference_mode", False):
                raise ValueError(
                    "LoRA adapter checkpoints require a loaded base LLM (not an empty inference skeleton). "
                    "Instantiate the LLM backbone with `inference_mode=False` before applying adapters."
                )
            lora_config = checkpoint.get("lora_config", None)
            if lora_config is None:
                raise ValueError(
                    "LoRA adapter checkpoint missing top-level `lora_config`; cannot reconstruct adapter modules."
                )
            vlm.llm_backbone.enable_lora(
                r=int(lora_config["r"]),
                lora_alpha=int(lora_config["lora_alpha"]),
                lora_dropout=float(lora_config["lora_dropout"]),
                target_modules=lora_config["target_modules"],
            )
            load_result = vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone_lora"], strict=False)
            if len(load_result.unexpected_keys) > 0:
                raise ValueError(f"Unexpected LoRA checkpoint keys: {load_result.unexpected_keys}")
        else:
            raise ValueError(
                "PrismaticVLM `from_pretrained` expects either `llm_backbone` or `llm_backbone_lora` in checkpoint."
            )

        # Inference loads should enable KV-cache regardless of how weights were materialized.
        vlm.llm_backbone.llm.config.use_cache = True

        # Freeze Weights
        vlm.requires_grad_(False)
        vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" >
        """
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "finetune":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "full-finetune":
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"PrismaticVLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    @staticmethod
    def _flatten_multimodal_pixel_values(
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        image_attention_mask: torch.Tensor,
        multimodal_indices: torch.Tensor,
    ) -> tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], List[int]]:
        selected_image_mask = image_attention_mask[multimodal_indices]
        image_counts = selected_image_mask.sum(dim=1).tolist()

        if isinstance(pixel_values, dict):
            selected_pixel_values = {key: value[multimodal_indices] for key, value in pixel_values.items()}
            flattened_pixel_values = {key: value[selected_image_mask] for key, value in selected_pixel_values.items()}
        else:
            flattened_pixel_values = pixel_values[multimodal_indices][selected_image_mask]

        return flattened_pixel_values, image_counts

    def _build_interleaved_sequences(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        input_embeddings: torch.Tensor,
        projected_patch_embeddings: torch.Tensor,
        multimodal_indices: torch.Tensor,
        image_counts: List[int],
        labels: Optional[torch.LongTensor] = None,
    ) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Optional[Dict[int, torch.Tensor]]]:
        if self.llm_backbone.image_token_id is None:
            raise ValueError("LLM backbone is missing `image_token_id`; cannot build interleaved multimodal sequence.")

        multimodal_embeddings: Dict[int, torch.Tensor] = {}
        multimodal_attention_masks: Dict[int, torch.Tensor] = {}
        multimodal_labels: Optional[Dict[int, torch.Tensor]] = {} if labels is not None else None
        patch_offset = 0

        for mm_idx, batch_idx in enumerate(multimodal_indices.tolist()):
            n_images = int(image_counts[mm_idx])
            if n_images <= 0:
                raise ValueError(f"Multimodal sample at batch index {batch_idx} has no valid images.")

            seq_len = int(attention_mask[batch_idx].sum().item())
            sample_input_ids = input_ids[batch_idx, :seq_len]
            sample_attention_mask = attention_mask[batch_idx, :seq_len]
            sample_input_embeddings = input_embeddings[batch_idx, :seq_len]
            sample_labels = labels[batch_idx, :seq_len] if labels is not None else None

            placeholder_positions = (sample_input_ids == self.llm_backbone.image_token_id).nonzero(as_tuple=False).flatten()
            if len(placeholder_positions) != n_images:
                raise ValueError(
                    f"Sample at batch index {batch_idx} has {n_images} image(s) but {len(placeholder_positions)} "
                    f"`<image>` token(s) after tokenization."
                )

            image_patch_blocks = projected_patch_embeddings[patch_offset : patch_offset + n_images]
            patch_offset += n_images

            embedding_chunks: List[torch.Tensor] = []
            attention_chunks: List[torch.Tensor] = []
            label_chunks: List[torch.Tensor] = []
            cursor = 0

            for placeholder_position, patch_block in zip(placeholder_positions.tolist(), image_patch_blocks):
                if placeholder_position > cursor:
                    embedding_chunks.append(sample_input_embeddings[cursor:placeholder_position])
                    attention_chunks.append(sample_attention_mask[cursor:placeholder_position])
                    if sample_labels is not None:
                        label_chunks.append(sample_labels[cursor:placeholder_position])

                embedding_chunks.append(patch_block)
                attention_chunks.append(
                    torch.ones(patch_block.shape[0], dtype=sample_attention_mask.dtype, device=sample_attention_mask.device)
                )
                if sample_labels is not None:
                    label_chunks.append(
                        torch.full(
                            (patch_block.shape[0],),
                            IGNORE_INDEX,
                            dtype=sample_labels.dtype,
                            device=sample_labels.device,
                        )
                    )

                cursor = placeholder_position + 1

            if cursor < seq_len:
                embedding_chunks.append(sample_input_embeddings[cursor:seq_len])
                attention_chunks.append(sample_attention_mask[cursor:seq_len])
                if sample_labels is not None:
                    label_chunks.append(sample_labels[cursor:seq_len])

            multimodal_embeddings[batch_idx] = torch.cat(embedding_chunks, dim=0)
            multimodal_attention_masks[batch_idx] = torch.cat(attention_chunks, dim=0)
            if multimodal_labels is not None:
                multimodal_labels[batch_idx] = torch.cat(label_chunks, dim=0)

        return multimodal_embeddings, multimodal_attention_masks, multimodal_labels

    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`

    # ruff: noqa: C901
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        if attention_mask is None:
            attention_mask = input_ids.ne(self.llm_backbone.pad_token_id)

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if image_attention_mask is None:
            raise ValueError("Multimodal forward requires `image_attention_mask`.")

        # Get Input Embeddings from LLM Backbone before we replace `<image>` placeholders with patch blocks.
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

        # Run Visual Feature Extraction
        flattened_pixel_values, image_counts = self._flatten_multimodal_pixel_values(
            pixel_values, image_attention_mask, multimodal_indices
        )
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(flattened_pixel_values, dict):
                patch_features = self.vision_backbone(flattened_pixel_values)
            else:
                patch_features = self.vision_backbone(flattened_pixel_values)

        # Projection Logic :: [n_images, num_patches, llm_embed_dim]
        projected_patch_embeddings = self.projector(patch_features)

        multimodal_embeddings, multimodal_attention_masks, multimodal_labels = self._build_interleaved_sequences(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_embeddings=input_embeddings,
            projected_patch_embeddings=projected_patch_embeddings,
            multimodal_indices=multimodal_indices,
            image_counts=image_counts,
            labels=labels,
        )

        fused_embedding_sequences: List[torch.Tensor] = []
        fused_attention_sequences: List[torch.Tensor] = []
        fused_label_sequences: Optional[List[torch.Tensor]] = [] if labels is not None else None

        for batch_idx in range(len(input_ids)):
            if batch_idx in multimodal_embeddings:
                fused_embedding_sequences.append(multimodal_embeddings[batch_idx])
                fused_attention_sequences.append(multimodal_attention_masks[batch_idx])
                if fused_label_sequences is not None and multimodal_labels is not None:
                    fused_label_sequences.append(multimodal_labels[batch_idx])
                continue

            seq_len = int(attention_mask[batch_idx].sum().item())
            fused_embedding_sequences.append(input_embeddings[batch_idx, :seq_len])
            fused_attention_sequences.append(attention_mask[batch_idx, :seq_len])
            if fused_label_sequences is not None and labels is not None:
                fused_label_sequences.append(labels[batch_idx, :seq_len])

        fused_embeddings = pad_sequence(fused_embedding_sequences, batch_first=True, padding_value=0.0)
        fused_attention_mask = pad_sequence(fused_attention_sequences, batch_first=True, padding_value=False)
        fused_labels = (
            pad_sequence(fused_label_sequences, batch_first=True, padding_value=IGNORE_INDEX)
            if fused_label_sequences is not None
            else None
        )

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        return self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === GenerationMixin Methods ===
    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_attention_mask": image_attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )

        return model_inputs

    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        image_attention_mask: Optional[torch.Tensor] = None,
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(pixel_values, torch.Tensor):
            if pixel_values.ndim == 4:
                pixel_values = pixel_values[None, ...]
            pixel_values = pixel_values.to(self.device)
            batch_size, max_n_images = pixel_values.shape[:2]
        elif isinstance(pixel_values, dict):
            first_value = next(iter(pixel_values.values()))
            if first_value.ndim == 4:
                pixel_values = {k: v[None, ...] for k, v in pixel_values.items()}
                first_value = next(iter(pixel_values.values()))
            pixel_values = {k: v.to(self.device) for k, v in pixel_values.items()}
            batch_size, max_n_images = first_value.shape[:2]
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        if image_attention_mask is None:
            image_attention_mask = torch.ones((batch_size, max_n_images), dtype=torch.bool, device=self.device)
        else:
            image_attention_mask = image_attention_mask.to(self.device)
        if batch_size != len(texts):
            raise ValueError(f"`pixel_values` batch size {batch_size} does not match number of texts {len(texts)}.")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    sample_pixel_values = pixel_values[idx : idx + 1]
                elif isinstance(pixel_values, dict):
                    sample_pixel_values = {k: pixel_values[k][idx : idx + 1] for k in pixel_values}
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
                sample_image_attention_mask = image_attention_mask[idx : idx + 1]

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(
                        input_ids=input_ids,
                        pixel_values=sample_pixel_values,
                        image_attention_mask=sample_image_attention_mask,
                        **kwargs,
                    )
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=sample_pixel_values,
                        image_attention_mask=sample_image_attention_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(self, image: Union[Image, List[Image]], prompt_text: str, **kwargs: str) -> str:
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        images = image if isinstance(image, list) else [image]
        transformed_images = [image_transform(single_image) for single_image in images]
        if isinstance(transformed_images[0], torch.Tensor):
            pixel_values = torch.stack(transformed_images).unsqueeze(0).to(self.device)
        elif isinstance(transformed_images[0], dict):
            pixel_values = {
                key: torch.stack([transformed_image[key] for transformed_image in transformed_images]).unsqueeze(0).to(
                    self.device
                )
                for key in transformed_images[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(transformed_images[0])}")
        image_attention_mask = torch.ones((1, len(images)), dtype=torch.bool, device=self.device)

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # Shape: [1, seq]
                pixel_values=pixel_values,      # Shape: [1, n_img, 3, res, res] or Dict[str, Shape[1, n_img, 3, res, res]]
                image_attention_mask=image_attention_mask,
                **kwargs
            )
            # fmt: on

        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text
