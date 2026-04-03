"""
qwen35_text.py

LLM backbone definition for Qwen3.5 text-only decoding.
This wrapper intentionally exposes only the text decoder and lm_head as the LLM backbone.
"""

from __future__ import annotations

import importlib
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Type

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.base_llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder, Qwen35PurePromptBuilder
from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def _patch_torch_autocast_api_for_transformers5() -> None:
    """
    Compatibility fix for the runtime error seen in training:
    `TypeError: torch.is_autocast_enabled() takes no arguments (1 given)`.
    """
    try:
        torch.is_autocast_enabled("cuda")
        return
    except TypeError:
        pass

    old_is_autocast_enabled = torch.is_autocast_enabled

    def _compat_is_autocast_enabled(device_type: Optional[str] = None) -> bool:
        del device_type
        return old_is_autocast_enabled()

    torch.is_autocast_enabled = _compat_is_autocast_enabled


_patch_torch_autocast_api_for_transformers5()


QWEN35_TEXT_MODELS = {
    "qwen35-text-9b-base": {
        "llm_family": "qwen35-text",
        "hf_hub_path": "/home/max/.cache/modelscope/hub/models/Qwen/Qwen3___5-9B-Base",
    },
}


def _get_qwen35_for_conditional_generation_cls() -> type[nn.Module]:
    try:
        module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    except ImportError as e:
        raise ImportError(
            "Qwen3.5 support requires a `transformers` build that provides "
            "`transformers.models.qwen3_5.modeling_qwen3_5`."
        ) from e

    if not hasattr(module, "Qwen3_5ForConditionalGeneration"):
        raise ImportError(
            "Loaded `transformers.models.qwen3_5.modeling_qwen3_5`, but "
            "`Qwen3_5ForConditionalGeneration` is unavailable."
        )
    return module.Qwen3_5ForConditionalGeneration


class Qwen35TextLLMBackbone(LLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 32768,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(llm_backbone_id)
        cfg = QWEN35_TEXT_MODELS[llm_backbone_id]
        self.llm_family = cfg["llm_family"]
        self.hf_hub_path = cfg["hf_hub_path"]
        self.llm_max_length = llm_max_length
        self.inference_mode = inference_mode
        self.lm_head: nn.Module = None
        self._transformer_layer_cls: Optional[type[nn.Module]] = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_hub_path,
            model_max_length=self.llm_max_length,
            token=hf_token,
            padding_side="right",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self._added_image_token = self._ensure_image_token_in_tokenizer()

        qwen35_cls = _get_qwen35_for_conditional_generation_cls()

        if not self.inference_mode:
            overwatch.info(
                f"Loading [bold]{self.llm_family}[/] text decoder from [underline]`{self.hf_hub_path}`[/]",
                ctx_level=1,
            )
            full_model = qwen35_cls.from_pretrained(
                self.hf_hub_path,
                token=hf_token,
                trust_remote_code=True,
                local_files_only=local_files_only,
                torch_dtype="auto",
            )
        else:
            overwatch.info(
                f"Building empty [bold]{self.llm_family}[/] text decoder from [underline]`{self.hf_hub_path}`[/]",
                ctx_level=1,
            )
            full_cfg = AutoConfig.from_pretrained(
                self.hf_hub_path,
                token=hf_token,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            full_cfg.text_config.vocab_size = len(self.tokenizer)
            full_model = qwen35_cls._from_config(full_cfg)

        if self._added_image_token and hasattr(full_model, "resize_token_embeddings"):
            full_model.resize_token_embeddings(len(self.tokenizer))

        self.llm, self.lm_head = self._extract_text_decoder_and_lm_head(full_model)
        del full_model

        self.llm.config.use_cache = False if not self.inference_mode else True
        if not self.inference_mode:
            self.llm.enable_input_require_grads()

        if not hasattr(self.llm, "generation_config"):
            self.llm.generation_config = GenerationConfig.from_model_config(self.llm.config)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if self.image_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Failed to register `{self.image_token}` as a dedicated tokenizer token.")

        if not hasattr(self.llm, "layers") or len(self.llm.layers) == 0:
            raise ValueError(
                f"Qwen3.5 text decoder `{type(self.llm)}` does not expose `.layers`; "
                "cannot derive FSDP transformer layer class."
            )
        self._transformer_layer_cls = type(self.llm.layers[0])

    @staticmethod
    def _extract_text_decoder_and_lm_head(full_model: nn.Module) -> tuple[nn.Module, nn.Module]:
        lm_head = getattr(full_model, "lm_head", None)
        if lm_head is None:
            raise ValueError(f"Qwen3.5 model `{type(full_model)}` is missing `lm_head`.")

        candidate_paths = (
            ("model", "language_model"),
            ("language_model",),
            ("model", "text_model"),
            ("text_model",),
        )
        for path in candidate_paths:
            node = full_model
            for attr in path:
                node = getattr(node, attr, None)
                if node is None:
                    break
            if node is not None:
                return node, lm_head

        raise ValueError(
            f"Could not locate Qwen3.5 text decoder inside `{type(full_model)}`. "
            "Tried `model.language_model`, `language_model`, `model.text_model`, and `text_model`."
        )

    def _ensure_image_token_in_tokenizer(self) -> bool:
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if image_token_id != self.tokenizer.unk_token_id:
            return False

        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": [self.image_token]})
        return num_added > 0

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls})

    def enable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable()

    def enable_lora(
        self,
        *,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: Sequence[str],
    ) -> None:
        if r <= 0:
            raise ValueError(f"`lora_r` must be > 0, got {r}.")
        if len(target_modules) == 0:
            raise ValueError("`lora_target_modules` must not be empty when LoRA is enabled.")

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise ImportError(
                "LoRA requested but `peft` is not installed. Install with `pip install peft`."
            ) from e

        self.llm.requires_grad_(False)
        self.lm_head.requires_grad_(False)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        self._lora_config = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": list(target_modules),
        }

        lora_param_count = 0
        for name, param in self.llm.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(dtype=self.half_precision_dtype)
                lora_param_count += param.numel()

        n_total = sum(param.numel() for param in self.llm.parameters())
        n_trainable = sum(param.numel() for param in self.llm.parameters() if param.requires_grad)
        if n_trainable == 0:
            raise RuntimeError("LoRA injection produced zero trainable LLM parameters; check `lora_target_modules`.")

        overwatch.info(
            "Enabled LoRA for Qwen3.5 text decoder "
            f"(r={r}, alpha={lora_alpha}, dropout={lora_dropout}, "
            f"target_modules={list(target_modules)}, "
            f"lora_dtype={self.half_precision_dtype}, "
            f"lora_params={lora_param_count}, trainable={n_trainable}/{n_total})",
            ctx_level=1,
        )

    def has_lora_enabled(self) -> bool:
        return hasattr(self.llm, "peft_config")

    def get_lora_config(self) -> Optional[Dict[str, object]]:
        return getattr(self, "_lora_config", None)

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Sequence[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        text_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = text_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict is False:
            output = (logits, text_outputs.past_key_values, text_outputs.hidden_states, text_outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=text_outputs.past_key_values,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return Qwen35PurePromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        assert self._transformer_layer_cls is not None, "Transformer layer class is not initialized."
        return self._transformer_layer_cls

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.embed_tokens, self.llm.layers[-1], self.lm_head)
