"""
generate_multivision.py

Unified inference CLI for the current multi-vision-tower Prismatic models.

This script supports three checkpoint formats:
1) full checkpoint         -> projector + llm_backbone
2) LoRA adapter checkpoint -> projector + llm_backbone_lora + top-level lora_config
3) projector-only          -> projector only (e.g. align-stage checkpoints)

Run with:
  python scripts/generate_multivision.py \
    --checkpoint /path/to/run_dir_or_checkpoint.pt
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import requests
import torch
from PIL import Image

from prismatic.models import get_llm_backbone_and_tokenizer, get_vlm, get_vision_backbone_and_transform
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

DEFAULT_IMAGE_SOURCE = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
)


@dataclass
class GenerateMultiVisionConfig:
    checkpoint: Union[str, Path]
    model_config: Optional[Union[str, Path]] = None
    hf_token: Union[str, Path] = Path(".hf_token")
    image_source: str = DEFAULT_IMAGE_SOURCE
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 512
    min_length: int = 1


def _resolve_checkpoint_path(checkpoint: Union[str, Path]) -> Path:
    path = Path(checkpoint)
    if path.is_dir():
        # Priority: best-val checkpoint > latest checkpoint
        best_candidates = sorted(path.glob("best-val-*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if best_candidates:
            return best_candidates[0]

        run_candidate = path / "checkpoints" / "latest-checkpoint.pt"
        direct_candidate = path / "latest-checkpoint.pt"
        if run_candidate.exists():
            return run_candidate
        if direct_candidate.exists():
            return direct_candidate
        raise FileNotFoundError(f"Could not find a checkpoint (.pt) under `{path}`")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: `{path}`")
    return path


def _resolve_model_config_path(checkpoint_path: Path, model_config: Optional[Union[str, Path]]) -> Path:
    if model_config is not None:
        config_path = Path(model_config)
        if not config_path.exists():
            raise FileNotFoundError(f"Model config does not exist: `{config_path}`")
        return config_path

    # Case 1: checkpoint in checkpoints/ subdir -> config.json is in parent run dir
    if checkpoint_path.parent.name == "checkpoints":
        candidate = checkpoint_path.parent.parent / "config.json"
        if candidate.exists():
            return candidate

    # Case 2: checkpoint directly in run dir (e.g. best-val-*.pt)
    candidate = checkpoint_path.parent / "config.json"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not infer model config. Please pass `--model_config /path/to/config.json` explicitly."
    )


def _load_image(source: str) -> Image.Image:
    if source.startswith("http://") or source.startswith("https://"):
        return Image.open(requests.get(source, stream=True, timeout=30).raw).convert("RGB")
    return Image.open(Path(source)).convert("RGB")


def _summarize_checkpoint_format(checkpoint_obj: dict) -> str:
    model_state = checkpoint_obj.get("model", checkpoint_obj)
    if not isinstance(model_state, dict):
        raise ValueError("Unexpected checkpoint format: top-level `model` is not a dict.")

    if "projector" not in model_state:
        raise ValueError("Checkpoint missing `projector` weights.")
    if "llm_backbone" in model_state:
        return "full"
    if "llm_backbone_lora" in model_state:
        return "lora"
    return "projector-only"


def _load_checkpoint_into_vlm(vlm: PrismaticVLM, checkpoint: dict, checkpoint_path: Path) -> tuple:
    """Load checkpoint weights into the VLM. Returns (checkpoint_format, vlm)."""
    checkpoint_format = _summarize_checkpoint_format(checkpoint)

    if checkpoint_format == "projector-only":
        model_state = checkpoint["model"]
        load_result = vlm.projector.load_state_dict(model_state["projector"], strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "Projector load mismatch: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        # Enable KV-cache for generation (projector-only loads the base LLM without it)
        vlm.llm_backbone.llm.config.use_cache = True
    else:
        vlm = PrismaticVLM.from_pretrained(
            checkpoint_path,
            vlm.model_id,
            vlm.vision_backbone,
            vlm.llm_backbone,
            enable_mixed_precision_training=vlm.enable_mixed_precision_training,
            arch_specifier=vlm.arch_specifier,
        )

    return checkpoint_format, vlm


def _get_hf_token(hf_token: Union[str, Path]) -> str:
    if isinstance(hf_token, Path):
        return hf_token.read_text().strip()
    return os.environ[hf_token]


def _describe_image_inputs(vlm: PrismaticVLM, image: Image.Image) -> str:
    transformed = vlm.vision_backbone.image_transform(image)
    if isinstance(transformed, torch.Tensor):
        return f"single-tower tensor input: shape={tuple(transformed.shape)}"
    if isinstance(transformed, dict):
        parts = [f"{key}={tuple(value.shape)}" for key, value in transformed.items()]
        return "multi-tower dict input: " + ", ".join(parts)
    return f"unsupported transformed image type: {type(transformed)}"


@draccus.wrap()
def generate_multivision(cfg: GenerateMultiVisionConfig) -> None:
    checkpoint_path = _resolve_checkpoint_path(cfg.checkpoint)
    config_path = _resolve_model_config_path(checkpoint_path, cfg.model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_format = _summarize_checkpoint_format(checkpoint)

    hf_token = _get_hf_token(cfg.hf_token)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with open(config_path, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    model_cfg = config_json["model"] if "model" in config_json else config_json

    # For "full" checkpoints, LLM weights come from the checkpoint itself (skip loading base)
    llm_inference_mode = checkpoint_format == "full"

    overwatch.info(f"Building base model from config `{config_path}`")
    overwatch.info(
        f"vision=`{model_cfg['vision_backbone_id']}`, llm=`{model_cfg['llm_backbone_id']}`, "
        f"arch=`{model_cfg['arch_specifier']}`, checkpoint_format=`{checkpoint_format}`"
    )

    vision_backbone, _ = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=llm_inference_mode,
    )
    vlm = get_vlm(
        model_cfg["model_id"],
        model_cfg["arch_specifier"],
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=model_cfg.get("enable_mixed_precision_training", True),
    )

    checkpoint_format, loaded_vlm = _load_checkpoint_into_vlm(vlm, checkpoint, checkpoint_path)
    vlm = loaded_vlm
    vlm.requires_grad_(False)
    vlm.eval()
    vlm.to(device, dtype=dtype)

    image = _load_image(cfg.image_source)
    is_align_only = checkpoint_format == "projector-only"

    prompt_builder = vlm.get_prompt_builder(system_prompt=cfg.system_prompt)
    system_prompt = prompt_builder.system_prompt

    overwatch.info(f"Loaded checkpoint from `{checkpoint_path}`")
    overwatch.info(_describe_image_inputs(vlm, image))

    def _build_prompt(user_message: str, *, is_first_turn: bool = True) -> str:
        """Build prompt text matching the training format.

        For align (projector-only) checkpoints the training data is just ``<image>\\n{caption}<eos>``
        so we feed ``<image>\\n`` and let the model generate the caption.
        For finetuned checkpoints we use the PurePromptBuilder ``In: ... Out: `` template.
        """
        if is_align_only:
            return "<image>\n"
        if is_first_turn:
            user_message = f"<image>\n{user_message}"
        prompt_builder.add_turn(role="human", message=user_message)
        return prompt_builder.get_prompt()

    if cfg.prompt is not None:
        prompt_text = _build_prompt(cfg.prompt)
        generated_text = vlm.generate(
            image,
            prompt_text,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            min_length=cfg.min_length,
        )
        print(generated_text)
        return

    if is_align_only:
        print(
            "[*] Multi-vision inference REPL ready (align / projector-only mode):\n"
            "       => Prompt: <image>\\n  (model generates caption directly)\n"
            f"       => Image source: `{cfg.image_source}`\n"
            f"       => Checkpoint format: `{checkpoint_format}`\n===\n"
        )
    else:
        print(
            "[*] Multi-vision inference REPL ready:\n"
            f"       => Prompt template:\n\n{prompt_builder.get_potential_prompt('<INSERT PROMPT HERE>')}\n\n"
            f"       => Image source: `{cfg.image_source}`\n"
            f"       => Checkpoint format: `{checkpoint_format}`\n===\n"
        )

    repl_prompt = (
        "|=>> Enter (i)mage source(URL/path), (p)rompt template, (q)uit, or any other key to start chatting: "
    )
    while True:
        user_input = input(repl_prompt)

        if user_input.lower().startswith("q"):
            print("\n|=>> Received (q)uit signal => Exiting...")
            return

        if user_input.lower().startswith("i"):
            source = input("\n|=>> Enter Image URL or local path: ")
            image = _load_image(source)
            if not is_align_only:
                prompt_builder = vlm.get_prompt_builder(system_prompt=system_prompt)
            overwatch.info(_describe_image_inputs(vlm, image))
            continue

        if user_input.lower().startswith("p"):
            if is_align_only:
                print("\n|=>> Align-only checkpoint does not support prompt templates!")
                continue
            if system_prompt is None:
                print("\n|=>> Model does not support `system_prompt`!")
                continue

            system_prompt = input("\n|=>> Enter New System Prompt: ")
            prompt_builder = vlm.get_prompt_builder(system_prompt=system_prompt)
            print(
                "\n[*] Set New System Prompt:\n"
                f"    => Prompt Template:\n{prompt_builder.get_potential_prompt('<INSERT PROMPT HERE>')}\n\n"
            )
            continue

        if is_align_only:
            # Align-only: no multi-turn -- just caption the current image once
            print("\n[*] Generating caption for current image...\n")
            prompt_text = _build_prompt("")
            generated_text = vlm.generate(
                image,
                prompt_text,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                min_length=cfg.min_length,
            )
            print(f"\t|=>> VLM Response >>> {generated_text}\n")
            continue

        print("\n[*] Entering chat session - CTRL-C to reset conversation!\n===\n")
        try:
            first_turn = True
            while True:
                message = input("|=>> Enter Prompt: ")
                prompt_text = _build_prompt(message, is_first_turn=first_turn)
                if first_turn:
                    first_turn = False

                generated_text = vlm.generate(
                    image,
                    prompt_text,
                    do_sample=cfg.do_sample,
                    temperature=cfg.temperature,
                    max_new_tokens=cfg.max_new_tokens,
                    min_length=cfg.min_length,
                )
                prompt_builder.add_turn(role="gpt", message=generated_text)
                print(f"\t|=>> VLM Response >>> {generated_text}\n")
        except KeyboardInterrupt:
            print("\n===\n")
            continue


if __name__ == "__main__":
    generate_multivision()
