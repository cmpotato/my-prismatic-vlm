"""
generate_projector.py

CLI script for interactive generation using:
1) a base Prismatic model configuration/backbones, and
2) a projector-only checkpoint (e.g., align-stage output).

Run with:
  python scripts/generate_projector.py \
    --projector_checkpoint <PATH-TO-CHECKPOINT-OR-RUN_DIR> \
    --model_config <OPTIONAL-PATH-TO-CONFIG.JSON>
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
from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

DEFAULT_IMAGE_SOURCE = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
)


@dataclass
class GenerateProjectorConfig:
    # Path to projector checkpoint or run directory containing `checkpoints/latest-checkpoint.pt`.
    projector_checkpoint: Union[str, Path]

    # Optional explicit model config path (expects a training config.json with top-level "model" key).
    model_config: Optional[Union[str, Path]] = None

    # HF Hub credential (used when base LLM/vision need downloads).
    hf_token: Union[str, Path] = Path(".hf_token")

    # Default generation parameters.
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 512
    min_length: int = 1


def _resolve_checkpoint_path(projector_checkpoint: Union[str, Path]) -> Path:
    path = Path(projector_checkpoint)
    if path.is_dir():
        run_candidate = path / "checkpoints" / "latest-checkpoint.pt"
        direct_candidate = path / "latest-checkpoint.pt"
        if run_candidate.exists():
            return run_candidate
        if direct_candidate.exists():
            return direct_candidate
        raise FileNotFoundError(f"Could not find `latest-checkpoint.pt` under `{path}`")

    if not path.exists():
        raise FileNotFoundError(f"Projector checkpoint does not exist: `{path}`")
    return path


def _resolve_model_config_path(checkpoint_path: Path, model_config: Optional[Union[str, Path]]) -> Path:
    if model_config is not None:
        config_path = Path(model_config)
        if not config_path.exists():
            raise FileNotFoundError(f"Model config does not exist: `{config_path}`")
        return config_path

    # Default inference: if checkpoint is under <run_dir>/checkpoints/*.pt => use <run_dir>/config.json
    if checkpoint_path.parent.name == "checkpoints":
        candidate = checkpoint_path.parent.parent / "config.json"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not infer model config. Please pass `--model_config /path/to/config.json` explicitly."
    )


def _extract_projector_state_dict(ckpt_obj: dict) -> dict:
    # Common format in this repo: {"model": {"projector": {...}}}
    if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        model_dict = ckpt_obj["model"]
        if "projector" in model_dict and isinstance(model_dict["projector"], dict):
            return model_dict["projector"]

        # Fallback: full-model dictionary where keys are prefixed with "projector."
        projector_keys = [k for k in model_dict.keys() if k.startswith("projector.")]
        if projector_keys:
            return {k.removeprefix("projector."): model_dict[k] for k in projector_keys}

    # Also support direct projector state dict.
    if all(isinstance(k, str) for k in ckpt_obj.keys()) and any(k.startswith("projector.") for k in ckpt_obj.keys()):
        return {k.removeprefix("projector."): v for k, v in ckpt_obj.items() if k.startswith("projector.")}

    raise ValueError("Could not locate projector weights in checkpoint.")


def _load_image(source: str) -> Image.Image:
    if source.startswith("http://") or source.startswith("https://"):
        return Image.open(requests.get(source, stream=True, timeout=30).raw).convert("RGB")

    return Image.open(Path(source)).convert("RGB")


@draccus.wrap()
def generate_projector(cfg: GenerateProjectorConfig) -> None:
    checkpoint_path = _resolve_checkpoint_path(cfg.projector_checkpoint)
    config_path = _resolve_model_config_path(checkpoint_path, cfg.model_config)

    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with open(config_path, "r") as f:
        config_json = json.load(f)
    model_cfg = config_json["model"] if "model" in config_json else config_json

    overwatch.info(f"Building base model from config `{config_path}`")
    overwatch.info(
        f"Base model => vision: `{model_cfg['vision_backbone_id']}`, llm: `{model_cfg['llm_backbone_id']}`, "
        f"arch: `{model_cfg['arch_specifier']}`"
    )

    # Important: inference_mode=False loads pretrained text-model weights for base LLM.
    vision_backbone, _ = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )
    llm_backbone, _ = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=False,
    )
    vlm = get_vlm(
        model_cfg["model_id"],
        model_cfg["arch_specifier"],
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=model_cfg.get("enable_mixed_precision_training", True),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    projector_state = _extract_projector_state_dict(ckpt)
    missing, unexpected = vlm.projector.load_state_dict(projector_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Projector load mismatch: missing={missing}, unexpected={unexpected}")

    vlm.requires_grad_(False)
    vlm.eval()
    vlm.to(device, dtype=dtype)
    overwatch.info(f"Loaded projector checkpoint from `{checkpoint_path}`")

    image = _load_image(DEFAULT_IMAGE_SOURCE)
    prompt_builder = vlm.get_prompt_builder()
    system_prompt = prompt_builder.system_prompt

    print(
        "[*] Projector-only inference REPL ready:\n"
        f"       => Prompt template:\n\n{prompt_builder.get_potential_prompt('<INSERT PROMPT HERE>')}\n\n"
        f"       => Default image source: `{DEFAULT_IMAGE_SOURCE}`\n===\n"
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
            prompt_builder = vlm.get_prompt_builder(system_prompt=system_prompt)
            continue

        if user_input.lower().startswith("p"):
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

        print("\n[*] Entering chat session - CTRL-C to reset conversation!\n===\n")
        try:
            while True:
                message = input("|=>> Enter Prompt: ")
                prompt_builder.add_turn(role="human", message=message)
                prompt_text = prompt_builder.get_prompt()

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
    generate_projector()
