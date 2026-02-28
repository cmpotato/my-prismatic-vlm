"""
pretrain.py

Pretraining script for Prismatic VLM pretraining in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed training across GPUs. By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).


Notes & Prerequisites:
    - We're loading LLaMa-2 (and possibly other) gated models from HuggingFace (HF Hub); these require an auth_token.
      For LLaMa-2, make sure to first get Meta approval, then fill out the form at the top of the HF LLaMa-2 page:
        => Link: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
        => Generate Token (from `huggingface.co`): Settings / Access Tokens / New "Read" Token
        => Set `cfg.hf_token` to file path with token (as single line text file) or environment variable name

    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/pretrain.py
    - [Multi-Node/AWS Sagemaker] Depends on your individual setup; file an issue if you have trouble!
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import Subset

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing import get_dataset_and_collator
from prismatic.training import Metrics, get_train_strategy
from prismatic.util import set_global_seed

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PretrainConfig:
    # fmt: off

    # ModelConfig (`prismatic/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.PRISM_DINOSIGLIP_7B.model_id)
    )

    # DatasetConfig (`prismatic/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id)
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    # ---
    stage: str = "finetune"                                         # Pretraining Stage in < align | finetune >
    pretrained_checkpoint: Optional[Path] = None                    # Pretrained Checkpoint to Load (for `finetune`)
                                                                    #   if None =>> will match on (run_dir / `align`)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    # wandb_project: str = "prismatic"                                # Name of W&B project (default: `prismatic`)
    # wandb_entity: Optional[str] = None                              # Name of W&B entity (default: None)
    wandb_project: str = "onyx-vlms"
    wandb_entity: str = "stanford-voltron"

    # Evaluation Parameters (disabled if `eval_every_n_steps` is None)
    eval_every_n_steps: Optional[int] = 200
    eval_max_batches: Optional[int] = None
    finetune_val_ratio: float = 0.2

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage` in {"align", "finetune"}."""
        if not (0.0 <= self.finetune_val_ratio < 1.0):
            raise ValueError("`finetune_val_ratio` must satisfy 0.0 <= ratio < 1.0")

        if self.stage == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

        elif self.stage.endswith("finetune"):
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

    # fmt: on


@draccus.wrap()
def pretrain(cfg: PretrainConfig) -> None:
    overwatch.info("Prismatic VLM Training :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := (overwatch.local_rank()))
    torch.cuda.empty_cache()

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    if (dataset_id := cfg.dataset.dataset_id) == "llava-v15":
        cfg.run_id = f"{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    else:
        cfg.run_id = f"{dataset_id}+{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Life is like a prism; what you see depends on how you turn the glass."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    if overwatch.is_rank_zero():
        # Additionally save a JSON version of the config
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load Vision Backbone --> on CPU, in Full Precision (initializing model, image_transform via TIMM)
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )

    # Load LLM Backbone --> on CPU, in Full Precision (initializing Tokenizer + handling special tokens if necessary)
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token
    )

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating PrismaticVLM `{model_id}` for Training Stage = `{cfg.stage}`")
    vlm = get_vlm(
        model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    )

    # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.freeze_backbones(cfg.stage)

    # Load Weights from Checkpoint (depends on stage, config)
    overwatch.info(f"Invoking `VLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    # Optional LoRA injection for finetune stage.
    if cfg.model.finetune_use_lora:
        if cfg.stage != "finetune":
            raise ValueError("LoRA is currently supported only when `--stage finetune`.")
        if not hasattr(vlm.llm_backbone, "enable_lora"):
            raise ValueError(
                f"LLM backbone `{cfg.model.llm_backbone_id}` does not implement `enable_lora(...)`."
            )

        overwatch.info(f"Invoking `LLM.enable_lora()` for `{cfg.model.llm_backbone_id}`", ctx_level=1)
        vlm.llm_backbone.enable_lora(
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            target_modules=cfg.model.lora_target_modules,
        )

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")
    train_dataset, collator = get_dataset_and_collator(
        cfg.stage,
        cfg.dataset,
        image_transform,
        tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        default_image_resolution=vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side,
        split="train",
    )

    val_dataset = None
    if cfg.eval_every_n_steps is not None and cfg.eval_every_n_steps > 0:
        if cfg.stage == "align":
            if cfg.dataset.align_val_stage_components is None:
                overwatch.warning(
                    "Validation is enabled, but `dataset.align_val_stage_components` is not set; skipping val loss."
                )
            else:
                overwatch.info(f"Creating Validation Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")
                val_dataset, _ = get_dataset_and_collator(
                    cfg.stage,
                    cfg.dataset,
                    image_transform,
                    tokenizer,
                    prompt_builder_fn=llm_backbone.prompt_builder_fn,
                    default_image_resolution=vision_backbone.default_image_resolution,
                    padding_side=tokenizer.padding_side,
                    split="val",
                )

        elif cfg.stage.endswith("finetune"):
            if cfg.finetune_val_ratio <= 0:
                overwatch.warning("`finetune_val_ratio <= 0`; skipping finetune validation split.")
            else:
                n_total = len(train_dataset)
                if n_total < 2:
                    overwatch.warning(
                        f"Dataset has only {n_total} example(s); skipping finetune validation split."
                    )
                else:
                    n_val = int(n_total * cfg.finetune_val_ratio)
                    if n_val <= 0:
                        n_val = 1
                    if n_val >= n_total:
                        n_val = n_total - 1

                    split_generator = torch.Generator()
                    split_generator.manual_seed(cfg.seed)
                    indices = torch.randperm(n_total, generator=split_generator).tolist()
                    val_indices, train_indices = indices[:n_val], indices[n_val:]

                    full_train_dataset = train_dataset
                    train_dataset = Subset(full_train_dataset, train_indices)
                    val_dataset = Subset(full_train_dataset, val_indices)
                    overwatch.info(
                        f"Created finetune train/val split from training data "
                        f"(train={len(train_dataset)}, val={len(val_dataset)}, val_ratio={cfg.finetune_val_ratio:.2f})"
                    )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        save_lora_adapter_only=cfg.model.finetune_save_lora_adapter_only,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        cfg.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Log one-shot run metadata so trackers capture all key run context.
    total_params = sum(param.numel() for param in vlm.parameters())
    trainable_params = sum(param.numel() for param in vlm.parameters() if param.requires_grad)
    stage_prefix = cfg.stage.capitalize()
    metrics.log(
        0,
        metrics={
            f"{stage_prefix}/Step": 0,
            f"{stage_prefix}/Train Examples": len(train_dataset),
            f"{stage_prefix}/Val Examples": len(val_dataset) if val_dataset is not None else 0,
            f"{stage_prefix}/Eval Every N Steps": cfg.eval_every_n_steps if cfg.eval_every_n_steps is not None else -1,
            f"{stage_prefix}/Eval Max Batches": cfg.eval_max_batches if cfg.eval_max_batches is not None else -1,
            f"{stage_prefix}/Global Batch Size": cfg.global_batch_size,
            f"{stage_prefix}/Per Device Batch Size": cfg.per_device_batch_size,
            f"{stage_prefix}/Grad Accumulation Steps": train_strategy.grad_accumulation_steps,
            f"{stage_prefix}/Max Steps": cfg.max_steps if cfg.max_steps is not None else -1,
            f"{stage_prefix}/Total Parameters": total_params,
            f"{stage_prefix}/Trainable Parameters": trainable_params,
            f"{stage_prefix}/Trainable Parameter Ratio": trainable_params / total_params,
        },
    )

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(
        train_dataset,
        collator,
        metrics,
        val_dataset=val_dataset,
        eval_every_n_steps=cfg.eval_every_n_steps,
        eval_max_batches=cfg.eval_max_batches,
        stage=cfg.stage,
        seed=cfg.seed,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    pretrain()
