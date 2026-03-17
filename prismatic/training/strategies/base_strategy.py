"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        save_lora_adapter_only: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.save_lora_adapter_only = save_lora_adapter_only
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    @staticmethod
    def _get_finetune_modality_lengths(dataset: Dataset) -> list[tuple[bool, int]]:
        """
        Resolve `(is_multimodal, length)` metadata for finetune datasets.

        Supports `Subset` wrappers so train/val splits can still use split-modality batching.
        """
        if isinstance(dataset, Subset):
            parent_modality_lengths = TrainingStrategy._get_finetune_modality_lengths(dataset.dataset)
            return [parent_modality_lengths[idx] for idx in dataset.indices]

        if hasattr(dataset, "get_modality_lengths"):
            return dataset.get_modality_lengths()

        raise ValueError(
            f"Dataset of type `{type(dataset)}` does not expose `get_modality_lengths()` required for split-modality."
        )

    def run_evaluation(
        self,
        val_dataloader: DataLoader,
        eval_max_batches: Optional[int] = None,
    ) -> tuple[float, int, float]:
        """Compute mean validation loss across all processes."""
        eval_start_time = time.time()
        self.vlm.eval()
        device = torch.device("cuda", self.device_id)
        loss_sum = torch.zeros(1, device=device, dtype=torch.float32)
        n_batches = torch.zeros(1, device=device, dtype=torch.float32)

        with torch.no_grad():
            for val_idx, batch in enumerate(val_dataloader):
                if eval_max_batches is not None and val_idx >= eval_max_batches:
                    break

                with torch.autocast(
                    "cuda",
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision_training,
                ):
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        image_attention_mask=batch["image_attention_mask"],
                        labels=batch["labels"],
                        multimodal_indices=batch["multimodal_indices"],
                    )

                loss_sum += output.loss.detach().to(dtype=torch.float32)
                n_batches += 1

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)

        self.vlm.train()
        if n_batches.item() == 0:
            return float("nan"), 0, time.time() - eval_start_time

        return (loss_sum / n_batches).item(), int(n_batches.item()), time.time() - eval_start_time

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        val_dataset: Optional[Dataset] = None,
        eval_every_n_steps: Optional[int] = None,
        eval_max_batches: Optional[int] = None,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        checkpoint_every_steps = 100

        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = self._get_finetune_modality_lengths(dataset)
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        val_dataloader = None
        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=False,
                seed=seed,
                drop_last=False,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.per_device_batch_size,
                sampler=val_sampler,
                collate_fn=collator,
                num_workers=2,
                worker_init_fn=self.worker_init_fn,
            )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 1000000

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            image_attention_mask=batch["image_attention_mask"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip gradients only when enabled (`max_grad_norm > 0`).
                        if self.max_grad_norm > 0:
                            # This is custom, per-strategy because of DDP vs. FSDP locality-assumptions.
                            self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        did_eval_this_step = False
                        if (
                            val_dataloader is not None
                            and eval_every_n_steps is not None
                            and eval_every_n_steps > 0
                            and metrics.global_step % eval_every_n_steps == 0
                        ):
                            val_loss, val_batches, eval_time = self.run_evaluation(
                                val_dataloader, eval_max_batches=eval_max_batches
                            )
                            metrics.log(
                                metrics.global_step,
                                metrics={
                                    f"{stage.capitalize()}/Step": metrics.global_step,
                                    f"{stage.capitalize()}/Val Loss": val_loss,
                                    f"{stage.capitalize()}/Val Batches": val_batches,
                                    f"{stage.capitalize()}/Val Time": eval_time,
                                },
                            )
                            did_eval_this_step = True

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            # Always log a final validation loss before returning (if a val set exists) so short smoke
                            # runs still contain val metrics even when `eval_every_n_steps` > `max_steps`.
                            if val_dataloader is not None and not did_eval_this_step:
                                val_loss, val_batches, eval_time = self.run_evaluation(
                                    val_dataloader, eval_max_batches=eval_max_batches
                                )
                                metrics.log(
                                    metrics.global_step,
                                    metrics={
                                        f"{stage.capitalize()}/Step": metrics.global_step,
                                        f"{stage.capitalize()}/Val Loss": val_loss,
                                        f"{stage.capitalize()}/Val Batches": val_batches,
                                        f"{stage.capitalize()}/Val Time": eval_time,
                                    },
                                )

                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Periodic checkpointing for long runs.
                        if metrics.global_step % checkpoint_every_steps == 0:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                if val_dataloader is not None:
                    val_loss, val_batches, eval_time = self.run_evaluation(
                        val_dataloader, eval_max_batches=eval_max_batches
                    )
                    metrics.log(
                        metrics.global_step,
                        metrics={
                            f"{stage.capitalize()}/Step": metrics.global_step,
                            f"{stage.capitalize()}/Val Loss": val_loss,
                            f"{stage.capitalize()}/Val Batches": val_batches,
                            f"{stage.capitalize()}/Val Time": eval_time,
                        },
                    )

                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()
