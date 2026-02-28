# Qwen Two-Stage Projector 关键改动日志

日期: 2026-03-01
作者: Codex
范围: `dinov3 + qwen3vl-text` 微调链路

## 1. 背景与问题

在现有配置中，`model.type=dinov3-qwen3vltext-align` 实际使用 `arch_specifier=no-align+gelu-mlp`。
这会触发 `PrismaticVLM.load_from_checkpoint()` 的 `no-align` 分支，导致 `stage=finetune` 不加载 projector 预训练权重。

结果:

1. 虽然可以训练，但 projector 不是“对齐后再微调”，而是随机初始化参与训练。
2. `--pretrained_checkpoint` 在该 model type 下对 projector 加载不生效（语义上冗余）。

## 2. 关键目标

在不破坏现有 one-stage 路径的前提下，新增可用的 two-stage 路径，使得:

1. 可先跑 `stage=align` 训练 projector。
2. 再跑 `stage=finetune` 时加载 align projector 权重。
3. 现有 `dinov3-qwen3vltext-align` 行为保持不变（避免影响已跑实验）。

## 3. 代码改动详情

### 3.1 新增 Two-Stage ModelConfig

文件: `prismatic/conf/models.py`

新增 dataclass:

1. `Prism_DINOv3_Qwen3VLText_8B_TwoStage`
2. `model_id = "dinov3-qwen3vltext-2stage"`
3. `arch_specifier = "gelu-mlp"`
4. 继承自 `Prism_DINOv3_Qwen3VLText_8B`，复用同一套视觉/语言骨干与基础超参。

设计要点:

1. 不使用 `no-align+...`，从而允许 `load_from_checkpoint()` 在 finetune 阶段执行 projector 加载逻辑。
2. 与旧配置并存，避免破坏历史命令和结果可复现性。

### 3.2 注册新 model type

文件: `prismatic/conf/models.py`

在 `ModelRegistry` 中新增:

1. `PRISM_DINOV3_QWEN3VLTEXT_8B_2STAGE = Prism_DINOv3_Qwen3VLText_8B_TwoStage`

结果:

1. CLI 可直接使用 `--model.type dinov3-qwen3vltext-2stage`。

## 4. 行为变化矩阵

### 4.1 保持不变（旧 one-stage）

`--model.type dinov3-qwen3vltext-align`

1. `arch_specifier = no-align+gelu-mlp`
2. finetune 不强制依赖 align projector
3. `pretrained_checkpoint` 对 projector 加载无效（按代码分支设计）

### 4.2 新增能力（two-stage）

`--model.type dinov3-qwen3vltext-2stage`

1. `arch_specifier = gelu-mlp`
2. 支持标准两阶段:
   - `stage=align` 产出 projector checkpoint
   - `stage=finetune` 通过 `--pretrained_checkpoint` 加载 projector

## 5. 推荐命令模板

### 5.1 Align 阶段

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type dinov3-qwen3vltext-2stage \
  --dataset.type carpaint-binary \
  --stage align \
  --trackers '["jsonl"]' \
  --run_root_dir /home/max/my-prismatic-vlm/runs \
  --run_id carpaint-qwen-2stage-align-x7
```

### 5.2 Finetune 阶段（LoRA + 轻量保存）

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type dinov3-qwen3vltext-2stage \
  --dataset.type carpaint-binary \
  --stage finetune \
  --pretrained_checkpoint /home/max/my-prismatic-vlm/runs/dinov3-qwen3vltext-align+stage-align+x7/checkpoints/latest-checkpoint.pt \
  --model.finetune_use_lora true \
  --model.finetune_save_lora_adapter_only true \
  --model.finetune_max_steps 300000 \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 16 \
  --model.finetune_max_grad_norm 0 \
  --trackers '["jsonl"]' \
  --run_root_dir /home/max/my-prismatic-vlm/runs \
  --run_id carpaint-qwen-2stage-lora-ft-300k-bs128-pd16
```

## 6. 与精度问题的关系

本次改动不改变主训练数值路径，仅改变“是否加载 projector 预训练权重”的入口选择。

已知仍需注意:

1. FSDP 梯度裁剪在混合梯度 dtype 条件下可能报错，因此沿用 `--model.finetune_max_grad_norm 0`。
2. LoRA dtype 对齐逻辑（bfloat16）仍建议保留，不受本次 model type 新增影响。

## 7. 验证与自检

1. 已完成 `py_compile` 语法检查（`prismatic/conf/models.py`）。
2. 旧 model type 未改名、未删除，兼容历史训练命令。
3. 新 model type 已注册，可通过 CLI 选择。

## 8. 回滚方案

若需回滚该关键改动，仅需在 `prismatic/conf/models.py` 中删除:

1. `Prism_DINOv3_Qwen3VLText_8B_TwoStage` 类定义
2. `ModelRegistry` 中对应枚举项

其余训练链路代码无需调整。
