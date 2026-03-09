# Align 阶段训练计划（DINOv3 + Qwen3VL TextModel，仅训练 Projector）

本文档是本仓库中 `align` 阶段训练的实用操作手册。

## 1. 目标

总目标保持不变：`stage=align` 只训练 `projector`，并冻结视觉骨干和 LLM 骨干。

本次方案改为：
- 视觉骨干：`DINOv3`
- 语言骨干：`Qwen3VL` 中抽取的 `TextModel`

关键参考：
- `prismatic/models/vlms/prismatic.py:129`
- `prismatic/models/vlms/prismatic.py:140`
- `prismatic/models/vlms/prismatic.py:142`
- `prismatic/models/vlms/prismatic.py:145`

## 2. 前置条件 ✅

1. CUDA 和 GPU 可用。
2. 当前环境中 `torchrun` 可正常运行。
3. 已准备 Hugging Face Token（如使用 gated/private 模型）。
4. 数据集路径有效。

脚本入口参考：
- `scripts/pretrain.py:18`
- `scripts/pretrain.py:117`

## 3. Hugging Face Token 配置 ✅

二选一：

1. 在仓库根目录放置 `.hf_token`（单行文本 token）。
2. 设置环境变量，并通过 `--hf_token` 传入变量名。

参考：
- `scripts/pretrain.py:73`
- `scripts/pretrain.py:74`
- `scripts/pretrain.py:128`

## 4. 模型适配改动清单（DINOv3 + Qwen3VL TextModel）

当前仓库默认注册的是 DINOv2 / CLIP / SigLIP 与 LLaMA2 / Mistral / Phi，
因此要先完成以下适配，才能启动新方案训练：

1. 视觉骨干接入（DINOv3）
- 新增或扩展视觉 backbone 封装（建议放在 `prismatic/models/backbones/vision/`）。
- 在 `prismatic/models/materialize.py` 的 `VISION_BACKBONES` 注册 `dinov3-*` 标识。

2. 语言骨干接入（Qwen3VL TextModel）
- 新增 LLM backbone 封装（建议放在 `prismatic/models/backbones/llm/`），只暴露 text model 作为解码骨干。
- 在 `prismatic/models/materialize.py` 的 `LLM_BACKBONES` 注册 `qwen3vl-text-*` 标识。

3. 组合模型配置
- 在 `prismatic/conf/models.py` 新增一个 model config（例如 `dinov3-qwen3vltext-align`）。
- 配置项至少包括：
  - `vision_backbone_id`
  - `llm_backbone_id`
  - `arch_specifier`
  - `align_*` 一组训练超参数
- 同时在 `ModelRegistry` 中注册该 model config，保证 `--model.type` 可识别。

4. 冻结行为核验
- 训练日志中应明确显示 `align` 阶段仅 projector 可训练。
- 核验入口：`vlm.freeze_backbones(cfg.stage)`（`scripts/pretrain.py:169`）。

## 5. 数据集配置检查

检查配置中的根目录和 align 阶段组件：
- `dataset_root_dir`
- `align_stage_components`

参考：
- `prismatic/conf/datasets.py:21`
- `prismatic/conf/datasets.py:27`
- `prismatic/conf/datasets.py:35`

如果你的数据集不在默认路径，可通过以下参数覆盖：
- `--dataset.dataset_root_dir /your/dataset/root`

### 5.0 本计划当前训练命令实际使用的数据集（align）

本计划中的冒烟与正式 `align` 命令使用：
- `--dataset.type llava-v15`
- `--dataset.dataset_root_dir /home/max/my-prismatic-vlm/data`

对应到本仓库当前配置，实际读取为：
- 训练集：`/home/max/my-prismatic-vlm/data/download/llava-laion-cc-sbu-558k/chat_train.json`
- 验证集：`/home/max/my-prismatic-vlm/data/download/llava-laion-cc-sbu-558k/chat_val.json`
- 图片根目录：`/home/max/my-prismatic-vlm/data/download/llava-laion-cc-sbu-558k/`

当前本地数据规模（用于 align）：
- `chat_train.json`: 482,558 条
- `chat_val.json`: 75,570 条

说明：
- `llava-v1.5-instruct/llava_v1_5_mix665k.json` 属于 `finetune` 阶段数据，不参与本计划的 `align` 训练。

### 5.1 使用 VQA 数据生成 Align 所需 `chat.json`（执行计划）

你当前在 `/home/max/my-prismatic-vlm/data` 已准备 VQA 原始数据，目标是产出 `align` 阶段可直接读取的数据形态：
- 一个 `chat.json`
- 一个图片目录

并满足 `AlignDataset` 约束：
- 每条样本必须含 `image` 与 `conversations`
- `conversations` 长度固定为 2（human + gpt）

#### 5.1.1 输入数据（VQA）

最少需要以下 3 类文件：

1. 问题文件（questions）
- `.../v2_OpenEnded_mscoco_train2014_questions.json`
- `.../v2_OpenEnded_mscoco_val2014_questions.json`

2. 标注文件（annotations）
- `.../v2_mscoco_train2014_annotations.json`
- `.../v2_mscoco_val2014_annotations.json`

3. 图片目录
- `train2014/` 与 `val2014/`（建议都准备）

#### 5.1.2 映射规则（VQA -> align chat）

对每个 `question_id` 进行 questions + annotations 关联，生成一条样本：

1. `image`
- 由 `image_id` 和 split 生成文件名：
  - `train`: `COCO_train2014_{image_id:012d}.jpg`
  - `val`: `COCO_val2014_{image_id:012d}.jpg`
- `image` 字段写相对路径（相对后续图片根目录）。

2. `conversations[0]`（human）
- 固定模板（建议）：
  - `"Answer the question based on the image.\\n<image>\\nQuestion: {question}"`

3. `conversations[1]`（gpt）
- 使用 `annotations.multiple_choice_answer` 作为监督答案。

4. 过滤规则
- 图片文件不存在：丢弃
- `question` 为空或 `multiple_choice_answer` 为空：丢弃

说明：VQA 是问答监督，不是图像描述。用它做 align 可跑通流程，但语义目标与 LLaVA caption 对齐任务不同，效果预期应单独评估。

#### 5.1.3 输出目录（建议）

为了复用当前 `llava-v15` 的 `align_stage_components` 结构，建议产出到：

```text
/home/max/my-prismatic-vlm/data/vqa-align/
  download/
    llava-laion-cc-sbu-558k/
      chat.json
      train2014/...
      val2014/...
```

其中：
- `chat.json` 为转换结果
- 图片可优先采用软链接（减少磁盘占用），不建议重复复制

#### 5.1.4 实施步骤

1. 新增转换脚本（建议路径）
- `scripts/data/convert_vqa_to_align_chat.py`

2. 脚本参数建议
- `--questions-train`
- `--questions-val`
- `--annotations-train`
- `--annotations-val`
- `--images-train-dir`
- `--images-val-dir`
- `--output-chat-json`
- `--output-image-root`
- `--link-images`（是否软链接）

3. 运行后生成 `chat.json` + 图片目录，并打印：
- 样本总数
- train/val 各自样本数
- 过滤数（缺图、空文本等）

#### 5.1.5 数据校验（必须）

运行前后执行以下核验：

1. JSON 可读且为 list：
- `python -c "import json; x=json.load(open('chat.json')); print(type(x).__name__, len(x))"`

2. 抽样检查字段：
- 每条必须有 `image`、`conversations`
- `len(conversations) == 2`

3. 抽样检查图片路径可访问：
- `os.path.exists(image_root / sample['image']) == True`

#### 5.1.6 训练接入

转换完成后，直接使用：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --stage align \
  --model.type dinov3-qwen3vltext-align \
  --model.align_max_steps 20 \
  --dataset.type llava-v15 \
  --dataset.dataset_root_dir /home/max/my-prismatic-vlm/data \
  --run_root_dir /home/max/my-prismatic-vlm/runs \
  --trackers '["jsonl"]'
```

如果后续不想复用 `llava-v15` 目录约定，可在 `prismatic/conf/datasets.py` 新增一个 `vqa-align` 数据集配置项。

## 6. Align 需要调的超参数

`align` 使用 `model.align_*` 字段：
- `align_epochs`
- `align_max_steps`
- `align_global_batch_size`
- `align_per_device_batch_size`
- `align_learning_rate`
- `align_weight_decay`
- `align_max_grad_norm`
- `align_lr_scheduler_type`
- `align_warmup_ratio`
- `align_train_strategy`

参考：
- `prismatic/conf/models.py:37`
- `prismatic/conf/models.py:49`
- `prismatic/conf/models.py:89`
- `prismatic/conf/models.py:100`

## 7. 冒烟测试（8卡、短跑）✅

已验证成功的冒烟命令（20 step）：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --stage align \
  --model.type dinov3-qwen3vltext-align \
  --model.align_max_steps 20 \
  --dataset.type llava-v15 \
  --dataset.dataset_root_dir /home/max/my-prismatic-vlm/data \
  --run_root_dir /home/max/my-prismatic-vlm/runs \
  --trackers '["jsonl"]'
```

原因：
- `--stage align` 会触发 `freeze_backbones("align")`，仅训练 projector。

参考：
- `scripts/pretrain.py:168`
- `scripts/pretrain.py:169`
- `scripts/pretrain.py:225`

## 8. 正式训练（多卡）

冒烟测试通过后，使用以下正式训练命令（`align_max_steps=300000`，显式唯一 `run_id`，并切换为二阶段模型配置）：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --stage align \
  --model.type dinov3-qwen3vltext-2stage \
  --model.align_max_steps 300000 \
  --dataset.type llava-v15 \
  --dataset.dataset_root_dir /home/max/my-prismatic-vlm/data \
  --eval_every_n_steps 100 \
  --run_root_dir /home/max/my-prismatic-vlm/runs \
  --run_id dinov3-qwen3vltext-2stage-align-s300k-x7-20260309-1900 \
  --trackers '["jsonl"]'
```

说明：
- `align` 阶段本身按当前训练策略仍是每 100 step 保存一次 checkpoint。
- `--eval_every_n_steps 100` 会每 100 step 额外计算一次 eval loss。
- 建议在后续 `finetune` 阶段始终显式传 `--pretrained_checkpoint`，避免依赖自动发现路径（run 目录命名变化时容易失配）。例如：

```bash
--pretrained_checkpoint /home/max/my-prismatic-vlm/runs/dinov3-qwen3vltext-2stage-align-s300k-x7-20260309-1900/checkpoints/latest-checkpoint.pt
```

### 8.1 本任务实际生效超参数（对应上面的正式训练命令）

```text
stage: align
nproc_per_node (torchrun): 8
run_id: dinov3-qwen3vltext-2stage-align-s300k-x7-20260309-1900
run_root_dir: /home/max/my-prismatic-vlm/runs
seed: 7
trackers: ('jsonl',)
eval_every_n_steps: 100
eval_max_batches: None

[model]
model.type: dinov3-qwen3vltext-2stage
arch_specifier: gelu-mlp
vision_backbone_id: dinov3-vit-l
llm_backbone_id: qwen3vl-text-8b-instruct
image_resize_strategy: resize-naive
llm_max_length: 32768
enable_gradient_checkpointing: True
enable_mixed_precision_training: True
reduce_in_full_precision: False

[align optimization]
align_epochs: 1
align_max_steps: 300000
align_global_batch_size: 256
align_per_device_batch_size: 16
grad_accumulation_steps (derived): 2
align_learning_rate: 0.001
align_weight_decay: 0.0
align_max_grad_norm: 1.0
align_lr_scheduler_type: linear-warmup+cosine-decay
align_warmup_ratio: 0.03
align_train_strategy: fsdp-shard-grad-op

[dataset]
dataset.type: llava-v15
dataset_root_dir: /home/max/my-prismatic-vlm/data
align_stage_components:
  - download/llava-laion-cc-sbu-558k/chat_train.json
  - download/llava-laion-cc-sbu-558k
align_val_stage_components:
  - download/llava-laion-cc-sbu-558k/chat_val.json
  - download/llava-laion-cc-sbu-558k
```

## 9. 训练中监控项

1. 训练 loss 应下降且整体稳定。
2. GPU 显存应处于可控范围。
3. 日志应显示 projector 可训练，vision 与 LLM 冻结。

## 10. 训练风险与注意事项（当前方案）

1. VQA 在 `align` 阶段会丢弃问题文本
- `AlignDataset` 只使用 `conversations[-1]` 作为监督目标，human 问题不会进入训练目标。
- 对 VQA 数据来说，这更接近“看图直接生成短答案”，不是“图+问题 -> 答案”。
- 参考：`prismatic/preprocessing/datasets/datasets.py:51`，`prismatic/preprocessing/datasets/datasets.py:72`。

2. Qwen3VL tokenizer 无 BOS，与当前图像 token 插入假设存在偏差
- 现有多模态拼接逻辑默认把图像 token 插入到首 token 后，并把首 token 的 label 置为 `IGNORE`。
- Qwen3VL tokenizer 的 `bos_token_id` 为 `None`，会导致真实首文本 token 被忽略监督。
- VQA 短答案比例高时，这会进一步削弱有效训练信号。
- 参考：`prismatic/preprocessing/datasets/datasets.py:84`，`prismatic/models/vlms/prismatic.py:391`，`prismatic/models/vlms/prismatic.py:408`。

3. 长跑 checkpoint 与恢复语义
- 当前已改为每 100 step 保存一次 checkpoint（并保留 `max_steps` 到点保存）。
- checkpoint 默认只保存可训练模块（`align` 阶段通常仅 projector），属于预期行为。
- 若训练中断，请确认从正确的 `run_id/checkpoints/` 手动续跑路径恢复。

参考：
- `prismatic/models/vlms/prismatic.py:151`
- `prismatic/models/vlms/prismatic.py:153`

## 10. Checkpoint 校验

align 阶段保存的模型状态应以可训练模块为主（重点是 projector）。

参考：
- `prismatic/training/strategies/ddp.py:37`
- `prismatic/training/strategies/ddp.py:40`
- `prismatic/training/strategies/fsdp.py:101`
- `prismatic/training/strategies/fsdp.py:110`
