# Qwen Finetune 兼容与 LoRA 改造计划

## 1. 目标

让当前仓库在以下组合下可稳定运行微调（`stage=finetune`）：

- 视觉骨干：`dinov3-vit-l`
- 语言骨干：`qwen3vl-text-8b-instruct`
- 数据格式：现有 LLaVA/Prismatic 风格 `conversations` JSON

分两阶段目标：

1. 先打通 Qwen 微调链路与损失回传（可运行、可复现）。
2. 在尽量保持原有框架逻辑下，加入 LoRA 微调能力（最小侵入）。

## 2. 当前已知阻塞

1. `FinetuneDataset` 对 tokenizer 类型判断过窄，仅显式支持：
- `LlamaTokenizerFast`
- `CodeGenTokenizerFast`

Qwen3VL 文本 tokenizer 当前为 `Qwen2Tokenizer`，会在数据构建阶段报错。

2. Qwen 路线在 `finetune` 阶段没有经过完整实跑验证，缺少最小闭环 smoke 测试与可复现命令。

3. 当前无 LoRA/PEFT 接入；`stage=finetune` 默认训练 `llm_backbone + projector`，参数量较大。

## 3. 改造范围

仅做与“Qwen 微调可运行 + LoRA 可用”直接相关改造：

1. 预处理层（`prismatic/preprocessing/datasets/datasets.py`）  
2. LLM 适配层（Qwen LoRA 注入与参数冻结策略）  
3. 必要的配置接入（数据集路径/训练参数覆盖，不改核心算法）  
4. 最小训练闭环验证（1 卡短步数）  

不在本轮处理：

1. 模型效果调优（提示词工程、损失加权、采样策略优化）  
2. 全量多卡长训稳定性优化  
3. 推理模板与产品化接口变更  
4. 大规模重构训练策略（DDP/FSDP 主逻辑保持不动）  

## 4. 具体执行步骤

### Step 1: 放宽 FinetuneDataset 的 tokenizer 兼容逻辑

文件：`prismatic/preprocessing/datasets/datasets.py`

执行项：

1. 将“仅白名单类名可通过”的分支改为“默认可通过，保留少量特例处理”。
2. 保留 Llama 系 `rstrip()` 兼容逻辑，Qwen 走通用分支。
3. 对未知 tokenizer 不直接抛错，改为可训练的通用行为。

产出标准：

1. `qwen3vl-text-8b-instruct` 不再因 tokenizer 类型报 `ValueError`。  
2. 原有 Llama/Phi 路线行为不回归（至少语法与基础构图不变）。  

### Step 2: 核验 Qwen 下的 prompt + tokenization 结果

执行项：

1. 对单条样本跑 `FinetuneDataset.__getitem__`，检查返回：
- `input_ids` 非空
- `labels` 中存在非 `-100` 的监督 token
- 多模态样本 `pixel_values` 正常

2. 对文本长度做边界检查：
- 确认不会把有效 `gpt` 监督全部截断

产出标准：

1. 单样本张量构建无异常。  
2. `labels` 中可学习 token 数量大于 0。  

### Step 3: 接入车漆数据集到 finetune 配置

执行项：

1. 通过 CLI 覆盖或新增 dataset 配置，使 `finetune_stage_components` 指向：
- `data/labeled_jpg/carpaint_finetune_chat.json`
- `data/labeled_jpg`

2. 确保路径解析和图片相对路径匹配：
- JSON 内 `image` 例如 `OK/xxxx.jpg`、`NG/xxxx.jpg`

产出标准：

1. DataLoader 能完整迭代车漆数据集。  
2. 不出现图片路径缺失错误。  

### Step 4: 基线微调（非 LoRA）最小闭环 smoke 训练

执行项：

1. 在单卡上运行 `stage=finetune` 短步数（例如 `max_steps=10~30`）。  
2. 显式指定 `pretrained_checkpoint`（若要接续 align projector）。  
3. 保存日志与 checkpoint，确认 loss 可下降或至少稳定为有限值。  

产出标准：

1. 训练启动成功并完成指定步数。  
2. 无 tokenizer 类型报错、无 batch 构造异常、无 NaN loss。  

参考命令（8 卡 smoke，关闭 grad clipping 以规避 FSDP 混合梯度 dtype 裁剪报错）：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type dinov3-qwen3vltext-align \
  --dataset.type carpaint-binary \
  --stage finetune \
  --pretrained_checkpoint runs/dinov3-qwen3vltext-align+stage-align+x7/checkpoints/latest-checkpoint.pt \
  --model.finetune_max_steps 2000 \
  --model.finetune_global_batch_size 8 \
  --model.finetune_per_device_batch_size 1 \
  --model.finetune_max_grad_norm 0 \
  --trackers '["jsonl"]' \
  --run_root_dir runs_smoke \
  --run_id smoke-finetune-carpaint-qwen-8gpu-eval
```

### Step 5: LoRA 最小侵入接入（保持原框架主流程）

执行项：

1. 新增 LoRA 配置项（优先放在 `ModelConfig`，默认关闭）：
- `finetune_use_lora: bool = False`
- `lora_r`, `lora_alpha`, `lora_dropout`
- `lora_target_modules`（可先给 Qwen 常用默认值）

2. 在 Qwen LLM backbone 中提供 LoRA 注入入口（例如 `enable_lora(...)`）：
- 基于 PEFT 注入到 `llm_backbone.llm`
- 注入后仅 LoRA 参数 `requires_grad=True`
- 原始 LLM 参数保持冻结

3. 尽量不改训练主干逻辑：
- 不新增新 stage，仍使用 `stage=finetune`
- `TrainingStrategy` / `DDP` / `FSDP` 主训练循环不改
- 继续依赖现有“按 `requires_grad` 聚合优化参数”的机制

产出标准：

1. 开启 `finetune_use_lora=True` 时，LLM 主体参数冻结、LoRA 参数可训练。  
2. 不开启 LoRA 时，行为与基线保持一致。  

### Step 6: LoRA 路径 smoke 训练

执行项：

1. 在与 Step 4 相同数据和命令模板下，只切换 LoRA 开关并短步训练。  
2. 记录 trainable 参数量、显存占用、loss 曲线起点。  
3. 验证 checkpoint 可正常保存并可继续训练。  

产出标准：

1. LoRA 路径可完成指定步数训练。  
2. trainable 参数量显著低于基线 finetune。  

### Step 7: 回归与提交

执行项：

1. 语法检查：`py_compile` 覆盖改动文件。  
2. 记录运行命令与结果摘要。  
3. 提交代码与说明文档。  

产出标准：

1. 改动可复现、可追踪。  
2. 提交信息包含“改了什么/为什么改/如何验证”。  

## 5. 风险与缓解

1. 风险：放宽 tokenizer 分支可能影响旧模型行为。  
缓解：保留 Llama 特殊处理，其他模型走“最小通用逻辑”；补单样本回归检查。

2. 风险：Qwen tokenizer 特性（BOS/EOS）引发标签错位。  
缓解：在 smoke 阶段打印首批样本 token/label 对齐统计。

3. 风险：数据集监督过短（yes/no）导致学习信号弱。  
缓解：先确保链路可跑，后续再评估任务提示与标签设计。

4. 风险：LoRA 与 FSDP 组合下参数包装或保存行为异常。  
缓解：先单卡短跑验证，再扩到多卡；必要时先在 DDP 路径验证 LoRA 正确性。

5. 风险：LoRA 目标模块命名与 Qwen 实际层名不匹配。  
缓解：注入前打印可选模块名，失败时给出明确报错并保留可配置覆盖。

## 6. 验收标准（分层）

### 6.1 基线兼容验收（必须）

1. `stage=finetune` + `qwen3vl-text-8b-instruct` 可正常进入训练循环。  
2. 车漆数据集可被正确读取并完成若干步训练。  
3. 全程无 tokenizer 类型相关异常。  
4. 产出 checkpoint 与基础训练日志。  

### 6.2 LoRA 验收（本轮新增）

1. 开启 LoRA 后可正常训练并完成 smoke 步数。  
2. trainable 参数量明显下降，且 loss 为有限值。  
3. 关闭 LoRA 时仍保持原 `finetune` 行为不变。  
4. 改造不引入新的训练 stage，不破坏现有训练框架入口。  

## 7. 执行记录（代码逻辑改动与问题）

### 7.1 已完成改动（截至 Step 3）

1. Step 1（Tokenizer 兼容）
- 文件：`prismatic/preprocessing/datasets/datasets.py`
- 改动：移除 `FinetuneDataset` 中对 tokenizer 的硬白名单限制。
- 新逻辑：保留 `LlamaTokenizerFast` 的 `rstrip()` 特例；其余 tokenizer（含 Qwen）走通用分支，不再抛 `ValueError`。

2. Step 3（车漆数据集接入）
- 文件：`prismatic/conf/datasets.py`
- 改动：新增 `CarPaint_Binary_Config`，`dataset_id = "carpaint-binary"`。
- 关键路径：
  - `finetune_stage_components = ("labeled_jpg/carpaint_finetune_chat.json", "labeled_jpg/")`
  - `dataset_root_dir = "data"`
- 注册：`DatasetRegistry.CARPAINT_BINARY`。

### 7.2 本地验证结果（Step 2 + Step 3）

1. Qwen tokenizer 单样本构建验证
- tokenizer 类型：`Qwen2Tokenizer`
- `sample_input_len = 24`
- `sample_trainable_tokens = 4`
- 首 16 条样本监督 token 最小值：`4`（均大于 0）

2. 数据集配置与 materialize 验证
- `DatasetConfig.get_choice_class("carpaint-binary")` 可正常实例化
- `get_dataset_and_collator(stage="finetune", dataset_cfg=carpaint-binary, ...)` 可正常构建
- materialized train dataset 长度：`2000`

### 7.3 发现的问题与处理

1. 问题：系统 `python` 环境缺少训练依赖。  
处理：统一使用仓库虚拟环境 `./.venv/bin/python` 进行验证命令。

2. 问题：当前环境打印 CUDA 初始化告警（`cudaGetDeviceCount`）。  
处理：本阶段仅做 CPU 侧数据构建验证，不影响 Step 1-3 的代码逻辑正确性；训练阶段前需在实际 GPU 训练环境复核。

### 7.4 HF Token 加载修复记录

1. 触发问题
- 现象：训练命令使用 `--hf_token .hf_token` 时，程序将 `.hf_token` 当作环境变量名读取，抛出 `KeyError: '.hf_token'`。
- 根因：原逻辑仅区分 `Path` 与 `str`，字符串分支固定按环境变量读取。

2. 修复方案（保持原框架逻辑）
- 新增统一解析函数：`prismatic/util/hf_utils.py::resolve_hf_token(...)`
- 解析优先级：
  - `Path` 对象 -> 按文件读取
  - 字符串且是现有文件路径 -> 按文件读取（支持 `.hf_token`）
  - 字符串命中环境变量名 -> 按环境变量读取
  - 字符串以 `hf_` 开头 -> 视作原始 token
- 其余情况抛出清晰错误信息，避免隐式失败。

3. 改动文件
- `prismatic/util/hf_utils.py`（新增）
- `prismatic/util/__init__.py`（导出 `resolve_hf_token`）
- `scripts/pretrain.py`（改为统一解析）
- `scripts/generate.py`（改为统一解析）
- `scripts/generate_projector.py`（改为统一解析）

4. 验证结论
- `resolve_hf_token(Path('.hf_token'))` 正常
- `resolve_hf_token('.hf_token')` 正常
- `resolve_hf_token('<ENV_NAME>')` 正常
- `resolve_hf_token('hf_xxx...')` 正常

### 7.5 Finetune 验证集切分与 Eval Loss 记录

1. 触发背景
- `finetune` 阶段原先仅使用训练集，`val_dataset` 仅在 `align` 阶段创建，导致无法在微调中记录 `Val Loss`。

2. 改动内容
- 文件：`scripts/pretrain.py`
  - 新增配置项：`finetune_val_ratio: float = 0.2`。
  - 当 `stage.endswith("finetune")` 且启用 eval 时，从训练集按固定随机种子切分 train/val（默认 80/20）。
  - 对极小数据集增加保护：确保 train/val 至少各保留 1 条（总样本 >= 2 时）。

- 文件：`prismatic/training/strategies/base_strategy.py`
  - 为 `split-modality` 采样新增 `Subset` 兼容：切分后训练子集仍可正确构建 `modality_lengths`。
  - `max_grad_norm <= 0` 时跳过梯度裁剪，避免 FSDP 混合梯度 dtype 裁剪报错阻塞 smoke 验证。

3. 行为结果
- `finetune` 阶段可在不新增训练 stage 的前提下，周期性计算并记录 `Finetune/Val Loss`。
- 训练集切分后仍保持原有 `split-modality` 批构造逻辑。

### 7.6 LoRA 最小侵入接入（代码已完成，待你实跑）

1. 配置层改动
- 文件：`prismatic/conf/models.py`
- 新增 LoRA 配置项（默认关闭，保持旧行为）：
  - `finetune_use_lora: bool = False`
  - `lora_r: int = 16`
  - `lora_alpha: int = 32`
  - `lora_dropout: float = 0.05`
  - `lora_target_modules: Tuple[str, ...]`

2. Qwen LLM 注入逻辑
- 文件：`prismatic/models/backbones/llm/qwen3vl_text.py`
- 新增 `enable_lora(...)`：
  - 惰性导入 `peft`（仅开启 LoRA 时需要）。
  - 先冻结 dense LLM 与 `lm_head`，再注入 LoRA。
  - 注入后仅 LoRA 参数保持可训练，并打印 trainable/total 参数量。

3. 训练入口接线
- 文件：`scripts/pretrain.py`
- 在 `vlm.load_from_checkpoint(...)` 后增加 LoRA 分支：
  - 当 `finetune_use_lora=True` 且 `stage=finetune` 时调用 `vlm.llm_backbone.enable_lora(...)`。
  - 非 `finetune` stage 显式报错，避免误用。

4. 依赖说明
- 文件：`pyproject.toml`
- 新增依赖：`peft>=0.12.0`（当前环境若未安装，实跑 LoRA 前需先安装）。
- 历史检查曾出现：`ModuleNotFoundError: No module named 'peft'`。
- 当前环境复核：`peft` 已可导入（版本 `0.18.1`）。

### 7.7 冒烟命令可行性确认（LoRA + Eval）

1. 静态核查结论
- `scripts/pretrain.py --help` 已确认参数存在：`--model.finetune_use_lora`、`--eval_every_n_steps`、`--finetune_val_ratio`。
- 当前默认值可用：`stage=finetune`、`eval_every_n_steps=200`、`finetune_val_ratio=0.2`。
- 当前环境依赖检查：`peft` 已可导入（版本 `0.18.1`）。
- `.hf_token` 文件存在，可不显式传 `--hf_token`。

2. 推荐 LoRA 冒烟命令（显式保留关键参数，含 eval）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type dinov3-qwen3vltext-align \
  --dataset.type carpaint-binary \
  --stage finetune \
  --pretrained_checkpoint runs/dinov3-qwen3vltext-align+stage-align+x7/checkpoints/latest-checkpoint.pt \
  --model.finetune_use_lora true \
  --model.finetune_max_steps 200 \
  --model.finetune_global_batch_size 8 \
  --model.finetune_per_device_batch_size 1 \
  --model.finetune_max_grad_norm 0 \
  --eval_every_n_steps 50 \
  --trackers '["jsonl"]' \
  --run_root_dir runs_smoke \
  --run_id smoke-finetune-carpaint-qwen-8gpu-lora-eval-v1
```

3. 合理性说明
- `max_steps=200` 更符合“冒烟”目标，能较快验证 LoRA 训练链路是否闭环。
- `eval_every_n_steps=50` 可在短跑内获得多次 `Val Loss` 观测点（50/100/150/200）。
- `finetune_max_grad_norm=0` 与当前 FSDP 混合梯度 dtype 现状兼容，可避免裁剪触发报错。

### 7.8 LoRA 注入报错修复（Qwen3VLTextModel + PEFT）

1. 触发问题
- 开启 LoRA 后在 `get_peft_model(...)` 报错：
  - `AttributeError: 'Qwen3VLTextModel' object has no attribute 'prepare_inputs_for_generation'`
- 根因：`TaskType.CAUSAL_LM` 会走 `PeftModelForCausalLM`，其构造要求底座模型实现 `prepare_inputs_for_generation`。

2. 修复方案
- 文件：`prismatic/models/backbones/llm/qwen3vl_text.py`
- 改动：LoRA 配置中的 `task_type` 从 `TaskType.CAUSAL_LM` 调整为 `TaskType.FEATURE_EXTRACTION`。
- 说明：当前注入目标是 decoder-only 的 `Qwen3VLTextModel`（非 `*ForCausalLM`），应避免走 CausalLM 专用封装分支。

3. 影响评估
- `finetune_use_lora=False` 路径不受影响。
- `finetune_use_lora=True` 仅修复 PEFT 包装层适配，不改变主训练循环（FSDP/DDP、优化器、eval、checkpoint）逻辑。

### 7.9 LoRA 轻量 checkpoint（仅保存 adapter + projector）

1. 目标
- 避免 LoRA 微调时保存整套 LLM 权重（单个 checkpoint 体积过大），改为可选轻量保存。

2. 改动内容
- 文件：`prismatic/conf/models.py`
  - 新增开关：`finetune_save_lora_adapter_only: bool = False`（默认关闭，保持旧行为）。

- 文件：`scripts/pretrain.py`
  - 将该开关透传到训练策略。

- 文件：`prismatic/training/materialize.py`
  - `get_train_strategy(...)` 新增 `save_lora_adapter_only` 参数并向下游传递。

- 文件：`prismatic/training/strategies/base_strategy.py`
  - 新增成员：`self.save_lora_adapter_only`。

- 文件：`prismatic/training/strategies/fsdp.py`、`prismatic/training/strategies/ddp.py`
  - 当 `save_lora_adapter_only=True` 且检测到 LoRA 训练时：
    - `model["llm_backbone"]` 改为 `model["llm_backbone_lora"]`（仅 `lora_` 参数）。
    - 保留 `model["projector"]`。
    - 在 checkpoint 顶层写入：
      - `llm_backbone_format: lora-adapter-only`
      - `lora_config`（r/alpha/dropout/target_modules）

- 文件：`prismatic/models/vlms/prismatic.py`
  - `from_pretrained(...)` 支持 LoRA adapter checkpoint：
    - 若检测到 `llm_backbone_lora`，先按 `lora_config` 调用 `enable_lora(...)` 再加载 adapter 权重。
    - 兼容旧格式（`llm_backbone` 全量权重）不变。

- 文件：`prismatic/models/load.py`
  - 若配置中开启 `finetune_save_lora_adapter_only`，加载时改为先拉起 HF base model（`inference_mode=False`），再由 `from_pretrained(...)` 叠加 LoRA adapter。

- 文件：`prismatic/models/backbones/llm/qwen3vl_text.py`
  - LoRA 注入后缓存 `lora_config`（供 checkpoint 持久化）。
  - 新增 `has_lora_enabled()` / `get_lora_config()`。
  - 允许推理加载路径调用 `enable_lora(...)`（去除训练态限制）。

3. 使用方式
- 开启轻量保存时，在训练命令中增加：
  - `--model.finetune_save_lora_adapter_only true`

4. 兼容性说明
- 默认值为 `False`，不影响现有全量 checkpoint 行为。
- 开启后，checkpoint 更小；加载逻辑已补齐，支持 `prismatic.models.load()` 路径。

### 7.10 正式训练命令（LoRA + Adapter 轻量保存）

1. 固定参数
- 8 卡训练：`--nproc-per-node 8`
- 数据与模型保持当前链路：`dinov3-qwen3vltext-2stage` + `carpaint-binary`
- LoRA 与轻量保存均开启：`finetune_use_lora=true`、`finetune_save_lora_adapter_only=true`
- 总步数：`300000`
- `per_device_batch_size=16`，配套 `global_batch_size=128`（8 卡下无梯度累积）
- 继续关闭梯度裁剪以规避已知 FSDP 混合梯度 dtype 裁剪报错：`finetune_max_grad_norm=0`
- 仅使用 `jsonl` 追踪器（避免默认 `wandb`）

2. 命令（运行目录：`/home/max/my-prismatic-vlm/runs`）

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type dinov3-qwen3vltext-2stage \
  --dataset.type carpaint-binary \
  --stage finetune \
  --pretrained_checkpoint /home/max/my-prismatic-vlm/runs/dinov3-qwen3vltext-2stage-align-s300k-x7-20260309-1900/best-val-step-013100.pt \
  --model.finetune_use_lora true \
  --model.finetune_save_lora_adapter_only true \
  --model.finetune_max_steps 300000 \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 16 \
  --model.finetune_max_grad_norm 0 \
  --trackers '["jsonl"]' \
  --run_root_dir /home/max/my-prismatic-vlm/runs \
  --run_id carpaint-qwen-lora-ft-20260312-bestvalalign
```
