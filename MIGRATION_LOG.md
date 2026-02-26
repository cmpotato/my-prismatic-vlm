# Migration Log

## 2026-02-17 00:36:59 +0800

### 任务
将 `/home/max/openvla/ALIGN_TRAINING_PLAN.md` 中与 `align` 训练方案（DINOv3 + Qwen3VL TextModel，仅训练 projector）直接相关的代码迁移到 `/home/max/my-prismatic-vlm`，并做一致性核对。

### 已迁移/补齐内容
1. 视觉骨干 DINOv3
- 新增 `prismatic/models/backbones/vision/dinov3_vit.py`
- 导出 `prismatic/models/backbones/vision/__init__.py`
- 注册 `prismatic/models/materialize.py` 中 `dinov3-*`

2. 语言骨干 Qwen3VL TextModel
- 新增 `prismatic/models/backbones/llm/qwen3vl_text.py`
- 导出 `prismatic/models/backbones/llm/__init__.py`
- 注册 `prismatic/models/materialize.py` 中 `qwen3vl-text-8b-instruct`

3. 组合模型配置
- 新增 `prismatic/conf/models.py` 配置类 `Prism_DINOv3_Qwen3VLText_8B`
- 注册 `ModelRegistry.PRISM_DINOV3_QWEN3VLTEXT_8B`
- 对应 `--model.type dinov3-qwen3vltext-align`

4. 训练流程行为对齐（计划文档中的 checkpoint 说明）
- 在 `prismatic/training/strategies/base_strategy.py` 增加 `checkpoint_every_steps = 100`
- 增加周期性 checkpoint 保存逻辑

5. 依赖下限（保证新增骨干可用）
- 更新 `pyproject.toml`:
  - `timm>=1.0.24`
  - `torch>=2.2.0`
  - `transformers>=5.0.0`

### 计划项核对结论
- `prismatic/models/materialize.py`：已覆盖（VISION/LLM 注册齐全）。
- `prismatic/conf/models.py`：已覆盖（新 model config + registry）。
- `scripts/pretrain.py`：`vlm.freeze_backbones(cfg.stage)` 调用存在。
- `prismatic/conf/datasets.py`：`dataset_root_dir` 与 `align_stage_components` 结构存在并可通过 CLI 覆盖。
- `prismatic/preprocessing/datasets/datasets.py`：与 openvla 一致（关键逻辑未丢失）。
- `prismatic/models/vlms/prismatic.py`：`align` 阶段冻结语义存在（vision/llm 冻结，仅 projector 训练）。
- `scripts/data/convert_vqa_to_align_chat.py`：openvla 原仓库中即不存在，故无可迁移源文件。

### 验证记录
- 语法检查：`python3 -m compileall` 通过（新增/改动文件）。
- 配置可见性检查：
  - `ModelConfig.get_choice_class('dinov3-qwen3vltext-align')` 成功。
  - `VISION_BACKBONES` 包含 `dinov3-vit-l`。
  - `LLM_BACKBONES` 包含 `qwen3vl-text-8b-instruct`。

### 说明
本次仅迁移并补齐 `ALIGN_TRAINING_PLAN.md` 对应训练方案的必要代码，未引入 openvla 中与 VLA 训练链路相关的非必要差异。

## 2026-02-18 19:18:41 +0800

### 任务
继续对照 `/home/max/openvla`，修复 DINOv3 视觉加载链路与当前仓库逻辑不一致问题，并核查：
- `prismatic/models/backbones/llm/qwen3vl_text.py`
- `prismatic/models/backbones/llm/base_llm.py`

### 已迁移/修复内容
1. 视觉骨干加载兼容性（DINOv3/EVA）
- 同步 `prismatic/models/backbones/vision/base_vision.py` 的兼容分支：
  - 当 `featurizer` 无 `get_intermediate_layers` 时回退到 `forward_features`
  - 解决 `AttributeError: 'Eva' object has no attribute 'get_intermediate_layers'`
- 同步 ViT-like 校验逻辑（`patch_embed` + `blocks`）
- 同步 FSDP wrap 逻辑为动态类型（`type(self.featurizer)` / block 实际类型）

2. LLM 文件核查
- `prismatic/models/backbones/llm/qwen3vl_text.py` 与 openvla 对照一致（无差异）
- `prismatic/models/backbones/llm/base_llm.py` 存在差异，但当前仓库语义可接受：
  - 未声明 `last_layer_finetune_modules` 抽象接口
  - 增加了 `tokenizer.padding_side == "right"` 显式断言

### 验证记录
- 语法检查通过：
  - `python -m py_compile prismatic/models/backbones/vision/base_vision.py`
  - `python -m py_compile prismatic/models/backbones/llm/base_llm.py`
  - `python -m py_compile prismatic/models/backbones/llm/qwen3vl_text.py`
- 最小前向验证通过：
  - `dinov3-vit-l` 可实例化并前向，输出形状 `(1, 201, 1024)`

### 说明
此次修复后，DINOv3 加载路径与 openvla 兼容行为保持一致，避免了 EVA 模型接口差异导致的训练启动失败。

## 2026-02-26 20:09:29 +0800

### 任务
整理并提交本轮迁移与训练相关改动，提交主题为“完成基础projector训练”，同时补充可追溯的细粒度变更日志。

### 代码与配置变更
1. 模型与骨干注册
- `prismatic/conf/models.py`
  - 新增 `Prism_DINOv3_Qwen3VLText_8B`
  - 新增 `ModelRegistry.PRISM_DINOV3_QWEN3VLTEXT_8B`
- `prismatic/models/materialize.py`
  - 视觉注册新增：`dinov3-vit-s/b/l/h+/7b`
  - LLM 注册新增：`qwen3vl-text-8b-instruct`（`local_files_only=True`）
- `prismatic/models/backbones/vision/__init__.py`
  - 导出 `DinoV3ViTBackbone`
- `prismatic/models/backbones/llm/__init__.py`
  - 导出 `Qwen3VLTextLLMBackbone`
- 新增文件：
  - `prismatic/models/backbones/vision/dinov3_vit.py`
  - `prismatic/models/backbones/llm/qwen3vl_text.py`

2. 视觉前向兼容修复
- `prismatic/models/backbones/vision/base_vision.py`
  - `get_intermediate_layers` 不可用时回退 `forward_features`
  - 放宽校验为 ViT-like（`patch_embed` + `blocks`）
  - FSDP wrapping 改为运行时动态类型，兼容 EVA/DINOv3

3. 训练策略与长跑行为
- `prismatic/training/strategies/base_strategy.py`
  - 保留周期 checkpoint（每 100 step）
  - 将“large epochs”从固定 `100` 提升为 `1000000`，避免 `max_steps` 长跑被 epoch 上限提前截断

4. 依赖与仓库行为
- `pyproject.toml`
  - `timm>=1.0.24`
  - `torch>=2.2.0`
  - `transformers>=5.0.0`
- `.gitignore`
  - 新增忽略：`runs/`

### 文档变更
1. `ALIGN_TRAINING_PLAN.md`
- 路径统一为当前仓库 `/home/max/my-prismatic-vlm/*`
- 训练命令中的 `dataset_root_dir` 对齐到 `/home/max/my-prismatic-vlm/data`

2. `PROJECTOR_EVAL_PLAN.md`（新增）
- 新增“最小 loss projector 快速验证计划”
- 明确输入、输出、执行命令、判定标准与风险说明

### 结果与说明
- 本轮改动覆盖“基础 projector 训练”所需的模型注册、训练长跑、兼容性修复与验证计划文档。
- 当前提交不包含 `runs/` 训练产物，仅包含代码与文档。
