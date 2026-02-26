# Projector 最小 Loss 权重验证计划（简版、可执行）

## 1. 目标

验证 `/home/max/my-prismatic-vlm/runs/dinov3-qwen3vltext-align+stage-align+x7/checkpoints` 中训练出的 projector 是否有效。  
只使用一个 checkpoint：**loss 最小的权重**。

本计划采用两种简单直观的验证：

1. 定量：同一批样本上，对比 `trained projector` 与 `随机初始化 projector` 的平均 teacher-forced loss。  
2. 定性：同一批图像上，对比两者的生成文本（side-by-side）。

---

## 2. 关键事实（已核对）

1. 当前 run 的 checkpoint 只包含 `projector`，不包含 `llm_backbone`：  
`torch.load(... )["model"].keys() == ["projector"]`
2. 因此不能直接把该 checkpoint 当作完整推理模型目录给 `prismatic.load()` 使用。  
3. 需要手动构建模型骨架（vision + llm + vlm），再加载 projector 权重。

---

## 3. 输入与输出

### 输入

- Run 目录：  
`/home/max/my-prismatic-vlm/runs/dinov3-qwen3vltext-align+stage-align+x7`
- 最小 loss checkpoint（当前扫描结果）：  
`/home/max/my-prismatic-vlm/runs/dinov3-qwen3vltext-align+stage-align+x7/checkpoints/step-202800-epoch-93-loss=0.4726.pt`

### 输出（建议）

放到：

`/home/max/my-prismatic-vlm/runs/dinov3-qwen3vltext-align+stage-align+x7/eval_projector_minloss/`

包含：

1. `summary.json`：平均 loss 对比和差值  
2. `per_sample_loss.csv`：每个样本的 loss 明细  
3. `qualitative.jsonl`：定性生成对比（baseline vs trained）  
4. `picked_samples.json`：本次评估使用的样本索引与图片路径

---

## 4. 执行步骤

### Step 0: 环境准备

```bash
cd /home/max/my-prismatic-vlm
source .venv/bin/activate
```

### Step 1: 自动确认最小 loss checkpoint（避免手填错误）

```bash
python - <<'PY'
import re
from pathlib import Path
ckpt_dir = Path("runs/dinov3-qwen3vltext-align+stage-align+x7/checkpoints")
pat = re.compile(r"loss=([0-9]+\.[0-9]+)\.pt$")
best = None
for p in ckpt_dir.glob("step-*-epoch-*-loss=*.pt"):
    m = pat.search(p.name)
    if not m:
        continue
    loss = float(m.group(1))
    if best is None or loss < best[0]:
        best = (loss, p)
print("best_loss =", best[0])
print("best_ckpt =", best[1])
PY
```

### Step 2: 新建评估目录

```bash
mkdir -p runs/dinov3-qwen3vltext-align+stage-align+x7/eval_projector_minloss
```

### Step 3: 创建快速评估脚本（建议路径）

创建：

`scripts/eval_projector_quick.py`

脚本参数建议：

- `--run_dir`
- `--checkpoint_path`
- `--n_samples`（建议 256）
- `--n_qualitative`（建议 8）
- `--seed`（建议 7）
- `--max_new_tokens`（建议 64）
- `--output_dir`

脚本内部流程：

1. 读取 `run_dir/config.json` 获取：
   - `vision_backbone_id`
   - `llm_backbone_id`
   - `arch_specifier`
   - `image_resize_strategy`
   - `dataset_root_dir`
   - `align_stage_components`
2. 从 `align chat.json` 固定随机采样 `n_samples` 条，写入 `picked_samples.json`。
3. 分别构建两套模型（结构相同）：
   - baseline：随机 projector（不加载 checkpoint）
   - trained：仅加载 `checkpoint["model"]["projector"]`
4. 对每条样本计算 teacher-forced loss（align 逻辑）：
   - caption = `conversation[-1]["value"] + eos`
   - `labels[0] = -100`
   - 图像经当前 `image_transform` 处理
   - 记录 baseline 与 trained loss
5. 再抽 `n_qualitative` 条，固定 prompt（如 `"Describe this image in one sentence."`）做生成对比。
6. 输出：
   - `summary.json`
   - `per_sample_loss.csv`
   - `qualitative.jsonl`

### Step 4: 运行评估

```bash
python scripts/eval_projector_quick.py \
  --run_dir runs/dinov3-qwen3vltext-align+stage-align+x7 \
  --checkpoint_path runs/dinov3-qwen3vltext-align+stage-align+x7/checkpoints/step-202800-epoch-93-loss=0.4726.pt \
  --n_samples 256 \
  --n_qualitative 8 \
  --seed 7 \
  --max_new_tokens 64 \
  --output_dir runs/dinov3-qwen3vltext-align+stage-align+x7/eval_projector_minloss
```

---

## 5. 判定标准（简单直接）

满足以下两条即可判定 projector 训练有效：

1. 定量：`trained_avg_loss < baseline_avg_loss`（差值越大越好）  
2. 定性：`qualitative.jsonl` 中多数样本的 trained 输出更贴图、语义更完整

---

## 6. 风险与注意事项

1. 该验证是快速验证，不是正式 benchmark。  
2. 样本来自训练数据分布（align 数据），结论是“是否学到有效 projector”，不是泛化能力结论。  
3. 若要更严谨，后续可增加：
   - 未参与训练的外部图像集
   - 固定评估模板与人工打分规则
