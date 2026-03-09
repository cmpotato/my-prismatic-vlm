# Carpaint/Qwen 训练任务简要总结

更新时间：2026-03-04

## 总览
| Run ID | 任务定位 | 主要输入数据 | 关键训练结果 | 关键产物 |
|---|---|---|---|---|
| `dinov3-qwen3vltext-align+stage-align+x7` | 基座对齐（align stage） | `llava-laion-cc-sbu-558k` | `Align/Loss` 最低约 `0.7093`；日志无 `Val Loss`（未做验证集评估） | `checkpoints/latest-checkpoint.pt` |
| `carpaint-qwen-2stage-lora-ft-300k-bs128-pd16` | 基于上面 align 权重做车漆缺陷 LoRA 微调（旧划分） | `labeled_jpg/carpaint_finetune_chat.json`（当时配置含随机 `val_ratio=0.2`） | `Finetune/Val Loss` 最低 `0.0718558`（step `3200`） | `best-val-step-003200.pt` |
| `carpaint-qwen-lora-ft-300k-bs128-pd16` | 基于同一 align 权重做车漆缺陷 LoRA 微调（新固定 train/val 拆分） | `carpaint_finetune_chat_train.json` + `carpaint_finetune_chat_val.json` | `Finetune/Val Loss` 最低 `0.1938285`（step `1800`） | `best-val-step-001800.pt` |

## 分任务说明

### 1) `dinov3-qwen3vltext-align+stage-align+x7`
- 目标：完成 DINOv3 视觉特征与 Qwen3VL-text 的对齐训练，为后续任务微调提供初始化权重。
- 训练阶段：`stage: align`，无显式验证损失记录。
- 结果：输出可用于下游微调的对齐 checkpoint（后续两个 carpaint 任务都以此为 `pretrained_checkpoint`）。
- 目录：`/home/max/my-prismatic-vlm/runs/dinov3-qwen3vltext-align+stage-align+x7`

### 2) `carpaint-qwen-2stage-lora-ft-300k-bs128-pd16`
- 目标：车漆缺陷二分类（提示词回答 yes/no）的 LoRA 微调。
- 训练方式：`stage: finetune`，LoRA（`r=16, alpha=32, dropout=0.05`），并定期做 `Val Loss` 评估。
- 结果：
  - 最优验证损失：`0.0718558 @ step 3200`
  - 产出最佳权重：`/home/max/my-prismatic-vlm/runs/carpaint-qwen-2stage-lora-ft-300k-bs128-pd16/best-val-step-003200.pt`
  - 历史 sample20 推理评估（`inference_results_sample20_v2.json`）表现较弱：Precision/Recall/F1（以 `NG=yes` 为正类）约 `0.1042 / 0.0588 / 0.0752`
- 目录：`/home/max/my-prismatic-vlm/runs/carpaint-qwen-2stage-lora-ft-300k-bs128-pd16`

### 3) `carpaint-qwen-lora-ft-300k-bs128-pd16`
- 目标：在修正标签并固定 train/val 拆分后，重新进行 LoRA 微调。
- 训练方式：与前次总体方法一致，但数据加载改为显式 train/val 文件，不再随机切分。
- 结果：
  - 最优验证损失：`0.1938285 @ step 1800`
  - 产出最佳权重：`/home/max/my-prismatic-vlm/runs/carpaint-qwen-lora-ft-300k-bs128-pd16/best-val-step-001800.pt`
  - 验证集推理（`inference_results_val.json`）指标（以 `NG=yes` 为正类）：
    - Precision `0.7086`
    - Recall `0.7700`
    - F1 `0.7380`
    - Accuracy `0.6356`
- 目录：`/home/max/my-prismatic-vlm/runs/carpaint-qwen-lora-ft-300k-bs128-pd16`

## 结论（简版）
- 三次任务形成了完整链路：`align 基座` -> `第一次 carpaint LoRA` -> `修正数据后的第二次 carpaint LoRA`。
- 从当前验证集结果看，第三次任务已经达到可用的缺陷识别能力（召回率较高，精度中等），可作为后续阈值策略或再训练迭代的基线模型。
