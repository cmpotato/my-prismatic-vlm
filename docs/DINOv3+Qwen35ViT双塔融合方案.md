# DINOv3 + Qwen3.5-ViT 双视觉塔特征融合方案

> **参考实现**：DinoV2-SigLIP-Phi3-LoRA-VLM 项目中的 DinoSigLIP 双塔模式
> **目标**：将 DINOv3-ViT-L（自监督，强局部特征） 与 Qwen3.5 自带 ViT（多模态预训练，强语义特征）的输出做通道拼接，经 FusedMLPProjector 投影后送入 Qwen3.5 文本模型

---

## 1. 架构总览

```
原始图像 [3, H, W]
    │
    ├─────────────────────────────┐
    ▼                             ▼
 DINOv3-ViT-L                  Qwen3.5-ViT
 (TIMM, patch=16)              (transformers, patch=16)
 24层, 1024d                    27层, 1152d
    │                             │
    ▼                             ▼
 [196, 1024]                   [196, 1152]
    │                             │
    └──────── concat dim=-1 ──────┘
                  │
                  ▼
            [196, 2176]
                  │
                  ▼
         FusedMLPProjector
         Linear(2176, 8704) → GELU
         Linear(8704, 4096) → GELU
         Linear(4096, 4096)
                  │
                  ▼
            [196, 4096]
                  │
                  ▼  替换 input_ids 中 <image> token
         Qwen3.5 Text LLM (32层, 4096d)
```

### 关键维度

| 组件 | embed_dim | patch_size | 224px时 patch 数 | 参数量 |
|------|-----------|-----------|-----------------|--------|
| DINOv3-ViT-L | 1024 | 16 | 196 (14×14) | ~307M |
| Qwen3.5-ViT | 1152 | 16 | 196 (14×14) | ~416M (不含 merger) |
| 拼接后 | **2176** | — | 196 | — |
| FusedMLPProjector | 2176→4096 | — | — | ~48M |
| Qwen3.5 Text | 4096 | — | — | ~8.5B |

**patch 数对齐**：两个 ViT 的 patch_size 同为 16，在相同分辨率输入下 patch 数完全一致（224px→196），可以直接 concat——这是方案可行的**核心前提**。

---

## 2. 与参考方案（DinoSigLIP）的对比

| 维度 | DinoSigLIP（参考） | DINOv3 + Qwen3.5-ViT（本方案） |
|------|-------------------|-------------------------------|
| 视觉塔 A | DINOv2-ViT-L (TIMM) | DINOv3-ViT-L (TIMM) |
| 视觉塔 B | SigLIP-SO (TIMM) | **Qwen3.5-ViT (transformers)** |
| 特征拼接 | `cat([dino, siglip], dim=2)` | `cat([dinov3, qwen_vit], dim=2)` |
| patch_size | 都是 14 | **都是 16** |
| embed_dim | 1024+1152=2176 | 1024+1152=**2176** (巧合相同) |
| Projector | FusedMLPProjector | FusedMLPProjector |
| LLM | Phi-3 / LLaMA | Qwen3.5-9B-Base |
| **关键差异** | 两个都是 TIMM 模型 | Qwen3.5-ViT **不是 TIMM 模型**，需要适配 |

### 核心挑战

DinoSigLIP 中两个 ViT 都是 TIMM 模型，接口统一。但 Qwen3.5-ViT 来自 transformers：
- **输入不同**：TIMM 接收 `[B, 3, H, W]` 张量；Qwen3.5-ViT 接收已展平的像素 patch + `grid_thw` 元信息
- **输出不同**：TIMM 返回 `[B, num_patches, embed_dim]`；Qwen3.5-ViT 返回 `BaseModelOutputWithPooling`（`last_hidden_state` = ViT 原始输出，`pooler_output` = merger 压缩后输出）
- **位置编码不同**：Qwen3.5-ViT 需要 `grid_thw` 来计算位置嵌入和 RoPE

---

## 3. 技术路线

### 3.1 新建 Qwen3.5-ViT 独立包装类

**文件**：`prismatic/models/backbones/vision/qwen35_vit.py`

将 `Qwen3_5VisionModel` 包装为 prismatic `VisionBackbone` 接口。核心逻辑：

```python
class Qwen35ViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id, image_resize_strategy, default_image_size=224):
        # 1. 从 Qwen3.5 config 中实例化 Qwen3_5VisionModel
        self.featurizer = Qwen3_5VisionModel(vision_config)

        # 2. 从 safetensors 中只加载 "model.visual.*" 权重（排除 merger）
        self._load_vision_weights(model_path)

        # 3. 图像预处理：Normalize(mean=0.5, std=0.5)
        self.image_transform = Compose([Resize(224), ToTensor(), Normalize(0.5, 0.5)])

    def _prepare_pixels(self, pixel_values):
        """[B, 3, H, W] → Qwen3.5-ViT 期望的 (flattened_patches, grid_thw) 格式"""
        # 复制一帧满足 temporal_patch_size=2 → reshape 为展平 patch 序列
        # grid_thw = [[1, H//16, W//16]] * B
        ...

    def forward(self, pixel_values):  # [B, 3, H, W] → [B, 196, 1152]
        prepared_pixels, grid_thw = self._prepare_pixels(pixel_values)
        outputs = self.featurizer(hidden_states=prepared_pixels, grid_thw=grid_thw)
        return outputs.last_hidden_state.reshape(B, -1, 1152)  # 不经过 merger

    @property
    def embed_dim(self): return 1152

    @property
    def num_patches(self): return (default_image_size // 16) ** 2  # 196
```

**设计要点**：
- **不使用 merger**：只取 `last_hidden_state`（27层 ViT 原始输出），空间压缩由 projector 统一处理
- **输入适配**：`_prepare_pixels()` 将 `[B, 3, H, W]` → 展平 patch + `grid_thw`
- **权重加载**：从完整 checkpoint 中只提取 `visual.*`（排除 merger），`strict=False`

### 3.2 新建双塔融合骨干

**文件**：`prismatic/models/backbones/vision/dinov3_qwen35vit.py`

复刻 DinoSigLIP 双塔模式：

```python
@dataclass
class DINOv3Qwen35ViTImageTransform:
    """双路预处理，各自独立 normalize"""
    dinov3_image_transform: ImageTransform     # ImageNet mean/std
    qwen35vit_image_transform: ImageTransform  # 0.5/0.5
    is_prismatic: bool = True

    def __call__(self, img):
        return {"dinov3": self.dinov3_image_transform(img),
                "qwen35vit": self.qwen35vit_image_transform(img)}


class DINOv3Qwen35ViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id, image_resize_strategy, default_image_size=224):
        # DINOv3 侧：TIMM 加载，monkey-patch forward 取倒数第 2 层
        self.dinov3_featurizer = timm.create_model("vit_large_patch16_dinov3", ...)
        # Qwen3.5-ViT 侧：用上面的包装类
        self.qwen35vit = Qwen35ViTBackbone(...)
        # 双路 transform
        self.image_transform = DINOv3Qwen35ViTImageTransform(dinov3_tx, qwen35_tx)

    def forward(self, pixel_values):  # {"dinov3": [B,3,H,W], "qwen35vit": [B,3,H,W]}
        dinov3_patches = self.dinov3_featurizer(pixel_values["dinov3"])   # [B, 196, 1024]
        qwen35_patches = self.qwen35vit(pixel_values["qwen35vit"])       # [B, 196, 1152]
        return torch.cat([dinov3_patches, qwen35_patches], dim=2)        # [B, 196, 2176]

    @property
    def embed_dim(self): return 1024 + 1152  # = 2176
```

### 3.3 注册到框架

**`vision/__init__.py`** — 导入新类：
```python
from .qwen35_vit import Qwen35ViTBackbone
from .dinov3_qwen35vit import DINOv3Qwen35ViTBackbone
```

**`materialize.py`** — VISION_BACKBONES 新增：
```python
"qwen35-vit":          {"cls": Qwen35ViTBackbone,        "kwargs": {"default_image_size": 224}},
"dinov3qwen35vit-224px": {"cls": DINOv3Qwen35ViTBackbone, "kwargs": {"default_image_size": 224}},
```

**`models.py`** — 模型配置 + ModelRegistry：
```python
@dataclass
class Prism_DINOv3Qwen35ViT_Qwen35Text_9B(Exp_7B_One_Stage):
    model_id: str = "dinov3qwen35vit-qwen35text-align"
    vision_backbone_id: str = "dinov3qwen35vit-224px"
    llm_backbone_id: str = "qwen35-text-9b-base"
    arch_specifier: str = "no-align+fused-gelu-mlp"   # ← fused projector

@dataclass
class Prism_DINOv3Qwen35ViT_Qwen35Text_9B_TwoStage(...):
    model_id: str = "dinov3qwen35vit-qwen35text-2stage"
    arch_specifier: str = "fused-gelu-mlp"
```

---

## 4. 训练策略

### 4.1 两阶段训练

与现有 Prismatic 框架完全一致：

**阶段 1：对齐（Align）**
- `arch_specifier = "no-align+fused-gelu-mlp"`
- 冻结：DINOv3 ViT ✅ + Qwen3.5 ViT ✅ + LLM ✅
- 只训练：**FusedMLPProjector**（~48M 参数）
- 数据：LLaVA-Pretrain 558K（image-caption 对）
- 目的：学习将双塔拼接特征映射到 LLM 的嵌入空间

**阶段 2：微调（Finetune）**
- `arch_specifier = "fused-gelu-mlp"`
- 冻结：DINOv3 ViT ✅ + Qwen3.5 ViT ✅
- 训练：**Projector + LLM（LoRA）**
- 数据：LLaVA-Instruct 665K / 或领域数据
- 目的：指令跟随和任务适配

### 4.2 冻结策略

| 组件 | 对齐阶段 | 微调阶段 | 理由 |
|------|---------|---------|------|
| DINOv3-ViT-L (307M) | 冻结 | 冻结 | 自监督特征已经很强，不需要动 |
| Qwen3.5-ViT (416M) | 冻结 | 冻结 | 预训练多模态特征保持稳定 |
| FusedMLPProjector (48M) | **训练** | **训练** | 学习特征对齐 |
| Qwen3.5 Text (8.5B) | 冻结 | **LoRA** | 资源约束下的高效微调 |

### 4.3 训练命令示例

```bash
# 阶段 1：对齐
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
    --model.type "dinov3qwen35vit-qwen35text-align" \
    --model.model_id "dinov3qwen35vit-qwen35text-align" \
    --model.vision_backbone_id "dinov3qwen35vit-224px" \
    --model.llm_backbone_id "qwen35-text-9b-base" \
    --model.arch_specifier "no-align+fused-gelu-mlp" \
    --dataset.type "llava-v15" \
    --stage "align"

# 阶段 2：微调（LoRA）
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
    --model.type "dinov3qwen35vit-qwen35text-2stage" \
    --model.model_id "dinov3qwen35vit-qwen35text-2stage" \
    --model.arch_specifier "fused-gelu-mlp" \
    --dataset.type "llava-v15" \
    --stage "finetune" \
    --model.finetune_global_batch_size 128 \
    --model.finetune_per_device_batch_size 16
```

---

## 5. 数据流详解

### 5.1 图像预处理（双路）

```
原始图片 (PIL.Image)
    │
    ▼  DINOv3Qwen35ViTImageTransform.__call__()
    │
    ├── "dinov3":     Resize(224) → ToTensor → Normalize(ImageNet mean/std)
    │                 → [3, 224, 224]
    │
    └── "qwen35vit":  Resize(224) → ToTensor → Normalize(0.5, 0.5)
                      → [3, 224, 224]
```

两条路各自独立预处理，归一化参数不同：
- DINOv3：ImageNet 标准（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）
- Qwen3.5-ViT：简单归一化（mean=std=0.5）

### 5.2 特征提取

```
pixel_values = {"dinov3": [B, 3, 224, 224], "qwen35vit": [B, 3, 224, 224]}
                    │                              │
                    ▼                              ▼
           DINOv3-ViT-L                    Qwen35ViTBackbone
    TIMM forward_features()          _prepare_pixels() → grid_thw
           24 层 ViT                        27 层 ViT
      取倒数第 2 层输出              取 last_hidden_state
                    │                              │
                    ▼                              ▼
             [B, 196, 1024]                 [B, 196, 1152]
                    │                              │
                    └────────── cat(dim=2) ─────────┘
                                   │
                                   ▼
                            [B, 196, 2176]
```

### 5.3 投影 + LLM

```
[B, 196, 2176]
      │
      ▼  FusedMLPProjector
      │   Linear(2176, 8704) → GELU       # 4x 扩展
      │   Linear(8704, 4096) → GELU       # 压缩到 LLM dim
      │   Linear(4096, 4096)              # 精调
      │
      ▼
[B, 196, 4096]
      │
      ▼  替换 input_ids 中的 <image> token
      │
      ▼  Qwen3.5 Text LLM (32 层, 4096d, 混合注意力)
```

---

## 6. 需要新增/修改的文件清单

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| **新建** | `prismatic/models/backbones/vision/qwen35_vit.py` | Qwen3.5 ViT 独立包装类 |
| **新建** | `prismatic/models/backbones/vision/dinov3_qwen35vit.py` | 双塔融合骨干类 |
| **修改** | `prismatic/models/backbones/vision/__init__.py` | 导入新类 |
| **修改** | `prismatic/models/materialize.py` | 注册到 VISION_BACKBONES |
| **修改** | `prismatic/conf/models.py` | 新增模型配置 + ModelRegistry |

**不需要修改的文件**（框架已有通用支持）：
- `prismatic/util/nn_utils.py` — FusedMLPProjector 已存在
- `prismatic/models/vlms/prismatic.py` — `arch_specifier.endswith("fused-gelu-mlp")` 已处理
- `prismatic/training/strategies/fsdp.py` — FSDP wrapping 已通用化

---

## 7. 风险点与应对

### 7.1 Qwen3.5-ViT 输入格式适配

**风险**：Qwen3.5-ViT 期望展平的 patch 像素 + `grid_thw`，而非标准 `[B, 3, H, W]`。
**应对**：在 `Qwen35ViTBackbone._prepare_pixels()` 中完成格式转换。需要仔细对齐 temporal_patch_size=2 的帧复制逻辑。

**验证方法**：
```python
# 用原始 Qwen3.5 processor 的输出与我们的 _prepare_pixels 做交叉验证
from transformers import Qwen3_5ForConditionalGeneration
model = Qwen3_5ForConditionalGeneration.from_pretrained(...)
# 对比 model.model.visual(our_prepared_input) 与 model.get_image_features(original_input) 的输出
```

### 7.2 特征层选择

**风险**：DINOv3 取倒数第 2 层（第 22 层），Qwen3.5-ViT 取最后一层（第 26 层）。特征语义深度不一致。

**应对**：
- 方案 A（推荐先用）：保持现状，FusedMLPProjector 的 3 层 MLP 有足够容量对齐
- 方案 B（可选优化）：Qwen3.5-ViT 也取倒数第 2 层（修改 forward 取 `blocks[-2]` 的输出）

### 7.3 显存预算

| 组件 | 显存（bf16） |
|------|-------------|
| DINOv3-ViT-L (冻结) | ~614 MB |
| Qwen3.5-ViT (冻结) | ~832 MB |
| FusedMLPProjector | ~96 MB |
| Qwen3.5 Text 9B | ~17 GB (推理) |
| LoRA 适配器 | ~200 MB |
| 激活值 + 优化器 | ~剩余 |
| **总计** | ~19-20 GB |

在 8×A800-40GB 集群上，FSDP 分片后单卡显存约 **5-6 GB**，完全可行。

### 7.4 patch_size 一致性

**前提**：两个 ViT 都是 patch_size=16。如果未来想换用 DINOv2（patch_size=14），需要在输入分辨率上做调整使 patch 数对齐（如 DINOv2 用 224px→16×16=256 patches，DINOv3/Qwen3.5 用 256px→16×16=256 patches）。

**当前状态**：DINOv3 和 Qwen3.5-ViT 都是 patch_size=16，在同一分辨率下 patch 数**天然对齐**，无需额外处理。

---

## 8. 实施步骤

```
第 1 步：实现 qwen35_vit.py
         ├── Qwen35ViTBackbone 类
         ├── 权重加载
         └── 单元测试：验证输出 shape 和数值正确性

第 2 步：实现 dinov3_qwen35vit.py
         ├── DINOv3Qwen35ViTBackbone 类
         ├── 双路 transform
         └── 单元测试：验证 concat 输出 [B, 196, 2176]

第 3 步：注册 & 配置
         ├── __init__.py 导入
         ├── materialize.py 注册
         └── models.py 配置类 + ModelRegistry

第 4 步：对齐训练（阶段 1）
         ├── LLaVA-Pretrain 558K
         ├── 只训练 FusedMLPProjector
         └── 验证 loss 收敛

第 5 步：微调训练（阶段 2）
         ├── LLaVA-Instruct / 领域数据
         ├── Projector + LLM LoRA
         └── 评估指标
```
