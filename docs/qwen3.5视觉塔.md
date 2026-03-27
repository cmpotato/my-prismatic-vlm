
## Qwen3.5-9B-Base 视觉塔完整解析

> **重要说明**：Qwen3.5 ≠ Qwen3-VL。它们是不同的模型系列，`model_type` 分别为 `qwen3_5` 和 `qwen3_vl`，使用独立的 transformers modeling 代码。本文分析的是 **Qwen3.5-9B-Base**。

本地 ModelScope 路径：`~/.cache/modelscope/hub/models/Qwen/Qwen3___5-9B-Base/`
transformers 源码位置：`transformers/models/qwen3_5/modeling_qwen3_5.py`

---

### 1. 总体架构

视觉塔总计 **456.0M 参数**，由 4 个组件构成（**无 DeepStack**）：

| 组件 | 参数量 | 作用 |
|------|--------|------|
| **Patch Embed** | 1.8M | 3D 卷积切片（Conv3d） |
| **Pos Embed** | 2.7M | 2D 可学习位置编码（Embedding） |
| **ViT Blocks ×27** | 411.5M | 视觉 Transformer 主干 |
| **Merger** | 40.1M | 空间合并 → 投影到 LLM 维度 |

与 Qwen3-VL 相比，**Qwen3.5 去掉了 DeepStack 机制**（`deepstack_visual_indexes: []`），因此没有 deepstack_merger 组件，参数量少了约 120M。

### 2. Patch Embed — 3D 卷积切片

```python
# Qwen3_5VisionPatchEmbed
Conv3d(3, 1152, kernel_size=[2, 16, 16], stride=[2, 16, 16], bias=True)
# 权重形状: [1152, 3, 2, 16, 16]
```

- 输入被视为 `[C=3, T, H, W]` 的时空体积
- **空间 patch_size = 16**：每 16×16 像素一个 patch
- **时间 temporal_patch_size = 2**：每 2 帧一个时间 patch（静态图片 T=1 时额外补 1 帧）
- 输出维度：`hidden_size = 1152`

**对单张图片**：假设 224×224 输入 → 14×14 = 196 个 patch，每个是 1152 维向量。

### 3. 位置编码 — 双轨制

视觉塔用了 **两套** 位置编码叠加：

**(a) 可学习 2D 位置嵌入**

```python
pos_embed = nn.Embedding(2304, 1152)   # 2304 = 48×48 网格
# 权重形状: [2304, 1152]
```

- 预训练了 48×48 的网格 (`num_position_embeddings=2304`)，支持最大 768×768 原生分辨率（48 × 16）
- 超过该分辨率时使用 **双线性插值**（`fast_pos_embed_interpolate`），从 4 个最近邻计算权重

插值后按 `spatial_merge_size=2` 做分块重排：
```python
# [t, h//2, 2, w//2, 2, dim] → permute → [t, h//2, w//2, 2, 2, dim] → flatten
```
使得合并后的 2×2 块内 patch 位置嵌入在空间上连续。

**(b) 2D RoPE（旋转位置编码）**

```python
rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(dim=36)  # head_dim/2 = 72/2 = 36
```

- 每个 attention head → dim=72（1152 / 16 heads），RoPE 只用一半 = 36 维
- 对每个 patch 计算 `(row_idx, col_idx)` → 分别查表得到 `[row_freq, col_freq]` → 拼接成 `[seq_len, 72]`
- 注入到 Q 和 K 向量上：`q_embed = q * cos + rotate_half(q) * sin`
- 位置 ID 按 `spatial_merge_size=2` 做分块展开，确保合并后的空间连续性

### 4. ViT Blocks ×27 — 视觉 Transformer 主干

每个 `Qwen3_5VisionBlock`：

```
Input → LayerNorm(1152) → Attention → + Residual
      → LayerNorm(1152) → MLP       → + Residual → Output
```

**Attention 层**（`Qwen3_5VisionAttention`）：
- **16 头**，每头 head_dim = 72（1152 / 16）
- **QKV 融合线性层**：`nn.Linear(1152, 3456, bias=True)` → 一次投影出 Q, K, V
- 对 Q, K 施加 2D RoPE
- 支持 Flash Attention（通过 `cu_seqlens` 做变长序列拼接处理）—— 多张图片在同一 batch 内各自独立注意
- 投影层：`nn.Linear(1152, 1152, bias=True)`
- **非因果注意力**（`is_causal=False`）

**MLP 层**（`Qwen3_5VisionMLP`）：
- `Linear(1152, 4304, bias=True)` → `GELU(approximate='tanh')` → `Linear(4304, 1152, bias=True)`
- 膨胀率 ≈ 3.74（不是常见的 4×，也不是 SwiGLU 门控结构）
- 激活函数 `hidden_act = "gelu_pytorch_tanh"`

**每个 Block 参数明细**（12 个权重张量）：
| 权重 | 形状 | 参数量 |
|------|------|--------|
| `attn.qkv.weight` | [3456, 1152] | 3,981,312 |
| `attn.qkv.bias` | [3456] | 3,456 |
| `attn.proj.weight` | [1152, 1152] | 1,327,104 |
| `attn.proj.bias` | [1152] | 1,152 |
| `mlp.linear_fc1.weight` | [4304, 1152] | 4,958,208 |
| `mlp.linear_fc1.bias` | [4304] | 4,304 |
| `mlp.linear_fc2.weight` | [1152, 4304] | 4,958,208 |
| `mlp.linear_fc2.bias` | [1152] | 1,152 |
| `norm1.weight` | [1152] | 1,152 |
| `norm1.bias` | [1152] | 1,152 |
| `norm2.weight` | [1152] | 1,152 |
| `norm2.bias` | [1152] | 1,152 |
| **合计（每 Block）** | | **15,239,504 ≈ 15.2M** |

27 层总计 = 27 × 15.2M ≈ **411.5M**

### 5. Merger（空间合并层）

```python
# Qwen3_5VisionPatchMerger
spatial_merge_size = 2  →  每 2×2=4 个相邻 patch 合并为 1 个 token
use_postshuffle_norm = False  # 注意：Qwen3.5 merger 用 pre-shuffle norm
```

流程：
```
[seq_len, 1152]
    ↓ 2×2 concat（沿 hidden_size 拼接 4 个相邻 patch）
[seq_len/4, 4608]        # 4608 = 1152 × 4
    ↓ LayerNorm(1152)    # pre-shuffle: 在 reshape 之前对原始 1152 维做 norm
    ↓ reshape to [seq_len/4, 4608]
    ↓ Linear(4608, 4608) → GELU → Linear(4608, 4096)
[seq_len/4, 4096]  ←────── 输出维度 = LLM hidden_size (out_hidden_size=4096)
```

**权重明细**（6 个张量）：
| 权重 | 形状 |
|------|------|
| `norm.weight` | [1152] |
| `norm.bias` | [1152] |
| `linear_fc1.weight` | [4608, 4608] |
| `linear_fc1.bias` | [4608] |
| `linear_fc2.weight` | [4096, 4608] |
| `linear_fc2.bias` | [4096] |

**压缩效果**：448×448 图 → 28×28=784 patch → merger 后 **196 个 token**（减少 4 倍）。

### 6. 关于 DeepStack — Qwen3.5 未启用

Qwen3-VL 的核心创新之一是 DeepStack（`deepstack_visual_indexes: [8, 16, 24]`），在 ViT 第 8/16/24 层抽取中间特征、通过独立的 deepstack_merger 投影后注入 LLM 对应层。

**Qwen3.5 配置为 `deepstack_visual_indexes: []`（空列表）**，因此：
- 无 `deepstack_merger_list` 权重
- 视觉信息只通过最终 merger 输出一次性注入 LLM 输入层
- 比 Qwen3-VL 少了 120.4M 参数（3 个 deepstack_merger × ~40.1M）

### 7. 文本模型架构差异 — 混合线性/全注意力

Qwen3.5 的文本模型（`Qwen3_5TextModel`）与 Qwen3-VL 有**本质区别**：

**混合注意力层**（`Qwen3_5DecoderLayer`）：

```python
# 32 层，每 4 层一个 full_attention，其余为 linear_attention
layer_types = [
    "linear_attention",  # 0
    "linear_attention",  # 1
    "linear_attention",  # 2
    "full_attention",    # 3
    "linear_attention",  # 4
    ...                  # 重复此模式
    "full_attention",    # 31
]
```

- **线性注意力层**（`Qwen3_5GatedDeltaNet`）：
  - 基于 Gated Delta Rule 的线性注意力，复杂度 O(n)
  - 包含 1D 因果卷积（`conv_kernel_size=4`），先对 QKV 做局部混合
  - 16 个 key head × 128d = 2048d，32 个 value head × 128d = 4096d
  - 使用 `beta` 门控和指数衰减 `g = -exp(A) * softplus(a + dt_bias)`
  - 输出经过 `RMSNormGated`（门控归一化）后投影
  - 适合长序列高效推理

- **全注意力层**（`Qwen3_5Attention`）：
  - 标准 Transformer 自注意力，GQA 16/4（16 个 Q head，4 个 KV head）
  - head_dim = 256（注意：比一般的 128 大一倍）
  - 仅用 25% 的 head_dim 做 RoPE（`partial_rotary_factor=0.25`，即 64 维旋转 + 192 维不旋转）

| 对比项 | Qwen3.5 文本模型 | Qwen3-VL 文本模型 |
|--------|-----------------|-------------------|
| 层数 | 32 | 36 |
| 注意力类型 | 混合（24 线性 + 8 全注意力） | 全 full_attention |
| GQA 配置 | 16 头 / 4 KV 头 | 32 头 / 8 KV 头 |
| head_dim | 256 | 128 |
| hidden_size | 4096 | 4096 |
| partial_rotary_factor | 0.25 | 1.0 |

### 8. 文本侧 3D M-RoPE

Qwen3.5 使用**交错式三维旋转位置编码**（Interleaved M-RoPE）：

```json
"mrope_section": [11, 11, 10],   // 32维（head_dim 256 × 0.25 = 64，64/2 = 32 对）
"mrope_interleaved": true,
"rope_theta": 10000000,
"partial_rotary_factor": 0.25
```

Qwen3.5 RoPE 的独特实现（`apply_interleaved_mrope`）：
```python
# 不是按 [T..TT, H..HH, W..WW] 拼接，而是交错排列：
# [T, H, W, T, H, W, ..., T, T]
# 对 11+11+10=32 对频率，按 stride=3 的方式交错 T/H/W 维度
```

对比 Qwen3-VL 的 M-RoPE：
| 对比项 | Qwen3.5 | Qwen3-VL |
|--------|---------|----------|
| mrope_section | [11, 11, 10] | [24, 20, 20] |
| 总 RoPE 维度 | 64 (= 256 × 0.25) | 128 |
| 排列方式 | **交错**（interleaved, stride=3） | 拼接（chunked） |
| rope_theta | 10,000,000 | 默认 |

对于**文本 token**，三个维度 position_id 相同（退化为 1D）。
对于**视觉 token**，分别编码 temporal、height、width 位置，空间感知。

### 9. 视觉-文本融合流程

在 `Qwen3_5Model.forward()` 中（源码 `modeling_qwen3_5.py`）：

```python
# 1. 文本 token 嵌入
inputs_embeds = self.get_input_embeddings()(input_ids)

# 2. 视觉特征提取
image_outputs = self.get_image_features(pixel_values, image_grid_thw)
#   内部: pixel_values → visual(PatchEmbed → PosEmbed + RoPE → 27 blocks → Merger)
#   输出: pooler_output = merger 的输出 [N/4, 4096]

# 3. 替换 placeholder
image_embeds = torch.cat(image_embeds, dim=0)
special_image_mask = (input_ids == 248056)  # image_token_id
inputs_embeds.masked_scatter_(image_mask, image_embeds)

# 4. 计算 3D position_ids (M-RoPE)
position_ids = self.compute_3d_position_ids(...)  # shape [3, batch, seq_len]

# 5. 送入文本模型（32 层混合注意力）
outputs = self.language_model(inputs_embeds=inputs_embeds, position_ids=position_ids, ...)
```

注意：**没有 DeepStack 注入步骤**，视觉信息仅在输入层一次性替换。

### 10. 数据流总结

```
原始图像 [3, H, W]
    │
    ▼  pad to T=2 (单图复制一帧)
[3, 2, H, W]
    │
    ▼  Conv3d(3→1152, k=[2,16,16])         Qwen3_5VisionPatchEmbed
[N_patches, 1152]          N = (H/16)*(W/16)
    │
    ▼  + pos_embed (Embedding 2304×1152, 双线性插值)
[N_patches, 1152]
    │
    ▼  ViT Block ×27 (with 2D RoPE)        Qwen3_5VisionBlock ×27
    │   （无中间层特征抽取，无 DeepStack）
    │
    ▼  最终输出
[N_patches, 1152]
    │
    ▼  Merger (2×2空间合并, pre-shuffle norm)  Qwen3_5VisionPatchMerger
[N/4, 4096]  →  替换 input_ids 中的 <image> placeholder (token_id=248056)
    │
    ▼  3D M-RoPE position_ids (interleaved [11,11,10], partial_rotary=0.25)
    │
    ▼  送入 Qwen3.5 Text LLM (32层, 4096d)  Qwen3_5TextModel
         ├── 24 层 Qwen3_5GatedDeltaNet (线性注意力, O(n))
         └──  8 层 Qwen3_5Attention (全注意力, GQA 16/4, head_dim=256)
```

### 11. Qwen3.5 vs Qwen3-VL 视觉塔完整对比

| 维度 | Qwen3.5-9B-Base | Qwen3-VL-8B-Instruct |
|------|-----------------|---------------------|
| model_type | `qwen3_5` | `qwen3_vl` |
| 视觉塔参数 | **456.0M** | **576.4M** |
| ViT 架构 | 27层, 1152d, 16头 | 27层, 1152d, 16头 (**相同**) |
| Patch size | 16 | 16 |
| 位置编码 | 学习式 Embedding + 2D RoPE | 学习式 Embedding + 2D RoPE (**相同**) |
| Merger | pre-shuffle norm | pre-shuffle norm (**相同**) |
| **DeepStack** | **❌ 无** (`[]`) | **✅ 有** (`[8, 16, 24]`, +120M) |
| LLM 层数 | 32 | 36 |
| LLM 注意力 | **混合** (线性+全) | 全 full_attention |
| LLM GQA | 16 头 / 4 KV头 | 32 头 / 8 KV头 |
| LLM head_dim | 256 | 128 |
| M-RoPE section | [11, 11, 10] (交错) | [24, 20, 20] (拼接) |
| partial_rotary | 0.25 | 1.0 |

**核心结论**：Qwen3.5 的视觉 ViT 部分（PatchEmbed + PosEmbed + 27 Blocks + Merger）与 Qwen3-VL **基本相同**，关键差异在于：
1. **去掉了 DeepStack** — 视觉信息只在输入层注入一次
2. **文本模型重新设计** — 用混合线性/全注意力替代全 full_attention，推理效率更高

### 12. 与项目中 DINOv3 视觉塔的对比

| 维度 | Qwen3.5 自带视觉塔 | 项目中的 DINOv3-ViT-L |
|------|---------------------|----------------------|
| 训练范式 | 多模态端到端训练 | 自监督预训练 (DINOv3) |
| 参数量 | 456M (含 merger) | ~307M |
| Patch size | 16 | 14 |
| Hidden dim | 1152 | 1024 |
| 深度 | 27 层 | 24 层 |
| 位置编码 | 学习式 + 2D RoPE | 学习式 |
| 到 LLM 的桥接 | Merger (2×2, → 4096) | GELU-MLP projector |
| 特点 | 原生多分辨率、视频支持 | 强局部特征、纹理敏感 |