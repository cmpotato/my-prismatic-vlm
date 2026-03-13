# 多图交错输入改动计划

## 目标

在这个仓库里实现**方案二**：多张图像与文本中的多个 `<image>` 占位符一一对齐，模型执行真正的图文交错条件建模，而不是把所有图像 patch 简单拼成一整段统一的视觉块。

这份计划有意聚焦在当前车漆任务正在使用的 **finetune 链路**：

- `scripts/pretrain.py`
- `stage=finetune`
- `PrismaticVLM`
- `Qwen3VLTextLLMBackbone`
- 当前基于 prompt builder 的对话格式化链路

数据集文件格式本身不在这份计划范围内，但下面的代码改动默认数据层最终能提供：**每个样本一组有序图像**，并且这组图像与文本中的 `<image>` 占位符顺序一致。

## 当前项目逻辑

### 训练与数据流

当前训练主链路如下：

1. `scripts/pretrain.py` 构建 vision backbone、LLM backbone、VLM、dataset 和 collator。
2. `prismatic/preprocessing/materialize.py` 在 `stage=finetune` 时选择 `FinetuneDataset`。
3. `prismatic/training/strategies/base_strategy.py` 构建 `DataLoader`，然后把 `input_ids`、`attention_mask`、`pixel_values`、`labels` 和 `multimodal_indices` 传入 `PrismaticVLM.forward()`。
4. `prismatic/models/vlms/prismatic.py` 运行 vision backbone，投影 patch 特征，并把整段视觉块一次性插入到 BOS 后面。

### 当前多模态约定

这个仓库目前默认 **每个样本只有一张图**：

- `FinetuneDataset.__getitem__()` 只读取一个 `example["image"]`
- prompt builder 会在 tokenization 之前删掉 `<image>`
- `PrismaticVLM.forward()` 只会在 BOS 后插入一次投影后的视觉 patch
- 生成接口也默认只有一张图

### 代码层面已经存在的重要约束

1. Prompt builder 会删除 `<image>` 标记

当前 builder 会调用 `message.replace("<image>", "").strip()`，因此占位符位置信息在 tokenization 前就丢失了。

2. `PrismaticVLM.forward()` 没有建模占位符位置

它当前只支持一个视觉插入点：

- 文本前缀：仅第一个 token
- 视觉块：投影后的 image patches
- 文本后缀：剩余 token

3. Collator 只支持每个样本一个图像载荷

它当前把 `pixel_values` 堆叠成：

- `Tensor[batch, C, H, W]`，或者
- `Dict[str, Tensor[batch, C, H, W]]`

样本内部没有独立的 image 维度。

4. Sampler 中写死了单图长度估计

`SplitModalitySampler` 当前在估算 batch 长度时硬编码了固定图像 patch 开销。支持多图时必须去掉这个假设。

## 方案二的目标行为

对于一个样本，如果它包含有序图像 `[img_1, img_2, ..., img_n]`，并且文本中包含恰好 `n` 个占位符：

```text
USER: compare <image> with <image> and answer ...
```

那么前向路径需要做到：

1. 在 tokenization 过程中保留占位符位置
2. 对每张图分别通过 vision backbone 编码
3. 将每张图分别投影成各自的 patch 序列
4. 用对应图像的 patch 块替换对应的占位符位置
5. 正确重建 `inputs_embeds`、`attention_mask` 和 `labels`
6. 将展开后的图文交错序列送入文本 LLM

这才是真正的交错式多图多模态路径，不是单纯把多个视觉块首尾拼接。

## 建议的实施范围

### 第一阶段范围

先让方案二在以下链路下正确工作：

- `stage=finetune`
- `Qwen3VLTextLLMBackbone`
- `PurePromptBuilder`
- 训练和评估循环
- `generate()` 和 `generate_batch()` 的多图推理

### 第二阶段范围

再把“保留占位符”的逻辑推广到其他 prompt builder：

- `LLaMa2ChatPromptBuilder`
- `MistralInstructPromptBuilder`
- `PhiPromptBuilder`
- `VicunaV15ChatPromptBuilder`

这样做的原因是：先保证当前车漆/Qwen 微调主链路稳定，而不是一开始就对全仓库所有模型家族做并行改造。

## 必要的内部接口变化

比较干净的 batch 内部协议建议如下：

- `pixel_values`
  - tensor 视觉 backbone：`Tensor[B, M, C, H, W]`
  - dict 视觉 backbone：`Dict[str, Tensor[B, M, C, H, W]]`
- `image_attention_mask`
  - `BoolTensor[B, M]`
  - `True` 表示真实图像，`False` 表示 padding 补位图像槽
- `input_ids`
  - 内部包含一个真实的 `<image>` tokenizer token
- `labels`
  - 占位符 token 位置提前置为 `IGNORE_INDEX`
- `multimodal_indices`
  - 含义保持不变：batch 中哪些样本包含图像

这个接口既足够支撑训练，也足够支撑生成。

## 核心设计决策

### 必须引入真正的图像占位 token

如果不把 `<image>` 变成 tokenizer 里的一级 token，方案二在这个仓库里并不成立。

原因：

- 当前 prompt builder 会删掉 `<image>`
- 当前 VLM forward 根本没有占位符位置输入
- 如果想靠普通字符串事后恢复占位位置，会非常脆弱，而且会高度依赖具体模型家族

因此推荐明确采用下面的设计：

1. 把 `<image>` 加成 tokenizer special token
2. 在 LLM backbone 上记录它的 token id
3. prompt formatting 阶段保留 `<image>`
4. 让 `PrismaticVLM.forward()` 直接通过 `input_ids` 中的这个 token id 查找占位符位置

## 按文件拆分的实现计划

### 1. `prismatic/models/backbones/llm/base_llm.py`

给 LLM backbone 抽象层增加统一的多模态 token 协议。

计划改动：

- 在基类上增加 `image_token: str = "<image>"`
- 增加 `image_token_id` 属性或初始化字段
- 在接口层文档里明确：凡是用于交错式多模态输入的 backbone，都必须暴露图像占位符的 tokenizer id

原因：

- `PrismaticVLM` 不应该临时直接深入 tokenizer 内部查 token
- 占位符检测应当属于 LLM/tokenizer 协议的一部分

### 2. `prismatic/models/backbones/llm/qwen3vl_text.py`

让当前在用的 Qwen 链路具备真实图像占位 token 能力。

计划改动：

- 如果 `<image>` 尚不存在，就把它加成 additional special token
- tokenizer 扩容后同步 resize 文本 embedding 矩阵
- 在 backbone 实例上记录 `image_token_id`
- 保证这套逻辑在训练模式和 inference skeleton 模式下都一致成立

说明：

- 占位 token 对应的 embedding 行本质上只是一个定位器，第一次前向时会被图像 patch 投影块替换
- 但 tokenizer 必须先有一个稳定、单独的 token id

兼容性风险：

- tokenizer 长度和 embedding 大小会与旧 checkpoint 前提不同
- 必须保证模型构建路径里这步是确定性的，训练和推理两边不能跑出不同 token 空间

### 3. Prompt builder

涉及文件：

- `prismatic/models/backbones/llm/prompting/base_prompter.py`
- 以及后续其他具体 builder 文件

计划改动：

- 在当前活跃 builder 路径中，不再删除 `<image>`
- 严格保留占位符顺序
- 继续清理多余空白，但不清理占位符本身

建议范围：

- 第一阶段：先改 `PurePromptBuilder`
- 第二阶段：再同步改其他 prompt builder

原因：

- 方案二要求占位符位置必须穿透到 tokenization 后
- 当前 builder 的行为从设计上就与方案二冲突

### 4. `prismatic/preprocessing/datasets/datasets.py`

在数据层保留文本侧的占位符位置，同时输出有序多图张量。

计划改动：

- 扩展 `FinetuneDataset.__getitem__()`，支持一个样本对应多张图
- 输出的图像结果改成：
  - tensor 路径：`[M, C, H, W]`
  - dict 路径：`{k: [M, C, H, W]}`
- 传给 tokenizer 前，保留 prompt 文本里的 `<image>`
- 验证样本中图像数量和 `<image>` 占位符数量一致
- 更新 `get_modality_lengths()`，把多图带来的真实序列膨胀计入长度估算

关键细节：

- 当前 finetune 的 `get_modality_lengths()` 只按文本词数估长度；一旦一个样本能扩成多个视觉 patch 块，这个估计会明显失真

### 5. `prismatic/util/data_utils.py`

把 collation 从“单图/样本”推广到“多图/样本”。

计划改动：

- 把每个样本的图像列表 pad 后堆叠成带 image 维度的 batch tensor
- 额外输出 `image_attention_mask`
- 继续兼容 tensor 型和 dict 型 vision transform 输出
- 继续保留 `multimodal_indices`

建议输出格式：

- tensor 路径：`pixel_values: [B, M, C, H, W]`
- dict 路径：`pixel_values[k]: [B, M, C, H, W]`
- `image_attention_mask: [B, M]`

为什么这一步应该放在 collator：

- “不同样本图像数不一致时如何 pad”是 batch 层问题，不是样本层问题
- 模型前向需要拿到统一规范的 batch 表示

### 6. `prismatic/models/vlms/base_vlm.py`

扩展抽象 VLM 接口，让新增的多图元数据成为显式前向参数。

计划改动：

- 扩展 `forward()` 签名，增加 `image_attention_mask`
- 更新多图 `pixel_values` 的类型注解

原因：

- 当前抽象接口只暴露单图 `pixel_values`
- 训练策略和生成接口都需要依赖一个稳定的模型 API

### 7. `prismatic/models/vlms/prismatic.py`

这是方案二的主要实现工作。

#### 7.1 前向逻辑改造

把当前“BOS 后单点插入视觉块”改造成“按占位符位置交错插入”。

建议的 forward 算法：

1. 保留当前 generation cache 的短路逻辑，用于首步之后的解码
2. 使用 `image_attention_mask` 将 batch 中所有真实图像展平
3. 将展平后的图像统一送入 vision backbone
4. 对 vision 特征做 projector 投影
5. 再按样本、按图像顺序把投影后的 patch 块 regroup 回来
6. 在 `input_ids == image_token_id` 的位置找到每个样本的占位符位置
7. 对每个样本做一致性校验：
   - 真实图像数
   - 占位符数
8. 重建每个样本的序列：用对应 patch 块替换对应占位符 token
9. 重新构造：
   - `inputs_embeds`
   - `attention_mask`
   - `labels`
10. 对展开后的多模态序列再做 batch 维 padding
11. 调用 `llm_backbone(..., inputs_embeds=..., attention_mask=..., labels=...)`

#### 7.2 Label 语义

插入到占位符位置上的 patch token 对应 label 必须全部置为 `IGNORE_INDEX`。

原始 `<image>` placeholder token 不应该进入最终展开后的序列。它只是一个定位符，不是模型应该预测的文本 token。

#### 7.3 Attention 语义

只有真实图像对应的 patch token 应该参与 attention。collator 里补出来的 padded image slot 不应该转成有效视觉 token。

#### 7.4 序列膨胀

这条路径不能再假设“每个样本插入固定数量 patch token”。展开后的真实长度应为：

- 原始文本长度
- 减去占位符 token 数
- 加上所有图像投影后的 patch token 总数

因此必须先按样本做展开，再做 batch padding。

#### 7.5 生成接口

以下方法都要同步改：

- `prepare_inputs_for_generation()`
- `generate_batch()`
- `generate()`

计划改动：

- 在 generation 输入里保留 `image_attention_mask`
- `generate()` 要能接受图像列表，而不是单张图
- `generate_batch()` 要能接受“每个样本一组图”的输入

当前这些 helper 都是单图签名，无法完整支撑方案二。

### 8. 训练与评估调用方

涉及文件：

- `prismatic/training/strategies/base_strategy.py`

计划改动：

- 在 train/eval forward 调用里把 `image_attention_mask` 一并透传下去
- 更新对 batch key 的假设

这部分改动不大，但属于必须改；否则新的 collator 输出永远到不了模型层。

### 9. 长度感知的 batch 构造

涉及文件：

- `prismatic/preprocessing/datasets/datasets.py`
- `prismatic/util/batching_utils.py`
- 如有必要，`prismatic/training/strategies/base_strategy.py`

计划改动：

- 去掉 `SplitModalitySampler` 里“单图 patch 开销”的硬编码
- 多模态长度估计改成基于：
  - 文本长度
  - 图像数 / 占位符数
  - 当前 vision backbone 的 patch 数

原因：

- 当前 sampler 默认一个样本只有一个视觉块
- 方案二下，多图会让多模态序列长度按图像数放大
- 如果不修正这里，训练时的 batch 排序、最长 batch 预警和 OOM 暴露都会变得不可靠

## 推荐的实施顺序

### Phase A：tokenizer + 占位符保留

交付内容：

- `<image>` 成为稳定 tokenizer token
- prompt builder 不再删除 `<image>`
- 单图训练仍然可以跑通

退出标准：

- 一个单图样本在进入 `input_ids` 后仍然能保留一个 placeholder token

### Phase B：训练/评估多图 forward

交付内容：

- dataset/collator 能提供多张图
- `PrismaticVLM.forward()` 能按占位符位置插入对应图像 patch 块
- train/eval 调用链能传递新增 batch 字段

退出标准：

- 两图 finetune smoke test 能完整跑通
- 单图路径仍然保持向后兼容

### Phase C：推理/生成支持

交付内容：

- `generate()` 和 `generate_batch()` 能接受多图
- 首个 decoding step 会按占位符做交错插入
- 后续 cache step 继续沿用现有快速路径

退出标准：

- 多图推理 prompt 可以完成 greedy generation

### Phase D：sampler 和性能收尾

交付内容：

- 长度估计逻辑适配多图展开
- 最长 batch 仍能尽早暴露 OOM 风险

退出标准：

- 在 `split-modality` 模式下没有明显的 batch 构造退化

## 验证计划

### 最低限度正确性测试

1. 单图向后兼容

- 单图 finetune 样本仍能训练
- loss 表现与旧路径在定性上保持一致

2. 多图占位符数量一致性

- 2 张图 + 2 个占位符的样本能成功运行
- 图像数和占位符数不一致时，能抛出明确错误

3. 顺序正确性

- 验证 placeholder 1 确实对应 image 1 的 patch
- 验证 placeholder 2 确实对应 image 2 的 patch
- 不允许在 collator 或 regroup 过程中发生隐式重排

4. Batch padding 正确性

- 一个 batch 中同时有 1 图样本和 3 图样本时仍能运行
- padded image slot 不会生成有效视觉 token

5. Generation 正确性

- 第一步生成会把 placeholder 展开成 patch 块
- 之后的 cache step 只绕过多模态扩展一次，并沿用正常增量解码

### 建议的 smoke test

- `max_steps=2~5`
- 单卡
- `per_device_batch_size=1`
- 一份只含单图样本的数据
- 一份只含双图样本的合成数据
- 一份图像数可变的混合数据

## 主要风险

### 1. 上下文长度快速膨胀

真正的交错式多图会让序列长度大致增加：

- `num_images * vision_backbone.num_patches`

如果图像数较多，很容易触发文本模型上下文长度上限或显存压力。

### 2. tokenizer / checkpoint 兼容性

加 special token 会改变 tokenizer 长度和 embedding 尺寸。模型构建和 checkpoint 加载必须严格保持一致。

### 3. 不同 prompt builder 的行为不一致

仓库里有多套 prompt builder，而且当前都在删 `<image>`。如果第一阶段只改 Qwen/Pure 路径，必须在设计和实现上明确这一点，避免误以为全仓库已经统一支持。

### 4. 主前向路径之外仍有隐藏的单图假设

`SplitModalitySampler` 已经暴露了一个案例，其他 utility 或推理封装里也可能还有类似假设。

## 第一版实现建议明确排除的非目标

为了让第一版可控，不建议一开始就同时做下面这些事：

- 把 `align` 阶段也一起改成多图
- 在数据层引入任意复杂的“每轮对话多图”语义
- 在 Qwen 路径稳定前就同步改全套 prompt builder
- 一开始就做多图性能优化或显存优化

## 第一版 PR 建议的最终形态

第一版 PR 应该做到：**对当前在用的 Qwen finetune 链路完整可用**，而不是一开始就对所有模型家族做到全局完备。

建议 PR 范围：

- tokenizer placeholder token 支持
- Qwen/Pure prompt 路径保留 `<image>`
- 多图 collator 协议
- `PrismaticVLM.forward()` 的占位符驱动交错插入
- train/eval 调用链更新
- generation helper 更新
- sampler 长度估计修正
- 单图 / 双图 smoke test

这是在工程上仍然诚实、且变动范围相对最小的一种方案二实现路径。
