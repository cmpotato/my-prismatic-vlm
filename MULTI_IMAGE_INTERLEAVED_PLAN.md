# 多图交错输入改动计划

正式训练命令

```
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type dinov3-qwen3vltext-2stage \
  --dataset.type carpaint-binary \
  --dataset.dataset_root_dir data \
  --dataset.finetune_stage_components '["jit_diff/carpaint_finetune_chat_train.json", "jit_diff/"]' \
  --dataset.finetune_val_stage_components '["jit_diff/carpaint_finetune_chat_val.json", "jit_diff/"]' \
  --stage finetune \
  --pretrained_checkpoint /home/max/my-prismatic-vlm/runs/dinov3-qwen3vltext-2stage-align-s300k-x7-20260309-1900/best-val-step-013100.pt \
  --model.finetune_use_lora true \
  --model.finetune_save_lora_adapter_only true \
  --model.finetune_global_batch_size 64 \
  --model.finetune_per_device_batch_size 8 \
  --model.finetune_max_steps 100000 \
  --model.finetune_max_grad_norm 0 \
  --eval_every_n_steps 100 \
  --trackers '["jsonl"]' \
  --run_root_dir /home/max/my-prismatic-vlm/runs \
  --run_id carpaint-jitdiff-3img-lora-ft
```

## 目标

在这个仓库里实现**方案二**：多张图像与文本中的多个 `<image>` 占位符一一对齐，模型执行真正的图文交错条件建模，而不是把所有图像 patch 简单拼成一整段统一的视觉块。

这份计划有意聚焦在当前车漆任务正在使用的 **唯一目标组合**：

- `scripts/pretrain.py`
- `stage=finetune`
- `DinoV3ViTBackbone`
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

只保留一个实施阶段，范围固定为：

- `stage=finetune`
- `DinoV3ViTBackbone`
- `Qwen3VLTextLLMBackbone`
- `PurePromptBuilder`
- 训练和评估循环
- `generate()` 和 `generate_batch()` 的多图推理

不在这一轮里考虑：

- 其他 vision backbone
- 其他 LLM family
- 其他 prompt builder
- `align` 阶段

这样做的原因是：当前分支目标不是做全仓库通用多图框架，而是先把正在用的 `DINOv3 + Qwen3VL textmodel` 微调链路完整打通。

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

- 仅修改 `PurePromptBuilder`

原因：

- 方案二要求占位符位置必须穿透到 tokenization 后
- 当前 builder 的行为从设计上就与方案二冲突
- 当前目标组合只走 `PurePromptBuilder`

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

### 3. 当前实现是组合定制，不是全仓库通用能力

这一轮只覆盖 `DINOv3 + Qwen3VL textmodel + PurePromptBuilder`。如果后续切到其他 backbone 或 prompt builder，默认不能认为多图交错逻辑已经自动成立。

### 4. 主前向路径之外仍有隐藏的单图假设

`SplitModalitySampler` 已经暴露了一个案例，其他 utility 或推理封装里也可能还有类似假设。

## 第一版实现建议明确排除的非目标

为了让第一版可控，不建议一开始就同时做下面这些事：

- 把 `align` 阶段也一起改成多图
- 在数据层引入任意复杂的“每轮对话多图”语义
- 在 Qwen 路径稳定前就同步改全套 prompt builder
- 一开始就做多图性能优化或显存优化

## 第一版 PR 建议的最终形态

第一版 PR 应该做到：**对当前在用的 `DINOv3 + Qwen finetune` 链路完整可用**，而不是一开始就对所有模型家族做到全局完备。

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

## 附录：多图适配核心代码审查总结

### 审查范围

这部分内容是对当前分支中“多图输入适配”核心代码修改的中文审查总结，审查范围限定在当前已经收敛过的目标路径：

- 视觉 backbone：`DinoV3ViTBackbone`
- 文本 backbone：`Qwen3VLTextLLMBackbone`
- 训练阶段：`finetune`
- prompt builder 路径：`PurePromptBuilder`

本次审查覆盖的 git 跟踪文件包括：

- `prismatic/models/backbones/llm/base_llm.py`
- `prismatic/models/backbones/llm/prompting/base_prompter.py`
- `prismatic/models/backbones/llm/qwen3vl_text.py`
- `prismatic/models/vlms/base_vlm.py`
- `prismatic/models/vlms/prismatic.py`
- `prismatic/preprocessing/datasets/datasets.py`
- `prismatic/training/strategies/base_strategy.py`
- `prismatic/util/data_utils.py`

审查目标不是简单罗列改动，而是判断这些改动在当前仓库中的逻辑是否闭环、是否正确、以及后续还存在哪些风险。

### 总体结论

当前这版多图适配，对于目标路径是成立的，核心逻辑是正确的。

这版实现做的不是“把多张图简单拼到 BOS 后面”，而是：

- 在文本里保留真实 `<image>` 占位符
- 数据集支持每条样本携带多张图
- collator 用 `image_attention_mask` 显式标记每条样本哪些图像槽位有效
- `PrismaticVLM.forward()` 根据 tokenized 后的 `<image>` 位置，把对应图像的 patch block 插入到对应位置

这套设计和“方案二”的目标一致，也就是：

- 图像顺序由 prompt 中的 `<image>` 位置决定
- 图像与文本是按占位符位置交错对齐，而不是统一插到开头

当前代码已经支持：

- 旧格式单图样本：`image`
- 新格式多图样本：`images`
- 图像数与 `<image>` 数量严格一致性校验
- 训练前向中的多图交错插入
- 推理接口中的多图输入

从工程角度看，这次改动是结构完整的，不是只改了 dataset 或只改了模型前向中的一处。

### 设计意图与实现方向

在修改前，仓库的单图假设存在于几个关键层：

- prompt builder 会直接删除 `<image>`
- finetune dataset 每条样本只读取一张图
- collator 只会堆叠每样本一张图
- `PrismaticVLM.forward()` 默认只在 BOS 后插一段视觉 patch

当前实现把这套单图约束改成了新的链路契约：

1. `<image>` 必须保留到 tokenization 之后。
2. 每条样本可以携带 0、1 或多张图。
3. 每张图先独立过 vision backbone 和 projector。
4. 每个 `<image>` token 都被替换成对应那一张图的 patch block。
5. patch 对应的位置不参与语言模型 loss。

对当前 Qwen 路径来说，这就是正确的多图交错语义。

### 分文件总结与正确性判断

#### 1. `prismatic/models/backbones/llm/base_llm.py`

相关位置：

- [base_llm.py](/home/max/my-prismatic-vlm/prismatic/models/backbones/llm/base_llm.py#L45)

修改内容：

- 在抽象 LLM backbone 基类中新增：
  - `self.image_token = "<image>"`
  - `self.image_token_id`

为什么这样改是对的：

- `PrismaticVLM.forward()` 需要知道 tokenized 之后哪个 token id 代表图像占位符。
- 这个信息属于 tokenizer / LLM backbone 范畴，不应该硬编码在 VLM 主体里。
- 放在基类里，能够明确多图对齐的通用契约。

结论：

- 改动很小，但位置是对的。
- 这一步把“图像占位符”从隐式字符串，提升成了显式模型契约。

#### 2. `prismatic/models/backbones/llm/prompting/base_prompter.py`

相关位置：

- [base_prompter.py](/home/max/my-prismatic-vlm/prismatic/models/backbones/llm/prompting/base_prompter.py#L42)

修改内容：

- `PurePromptBuilder.add_turn()` 不再删除 `<image>`，改为仅做 `strip()`

为什么这样改是对的：

- 如果这里继续删 `<image>`，那么后面的 tokenizer、dataset、forward 再怎么改都无法知道图像该插到哪里。
- 多图交错的第一前提就是 `<image>` 必须从 prompt 构造一路保留下来。

结论：

- 这是当前路径里最关键的启用性改动之一。
- 当前范围只覆盖 `PurePromptBuilder`，这个边界是明确且合理的。

#### 3. `prismatic/models/backbones/llm/qwen3vl_text.py`

相关位置：

- [qwen3vl_text.py](/home/max/my-prismatic-vlm/prismatic/models/backbones/llm/qwen3vl_text.py#L101)
- [qwen3vl_text.py](/home/max/my-prismatic-vlm/prismatic/models/backbones/llm/qwen3vl_text.py#L116)
- [qwen3vl_text.py](/home/max/my-prismatic-vlm/prismatic/models/backbones/llm/qwen3vl_text.py#L135)
- [qwen3vl_text.py](/home/max/my-prismatic-vlm/prismatic/models/backbones/llm/qwen3vl_text.py#L153)

修改内容：

- tokenizer 初始化后检查 `<image>` 是否已经是独立 token
- 若不存在，则把 `<image>` 注册成 additional special token
- 若词表长度发生变化，则 resize token embeddings
- 记录 `image_token_id`
- 如果 `<image>` 最终仍被解析成 `unk_token_id`，则直接报错

为什么这样改是对的：

- 方案二要求 `<image>` 在 tokenization 后仍能稳定定位。
- 如果 `<image>` 只是普通字符串片段，被拆成多个 subtoken，那么后续对齐逻辑会变脆弱甚至失效。
- 因此把 `<image>` 注册成真正的 special token 是正确做法。
- 增加 special token 后同步 resize embedding 是必须步骤，否则词表和 embedding 尺寸不一致。

结论：

- 这部分实现是必要且正确的。
- 失败路径也设计得清楚：如果 `<image>` 没注册成功，会直接报错，而不是静默坏掉。

#### 4. `prismatic/preprocessing/datasets/datasets.py`

相关位置：

- [datasets.py](/home/max/my-prismatic-vlm/prismatic/preprocessing/datasets/datasets.py#L123)
- [datasets.py](/home/max/my-prismatic-vlm/prismatic/preprocessing/datasets/datasets.py#L139)
- [datasets.py](/home/max/my-prismatic-vlm/prismatic/preprocessing/datasets/datasets.py#L187)
- [datasets.py](/home/max/my-prismatic-vlm/prismatic/preprocessing/datasets/datasets.py#L214)

修改内容：

- 新增 `_extract_image_paths()`：
  - 兼容旧的 `image`
  - 也支持新的 `images`
- 新增 `_count_image_placeholders()`
- `__getitem__()` 中：
  - 读取多张图
  - 校验图像数量必须和对话中的 `<image>` 数量一致
  - 支持把多张 transform 输出堆叠成 tensor 或 dict
- `get_modality_lengths()` 从只看 `"image" in example` 改成用统一抽取逻辑判断

为什么这样改是对的：

- 多图适配从数据层开始就必须升级协议。
- `image` 与 `images` 双兼容保证了对旧数据的回退兼容，不会强行打断已有链路。
- 最重要的是“图像数 == 占位符数”这个约束被放在 dataset 层就开始检查，这可以在最早阶段拦住脏数据。

结论：

- 数据层改法是稳妥的。
- 这一步避免了最危险的一类隐式错误：样本看似能读，但图像顺序和文本占位符不一致。

补充判断：

- 这里的 `get_modality_lengths()` 仍然只统计文本词数，没有把多图序列膨胀算进去。
- 这不影响功能正确性，但会影响 sampler 的长度估计质量。

#### 5. `prismatic/util/data_utils.py`

相关位置：

- [data_utils.py](/home/max/my-prismatic-vlm/prismatic/util/data_utils.py#L28)
- [data_utils.py](/home/max/my-prismatic-vlm/prismatic/util/data_utils.py#L52)
- [data_utils.py](/home/max/my-prismatic-vlm/prismatic/util/data_utils.py#L114)

修改内容：

- collator 支持多图 batch
- 对 tensor 路径，输出 shape 变为：
  - `[B, M, C, H, W]`
- 对 dict 路径，输出变为：
  - `dict[key] -> [B, M, ...]`
- 新增 `image_attention_mask`，shape 为：
  - `[B, M]`
- 对 unimodal 样本仍用 dummy image，但 mask 全为 false

为什么这样改是对的：

- batch 内样本的图片数不固定，所以必须按图片轴做 padding。
- 一旦做 padding，就必须显式告诉下游哪些图片槽位是真图、哪些是 pad 图。
- `image_attention_mask` 正是这个作用。
- 对 unimodal 样本统一沿用同一个 batch 协议，减少了模型前向中的分支复杂度。

结论：

- 这部分设计是正确的，而且和后续 `PrismaticVLM.forward()` 的 flatten 逻辑完全配套。

#### 6. `prismatic/training/strategies/base_strategy.py`

相关位置：

- [base_strategy.py](/home/max/my-prismatic-vlm/prismatic/training/strategies/base_strategy.py#L147)
- [base_strategy.py](/home/max/my-prismatic-vlm/prismatic/training/strategies/base_strategy.py#L271)
- [base_strategy.py](/home/max/my-prismatic-vlm/prismatic/training/strategies/base_strategy.py#L227)

修改内容：

- 训练和验证前向都把 `image_attention_mask` 传入模型

为什么这样改是对的：

- 仅仅在 train 传 mask、eval 不传，或者反过来，都会导致行为不一致。
- 当前实现使 train/eval 共用同一 forward 契约，符合预期。

结论：

- 这是必须的调用链打通改动，方向正确。

补充说明：

- 当前 validation dataloader 复用了 `per_device_batch_size`。
- 这也是后续正式大 batch 运行时在 eval 阶段更容易爆显存的一个原因。

#### 7. `prismatic/models/vlms/base_vlm.py`

相关位置：

- [base_vlm.py](/home/max/my-prismatic-vlm/prismatic/models/vlms/base_vlm.py#L82)

修改内容：

- 抽象 `forward()` 签名增加 `image_attention_mask`

为什么这样改是对的：

- 新字段既然是核心 forward 契约的一部分，就应该上升到抽象接口层。
- 不应该只在具体实现里偷偷增加参数。

结论：

- 这是正确的接口层同步。

#### 8. `prismatic/models/vlms/prismatic.py`

这是本轮改动的核心。

相关位置：

- `_flatten_multimodal_pixel_values()`： [prismatic.py](/home/max/my-prismatic-vlm/prismatic/models/vlms/prismatic.py#L286)
- `_build_interleaved_sequences()`： [prismatic.py](/home/max/my-prismatic-vlm/prismatic/models/vlms/prismatic.py#L303)
- 主 `forward()`： [prismatic.py](/home/max/my-prismatic-vlm/prismatic/models/vlms/prismatic.py#L387)
- `prepare_inputs_for_generation()`： [prismatic.py](/home/max/my-prismatic-vlm/prismatic/models/vlms/prismatic.py#L520)
- `generate_batch()`： [prismatic.py](/home/max/my-prismatic-vlm/prismatic/models/vlms/prismatic.py#L555)
- `generate()`： [prismatic.py](/home/max/my-prismatic-vlm/prismatic/models/vlms/prismatic.py#L649)

修改内容：

- 先根据 `image_attention_mask` 提取 batch 中所有真实图像，并展平成视觉 backbone 可处理的连续图像 batch
- 每张图独立经过 vision backbone 和 projector，得到一段 patch block
- 对每个多模态样本：
  - 找出 tokenized 后 `<image>` 的位置
  - 检查占位符数量是否与该样本图像数一致
  - 依次用对应图像 patch block 替换对应占位符
- 图像 patch block 对应位置的 label 统一设为 `IGNORE_INDEX`
- 因为替换后各样本长度不再相同，所以再通过 `pad_sequence` 拼回一个 batch
- 同时把 generation 路径也同步改成支持多图

为什么这样改是对的：

1. 只对真实图像做视觉编码  
   通过 `image_attention_mask` 先过滤掉 pad 图像，避免无效计算，这是正确的。

2. 用 placeholder 位置驱动插入  
   这正是方案二的定义。图像不是统一塞到 BOS 后，而是由 `<image>` 的文本位置决定。

3. patch 位置不参与语言建模 loss  
   图像 patch 不是文本 token，对这些位置设 `IGNORE_INDEX` 是标准做法。

4. 允许不同样本替换后序列长度不同  
   多图交错后，每条样本长度天然可变，用 `pad_sequence` 恢复批处理是合理的。

5. generation 路径同步升级  
   如果只改训练前向，不改 generation，那么推理会和训练契约不一致。当前实现把这一点补上了。

结论：

- 这部分是当前多图适配最核心、也是最关键正确的一段代码。
- 从逻辑闭环上看，数据、掩码、token 位置、视觉特征和 loss 屏蔽之间是自洽的。

需要明确的一点：

- 当前 `generate_batch()` 仍然是逐样本循环调用 `super().generate()`，不是一次性真正矢量化批量生成。
- 这不影响正确性，但会影响效率。

### 正确性论证

当前实现最重要的正确性来自于整条数据契约闭环：

1. Dataset 读取多图样本。
2. Dataset 校验图像数和 `<image>` 数量一致。
3. Collator 对图片轴做 padding，并输出 `image_attention_mask`。
4. Forward 根据 mask 只提取真实图像。
5. Forward 在 tokenized 后再次检查 `<image>` 数量和图像数一致。
6. Forward 逐个占位符插入对应图像的 patch block。
7. 图像 patch 对应位置不计入 loss。
8. 最终 LLM 收到的是已经完成对齐的 fused embeddings / attention / labels。

这条链路中没有缺关键步骤，因此从工程逻辑上看是完整的。

尤其重要的是，这版实现避免了两种常见错误：

- 图像数和占位符数不一致却静默继续训练
- padded image slot 被误当成真实图像送进模型

### 已有验证证据

#### 1. 静态验证

已对当前改动文件做过 `py_compile` 检查，语法通过。

#### 2. 成功的三图 smoke run

`runs_smoke/carpaint-jitdiff-3img-smoke` 已经成功跑通了一次真实的 train + val：

- [carpaint-jitdiff-3img-smoke.jsonl](/home/max/my-prismatic-vlm/runs_smoke/carpaint-jitdiff-3img-smoke/carpaint-jitdiff-3img-smoke.jsonl)

关键结果：

- Train Examples：`1550`
- Val Examples：`450`
- Step 1 Train Loss：`8.0859375`
- Step 1 Val Loss：`8.23828125`
- Val Batches：`8`

这说明：

- 三图 JSON 被 dataset 正确读取
- collator 正确构造了多图 batch
- `PrismaticVLM.forward()` 的交错插入路径能完整跑通
- eval 路径也能使用同样契约运行

#### 3. 大 batch 正式 run 的失败归因

后续正式 run 失败不是多图逻辑错误，而是显存问题：

- 失败日志见 [output.txt](/home/max/output.txt#L1632)
- 直接异常是 `torch.OutOfMemoryError`
- 发生位置在 Qwen loss 计算阶段
- 当时配置使用了更激进的：
  - `per_device_batch_size=16`
  - `global_batch_size=128`

因此应把这个失败解释为：

- 代码路径是可运行的
- 但三图输入把显存和序列长度压力显著放大了
- 单图时期的 batch 配置不能直接照搬

### 主要剩余风险

#### 1. sampler 长度估计仍偏单图假设

当前：

- `FinetuneDataset.get_modality_lengths()` 仍然只统计文本词数
- `SplitModalitySampler` 仍按旧思路估计长度

影响：

- 功能是对的
- 但 batch packing 和 OOM 预估会不够精确

#### 2. 当前支持范围是定制组合，不是全仓库通用能力

本轮只覆盖：

- Qwen3VLText
- PurePromptBuilder
- 当前这条 finetune 链路

影响：

- 不能把这次改动表述成“仓库已经全面支持多图”
- 换其他 prompt builder 或其他 LLM backbone 时，默认不能假设同样成立

#### 3. eval batch size 仍继承 train batch size

当前验证 dataloader 直接复用 `per_device_batch_size`。

影响：

- 训练可能能跑，eval 先爆
- 正式 run 中已经出现了这个现象

#### 4. 多图天然带来上下文长度与显存膨胀

每增加一张图，都会增加一整段视觉 patch block。

影响：

- 这不是实现 bug，而是这个方案天然的代价
- 配置必须更保守，尤其是 `per_device_batch_size`

### 后续建议

1. 保留当前多图核心实现，不建议回退成“统一插到 BOS 后”的简化版本。  
   当前实现才真正满足方案二。

2. 正式训练默认采用保守 batch 配置。  
   对三图版本，优先使用已经 smoke 证明可运行的小 batch。

3. 下一步最值得补的是 sampler 长度估计。  
   这会改善 batch 组织质量和 OOM 可预测性。

4. 如有必要，可单独引入更小的 eval batch size。  
   这能降低评估阶段的显存风险。

5. 如果后续要支持更广泛的模型组合，应把其他 prompt builder / backbone 作为新的独立工作项。  
   不建议在当前分支里默认外推。

### 最终判断

对于当前限定的 `DINOv3 + Qwen3VLText + finetune + PurePromptBuilder` 组合，这次多图适配的核心代码改动是成立的。

最关键的一点，即：

- 让多个 `<image>` 占位符在 `PrismaticVLM.forward()` 中各自替换成对应图像的 patch block

已经在数据层、collator、训练前向、验证前向和 generation 接口层面形成了闭环。

已经跑通的三图 smoke run 说明这套逻辑不仅静态上自洽，而且动态上能实际执行。

当前暴露出来的主要问题是大 batch 配置下的显存压力，而不是多图交错实现本身的逻辑错误。

## 阶段性完成日志

日期：2026-03-17

本阶段可以视为“当前目标范围内的多图适配代码基本完成”，范围限定为：

- `DINOv3 + Qwen3VLText + finetune + PurePromptBuilder`
- 当前 `jit_diff` 三图车漆数据链路
- 训练、验证、推理三条主路径

本阶段完成内容：

- `FinetuneDataset` 支持 `images` 多图字段，并校验 `images` 数量与 `<image>` 占位符数量一致
- `PaddedCollatorForLanguageModeling` 支持多图 batch，并显式输出 `image_attention_mask`
- `PrismaticVLM.forward()` 支持按 `<image>` token 位置交错插入对应图像 patch block
- `generate()` / `generate_batch()` 支持单样本多图输入
- `TrainingStrategy` 训练与验证路径透传多图所需字段
- `Qwen3VLTextLLMBackbone` 为 `<image>` 注册专用 token，支撑多图占位符对齐
- 保留 `PurePromptBuilder` 中的 `<image>`，使多图占位符可以从数据层一路传递到模型前向
- checkpoint 保存间隔已调整为每 `100` step 保存一次

阶段性验证结果：

- 三图 smoke run 已跑通，说明训练前向、验证前向与 checkpoint 保存链路可执行
- 当前限定路径下，单图旧格式与多图新格式均可被现有代码处理
- 三图数据集的图像数与 `<image>` 数量检查通过
- 当前三图样本不会触发序列长度截断

当前结论：

- 多图适配核心逻辑未发现明显实现错误
- 当前分支可以视为“多图适配阶段性完成”
- 后续若继续优化，重点应转向训练目标、prompt/输出格式、类别偏置与评估口径，而不是多图输入管线本身
