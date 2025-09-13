# Full Fine-tuning

## 简介

没有经过 Fine-tuning 的预训练模型只能续写文本，无法进行诸如聊天、翻译等任务。

比 Prompt 工程更彻底的解决方法是在基础模型上针对特定的大量内容进行 FFT（Full Fine-Tuning），FFT 指的是直接更新模型参数来适配具体任务的方法。比如我们有几万条“给文章写标题”的数据，可以通过 FFT 一个已有的大模型，把它变成一个专门用于“给文章写标题”的模型。这类机制在有些地方也被称为 Domain 或 Task Adapatation，用带标注的 Domain 或 Task 数据有监督微调基座模型。

### 不足

Full Fine-Tuning 比较适合特化的任务或风格，但也存在一些问题：

- 没有解决事实性问答可靠性的问题：
- 消耗的资源量虽然相对大模型预训练减少，但还是不容小觑的：比如 Alpaca  的微调，据作者介绍他们使用 8 个 显存 80GB A100 ，花费了 3 个小时。如果领域支持频繁更新，且需要需要较高的实时性，显然是无法满足要求的。
- 需要构建特定领域微调的训练语料：可以参考[Dataset Engineering for LLM finetuning](https://www.flowrite.com/blog/dataset-engineering-llm-finetuning)。如果想要获得较好的结果，高质量训练数据集的构建需要精心设计，开销也是不容忽视的。
- 微调的结果不一定符合预期：在[ChatGLM-6B微调及推理实践](https://km.woa.com/articles/show/575742) 一文中可以发现，作者在尝试使用ADGEN数据集微调后，模型对“广告词生成”任务的确变好，但其他任务的回答均不如原始模型。

## 标准流程

### 模型准备



### 数据准备

理想情况下，用于微调的数据集应该是以下格式的 JSONL 文件。根据需要应用的场景不同而用不同的姿势进行填入：

- prompt：通常是一些更抽象的描述，如一篇文章中的标题、摘要、一个人的头衔等。
- completion：通常是更详细具象的描述，如一篇文章优化部分的内容、一个人在某职位上的具体工作等。

```json
{"prompt": "<text>", "completion": "<text to be generated>"}
{"prompt": "<text>", "completion": "<text to be generated>"}
{"prompt": "<text>", "completion": "<text to be generated>"}
```

### 启动微调

对于分类任务，

```json
{"prompt":"Company: BHFF insurance\nProduct: allround insurance\nAd:One stop shop for all your insurance needs!\nSupported:", "completion":" yes"}
{"prompt":"Company: Loft conversion specialists\nProduct: -\nAd:Straight teeth in weeks!\nSupported:", "completion":" no"}
```

completion是一个小空间（相对的）中的结果，比如“是否”、“中高低”、“东南西北中”，“火之高兴和霜之哀伤”。

### 使用模型



## 开源LLM

FFT 使用下游特定领域的知识对基础模型进行更新，它改变了神经网络中参数的权重。业界已经不少 ChatGPT 的平替方案都支持 FFT。主流的开源大语言模型主要有三个：LLaMA、ChatGLM 和 BLOOM。基于这三个开源模型，业界进行了指令微调或强化学习，衍生出了许多不同的大模型。下面从训练数据、tokenizer 和模型结构上对这三个大语言模型进行比较。

| 模型       | 训练数据                             | 训练数据量     | 模型参数量                       | 词表大小 |
| ---------- | ------------------------------------ | -------------- | -------------------------------- | -------- |
| LLaMA      | 以英语为主的拉丁语系，不包含中日韩文 | 1T/1.4T tokens | 7B、13B、33B、65B                | 32000    |
| ChatGLM-6B | 中英双语，中英文比例为1:1            | 1T tokens      | 6B                               | 130528   |
| Bloom      | 46种自然语言和13种编程语言，包含中文 | 350B tokens    | 560M、1.1B、1.7B、3B、7.1B、176B | 250880   |

| 模型       | 模型结构       | 位置编码 | 激活函数 | layer norm     |
| ---------- | -------------- | -------- | -------- | -------------- |
| LLaMA      | Casual decoder | RoPE     | SwiGLU   | Pre RMS Norm   |
| ChatGLM-6B | Prefix decoder | RoPE     | GeGLU    | Post Deep Norm |
| Bloom      | Casual decoder | ALiBi    | GeLU     | Pre Layer Norm |

### LLaMA

LIMA 证明了 **LLM 的几乎所有知识都是在预训练过程中学习到的**，只需要有限的指令微调数据就可以生成高质量的回复。因此，基座模型的性能是至关重要的，如果基座模型的性能不够好，指令微调和强化学习也难以取得很好的效果。Fine-tuning 主流的思路就是用好的数据来微调 [LLaMA](https://huggingface.co/decapoda-research/llama-65b-hf) 这样的开源模型，LLaMA 是 Meta 开源的预训练模型，是现在市面上预训练做得最好的开源、类似 GPT-3 的模型，它只做了预训练。它提供了 70 亿、130 亿、650 亿 3 个参数规格的模型。它支持多种语言，其中英文效果最佳，中文和其他语言的效果比英语差一些。

LLaMA 是Meta提出的大语言模型。训练数据是以英语为主的拉丁语系，另外还包含了来自GitHub的代码数据。训练数据以英文为主，不包含中韩日文，所有训练数据都是开源的，分词之后大约有1400B 的 token。 

![c6af898572224be3889cb69c978d4820](figures/c6af898572224be3889cb69c978d4820.webp)

#### 子模型

按照模型参数量，LLaMA 模型有 7B、13B、33B、65B 这四个不同参数规模的模型版本。7B 和 13B 版本使用了 1T 的 token 进行训练，33B 和 65B 的版本使用了 1.4T 的 token进行训练。

#### 网络结构

模型结构上，与 GPT 相同，LLaMA 采用了 causal decoder-only 的 Transformer。在模型细节上，做了以下几点改动：

-  layer normalization：为了提升训练的稳定性，没有使用传统的 post layer norm，而是使用了pre layer Norm。具体地，去除了 layer normalization 中的偏置项，采用了 RMS Norm（即均方根 Norm）。
- 激活函数：没有采用 ReLU激 活函数，而是采用了 SwiGLU 激活函数。FFN 通常有两个权重矩阵，先将向量从维度 $d$ 升维到中间维度 $4d$，再从 $4d$ 降维到 $d$。而使用 SwiGLU 激活函数的 FFN 增加了一个权重矩阵，共有三个权重矩阵，为了保持参数量一致，中间维度采用了 $2/3\times 4d$，而不是 $4d$。
- 位置编码：去除了绝对位置编码，采用了旋转位置编码 RoPE。

在**训练目标**上，LLaMA 的训练目标是语言模型，即根据已有的上文去预测下一个词。

关于**tokenizer**，LLaMA 的训练语料以英文为主，使用了 Sentence Piece 作为 tokenizer，词表大小只有 32000。词表里的中文 token 很少，只有几百个，LLaMA tokenizer 对中文分词的编码效率比较低。

#### 微调后模型

基于 LLaMA 进行 Instruction Fine-tuning 后的模型有：

- [Vicuna-7b](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1)：是用 70k 指令对 LLaMA 进行了指令微调，它微调了 70 亿（7b）参数的 LLaMA 版本，是当前开源的做完了指令微调的模型里效果最好的。
- [Alpaca-lora-7b](https://huggingface.co/tloen/alpaca-lora-7b)：[Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 是在 Meta 提出的 LLaMA 7b 参数的模型基础上微调的结果，但它效果一般。但它开源了 52k 条很有价值的指令微调训练数据，使用 ChatGPT 来生成更多训练数据进行 fine-tuning 的模型。self-instruct（[SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions](https://arxiv.org/pdf/2212.10560.pdf)）这个思路非常有趣，其实大家都会自然而然有这种想法：既然有了 ChatGPT 这个效果特别好的模型，为什么不直接只搜集 QA 指令问题，然后用 ChatGPT 给我生成 QA 里的 Answer，并且生成更多类似的 QA 对呢？原生的 Alpaca 对中文的支持并不好，不过已经业界也做了些[扩充中文词表的开源方案](https://link.juejin.cn/?target=https://arxiv.org/pdf/2304.08177v1.pdf)。同时，Alpaca 是一个 LoRA 方法下的模型。

#### LLaMA 2

##### 超参数

- **模型尺寸**：LLaMA 2 提供了三种不同的模型尺寸：7B、13B 和 70B。其中，7B 和 13B 的架构与 LLaMA 1 相同，可直接用于商业应用。
- **超参数**：使用 AdamW 优化器进行训练，其中 β1=0.9，β2=0.95，eps=10−5。使用余弦学习率计划，预热 2000 步，衰减最终学习率降至峰值学习率的 10%。使用 0.1 的权重衰减和 1.0 的梯度裁剪。

##### 架构

- **模型架构**：LLaMA 2 采用了LLaMA  1 的大部分预训练设置和模型架构，使用标准 Transformer 架构，使用 RMSNorm 应用预归一化、使用 SwiGLU 激活函数和旋转位置嵌入 RoPE。与 LLaMA 1 的主要架构差异包括增加了上下文长度和分组查询注意力（GQA）。
- **分组查询注意力（GQA）**：这是一个新的注意力机制，可以提高大模型的推理可扩展性。它的工作原理是将键和值投影在多个头之间共享，而不会大幅降低性能。可以使用具有单个 KV 投影的原始多查询格式（MQA）或具有8KV投影的分组查询注意力变体（GQA）。
- **分词器**：LLaMA 2 使用与 LLaMA 1 相同的分词器；它采用字节对编码（BPE）算法，使用 SentencePiece 实现。与 LLaMA 1 一样，将所有数字拆分为单独的数字，并使用字节来分解未知的 UTF-8 字符，总数词汇量为 32k 个 token。

##### 训练流程

- **训练**：LLaMA 2 模型经过了 2 万亿个 taken 的训练，其上下文长度是 LLaMA 1 的两倍。此外，LLaMA-2-chat 模型还接受了超过 100 万个新的人类注释的训练。LLaMA 2 的训练语料比 LLaMA  1 多出40%，上下文长度从 2048 增加到 4096，使其能够理解和生成更长的文本。
- **预训练**：LLaMA 2 使用公开的在线数据进行预训练，然后通过有监督微调创建 LLaMA-2-chat 的初始版本。接下来，LLaMA-2-chat 使用 RLHF 进行迭代细化，其中包括 PPO。
- **微调**：LLaMA 2-Chat 是数月实验研究和对齐技术迭代应用的结果，包括指令微调和 RLHF，需要大量的计算和数据标注资源。有监督微调指令数据质量非常重要，包括多样性，注重隐私安全不包含任何元用户数据。

### ChatGLM

清华大学于 2023 年 3 月提出了支持中英双语的 [ChatGLM-6b](https://github.com/THUDM/ChatGLM-6B) 模型，它具有 62 亿参数，可以在消费级显卡上部署，INT4 量化级别下最低只需要 6 GB 显存。

ChatGLM 是基于清华大学自己设计的预训练模型架构 [General Language Model](https://arxiv.org/pdf/2103.10360.pdf) 微调而来的聊天模型，它采用了 Transformer 架构，更像 BERT 的 encoder。

#### 子模型

- ChatGLM-6b：预训练用了 1T token 的中英文语料，但未透露具体用了多少语料进行指令微调，其中文聊天的效果是开源模型里最好的。
- ChatGLM-13b：同时，ChatGLM 还有一个 130 亿的版本预训练模型，但没有对外开源。

#### 网络结构

从网络结构上，ChatGLM-6B 采用了 prefix decoder-only 的 Transformer，在输入上采用双向的注意力机制，在输出上采用单向注意力机制。在模型细节上，做了以下几点改动：

- Embedding 层梯度缩减：为了提升训练稳定性，减小了 Embedding 层的梯度。梯度缩减的效果相当于把 Embedding 层的梯度缩小了10倍，减小了梯度的范数。
- layer normalization：采用了基于 Deep Norm 的 post layer norm。
- 激活函数：采用了 GeGLU 激活函数。相比于普通的 FFN，使用线形门控单元的 GLU 新增了一个权重矩阵，共有三个权重矩阵，为了保持参数量一致，中间维度采用了 $8/3d$，而不是 $4d$。
- 位置编码：去除了绝对位置编码，采用了旋转位置编码 RoPE。

在**训练目标**上，ChatGLM-6B 的训练任务是自回归文本填空。相比于采用 causal decoder-only 结构的 LLM，采用 prefix decoder-only 结构的 ChatGLM-6B 存在一个劣势：训练效率低。causal decoder 结构会在所有的 token 上计算损失，而 prefix decoder 只会在输出上计算损失，而不计算输入上的损失。在有相同数量的训练 tokens 的情况下，prefix decoder 要比 causal  decoder 的效果差，因为训练过程中实际用到的 token 数量要更少。另外，ChatGPT 的成功已经证明了 causal  decode r结构的 LLM 可以获得非常好的 few-shot 和 zero-shot 生成能力，通过指令微调可以进一步激发模型的能力。至于 prefix decoder 结构的 LLM 能否获得相当的 few-shot 和 zero-shot 能力还缺少足够的验证。

关于**tokenizer**，ChatGLM 在 25GB 的中英双语数据上训练了 SentencePiece 作为 tokenizer，词表大小为 130528。

### Bloom

[Bloom](https://huggingface.co/bigscience/bloom) 是参数最多的开源预训练模型，模型结构和 GPT-3 很相似，是学术界很多人一起窜起来的一个预训练模型。它只做了预训练，做 QA 任务效果较差。

BLOOM 系列模型是由 BigScience 团队训练的 LLM。训练数据包含了英语、中文、法语、西班牙语、葡萄牙语等共 46 种语言，另外还包含 13 种编程语言。1.5TB 经过去重和清洗的文本，转换为 350B 的 tokens。训练数据的语言分布如下图所示，可以看到中文语料占比为16.2%。

![ad7a92467e724295955f2ec35fdc7447](figures/ad7a92467e724295955f2ec35fdc7447.svg)

#### 子模型

按照模型参数量，BLOOM 模型有 560M、1.1B、1.7B、3B、7.1B 和 176B 这几个不同参数规模的模型。

#### 网络结构

**模型结构**上，与 GPT 相同，BLOOM 采用了causal decoder-only 的 Transformer。在模型细节上，做了以下几点改动：

- Embedding layer norm：在 Embedding 层后添加了一个 layer normalization，来使训练更加稳定。
- layer normalization：为了提升训练的稳定性，没有使用传统的 post layer norm，而是使用了 pre layer Norm。
- 激活函数：采用了 GeLU 激活函数。
- 位置编码：去除了绝对位置编码，采用了相对位置编码 ALiBi。相比于绝对位置编码，ALiBi 的外推性更好，即虽然训练阶段的最大序列长度为 2048，模型在推理过程中可以处理更长的序列。

在**训练目标**上，BLOOM 的训练目标是语言模型，即根据已有的上文去预测下一个词。

关于 **tokenizer**，BLOOM 在多语种语料上使用 BPE（Byte Pair Encoding） 算法进行训练得到 tokenizer，词表大小为 250880。

#### 微调后模型

- BLOOMZ 系列模型是在 xP3 数据集上微调得到的，推荐用于英语提示的场景。
- BLOOMZ-MT 系列模型是在 xP3mt 数据集上微调得到的，推荐用于非英语提示的场景。

### Dolly



## Lab

### ChatBLM-6b微调

【1】



## Ref

1. [ChatGLM-6B 基础模型Fine-tuning(Mini版)](https://km.woa.com/articles/show/576645)

   
