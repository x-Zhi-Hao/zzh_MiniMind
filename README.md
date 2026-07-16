# Zzh MiniMind

> 一个 25.83M 参数的轻量中文对话模型复现项目，已完整走通 **预训练 → 指令微调（SFT）→ 本地 CPU 推理**。

## 项目成果

| 项目 | 实际结果 |
| --- | --- |
| 模型 | Decoder-only Transformer，25.83M 参数（`hidden_size=512`、8 层） |
| 预训练 | 约 1.2GB 文本语料，39,695 step，最终 loss 约 2.16 |
| SFT | 28.5 万条对话数据，训练 2 个 epoch，末 step loss 约 2.19 |
| 训练环境 | AutoDL RTX 4090D 24GB，bfloat16、梯度累积、断点保存与后台日志 |
| 产物 | 最终 SFT 权重 `full_sft_real_v2_512.pth`，约 56MB |
| 推理验证 | 已在本地 Python 3.12 + CPU 环境加载权重并完成中文问答 |

这个项目是学习型复现，不宣称训练了通用大模型；重点在于理解并实现一个语言模型从数据到对话生成的完整工程闭环。

## 能力与实现要点

- 实现 Causal Language Modeling、Causal Attention、RoPE、RMSNorm 与自回归生成。
- 预训练数据中将 PAD 的 label 置为 `-100`，避免 padding 参与交叉熵损失。
- SFT 数据中只对 assistant 回复计算 loss，user/system 部分仅作为上下文。
- 兼容普通多轮对话和工具调用数据：训练时将 JSON 字符串形式的 `tools`、`tool_calls` 解析为聊天模板需要的结构。
- 支持 bfloat16、梯度累积、checkpoint 保存与恢复，并提供本地命令行对话入口。

## 项目结构

```text
zzh_MiniMind/
├─ model/                  # Transformer 模型与 tokenizer
├─ dataset/                # 预训练与 SFT 数据集实现
├─ trainer/                # 预训练、SFT 训练与工具函数
├─ weights/                # 本地下载的模型权重（已忽略，不提交 Git）
├─ chat.py                 # 本地命令行推理入口
└─ README.md
```

## 快速开始：本地推理

准备好最终 SFT 权重后，将其放到：

```text
weights/full_sft_real_v2_512.pth
```

Windows 下运行单条问题：

```powershell
.\.venv\Scripts\python.exe chat.py --prompt "请用通俗的话解释什么是机器学习" --temperature 0
```

进入多轮对话：

```powershell
.\.venv\Scripts\python.exe chat.py
```

输入 `/exit` 退出。CPU 可以完成验证；GPU 会有更快的生成速度。

## 训练流程

```mermaid
flowchart LR
    A[文本语料] --> B[预训练\nNext Token Prediction]
    B --> C[pretrain_real_512.pth]
    D[对话语料] --> E[SFT\n仅监督 assistant 回复]
    C --> E
    E --> F[full_sft_real_v2_512.pth]
    F --> G[chat.py 本地推理]
```

训练脚本入口：

```powershell
# 查看参数
.\.venv\Scripts\python.exe -m trainer.train_pretrain --help
.\.venv\Scripts\python.exe -m trainer.train_full_sft --help
```

## 环境

- Python 3.12
- PyTorch（CPU 或 CUDA 环境均可）
- Transformers
- Hugging Face Datasets（训练时需要）

---

# 学习笔记与复现记录

## 感悟

只要每一步都朝着目标方向走，肯定最快到达。对吗？

当然不是。Dijkstra 算法提出，在寻找最短路径时，必须暂时放弃直奔终点，反而要探索那些看起来绕远路的节点。算法从开始节点出发，找到距离当前节点最短、且尚未被确定下来的节点，然后用它去更新所有邻居节点的潜在距离。一旦某个节点被确定为最短路径节点，这个距离就不会再被修改。整个搜索过程像一个不断扩大的涟漪，直到某一次波浪拍打到预设的终点。

我想说的是，最短路径一定是直线，最快路径不一定是直线。

Dijkstra 算法告诉我，最快的路径往往藏在那些看起来绕远的路径上。我愿意先走弯路，是因为真正的抵达从来不是靠对准方向，而是靠看清所有的代价，试所有的错。有些人直奔终点，却被困在半途；有些人独自走了很远，反而第一个敲开了门。

所以，兄弟，别怕绕路。怕的是只顾看着方向，却忘记了体验每个节点给你的代价。经验，在试错的基础上不断发展、进步。

一开始跟着MiKio视频，难绷，打算看源码问ai，结果豆包没水平，就想着整个gpt，倒腾了一下午，才在朋友的帮助下搞上了GPT plus。现在学习轻松多了。，

## 什么是复现？

什么是复现？我起初以为是下载别人的代码、配置环境、运行代码。其实不是，复现应该是：还原作者的逻辑思路！

2026 年，我是 27 届应届生。为了找一份实习，我海投简历。有个 HR 在 BOSS 上问了我几个问题，其中一个就是：是否复现过 GitHub 项目与论文算法，有无个人代码仓库、是否熟练使用 Git 命令？

于是我开始试着复现 MiniMind。如开头所说，我太天真了，事实没那么简单。好在 B 站找到了 UP 主，可以跟着学。UP 主声称“三小时三元，复现 MiniMind”，确实可以在三小时内复现，但前提是你得具备机器学习、深度学习的扎实基本功。我又天真了。这个复现至少要几周时间，像我这样 0.1 基础的人，只能跟着 UP 主学，不能自己写代码。

有句话很好：先装模做样，然后像模像样，最后有模有样。所以先简单跟着来一遍，再自己写代码。两次不行就三次，三次不行就再多来几次。

## MiniMind 复现流程

### 1. 复现四大块代码

### 2. 本质

按着下面流程图，逐个写代码块。

<p align="center">
  <img src="./images/LLM-structure.jpg" width="700" alt="MiniMind LLM 结构图">
</p>

<p align="center">
  <img src="./images/LLM-structure-moe.jpg" width="700" alt="MiniMind MoE 结构图">
</p>

0.1 基础的朋友，可以看一下：

- [MiniMind 复现视频（一）](https://www.bilibili.com/video/BV1TZ421j7Ke?vd_source=17a936cc4bdf336ce68bd6a091daa956)
- [MiniMind 复现视频（二）](https://www.bilibili.com/video/BV13z421U7cs?vd_source=17a936cc4bdf336ce68bd6a091daa956)

## 3. 基础知识

### 什么是 GPT？

GPT（Generative Pre-trained Transformer）是 OpenAI 提出的、基于 Transformer 的语言模型。GPT 的核心是 Transformer（模型），它使用 Attention 机制处理序列信息。

| 术语 | 说明 |
| --- | --- |
| Token | 单词切片，对应一组向量（试图表达片段含义）。类聚是空间中的方向，可以承载语义。 |
| Tensor | 高维数组。 |
| Weight | 权重。 |
| Embedding matrix | 嵌入矩阵。 |
| QKV 三元组 | Q：query matrix；V：value matrix；K：key matrix。 |
| Softmax | 激活函数。 |
