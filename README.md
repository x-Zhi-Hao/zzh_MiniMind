# 概述

## 什么是复现？

 <p style="text-indent: 2em;">什么是复现？我起初以为是下载别人代码，配置环境，运行代码。其实不是，复现应该是：还原作者逻辑思路！
<p style="text-indent: 2em;">今年2026年，我是27应届生，为了找一份实习，海投简历，有个HR在boss上问了我几个问题，其中一个就是：
是否复现过 GitHub 项目与论文算法，有无个人代码仓库、熟练使用 Git 命令？
<p style="text-indent: 2em;">于是我开始试着复现MiniMind，如开头所说，我太天真了，事实没那么简单。好在B站找到了up主，可以跟着学，up主声称：三小时三元，复现MiniMind,，确实可以在三小时内复现，但前提是你得具备机器学习，深度学习的扎实基本功。我又天真了。这个复现至少要几周时间，像我这样0.1基础的人，只能跟着up主学，不能自己写代码。
<p style="text-indent: 2em;">有句话很好：先装模做样，然后像模像样，最后有模有样。所以先简单跟着来一遍，再自己写代码。两次不行就三次，三次不行就再多来几次。

## MiniMind复现流程
### 1.复现四大块代码

### 2.本质

按着下面流程图，逐个写代码块


<img src="images\LLM-structure.jpg" width="700" alt="minmind">


<img src="images\LLM-structure-moe.jpg" width="700" alt="minmind">

0.1基础的朋友，可以看一下：
https://www.bilibili.com/video/BV1TZ421j7Ke?vd_source=17a936cc4bdf336ce68bd6a091daa956

https://www.bilibili.com/video/BV13z421U7cs?vd_source=17a936cc4bdf336ce68bd6a091daa956

### 3.基础知识
#### 什么是GPT？
<p style="text-indent: 2em;">GPT（Generative Pre-trained Transformer）是 Google 推出的一种基于 Transformer 的语言模型。GPT 的核心是 Transformer（模型），它是一种Attention机制的模型。
<p style="text-indent: 2em;">Token 单词切片 对应一组向量（试图表达片段含义） 类聚  空间中的方向，可以承载语义
<p style="text-indent: 2em;">Tensor 高维数组
<p style="text-indent: 2em;">Weight 权重
<p style="text-indent: 2em;">Embedding matrix 嵌入矩阵
<p style="text-indent: 2em;">QKV 三元组 Q: query matrix V: value matrix K: key matrix
<p style="text-indent: 2em;">Softmax 激活函数 
