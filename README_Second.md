# zzh的复现Minmind
1.本项目复现Minmind (项目链接：https://github.com/minmind/minmind)  
2.跟着up主（Mokio）视频：https://www.bilibili.com/video/BV1T2k6BaEeC?p=6&vd_source=17a936cc4bdf336ce68bd6a091daa956

3.记录自己的学习过程也是为了加深印象

# 复现过程
## 初始化项目

### 1. 创建项目zzh_minmind


### 2. vscode 初始化项目
#### 2.1 配置uv
<img src="images\uv_install.png" width="700" alt="uv安装演示">

    init uv

#### 2.2 安装依赖
<img src="images\explain_pyproject_toml.png" width="700" alt="安装依赖">

    
```toml
[project]
name = "zzh-minimind"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = 
[
     "numpy>=2.3.4",
    "pandas>=2.3.3",
    "torch>=2.9.0",
    "transformers>=4.57.1"

]

```
终端运行`uv sync`

#### 2.3 创建项目结构
<img src="images\explain_file.png" width="700" alt="项目结构">


### 2. 复现model.py

<p style="text-indent: 2em;">RMSNorm ROPE & Yarn   GQA   FFN这四样完成了，大功就基本搞成了。
本人计科专业，对于深度学习，机器学习没什么基础，很吃力。

#### 2.1 RMSNorm

<p style="text-indent: 2em;">全称 Root Mean Square Normalization，均方根归一化
<p style="text-indent: 2em;">作用：把输入特征向量的数值压缩到稳定区间，让深层 Transformer 训练不震荡、梯度不容易消失，LLaMA、MiniMind 这类大模型全部用它替代传统 LayerNorm，计算更快。
<p style="text-indent: 2em;">其实不用理解，记住公式就好：
<img src="images\RMSNorm.png" width="700" alt="NmsNorm">

#### 2.2 RoPE & Yarn

<p style="text-indent: 2em;">RoPE 是一种旋转位置编码，用于在 Transformer 模型中处理序列数据。
<p style="text-indent: 2em;">Yarn 是一种基于 RoPE 的位置编码，用于处理长序列数据。
<p style="text-indent: 2em;">RoPE 的作用是把序列数据中的位置信息进行编码，使得模型能够处理长序列数据。Yarn 的作用是把序列数据中的位置信息进行编码，使得模型能够处理长序列数据。
<p style="text-indent: 2em;">难的要死，先复制代码，跳过这部分，up主讲的不行，自己理解更不行。

#### 2.3 GQA

#### 2.4 FFN