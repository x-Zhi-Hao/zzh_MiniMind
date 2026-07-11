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