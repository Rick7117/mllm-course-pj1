# MLLM Course PJ1

这是《Multimodal Large Language Models: Algorithms, Applications \& Fine-tuning》课程的 PJ1 项目。

项目目标是在统一的 COCO val2017 数据集上，对三类视觉语言模型进行比较：

- `OpenCLIP`：contrastive alignment
- `BLIP`：matching-based alignment
- `BLIP-2`：query-based alignment

主要实验内容包括：

- 图文检索 `retrieval`
- 图像描述生成 `captioning`
- 表征可视化与最近邻分析 `representation analysis`
- 简单的组合泛化案例分析

## 项目结构

- `models.py`：模型加载与统一接口
- `dataset.py`：COCO 数据读取
- `evaluator.py`：检索与生成指标计算
- `task_retrieval.py`：图文检索实验
- `task_captioning.py`：图像描述生成实验
- `task_representation.py`：PCA / t-SNE 表征分析
- `task_nearest_neighbor.py`：最近邻与组合泛化分析
- `output/`：实验结果、缓存特征和可视化图像
- `latex_report/`：LaTeX 报告

## 数据准备

需要准备 COCO val2017 图像和标注文件，并放在项目配置所对应的 `data/` 目录下。

如果需要下载数据，可以使用：

```bash
python download_coco.py
```

## 安装依赖

建议先创建 Python 虚拟环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

## 如何使用

运行图文检索实验：

```bash
python task_retrieval.py
```

运行图像描述生成实验：

```bash
python task_captioning.py
```

运行表征可视化：

```bash
python task_representation.py
```

运行最近邻与组合泛化分析：

```bash
python task_nearest_neighbor.py
```

## 结果输出

实验结果默认保存在 `output/` 目录下：

- `output/results/`：json 格式结果
- `output/visualizations/`：PCA / t-SNE 图像
- `output/embeddings/`：缓存的特征文件

## 说明

- 当前项目主要面向课程实验复现与对比分析
- `BLIP-2` 的 retrieval 使用的是近似特征提取方案，结果主要用于实验分析，不代表其原生 retrieval 上限
