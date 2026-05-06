# MLLM Course PJ1

这是《Multimodal Large Language Models: Algorithms, Applications \& Fine-tuning》课程的 PJ1 项目。

项目基于统一的 COCO val2017 数据集，对三类视觉语言对齐范式进行比较：

- `CLIP`：contrastive alignment
- `BLIP`：matching-based alignment
- `BLIP-2`：query-based alignment

当前实验包括：

- 图文检索 `retrieval`
- 图像描述生成 `captioning`
- 表征可视化 `PCA / t-SNE`
- 最近邻案例分析 `nearest-neighbor`
- 简单组合泛化分析 `compositional analysis`

## 当前模型

项目默认使用本地 `ModelScope` 目录中的模型。

`Task 1` 为保证公平对比，`BLIP` 和 `BLIP-2` 的 retrieval 都选择使用 COCO 数据微调后的模型，对应下载命令如下：

```bash
modelscope download --model openai-mirror/clip-vit-base-patch32 --local_dir /your/path
modelscope download --model thomas/blip-itm-base-coco --local_dir /your/path
modelscope download --model Salesforce/blip2-itm-vit-g-coco --local_dir /your/path
```

`Task 2` 只涉及 `BLIP` 和 `BLIP-2` 的 captioning 模型，对应下载命令如下：

```bash
modelscope download --model Salesforce/blip-image-captioning-base --local_dir /your/path
modelscope download --model Salesforce/blip2-opt-2.7b-coco --local_dir /your/path
```

其中，`BLIP-2` 的 retrieval 现在使用的是retrieval-specific 模型。

## 项目结构

- `config.py`：数据路径、输出目录、样本数和本地模型目录配置
- `models.py`：CLIP / BLIP / BLIP-2 的统一加载、特征提取和 captioning 接口
- `dataset.py`：COCO val2017 图像与标注读取
- `evaluator.py`：retrieval 与 captioning 指标计算
- `task_retrieval.py`：图文检索实验
- `task_captioning.py`：BLIP / BLIP-2 图像描述实验
- `task_representation.py`：PCA / t-SNE 表征分析
- `task_nearest_neighbor.py`：最近邻与组合泛化分析
- `download_coco.py`：下载 COCO val2017 数据
- `main.py`：统一任务入口
- `output/results/`：实验结果 JSON
- `output/visualizations/`：PCA / t-SNE 图像与统计文件
- `output/embeddings/`：缓存的特征文件
- `latex_report/`：LaTeX 报告

## 数据准备

需要准备以下 COCO 数据，并放在项目的 `data/` 目录下：

- `data/val2017/`
- `data/annotations/captions_val2017.json`

如果本地还没有数据，可以直接运行：

```bash
python download_coco.py
```

## 安装依赖

```bash
pip install -r requirements.txt
```

如果需要重新下载模型，可以放到 `modelscope_models/` 目录下，路径由 `config.py` 统一管理。

## 如何运行

分别运行各个任务：

```bash
python task_retrieval.py
python task_captioning.py
python task_representation.py
python task_nearest_neighbor.py
```

也可以使用统一入口：

```bash
python main.py --task retrieval
python main.py --task captioning
python main.py --task representation
python main.py --task nearest_neighbor
python main.py --task all
```

## 结果输出

实验结果默认保存在 `output/` 目录下：

- `output/results/retrieval_results.json`：检索结果
- `output/results/captioning_results.json`：captioning 指标与定性案例
- `output/results/nearest_neighbor_results.json`：最近邻案例
- `output/results/compositional_analysis.json`：组合泛化分析
- `output/visualizations/representation_analysis.json`：表征统计结果
- `output/visualizations/*.png`：PCA / t-SNE 可视化图像

## 说明

- 当前项目主要面向课程实验复现与对比分析
- 项目会缓存 embedding 到 `output/embeddings/`，切换模型后建议重新生成
- 报告源码位于 `latex_report/pj1_report.tex`

## LaTeX 报告编译

推荐直接使用项目根目录下的脚本：

```bash
./build_report.sh
```

该脚本会自动调用 `xelatex` / `latexmk`，并将最终 PDF 和中间文件统一输出到 `latex_report/` 目录下。

如果需要手动编译，也可以运行：

```bash
export PATH="$PATH:/Users/bytedance/Library/TinyTeX/bin/universal-darwin"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=latex_report -auxdir=latex_report latex_report/pj1_report.tex
```
