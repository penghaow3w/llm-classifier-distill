# LLM Classifier Distill — 大模型知识蒸馏分类器

> **用大模型给数据打标，训练一个小模型做分类 — 速度提升 10 倍以上，准确率接近大模型**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)](https://huggingface.co/docs/transformers)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目概述

本项目实现了一套**完整的知识蒸馏流水线**，用于中文文本情感分类。核心思路是：用**大模型（Qwen2.5）作为教师**自动给无标签数据打标，然后用打标后的数据训练一个**轻量级模型（DistilBERT）作为学生**。最终效果：推理速度**提升 10 倍以上**，准确率**与大模型差距在 3% 以内**。

### 架构图

```
┌──────────────────────────────────────────────────────────────┐
│                       知识蒸馏流水线                           │
│                                                              │
│  ┌──────────────────┐          ┌──────────────────┐          │
│  │   教师模型 (大)    │          │   学生模型 (小)    │          │
│  │  (Qwen2.5-1.5B)  │ ──────► │   (DistilBERT)   │          │
│  │   15亿参数         │  打标    │   1.35亿参数       │          │
│  │   ~500ms/条       │         │   ~5ms/条          │          │
│  └──────────────────┘          └──────────────────┘          │
│         │                              │                     │
│         │   无标签文本                   │   快速推理           │
│         ▼                              ▼                     │
│  ┌──────────────────┐          ┌──────────────────┐          │
│  │  多源数据采集      │          │  FastAPI 服务     │          │
│  │  + 交叉验证打标    │          │  POST /predict   │          │
│  └──────────────────┘          └──────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

### 创新点

| 创新点 | 说明 |
|---|---|
| **多源数据融合** | 从多个 HuggingFace 数据集 + 内置语料库收集文本，覆盖科技、娱乐、体育、财经、时政、生活六大领域 |
| **交叉验证打标** | 每条样本从两个不同角度让大模型打标，只有结果一致才接受，不一致则取置信度高者 |
| **置信度加权训练** | 高置信度标签在损失函数中权重更大，有效降低噪声标签影响 |
| **文本数据增强** | 随机删除、交换、重复等策略增加训练数据多样性 |
| **监督对比学习** | 拉近同类样本表示、推远异类样本，提升分类边界质量 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 大模型数据打标

使用 Qwen2.5-1.5B-Instruct 自动为 2000+ 条中文文本打标：

```bash
python label_data.py --total 2000 --batch_size 8
```

**输出**: `data/labeled/labeled_data.jsonl` — 带置信度的标注数据

> **提示**: 如果 GPU 显存不足，可使用 `--model Qwen/Qwen2.5-0.5B-Instruct` 切换为更小的教师模型。

### 3. 训练学生分类器

用打标数据训练 DistilBERT 分类器：

```bash
python train_classifier.py --epochs 5 --batch_size 16
```

**输出**: `models/student/` — 训练好的学生模型

### 4. 模型对比

对比教师模型和学生模型的准确率与推理速度：

```bash
python compare.py --num_samples 200
```

### 5. 部署 API 服务

启动 FastAPI 推理服务：

```bash
python deploy.py --host 0.0.0.0 --port 8000
```

打开 http://localhost:8000/docs 查看交互式 Swagger 文档。

### API 调用示例

**单条预测**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "华为发布最新旗舰手机，性能强劲，用户好评如潮"}'
```

**批量预测**:
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["今天天气真好", "这个产品质量太差了", "明天开会讨论方案"]}'
```

## 项目结构

```
llm-classifier-distill/
├── config.py                 # 全局配置（集中管理所有参数）
├── utils.py                  # 工具函数（数据加载、评估、增强）
├── label_data.py             # 大模型数据打标流水线
├── train_classifier.py       # 小模型训练脚本
├── deploy.py                 # FastAPI 部署接口
├── compare.py                # 大模型 vs 小模型对比评测
├── requirements.txt          # Python 依赖
├── README.md                 # 英文文档
├── README_CN.md              # 中文文档
├── data/
│   ├── raw/                  # 原始无标签文本
│   └── labeled/              # 大模型打标结果 (JSONL)
├── models/
│   ├── teacher/              # 教师模型缓存
│   └── student/              # 训练好的学生模型
└── results/                  # 对比报告 & 训练日志
```

## 预期效果

| 指标 | 教师模型 (Qwen2.5-1.5B) | 学生模型 (DistilBERT) |
|---|---|---|
| 参数量 | 15亿 | 1.35亿 |
| 准确率 | ~92% | ~90% |
| F1 (Macro) | ~0.91 | ~0.89 |
| 推理耗时 | ~450ms | ~5ms |
| 速度提升 | 1x | **90x+** |
| 磁盘占用 | ~3GB | ~500MB |

> *实际效果因硬件、数据质量和训练配置而异。*

## 配置说明

所有可调参数集中在 `config.py` 中：

- `LabelConfig` — 教师模型、打标策略、批次大小
- `TrainConfig` — 学生模型、训练超参数、数据增强
- `DeployConfig` — API 地址、端口、并发限制
- `CompareConfig` — 评测设置

## 硬件需求

| 组件 | 最低配置 | 推荐配置 |
|---|---|---|
| GPU 显存 | 8GB（4bit 量化） | 16GB+ |
| 内存 | 16GB | 32GB |
| 磁盘 | 10GB | 20GB |

学生模型（部署阶段）支持 CPU 推理，无需 GPU。

## 知识蒸馏原理

本项目采用**数据蒸馏（Data Distillation）**方法：

1. **教师模型**（Qwen2.5-1.5B-Instruct）拥有强大的语言理解能力，但推理速度慢、资源消耗大
2. 教师模型对大量无标签文本进行情感标注，生成高质量训练数据
3. **学生模型**（DistilBERT-multilingual）参数量仅为教师的 1/11，通过学习教师标注的数据来模仿教师的行为
4. 学生模型推理速度快 90 倍以上，且准确率接近教师，适合生产环境部署

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。

## 引用

如果你在研究中使用了本项目，请引用：

```bibtex
@misc{llm-classifier-distill,
  author = {penghaow3w},
  title = {LLM Classifier Distill: 大模型知识蒸馏文本分类},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/penghaow3w/llm-classifier-distill}
}
```