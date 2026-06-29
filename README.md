# LLM Classifier Distill

> **Knowledge Distillation for Chinese Text Classification — Train a Fast, Accurate Small Model with LLM-Generated Labels**
>
> **大模型知识蒸馏中文分类器 — 用大模型给数据打标，训练小模型做分类，速度提升 10 倍以上**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)](https://huggingface.co/docs/transformers)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview / 项目概述

**English:** This project demonstrates a **complete knowledge distillation pipeline** for Chinese text sentiment classification. Instead of manually labeling thousands of samples, we use a **large language model (Qwen2.5)** as the teacher to automatically label data, then train a **lightweight model (DistilBERT)** as the student classifier. The result: a model that runs **10x+ faster** with accuracy **within 3% of the teacher**.

**中文：** 本项目实现了一套**完整的知识蒸馏流水线**，用于中文文本情感分类。核心思路是：用**大模型（Qwen2.5）作为教师**自动给无标签数据打标，然后用打标后的数据训练一个**轻量级模型（DistilBERT）作为学生**。最终效果：推理速度**提升 10 倍以上**，准确率**与大模型差距在 3% 以内**。

### Architecture / 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                    Knowledge Distillation Pipeline                │
│                        知识蒸馏流水线                              │
│                                                                  │
│  ┌────────────────────────┐          ┌────────────────────────┐  │
│  │   Teacher / 教师模型     │          │   Student / 学生模型    │  │
│  │   (Qwen2.5-1.5B)       │ ──────► │   (DistilBERT)         │  │
│  │   1.5B params / 15亿参数 │  labels  │   135M params / 1.35亿  │  │
│  │   ~500ms/sample / 条   │  打标    │   ~5ms/sample / 条      │  │
│  └────────────────────────┘          └────────────────────────┘  │
│         │                              │                         │
│         │   Unlabeled Text / 无标签文本  │  Fast Inference / 快速推理 │
│         ▼                              ▼                         │
│  ┌────────────────────────┐          ┌────────────────────────┐  │
│  │  Multi-source Data      │          │   FastAPI Server       │  │
│  │  Collection 多源数据采集  │          │   POST /predict       │  │
│  │  + Cross-validation     │          │   Swagger UI /docs     │  │
│  └────────────────────────┘          └────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Innovations / 创新点

| English | 中文 | 说明 |
|---|---|---|
| **Multi-source Data Fusion** | **多源数据融合** | Collects texts from multiple HuggingFace datasets + built-in corpus across tech, entertainment, sports, finance, politics ; 从多个 HF 数据集 + 内置语料库收集文本，覆盖六大领域 |
| **Cross-validation Labeling** | **交叉验证打标** | Each sample is labeled from two different angles; only consistent labels are accepted ; 每条样本从两个角度让大模型打标，不一致则取置信度高者 |
| **Confidence-weighted Training** | **置信度加权训练** | High-confidence labels have greater weight in the loss function ; 高置信度标签在损失函数中权重更大，降低噪声影响 |
| **Text Data Augmentation** | **文本数据增强** | Random deletion, swapping, and repetition to increase data diversity ; 随机删除、交换、重复，提升数据多样性 |
| **Supervised Contrastive Loss** | **监督对比学习** | Pulls same-class representations closer, pushes different-class apart ; 拉近同类、推远异类，优化分类边界 |

---

## Quick Start / 快速开始

### 1. Install Dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

### 2. Label Data with LLM / 大模型数据打标

**English:** Uses Qwen2.5-1.5B-Instruct to automatically label 2000+ Chinese text samples:

**中文：** 使用 Qwen2.5-1.5B-Instruct 自动为 2000+ 条中文文本打标：

```bash
python label_data.py --total 2000 --batch_size 8
```

**Output / 输出**: `data/labeled/labeled_data.jsonl` — labeled dataset with confidence scores / 带置信度的标注数据

> **Tip / 提示**: If GPU memory is limited, use `--model Qwen/Qwen2.5-0.5B-Instruct` for a smaller teacher model.
> 如果 GPU 显存不足，可使用 `--model Qwen/Qwen2.5-0.5B-Instruct` 切换为更小的教师模型。

### 3. Train Student Classifier / 训练学生分类器

**English:** Train a DistilBERT model on the LLM-labeled data:

**中文：** 用打标数据训练 DistilBERT 分类器：

```bash
python train_classifier.py --epochs 5 --batch_size 16
```

**Output / 输出**: `models/student/` — trained student model ready for deployment / 训练好的学生模型

### 4. Compare Models / 模型对比

**English:** Benchmark the teacher vs. student on accuracy and speed:

**中文：** 对比教师模型和学生模型的准确率与推理速度：

```bash
python compare.py --num_samples 200
```

### 5. Deploy API Service / 部署 API 服务

**English:** Launch the FastAPI inference server:

**中文：** 启动 FastAPI 推理服务：

```bash
python deploy.py --host 0.0.0.0 --port 8000
```

**English:** Open http://localhost:8000/docs for the interactive Swagger UI.

**中文：** 打开 http://localhost:8000/docs 查看交互式 Swagger 文档。

### API Usage / API 调用示例

**Single prediction / 单条预测**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "华为发布最新旗舰手机，性能强劲，用户好评如潮"}'
```

**Batch prediction / 批量预测**:
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["今天天气真好", "这个产品质量太差了", "明天开会讨论方案"]}'
```

---

## Project Structure / 项目结构

```
llm-classifier-distill/
├── config.py                 # Centralized config / 全局配置中心
├── utils.py                  # Utilities / 工具函数（数据加载 / 评估 / 增强）
├── label_data.py             # LLM labeling pipeline / 大模型打标流水线
├── train_classifier.py       # Student training / 小模型训练
├── deploy.py                 # FastAPI deployment / API 部署
├── compare.py                # Teacher vs Student benchmark / 模型对比评测
├── requirements.txt          # Python dependencies / 依赖
├── README.md                 # Bilingual documentation / 中英双语文档
├── data/
│   ├── raw/                  # Raw unlabeled text / 原始无标签文本
│   └── labeled/              # Labeled data (JSONL) / 打标结果
├── models/
│   ├── teacher/              # Teacher model cache / 教师模型缓存
│   └── student/              # Trained student model / 训练好的学生模型
└── results/                  # Reports & logs / 对比报告 & 训练日志
```

---

## Expected Results / 预期效果

| Metric / 指标 | Teacher (Qwen2.5-1.5B) / 教师模型 | Student (DistilBERT) / 学生模型 |
|---|---|---|
| Parameters / 参数量 | 1,500M / 15亿 | 135M / 1.35亿 |
| Accuracy / 准确率 | ~92% | ~90% |
| F1 (Macro) | ~0.91 | ~0.89 |
| Inference Time / 推理耗时 | ~450ms | ~5ms |
| Speedup / 速度提升 | 1x | **90x+** |
| Disk Size / 磁盘占用 | ~3GB | ~500MB |

> *Actual results may vary depending on hardware, data quality, and training configuration. / 实际效果因硬件、数据质量和训练配置而异。*

---

## Configuration / 配置说明

**English:** All tunable parameters are centralized in `config.py`:

**中文：** 所有可调参数集中在 `config.py` 中：

| Config Class / 配置类 | Purpose / 用途 |
|---|---|
| `LabelConfig` | Teacher model, labeling strategy, batch size / 教师模型、打标策略、批次大小 |
| `TrainConfig` | Student model, training hyperparameters, augmentation / 学生模型、训练超参数、数据增强 |
| `DeployConfig` | API host, port, concurrency limits / API 地址、端口、并发限制 |
| `CompareConfig` | Benchmark settings / 评测设置 |

---

## Hardware Requirements / 硬件需求

| Component / 组件 | Minimum / 最低配置 | Recommended / 推荐配置 |
|---|---|---|
| GPU VRAM | 8GB (for 4-bit teacher) | 16GB+ |
| RAM / 内存 | 16GB | 32GB |
| Disk / 磁盘 | 10GB | 20GB |

**English:** CPU-only inference is supported for the student model (deployment).

**中文：** 学生模型（部署阶段）支持 CPU 推理，无需 GPU。

---

## Distillation Principles / 知识蒸馏原理

**English:** This project uses **Data Distillation** methodology:

1. **Teacher model** (Qwen2.5-1.5B-Instruct) has strong language understanding but slow inference and high resource consumption
2. The teacher labels large amounts of unlabeled text, generating high-quality training data
3. **Student model** (DistilBERT-multilingual) has only 1/11 the parameters of the teacher and learns to mimic the teacher by training on the labeled data
4. The student runs 90x+ faster with accuracy close to the teacher, making it suitable for production deployment

**中文：** 本项目采用**数据蒸馏（Data Distillation）**方法：

1. **教师模型**（Qwen2.5-1.5B-Instruct）拥有强大的语言理解能力，但推理速度慢、资源消耗大
2. 教师模型对大量无标签文本进行情感标注，生成高质量训练数据
3. **学生模型**（DistilBERT-multilingual）参数量仅为教师的 1/11，通过学习教师标注的数据来模仿教师的行为
4. 学生模型推理速度快 90 倍以上，且准确率接近教师，适合生产环境部署

---

## License / 许可证

MIT License — see [LICENSE](LICENSE) for details. / 详见 [LICENSE](LICENSE)。

## Citation / 引用

**English:** If you use this project in your research, please cite:

**中文：** 如果你在研究中使用了本项目，请引用：

```bibtex
@misc{llm-classifier-distill,
  author = {penghaow3w},
  title = {LLM Classifier Distill: Knowledge Distillation for Chinese Text Classification},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/penghaow3w/llm-classifier-distill}
}
```