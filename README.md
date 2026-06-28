# LLM Classifier Distill

> **Knowledge Distillation for Text Classification: Train a Fast, Accurate Small Model with LLM-Generated Labels**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)](https://huggingface.co/docs/transformers)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project demonstrates a **complete knowledge distillation pipeline** for Chinese text sentiment classification. Instead of manually labeling thousands of samples, we use a **large language model (Qwen2.5)** as the teacher to automatically label data, then train a **lightweight model (DistilBERT)** as the student classifier. The result: a model that runs **10x+ faster** with accuracy **within 3% of the teacher**.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Knowledge Distillation                     │
│                                                              │
│  ┌──────────────────┐          ┌──────────────────┐          │
│  │   Teacher Model  │          │  Student Model   │          │
│  │  (Qwen2.5-1.5B)  │ ──────► │   (DistilBERT)   │          │
│  │   1.5B params     │  labels │   135M params     │          │
│  │   ~500ms/sample   │         │   ~5ms/sample     │          │
│  └──────────────────┘          └──────────────────┘          │
│         │                              │                     │
│         │   Unlabeled Text             │  Fast Inference     │
│         ▼                              ▼                     │
│  ┌──────────────────┐          ┌──────────────────┐          │
│  │  Multi-source     │          │   FastAPI Server  │          │
│  │  Data Collection  │          │   POST /predict   │          │
│  └──────────────────┘          └──────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

### Key Innovations

| Innovation | Description |
|---|---|
| **Multi-source Data Fusion** | Collects texts from multiple HuggingFace datasets + built-in corpus across tech, entertainment, sports, finance, politics |
| **Cross-validation Labeling** | Each sample is labeled from two different angles; only consistent labels are accepted |
| **Confidence-weighted Training** | High-confidence labels have greater weight in the loss function |
| **Text Data Augmentation** | Random deletion, swapping, and repetition to increase data diversity |
| **Supervised Contrastive Loss** | Pulls same-class representations closer, pushes different-class apart |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Label Data with LLM

Uses Qwen2.5-1.5B-Instruct to automatically label 2000+ Chinese text samples:

```bash
python label_data.py --total 2000 --batch_size 8
```

**Output**: `data/labeled/labeled_data.jsonl` — labeled dataset with confidence scores

> **Tip**: If GPU memory is limited, use `--model Qwen/Qwen2.5-0.5B-Instruct` for a smaller teacher model.

### 3. Train Student Classifier

Train a DistilBERT model on the LLM-labeled data:

```bash
python train_classifier.py --epochs 5 --batch_size 16
```

**Output**: `models/student/` — trained student model ready for deployment

### 4. Compare Models

Benchmark the teacher vs. student on accuracy and speed:

```bash
python compare.py --num_samples 200
```

### 5. Deploy API Service

Launch the FastAPI inference server:

```bash
python deploy.py --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for the interactive Swagger UI.

### API Usage

**Single prediction**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "华为发布最新旗舰手机，性能强劲，用户好评如潮"}'
```

**Batch prediction**:
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["今天天气真好", "这个产品质量太差了", "明天开会讨论方案"]}'
```

## Project Structure

```
llm-classifier-distill/
├── config.py                 # Centralized configuration
├── utils.py                  # Utility functions
├── label_data.py             # LLM-based data labeling pipeline
├── train_classifier.py       # Student model training
├── deploy.py                 # FastAPI deployment
├── compare.py                # Teacher vs. Student benchmark
├── requirements.txt          # Python dependencies
├── README.md                 # English documentation
├── README_CN.md              # Chinese documentation
├── data/
│   ├── raw/                  # Raw unlabeled text data
│   └── labeled/              # LLM-labeled data (JSONL)
├── models/
│   ├── teacher/              # Teacher model cache
│   └── student/              # Trained student model
└── results/                  # Comparison reports & logs
```

## Expected Results

| Metric | Teacher (Qwen2.5-1.5B) | Student (DistilBERT) |
|---|---|---|
| Parameters | 1,500M | 135M |
| Accuracy | ~92% | ~90% |
| F1 (Macro) | ~0.91 | ~0.89 |
| Inference Time | ~450ms | ~5ms |
| Speedup | 1x | **90x+** |
| Disk Size | ~3GB | ~500MB |

> *Actual results may vary depending on hardware, data quality, and training configuration.*

## Configuration

All tunable parameters are centralized in `config.py`:

- `LabelConfig` — Teacher model, labeling strategy, batch size
- `TrainConfig` — Student model, training hyperparameters, augmentation
- `DeployConfig` — API host, port, concurrency limits
- `CompareConfig` — Benchmark settings

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8GB (for 4-bit teacher) | 16GB+ |
| RAM | 16GB | 32GB |
| Disk | 10GB | 20GB |

CPU-only inference is supported for the student model (deployment).

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{llm-classifier-distill,
  author = {penghaow3w},
  title = {LLM Classifier Distill: Knowledge Distillation for Text Classification},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/penghaow3w/llm-classifier-distill}
}
```