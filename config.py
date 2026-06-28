"""
全局配置 — 所有可调参数集中管理，方便复现实验。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ── 项目根目录 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELED_DIR = DATA_DIR / "labeled"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for _d in (DATA_DIR, RAW_DIR, LABELED_DIR, MODEL_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class LabelConfig:
    """大模型打标配置"""

    # 教师模型 — 使用 Qwen2.5 小参数版本，兼顾质量与速度
    teacher_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # 若显存不足可切换为 0.5B 版本: "Qwen/Qwen2.5-0.5B-Instruct"

    # 打标数据量
    total_samples: int = 2000
    # 每批次打标条数（避免 OOM）
    batch_size: int = 8

    # 分类类别
    labels: List[str] = field(default_factory=lambda: ["正面", "负面", "中性"])

    # 生成参数
    max_new_tokens: int = 32
    temperature: float = 0.1  # 低温度保证标签一致性
    top_p: float = 0.9

    # 量化（加速推理，节省显存）
    use_4bit: bool = True
    use_8bit: bool = False

    # 主动学习策略：只对置信度低于阈值的样本重新打标
    confidence_threshold: float = 0.85

    # 输出文件
    labeled_output: str = str(LABELED_DIR / "labeled_data.jsonl")


@dataclass
class TrainConfig:
    """小模型训练配置"""

    # 学生模型
    student_model: str = "distilbert/distilbert-base-multilingual-cased"
    # 备选: "bert-base-chinese", "google-bert/bert-base-multilingual-cased"

    # 训练参数
    num_epochs: int = 5
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = 128

    # 数据划分
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # 早停策略
    early_stopping_patience: int = 3
    eval_steps: int = 50

    # 输出目录
    output_dir: str = str(MODEL_DIR / "student")
    # 日志
    logging_dir: str = str(RESULTS_DIR / "logs")

    # 混合精度训练
    fp16: bool = True

    # 数据增强（创新点）
    use_augmentation: bool = True
    aug_factor: int = 1  # 增强倍数


@dataclass
class DeployConfig:
    """FastAPI 部署配置"""

    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = str(MODEL_DIR / "student")
    max_seq_length: int = 128
    # 请求限流
    max_concurrent_requests: int = 10


@dataclass
class CompareConfig:
    """模型对比配置"""

    teacher_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    student_model_path: str = str(MODEL_DIR / "student")
    test_data_path: str = str(LABELED_DIR / "test.jsonl")
    num_samples: int = 200  # 对比测试样本数
    warmup_runs: int = 3    # 预热轮次
    benchmark_runs: int = 10  # 基准测试轮次


# ── 全局单例 ──────────────────────────────────────────────
label_cfg = LabelConfig()
train_cfg = TrainConfig()
deploy_cfg = DeployConfig()
compare_cfg = CompareConfig()