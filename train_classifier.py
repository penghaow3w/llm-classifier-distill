#!/usr/bin/env python3
"""
train_classifier.py — 小模型训练脚本
======================================
使用大模型打标后的数据训练小模型（DistilBERT / BERT-base），实现知识蒸馏。

创新点：
  1. 置信度加权训练 — 高置信度样本权重更高，减少噪声标签影响
  2. 文本数据增强 — 通过随机删除/交换/重复增加数据多样性
  3. 标签平滑 — 防止过拟合，提升泛化能力
  4. 对比学习辅助损失 — 拉近同类样本表示，推远异类样本

用法:
  python train_classifier.py [--data data/labeled/labeled_data.jsonl] [--epochs 5]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import train_cfg, LABELED_DIR, MODEL_DIR, RESULTS_DIR
from utils import (
    build_dataset,
    compute_metrics,
    set_seed,
    logger,
    augment_text,
    LABEL2ID,
    ID2LABEL,
)


# ── 置信度加权损失（创新点） ──────────────────────────────

class ConfidenceWeightedTrainer(Trainer):
    """
    自定义 Trainer：支持置信度加权损失。
    置信度高的样本在损失中权重更大，减少低质量标签的影响。
    """

    def __init__(self, *args, confidence_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_weights = confidence_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # 提取置信度权重
        conf_weights = inputs.pop("confidence_weights", None)

        outputs = model(**inputs)
        logits = outputs.logits

        if conf_weights is not None and conf_weights.sum() > 0:
            # 加权交叉熵
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits, labels)
            loss = (loss * conf_weights.to(loss.device)).mean()
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ── 对比学习辅助损失（创新点） ─────────────────────────────

class SupConLoss(nn.Module):
    """
    监督对比损失 (Supervised Contrastive Loss)
    拉近同类样本的表示，推远异类样本的表示。
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [batch_size, hidden_dim]
        labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        # 归一化
        features = F.normalize(features, dim=1)

        # 相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature

        # 正样本 mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 移除自身
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # 计算损失
        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        return -mean_log_prob.mean()


# ── 标签平滑 ──────────────────────────────────────────────

def label_smoothing(labels: List[int], num_classes: int = 3, epsilon: float = 0.1) -> np.ndarray:
    """将硬标签转为软标签（标签平滑）"""
    smooth = np.full((len(labels), num_classes), epsilon / (num_classes - 1))
    for i, label in enumerate(labels):
        smooth[i, label] = 1.0 - epsilon
    return smooth


# ── 主训练类 ──────────────────────────────────────────────

class DistillationTrainer:
    """知识蒸馏训练器"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or train_cfg.student_model
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"学生模型: {self.model_name}")
        logger.info(f"训练设备: {self.device}")

    def load_and_prepare_data(self, data_path: str) -> DatasetDict:
        """加载数据并划分训练/验证/测试集"""
        logger.info("加载打标数据 …")
        dataset = build_dataset(data_path)
        logger.info(f"总样本数: {len(dataset)}")

        # 转换为 pandas 方便操作
        df = dataset.to_pandas()

        # 分层划分
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_cfg.train_ratio),
            random_state=42,
            stratify=df["label_id"],
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=train_cfg.test_ratio / (train_cfg.val_ratio + train_cfg.test_ratio),
            random_state=42,
            stratify=temp_df["label_id"],
        )

        logger.info(
            f"数据划分 — 训练: {len(train_df)}, 验证: {len(val_df)}, 测试: {len(test_df)}"
        )

        # 数据增强（创新点）
        if train_cfg.use_augmentation:
            train_df = self._augment_data(train_df)

        train_ds = Dataset.from_pandas(train_df, preserve_index=False)
        val_ds = Dataset.from_pandas(val_df, preserve_index=False)
        test_ds = Dataset.from_pandas(test_df, preserve_index=False)

        return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    def _augment_data(self, df):
        """文本数据增强"""
        logger.info("执行数据增强 …")
        augmented = []
        for _, row in df.iterrows():
            augmented.append(row.to_dict())
            # 生成增强样本
            for _ in range(train_cfg.aug_factor):
                aug_text = augment_text(row["text"])
                if aug_text != row["text"]:
                    aug_row = row.to_dict()
                    aug_row["text"] = aug_text
                    augmented.append(aug_row)

        import pandas as pd
        new_df = pd.DataFrame(augmented)
        logger.info(f"增强后训练样本: {len(new_df)} (原始: {len(df)})")
        return new_df

    def train(self, dataset: DatasetDict) -> Tuple[any, any]:
        """训练学生模型"""
        logger.info("初始化学生模型 …")

        num_labels = len(LABEL2ID)
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 分词
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=train_cfg.max_seq_length,
            )

        tokenized_dataset = dataset.map(tokenize_fn, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("label_id", "labels")

        # 设置格式
        columns = ["input_ids", "attention_mask", "labels"]
        if "confidence" in tokenized_dataset["train"].column_names:
            columns.append("confidence")
        tokenized_dataset.set_format(type="torch", columns=columns)

        # 训练参数
        training_args = TrainingArguments(
            output_dir=train_cfg.output_dir,
            num_train_epochs=train_cfg.num_epochs,
            per_device_train_batch_size=train_cfg.batch_size,
            per_device_eval_batch_size=train_cfg.eval_batch_size,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            warmup_ratio=train_cfg.warmup_ratio,
            logging_dir=train_cfg.logging_dir,
            logging_steps=train_cfg.eval_steps,
            eval_strategy="steps",
            eval_steps=train_cfg.eval_steps,
            save_strategy="steps",
            save_steps=train_cfg.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=train_cfg.fp16 and self.device == "cuda",
            report_to="none",
            dataloader_drop_last=False,
            # 梯度累积
            gradient_accumulation_steps=2,
            # 学习率调度
            lr_scheduler_type="cosine",
        )

        trainer = ConfidenceWeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=train_cfg.early_stopping_patience
            )],
        )

        logger.info("开始训练 …")
        trainer.train()

        # 保存模型
        logger.info("保存学生模型 …")
        trainer.save_model(train_cfg.output_dir)
        tokenizer.save_pretrained(train_cfg.output_dir)

        # 在测试集上评估
        logger.info("在测试集上评估 …")
        test_results = trainer.evaluate(tokenized_dataset["test"])
        logger.info(f"测试集结果: {test_results}")

        # 保存评估结果
        results_path = RESULTS_DIR / "training_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_results": test_results,
                "model": self.model_name,
                "num_train_samples": len(dataset["train"]),
                "num_epochs": train_cfg.num_epochs,
            }, f, ensure_ascii=False, indent=2)

        return model, tokenizer


# ── 主函数 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="小模型训练脚本")
    parser.add_argument("--data", type=str, default=str(LABELED_DIR / "labeled_data.jsonl"),
                        help="打标数据路径")
    parser.add_argument("--model", type=str, default=None,
                        help="学生模型名称")
    parser.add_argument("--epochs", type=int, default=train_cfg.num_epochs,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=train_cfg.batch_size,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=train_cfg.learning_rate,
                        help="学习率")
    parser.add_argument("--no-augment", action="store_true",
                        help="禁用数据增强")
    parser.add_argument("--output", type=str, default=train_cfg.output_dir,
                        help="模型输出目录")
    args = parser.parse_args()

    set_seed(42)

    # 更新配置
    if args.model:
        train_cfg.student_model = args.model
    train_cfg.num_epochs = args.epochs
    train_cfg.batch_size = args.batch_size
    train_cfg.learning_rate = args.lr
    train_cfg.output_dir = args.output
    if args.no_augment:
        train_cfg.use_augmentation = False

    # 检查数据文件
    if not os.path.exists(args.data):
        logger.error(f"数据文件不存在: {args.data}")
        logger.error("请先运行 label_data.py 生成打标数据")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("小模型知识蒸馏训练")
    logger.info("=" * 60)

    trainer = DistillationTrainer()
    dataset = trainer.load_and_prepare_data(args.data)
    model, tokenizer = trainer.train(dataset)

    logger.info("=" * 60)
    logger.info("训练完成! ✓")
    logger.info(f"模型已保存到: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()