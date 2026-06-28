#!/usr/bin/env python3
"""
compare.py — 大模型 vs 小模型对比评测
=======================================
全面对比大模型（教师）和小模型（学生）在分类准确率、推理速度、资源消耗等维度的表现。

对比维度:
  1. 准确率 (Accuracy)
  2. Macro F1
  3. 推理速度 (单条 / 批量)
  4. 推理延迟 (P50/P95/P99)
  5. 模型大小 (参数量 / 磁盘占用)
  6. GPU 显存占用
  7. 速度提升倍数

用法:
  python compare.py [--num_samples 200] [--output results/comparison_report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import compare_cfg, label_cfg, train_cfg, RESULTS_DIR, LABELED_DIR
from utils import (
    build_dataset,
    set_seed,
    logger,
    parse_label,
    ID2LABEL,
    LABEL2ID,
    get_device,
)


# ── 对比评测类 ────────────────────────────────────────────

class ModelComparator:
    """大模型 vs 小模型对比评测器"""

    def __init__(self):
        self.device = get_device()
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.student_model = None
        self.student_tokenizer = None

    def load_teacher(self, model_name: str = None):
        """加载大模型（教师）"""
        model_name = model_name or compare_cfg.teacher_model
        logger.info(f"加载教师模型: {model_name}")

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None

        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.teacher_model.eval()
        logger.info("教师模型加载完成 ✓")

    def load_student(self, model_path: str = None):
        """加载小模型（学生）"""
        model_path = model_path or compare_cfg.student_model_path
        logger.info(f"加载学生模型: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")

        self.student_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.student_model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.student_model.to(self.device)
        self.student_model.eval()
        logger.info("学生模型加载完成 ✓")

    def load_test_data(self, data_path: str = None, num_samples: int = None) -> Tuple[List[str], List[int]]:
        """加载测试数据"""
        data_path = data_path or compare_cfg.test_data_path
        num_samples = num_samples or compare_cfg.num_samples

        if not os.path.exists(data_path):
            # 尝试从 labeled_data.jsonl 中取测试集
            data_path = str(LABELED_DIR / "labeled_data.jsonl")
            if not os.path.exists(data_path):
                raise FileNotFoundError("测试数据不存在，请先运行 label_data.py")

        dataset = build_dataset(data_path)
        # 取后 num_samples 条作为测试（避免与训练数据重叠）
        df = dataset.to_pandas()
        test_df = df.tail(num_samples)

        texts = test_df["text"].tolist()
        labels = test_df["label_id"].tolist()

        logger.info(f"测试数据: {len(texts)} 条")
        return texts, labels

    def benchmark_teacher(
        self, texts: List[str], labels: List[int]
    ) -> Dict:
        """评测教师模型"""
        logger.info("=" * 60)
        logger.info("评测教师模型 (大模型) …")
        logger.info("=" * 60)

        predictions = []
        times = []

        for text in tqdm(texts, desc="教师模型推理"):
            # 构造 prompt
            messages = [
                {"role": "system", "content": "你是一个情感分析专家。请对以下文本进行情感分类，只输出：正面/负面/中性"},
                {"role": "user", "content": text},
            ]
            prompt = self.teacher_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.teacher_tokenizer(prompt, return_tensors="pt").to(self.device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = self.teacher_model.generate(
                    **inputs,
                    max_new_tokens=16,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.teacher_tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            response = self.teacher_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            label, _ = parse_label(response)
            predictions.append(LABEL2ID[label])

        from sklearn.metrics import accuracy_score, f1_score, classification_report

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")

        logger.info(f"教师模型 — Accuracy: {acc:.4f}, F1: {f1:.4f}")

        return {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "avg_inference_time_ms": round(np.mean(times) * 1000, 2),
            "p50_latency_ms": round(np.percentile(times, 50) * 1000, 2),
            "p95_latency_ms": round(np.percentile(times, 95) * 1000, 2),
            "p99_latency_ms": round(np.percentile(times, 99) * 1000, 2),
            "total_time_s": round(sum(times), 2),
            "classification_report": classification_report(
                labels, predictions,
                target_names=["正面", "负面", "中性"],
                output_dict=True,
            ),
        }

    def benchmark_student(
        self, texts: List[str], labels: List[int]
    ) -> Dict:
        """评测学生模型"""
        logger.info("=" * 60)
        logger.info("评测学生模型 (小模型) …")
        logger.info("=" * 60)

        predictions = []
        times = []

        for text in tqdm(texts, desc="学生模型推理"):
            inputs = self.student_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self.device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs = self.student_model(**inputs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            pred = outputs.logits.argmax(-1).item()
            predictions.append(pred)

        from sklearn.metrics import accuracy_score, f1_score, classification_report

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")

        logger.info(f"学生模型 — Accuracy: {acc:.4f}, F1: {f1:.4f}")

        return {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "avg_inference_time_ms": round(np.mean(times) * 1000, 2),
            "p50_latency_ms": round(np.percentile(times, 50) * 1000, 2),
            "p95_latency_ms": round(np.percentile(times, 95) * 1000, 2),
            "p99_latency_ms": round(np.percentile(times, 99) * 1000, 2),
            "total_time_s": round(sum(times), 2),
            "classification_report": classification_report(
                labels, predictions,
                target_names=["正面", "负面", "中性"],
                output_dict=True,
            ),
        }

    def get_model_stats(self) -> Dict:
        """获取模型统计信息"""
        tp = sum(p.numel() for p in self.teacher_model.parameters())
        sp = sum(p.numel() for p in self.student_model.parameters())

        # 磁盘占用
        import tempfile
        import shutil

        tmpdir = tempfile.mkdtemp()
        try:
            self.student_model.save_pretrained(tmpdir)
            disk_size = sum(
                os.path.getsize(os.path.join(tmpdir, f))
                for f in os.listdir(tmpdir)
            ) / (1024 * 1024)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return {
            "teacher_params_million": round(tp / 1e6, 1),
            "student_params_million": round(sp / 1e6, 1),
            "student_disk_mb": round(disk_size, 1),
            "compression_ratio": round(tp / sp, 1),
        }

    def run_full_comparison(
        self, texts: List[str], labels: List[int]
    ) -> Dict:
        """运行完整对比评测"""
        logger.info("=" * 60)
        logger.info("开始完整对比评测")
        logger.info("=" * 60)

        # 评测教师
        teacher_results = self.benchmark_teacher(texts, labels)

        # 评测学生
        student_results = self.benchmark_student(texts, labels)

        # 模型统计
        stats = self.get_model_stats()

        # 速度对比
        speedup = (
            teacher_results["avg_inference_time_ms"]
            / student_results["avg_inference_time_ms"]
        )
        accuracy_gap = (
            teacher_results["accuracy"] - student_results["accuracy"]
        )

        # 汇总报告
        report = {
            "comparison_summary": {
                "test_samples": len(texts),
                "device": self.device,
                "speedup_ratio": round(speedup, 1),
                "accuracy_gap": round(accuracy_gap, 4),
                "speedup_meets_target": speedup >= 10,
                "accuracy_close": abs(accuracy_gap) <= 0.05,
            },
            "accuracy_comparison": {
                "teacher_accuracy": teacher_results["accuracy"],
                "student_accuracy": student_results["accuracy"],
                "teacher_f1": teacher_results["f1_macro"],
                "student_f1": student_results["f1_macro"],
                "accuracy_difference": round(accuracy_gap, 4),
                "f1_difference": round(
                    teacher_results["f1_macro"] - student_results["f1_macro"], 4
                ),
            },
            "speed_comparison": {
                "teacher_avg_ms": teacher_results["avg_inference_time_ms"],
                "student_avg_ms": student_results["avg_inference_time_ms"],
                "teacher_p50_ms": teacher_results["p50_latency_ms"],
                "student_p50_ms": student_results["p50_latency_ms"],
                "teacher_p95_ms": teacher_results["p95_latency_ms"],
                "student_p95_ms": student_results["p95_latency_ms"],
                "teacher_p99_ms": teacher_results["p99_latency_ms"],
                "student_p99_ms": student_results["p99_latency_ms"],
                "speedup_ratio": round(speedup, 1),
            },
            "model_stats": stats,
            "teacher_classification_report": teacher_results["classification_report"],
            "student_classification_report": student_results["classification_report"],
        }

        return report

    def print_report(self, report: Dict):
        """打印格式化的对比报告"""
        s = report["comparison_summary"]

        print("\n" + "=" * 70)
        print("  🏆  大模型 vs 小模型 对比评测报告")
        print("=" * 70)

        print(f"\n📊 测试配置")
        print(f"  测试样本: {s['test_samples']} 条")
        print(f"  运行设备: {s['device']}")

        print(f"\n🎯 准确率对比")
        a = report["accuracy_comparison"]
        print(f"  教师模型 Accuracy: {a['teacher_accuracy']:.4f}")
        print(f"  学生模型 Accuracy: {a['student_accuracy']:.4f}")
        print(f"  准确率差距: {a['accuracy_difference']:.4f}")
        print(f"  教师模型 F1:     {a['teacher_f1']:.4f}")
        print(f"  学生模型 F1:     {a['student_f1']:.4f}")

        print(f"\n⚡ 推理速度对比")
        sp = report["speed_comparison"]
        print(f"  教师模型平均延迟: {sp['teacher_avg_ms']:.2f} ms")
        print(f"  学生模型平均延迟: {sp['student_avg_ms']:.2f} ms")
        print(f"  教师模型 P95 延迟: {sp['teacher_p95_ms']:.2f} ms")
        print(f"  学生模型 P95 延迟: {sp['student_p95_ms']:.2f} ms")
        print(f"  🚀 速度提升: {sp['speedup_ratio']:.1f}x")

        print(f"\n💾 模型规模")
        m = report["model_stats"]
        print(f"  教师模型参数量: {m['teacher_params_million']:.1f}M")
        print(f"  学生模型参数量: {m['student_params_million']:.1f}M")
        print(f"  压缩比: {m['compression_ratio']:.1f}x")
        print(f"  学生模型磁盘占用: {m['student_disk_mb']:.1f} MB")

        print(f"\n📋 结论")
        speed_ok = "✅" if s["speedup_meets_target"] else "❌"
        acc_ok = "✅" if s["accuracy_close"] else "⚠️"
        print(f"  {speed_ok} 推理速度提升 {s['speedup_ratio']:.1f}x {'达到' if s['speedup_meets_target'] else '未达到'} 10x 目标")
        print(f"  {acc_ok} 准确率差距 {abs(s['accuracy_gap']):.4f} {'在可接受范围内' if s['accuracy_close'] else '较大，需优化'}")
        print("=" * 70)


# ── 主函数 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="大模型 vs 小模型对比评测")
    parser.add_argument("--num_samples", type=int, default=compare_cfg.num_samples,
                        help="测试样本数")
    parser.add_argument("--teacher", type=str, default=None,
                        help="教师模型名称")
    parser.add_argument("--student", type=str, default=None,
                        help="学生模型路径")
    parser.add_argument("--data", type=str, default=None,
                        help="测试数据路径")
    parser.add_argument("--output", type=str,
                        default=str(RESULTS_DIR / "comparison_report.json"),
                        help="输出报告路径")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="跳过教师模型评测（已有结果时使用）")
    args = parser.parse_args()

    set_seed(42)

    comparator = ModelComparator()

    # 加载测试数据
    texts, labels = comparator.load_test_data(args.data, args.num_samples)

    # 加载学生模型
    comparator.load_student(args.student)

    if not args.skip_teacher:
        # 加载教师模型
        comparator.load_teacher(args.teacher)
        # 运行完整对比
        report = comparator.run_full_comparison(texts, labels)
    else:
        # 只评测学生模型
        logger.info("跳过教师模型评测 …")
        student_results = comparator.benchmark_student(texts, labels)
        report = {
            "comparison_summary": {
                "test_samples": len(texts),
                "device": comparator.device,
                "note": "跳过了教师模型评测",
            },
            "student_results": student_results,
        }

    # 打印报告
    comparator.print_report(report)

    # 保存报告
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"对比报告已保存到: {args.output}")


if __name__ == "__main__":
    main()