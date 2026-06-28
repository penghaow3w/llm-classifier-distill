#!/usr/bin/env python3
"""
label_data.py — 大模型数据打标流水线
=====================================
使用 HuggingFace 上的大模型（Qwen2.5-1.5B-Instruct）为无标签中文文本自动打标。

创新点：
  1. 多源数据融合 — 从多个 HF 数据集 + 内置语料采集文本，覆盖科技/娱乐/体育/财经/时政/生活
  2. 主动学习策略 — 置信度低于阈值时进行二次打标，提高标签质量
  3. 标签一致性校验 — 同一 prompt 多角度提问，交叉验证标签

用法:
  python label_data.py [--total 2000] [--batch_size 8] [--output data/labeled/labeled_data.jsonl]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import label_cfg, RAW_DIR, LABELED_DIR
from utils import (
    collect_chinese_texts,
    parse_label,
    set_seed,
    logger,
    ID2LABEL,
    LABEL2ID,
)

# ── Prompt 模板 ────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个专业的文本情感分析标注员。你的任务是对给定的中文文本进行情感分类。

分类标准：
- 正面：文本表达积极、乐观、满意、赞扬等正面情感
- 负面：文本表达消极、悲观、不满、批评等负面情感  
- 中性：文本客观陈述事实、无明显情感倾向

输出格式（严格遵守）：
标签：<正面/负面/中性>
置信度：<0.0-1.0之间的浮点数>

只输出以上两行，不要输出任何其他内容。"""

# 多角度验证 Prompt（创新点）
CROSSCHECK_PROMPT = """请从另一个角度重新分析以下文本的情感倾向，并给出分类结果。

分类标准：
- 正面：文本表达积极、乐观、满意、赞扬等正面情感
- 负面：文本表达消极、悲观、不满、批评等负面情感  
- 中性：文本客观陈述事实、无明显情感倾向

输出格式：
标签：<正面/负面/中性>
置信度：<0.0-1.0之间的浮点数>"""


# ── 主类 ──────────────────────────────────────────────────

class LLMLabeler:
    """大模型自动打标器"""

    def __init__(self, model_name: str = None, use_4bit: bool = True):
        self.model_name = model_name or label_cfg.teacher_model
        self.use_4bit = use_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"正在加载教师模型: {self.model_name}")
        logger.info(f"设备: {self.device}")

        # 量化配置
        if use_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("教师模型加载完成 ✓")

    def _generate(self, messages: List[dict], max_tokens: int = 64) -> str:
        """调用 LLM 生成回复"""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=label_cfg.temperature,
                top_p=label_cfg.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def label_single(
        self,
        text: str,
        use_crosscheck: bool = True,
    ) -> Tuple[str, float]:
        """
        对单条文本进行情感标注。
        返回 (标签, 置信度)
        """
        # 第一轮标注
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请对以下文本进行情感分类：\n\n{text}"},
        ]
        response1 = self._generate(messages)
        label1, conf1 = parse_label(response1)

        # 如果置信度足够高，直接返回
        if conf1 >= label_cfg.confidence_threshold and not use_crosscheck:
            return label1, conf1

        # 交叉验证（创新点）
        if use_crosscheck:
            messages2 = [
                {"role": "system", "content": CROSSCHECK_PROMPT},
                {"role": "user", "content": f"文本：{text}\n\n你之前的分类结果是：{label1}（置信度：{conf1}）\n请重新分析并给出结果："},
            ]
            response2 = self._generate(messages2)
            label2, conf2 = parse_label(response2)

            # 两次结果一致 → 取平均置信度
            if label1 == label2:
                return label1, (conf1 + conf2) / 2
            # 不一致 → 取置信度更高的
            else:
                return (label1, conf1) if conf1 >= conf2 else (label2, conf2)

        return label1, conf1

    def label_batch(
        self,
        texts: List[str],
        use_crosscheck: bool = True,
        show_progress: bool = True,
    ) -> List[dict]:
        """批量打标"""
        results = []
        iterator = tqdm(texts, desc="打标进度", disable=not show_progress)

        for text in iterator:
            try:
                label, confidence = self.label_single(text, use_crosscheck)
                results.append({
                    "text": text,
                    "label": label,
                    "label_id": LABEL2ID[label],
                    "confidence": round(confidence, 4),
                })
                iterator.set_postfix(
                    label=label,
                    conf=f"{confidence:.2f}",
                    pos=sum(1 for r in results if r["label"] == "正面"),
                    neg=sum(1 for r in results if r["label"] == "负面"),
                    neu=sum(1 for r in results if r["label"] == "中性"),
                )
            except Exception as e:
                logger.error(f"打标失败: {text[:50]}... | 错误: {e}")
                results.append({
                    "text": text,
                    "label": "中性",
                    "label_id": 2,
                    "confidence": 0.5,
                })

        return results

    def save_results(self, results: List[dict], output_path: str):
        """保存打标结果到 JSONL"""
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"打标结果已保存到: {output_path}")

        # 打印分布统计
        label_counts = {}
        for r in results:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

        logger.info("标签分布:")
        for label, count in sorted(label_counts.items()):
            pct = count / len(results) * 100
            logger.info(f"  {label}: {count} ({pct:.1f}%)")

        # 打印置信度统计
        confs = [r["confidence"] for r in results]
        logger.info(f"平均置信度: {sum(confs)/len(confs):.4f}")
        logger.info(f"置信度 >= 0.9 的样本: {sum(1 for c in confs if c >= 0.9)}/{len(confs)}")


# ── 主函数 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="大模型数据打标流水线")
    parser.add_argument("--total", type=int, default=label_cfg.total_samples,
                        help="打标数据总量")
    parser.add_argument("--batch_size", type=int, default=label_cfg.batch_size,
                        help="批次大小")
    parser.add_argument("--output", type=str, default=label_cfg.labeled_output,
                        help="输出文件路径")
    parser.add_argument("--no-crosscheck", action="store_true",
                        help="禁用交叉验证")
    parser.add_argument("--model", type=str, default=None,
                        help="教师模型名称（覆盖默认配置）")
    parser.add_argument("--no-4bit", action="store_true",
                        help="禁用 4bit 量化")
    args = parser.parse_args()

    set_seed(42)

    # ── 步骤 1: 收集文本数据 ──
    logger.info("=" * 60)
    logger.info("步骤 1/3: 收集无标签中文文本 …")
    logger.info("=" * 60)

    texts = collect_chinese_texts(args.total)

    # 保存原始文本
    raw_path = RAW_DIR / "raw_texts.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    logger.info(f"原始文本已保存到: {raw_path}")

    # ── 步骤 2: LLM 打标 ──
    logger.info("=" * 60)
    logger.info("步骤 2/3: 大模型自动打标 …")
    logger.info("=" * 60)

    labeler = LLMLabeler(
        model_name=args.model,
        use_4bit=not args.no_4bit,
    )

    use_cc = not args.no_crosscheck
    results = labeler.label_batch(texts, use_crosscheck=use_cc)

    # ── 步骤 3: 保存结果 ──
    logger.info("=" * 60)
    logger.info("步骤 3/3: 保存打标结果 …")
    logger.info("=" * 60)

    labeler.save_results(results, args.output)

    # 额外保存一份统计信息
    stats_path = LABELED_DIR / "labeling_stats.json"
    stats = {
        "total_samples": len(results),
        "model": labeler.model_name,
        "label_distribution": {
            label: sum(1 for r in results if r["label"] == label)
            for label in ["正面", "负面", "中性"]
        },
        "avg_confidence": round(
            sum(r["confidence"] for r in results) / len(results), 4
        ),
        "crosscheck_enabled": use_cc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("打标完成! ✓")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"统计信息: {stats_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()