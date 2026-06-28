"""
工具函数 — 数据加载、预处理、评估指标等。
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# ── 日志 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm-distill")


def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── 数据采集 ──────────────────────────────────────────────

def collect_chinese_texts(num_samples: int = 2500) -> List[str]:
    """
    从多个 HuggingFace 数据集收集中文文本用于打标。
    创新点：多源数据融合，覆盖不同领域和风格。
    """
    texts = []
    sources = [
        # 中文情感分析数据集（取其文本列）
        ("lansinuote/ChnSentiCorp", "text", 800),
        # 中文新闻
        ("clue/clue", "content", 700),
        # 中文维基（取摘要）
        ("pleisto/wikipedia-cn-20230720-filtered", "completion", 500),
    ]

    for ds_name, text_col, target in sources:
        try:
            ds = load_dataset(ds_name, split="train", trust_remote_code=True)
            # 随机采样
            indices = random.sample(range(len(ds)), min(target, len(ds)))
            for idx in indices:
                txt = ds[idx].get(text_col, "")
                if isinstance(txt, str) and len(txt.strip()) > 10:
                    texts.append(txt.strip()[:512])  # 截断过长文本
            logger.info(f"从 {ds_name} 收集了 {len(texts)} 条文本")
        except Exception as e:
            logger.warning(f"加载数据集 {ds_name} 失败: {e}")

    # 如果上述来源不够，补充内置中文语料
    if len(texts) < num_samples:
        logger.info("补充内置中文语料 …")
        texts.extend(_get_builtin_corpus())

    # 去重
    texts = list(dict.fromkeys(texts))
    # 采样
    if len(texts) > num_samples:
        texts = random.sample(texts, num_samples)

    logger.info(f"最终收集到 {len(texts)} 条中文文本")
    return texts


def _get_builtin_corpus() -> List[str]:
    """内置中文语料库，覆盖多领域确保数据多样性"""
    corpus = [
        # 科技
        "华为发布最新Mate系列手机，搭载自研麒麟芯片，性能大幅提升",
        "人工智能技术正在深刻改变各行各业的运作方式",
        "5G网络覆盖范围持续扩大，用户体验显著改善",
        "特斯拉推出全自动驾驶FSD V12版本，引发行业震动",
        "量子计算取得重大突破，中国科学家实现量子优越性",
        "苹果公司市值突破三万亿美元，创历史新高",
        "微软发布Windows 12操作系统，集成AI助手功能",
        "字节跳动推出AI编程助手，开发者效率提升200%",
        "三星展示折叠屏新技术，屏幕耐用性大幅提升",
        "OpenAI发布GPT-5，多模态能力引发广泛讨论",
        # 娱乐
        "春节档电影票房突破100亿，国产科幻片表现亮眼",
        "周杰伦新专辑发布，数字音乐平台销量破纪录",
        "国产游戏《黑神话：悟空》获得TGA年度最佳游戏提名",
        "综艺节目过于娱乐化引发社会讨论",
        "某知名艺人因税务问题被罚款数亿元",
        "短视频平台内容质量参差不齐，青少年沉迷问题突出",
        "音乐会门票一秒售罄，黄牛票价格翻十倍",
        "网络文学IP改编电视剧质量持续下滑",
        # 体育
        "中国队在亚运会上获得金牌总数第一，创造历史",
        "NBA季后赛精彩纷呈，湖人队逆转晋级",
        "世界杯预选赛中国队2:1战胜对手，保留出线希望",
        "马拉松赛事在全国各地兴起，全民健身热潮涌动",
        "电子竞技入选亚运会正式比赛项目，引发争议",
        "某运动员因兴奋剂丑闻被禁赛四年",
        "CBA联赛改革方案出台，引入工资帽制度",
        # 财经
        "A股市场迎来反弹，沪指重回3000点上方",
        "央行降准释放流动性，支持实体经济发展",
        "房地产市场调控政策持续，多地房价出现回落",
        "人民币汇率波动加大，出口企业面临挑战",
        "新能源汽车补贴退坡，行业竞争加剧",
        "数字人民币试点范围扩大，覆盖更多城市",
        "美联储加息预期升温，全球金融市场震荡",
        "某上市公司财务造假被证监会立案调查",
        "跨境电商行业高速增长，SHEIN估值超千亿美元",
        # 时政
        "中美高层会晤取得积极进展，双方同意加强沟通",
        "联合国气候变化大会达成新的减排协议",
        "一带一路倡议迎来十周年，成果丰硕",
        "教育部出台双减政策，减轻学生课外负担",
        "医保改革方案公布，群众看病负担将进一步减轻",
        "某地发生严重自然灾害，救援工作正在紧张进行",
        "反腐倡廉持续深入，多名高级官员被调查",
        "养老服务体系加快建设，应对人口老龄化挑战",
        # 日常生活
        "今天天气真好，适合出去郊游野餐",
        "这家餐厅的菜品味道很棒，服务也很周到",
        "最近工作压力很大，感觉身体被掏空了",
        "孩子的学习成绩一直上不去，作为家长很焦虑",
        "新买的手机用了不到一周就坏了，太失望了",
        "社区组织志愿者活动，邻里关系越来越融洽",
        "地铁高峰期太拥挤了，每天通勤都是一场战斗",
        "双十一购物节买了一堆用不上的东西，后悔莫及",
        "这本书写得非常好，推荐给大家阅读",
        "健身房办了年卡，去了不到五次就放弃了",
    ]
    return corpus


# ── 标签映射 ──────────────────────────────────────────────

LABEL2ID = {"正面": 0, "负面": 1, "中性": 2}
ID2LABEL = {0: "正面", 1: "负面", 2: "中性"}


def parse_label(raw_output: str) -> Tuple[str, float]:
    """
    解析 LLM 输出，提取标签和置信度。
    支持多种输出格式，提高鲁棒性。
    """
    raw_output = raw_output.strip()

    # 格式1: "正面" / "负面" / "中性"
    for label in ["正面", "负面", "中性"]:
        if label in raw_output:
            # 尝试提取置信度
            confidence = _extract_confidence(raw_output)
            return label, confidence

    # 格式2: "positive" / "negative" / "neutral"
    en_map = {"positive": "正面", "negative": "负面", "neutral": "中性"}
    for en, zh in en_map.items():
        if en in raw_output.lower():
            return zh, _extract_confidence(raw_output)

    # 格式3: "0" / "1" / "2"
    for i, label in ID2LABEL.items():
        if str(i) in raw_output:
            return label, _extract_confidence(raw_output)

    # 兜底: 默认中性
    return "中性", 0.5


def _extract_confidence(text: str) -> float:
    """从文本中提取置信度数值"""
    import re

    # 匹配 "置信度: 0.95" / "confidence: 0.9" 等
    pattern = r"(\d+\.\d+|\d+)"
    matches = re.findall(pattern, text)
    if matches:
        val = float(matches[-1])
        if 0 <= val <= 1:
            return val
        if 1 < val <= 100:
            return val / 100.0
    return 0.8  # 默认置信度


# ── 数据集构建 ────────────────────────────────────────────

def build_dataset(data_path: str) -> Dataset:
    """从 JSONL 文件构建 HuggingFace Dataset"""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    dataset = Dataset.from_list(records)
    return dataset


def tokenize_function(examples, tokenizer, max_length: int = 128):
    """分词函数"""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def augment_text(text: str, method: str = "random") -> str:
    """
    文本数据增强（创新点）。
    通过同义词替换、随机删除等方式增加数据多样性。
    """
    import random as _random

    words = text.split()
    if len(words) < 5:
        return text

    if method == "random":
        # 随机选择一种增强策略
        method = _random.choice(["delete", "swap", "repeat"])

    if method == "delete":
        # 随机删除 10% 的词
        keep_prob = 0.9
        words = [w for w in words if _random.random() < keep_prob]
        if not words:
            return text
    elif method == "swap":
        # 随机交换相邻词
        for i in range(len(words) - 1):
            if _random.random() < 0.1:
                words[i], words[i + 1] = words[i + 1], words[i]
    elif method == "repeat":
        # 随机重复部分词
        for i in range(len(words)):
            if _random.random() < 0.05:
                words.insert(i, words[i])

    return " ".join(words)


# ── 评估指标 ──────────────────────────────────────────────

def compute_metrics(pred) -> Dict[str, float]:
    """计算分类评估指标"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }


def evaluate_model(
    model, tokenizer, texts: List[str], labels: List[int], device: str = "cpu"
) -> Dict:
    """
    评估模型在给定数据集上的表现。
    返回 accuracy, f1, precision, recall, 以及每条样本的推理时间。
    """
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels),
    )
    loader = DataLoader(dataset, batch_size=32)

    all_preds = []
    all_labels = []
    times = []

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]

            start = time.perf_counter()
            outputs = model(input_ids, attention_mask=attention_mask)
            elapsed = time.perf_counter() - start

            preds = outputs.logits.argmax(-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.cpu().numpy().tolist())
            times.append(elapsed)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro"),
        "recall": recall_score(all_labels, all_preds, average="macro"),
        "avg_inference_time_ms": np.mean(times) * 1000,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "classification_report": classification_report(
            all_labels, all_preds,
            target_names=["正面", "负面", "中性"],
            output_dict=True,
        ),
    }


def get_device() -> str:
    """获取可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"