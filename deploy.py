#!/usr/bin/env python3
"""
deploy.py — FastAPI 部署接口
==============================
提供 RESTful API 进行情感分类推理。

特性:
  - POST /predict — 单条文本分类
  - POST /predict_batch — 批量文本分类
  - GET /health — 健康检查
  - GET /model_info — 模型信息
  - 请求限流 & 并发控制
  - 自动生成 API 文档 (Swagger UI)

用法:
  python deploy.py [--host 0.0.0.0] [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import deploy_cfg, MODEL_DIR
from utils import ID2LABEL, logger, get_device


# ── 请求/响应模型 ─────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., description="待分类的中文文本", min_length=1, max_length=2048)


class PredictResponse(BaseModel):
    text: str = Field(..., description="原始文本")
    label: str = Field(..., description="分类结果 (正面/负面/中性)")
    label_id: int = Field(..., description="分类 ID (0/1/2)")
    confidence: float = Field(..., description="置信度")
    inference_time_ms: float = Field(..., description="推理耗时 (毫秒)")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="待分类的文本列表", min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse] = Field(..., description="分类结果列表")
    total_time_ms: float = Field(..., description="总耗时 (毫秒)")


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    num_parameters: int
    num_labels: int
    labels: List[str]
    device: str
    max_seq_length: int


# ── 全局模型 ──────────────────────────────────────────────

model = None
tokenizer = None
model_info_cache = {}


def load_model(model_path: str):
    """加载分类模型"""
    global model, tokenizer, model_info_cache

    device = get_device()
    logger.info(f"加载模型: {model_path}")
    logger.info(f"使用设备: {device}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型路径不存在: {model_path}\n请先运行 train_classifier.py 训练模型"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 缓存模型信息
    config = model.config
    num_params = sum(p.numel() for p in model.parameters())
    model_info_cache = {
        "model_name": config._name_or_path if hasattr(config, "_name_or_path") else "student",
        "model_type": config.model_type if hasattr(config, "model_type") else "transformer",
        "num_parameters": num_params,
        "num_labels": config.num_labels,
        "labels": [ID2LABEL.get(i, str(i)) for i in range(config.num_labels)],
        "device": device,
        "max_seq_length": deploy_cfg.max_seq_length,
    }

    logger.info(f"模型加载完成 ✓ ({num_params / 1e6:.1f}M 参数)")


def predict_single(text: str) -> dict:
    """单条文本推理"""
    device = get_device()
    start = time.perf_counter()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=deploy_cfg.max_seq_length,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_id].item()

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "text": text,
        "label": ID2LABEL[pred_id],
        "label_id": pred_id,
        "confidence": round(confidence, 4),
        "inference_time_ms": round(elapsed, 2),
    }


# ── FastAPI 应用 ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("启动分类服务 …")
    model_path = os.environ.get("MODEL_PATH", deploy_cfg.model_path)
    load_model(model_path)
    yield
    logger.info("关闭分类服务 …")


app = FastAPI(
    title="LLM 知识蒸馏 — 情感分类 API",
    description="""
使用大模型（Qwen）打标数据训练的小模型（DistilBERT）进行中文情感分类。

## 功能
- **单条文本分类**: 输入一段中文文本，返回正面/负面/中性及置信度
- **批量文本分类**: 一次提交多条文本，高效批量推理
- **模型信息**: 查看当前部署模型的详细信息

## 分类标签
- `正面 (0)`: 积极、乐观、满意等正面情感
- `负面 (1)`: 消极、悲观、不满等负面情感  
- `中性 (2)`: 客观陈述、无明显情感倾向
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 并发控制
request_semaphore = None


@app.middleware("http")
async def concurrency_limiter(request: Request, call_next):
    """请求并发限制中间件"""
    import asyncio

    global request_semaphore
    if request_semaphore is None:
        request_semaphore = asyncio.Semaphore(deploy_cfg.max_concurrent_requests)

    if request.url.path in ["/predict", "/predict_batch"]:
        async with request_semaphore:
            response = await call_next(request)
    else:
        response = await call_next(request)
    return response


# ── API 端点 ──────────────────────────────────────────────

@app.get("/", tags=["root"])
async def root():
    """API 根路径"""
    return {
        "service": "LLM 知识蒸馏 — 情感分类 API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "predict_batch": "POST /predict_batch",
            "health": "GET /health",
            "model_info": "GET /model_info",
        },
    }


@app.get("/health", tags=["system"])
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": get_device(),
    }


@app.get("/model_info", response_model=ModelInfoResponse, tags=["system"])
async def get_model_info():
    """获取模型信息"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")
    return ModelInfoResponse(**model_info_cache)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest):
    """
    单条文本情感分类。

    输入一段中文文本，返回分类结果和置信度。

    **示例请求:**
    ```json
    {"text": "华为发布最新旗舰手机，性能强劲，用户好评如潮"}
    ```

    **示例响应:**
    ```json
    {
        "text": "华为发布最新旗舰手机，性能强劲，用户好评如潮",
        "label": "正面",
        "label_id": 0,
        "confidence": 0.9523,
        "inference_time_ms": 3.21
    }
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")

    try:
        result = predict_single(request.text)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictResponse, tags=["inference"])
async def predict_batch(request: BatchPredictRequest):
    """
    批量文本情感分类。

    一次提交多条文本，高效批量推理。

    **示例请求:**
    ```json
    {"texts": ["今天天气真好", "这个产品质量太差了", "明天开会讨论方案"]}
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")

    try:
        start = time.perf_counter()
        results = []
        for text in request.texts:
            results.append(PredictResponse(**predict_single(text)))
        total_time = (time.perf_counter() - start) * 1000

        return BatchPredictResponse(
            results=results,
            total_time_ms=round(total_time, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")


# ── 主函数 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FastAPI 分类服务部署")
    parser.add_argument("--host", type=str, default=deploy_cfg.host,
                        help="监听地址")
    parser.add_argument("--port", type=int, default=deploy_cfg.port,
                        help="监听端口")
    parser.add_argument("--model-path", type=str, default=deploy_cfg.model_path,
                        help="模型路径")
    parser.add_argument("--reload", action="store_true",
                        help="开发模式（热重载）")
    args = parser.parse_args()

    deploy_cfg.host = args.host
    deploy_cfg.port = args.port
    deploy_cfg.model_path = args.model_path

    os.environ["MODEL_PATH"] = args.model_path

    logger.info("=" * 60)
    logger.info(f"启动分类服务: http://{args.host}:{args.port}")
    logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
    logger.info(f"模型路径: {args.model_path}")
    logger.info("=" * 60)

    uvicorn.run(
        "deploy:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()