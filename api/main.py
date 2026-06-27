# api/main.py
"""
FastAPI application entry point.

Design decisions:
- Lifespan context manager for startup/shutdown events
- Single model instance (singleton) shared across requests
- Preprocessing applied before inference (same pipeline as training)
- Structured logging with loguru
"""

import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ScoreDetail,
    RoutingDecision
)
from api.model_loader import model_loader

# Add src to path for preprocessor
sys.path.append('.')
from src.data.preprocessor import HinglishPreprocessor

# ── Preprocessor (same config as training) ────────────────
preprocessor = HinglishPreprocessor(
    remove_urls=True,
    remove_mentions=True,
    remove_hashtags=False,
    remove_emojis=False,
    lowercase=True,
    normalize_whitespace=True,
    min_length=3
)


# ── Lifespan (startup + shutdown) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — runs before first request
    logger.info("🚀 SafeShield AI API starting up...")
    model_loader.load()
    logger.info("✅ API ready to serve requests")
    yield
    # Shutdown — runs when app stops
    logger.info("👋 SafeShield AI API shutting down...")


# ── App initialization ────────────────────────────────────
app = FastAPI(
    title="SafeShield AI",
    description="""
    ## Multilingual Hate Speech Detection API
    
    Detects hate speech and offensive content in **Hinglish** 
    (Hindi-English code-mixed) text using a fine-tuned MuRIL model.
    
    ### Labels
    - **NOT** — Non-hate, non-offensive content
    - **HOF** — Hate speech or Offensive content
    
    ### Routing Decisions
    - **auto_classify** — Confidence ≥ 0.65, reliable prediction
    - **classify_log** — Confidence 0.60-0.65, log for review
    - **human_review** — Confidence < 0.60, needs human review
    """,
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS middleware ────────────────────────────────────────
# Allows Streamlit UI (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ──────────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Process-Time"] = f"{duration:.3f}s"
    return response


# ── Routes ────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    """API root — basic info."""
    return {
        "name"       : "SafeShield AI",
        "version"    : "1.0.0",
        "description": "Multilingual Hate Speech Detection for Hinglish text",
        "docs"       : "/docs",
        "health"     : "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check — verify model is loaded."""
    return HealthResponse(
        status      ="healthy" if model_loader.is_loaded else "loading",
        model_loaded=model_loader.is_loaded,
        model_name  =model_loader.model_name,
        device      =model_loader.device
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Classify a single text as hate speech (HOF) or not (NOT).
    
    - Applies same preprocessing pipeline used during training
    - Returns confidence score and routing decision
    """
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please try again."
        )

    # Preprocess — same pipeline as training!
    clean_text = preprocessor.clean(request.text)
    if clean_text is None:
        raise HTTPException(
            status_code=422,
            detail="Text too short after preprocessing."
        )

    try:
        results = model_loader.predict(
            [clean_text],
            threshold=request.threshold
        )
        result = results[0]

        return PredictResponse(
            text          =request.text,   # Return original text
            label         =result['label'],
            confidence    =result['confidence'],
            scores        =ScoreDetail(**result['scores']),
            routing       =RoutingDecision(result['routing']),
            text_length   =len(request.text),
            model_version ="muril-base-cased-v1"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest):
    """
    Classify multiple texts in one request (max 32).
    More efficient than calling /predict multiple times.
    """
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model loading...")

    # Preprocess all texts
    clean_texts = []
    valid_indices = []

    for i, text in enumerate(request.texts):
        clean = preprocessor.clean(text)
        if clean:
            clean_texts.append(clean)
            valid_indices.append(i)

    if not clean_texts:
        raise HTTPException(
            status_code=422,
            detail="All texts were too short after preprocessing."
        )

    try:
        raw_results = model_loader.predict(
            clean_texts,
            threshold=request.threshold
        )

        responses = []
        for i, result in enumerate(raw_results):
            responses.append(PredictResponse(
                text          =request.texts[valid_indices[i]],
                label         =result['label'],
                confidence    =result['confidence'],
                scores        =ScoreDetail(**result['scores']),
                routing       =RoutingDecision(result['routing']),
                text_length   =len(request.texts[valid_indices[i]]),
                model_version ="muril-base-cased-v1"
            ))

        hof_count    = sum(1 for r in responses if r.label == "HOF")
        review_count = sum(1 for r in responses if r.routing == "human_review")

        return BatchPredictResponse(
            results     =responses,
            total       =len(responses),
            hof_count   =hof_count,
            not_count   =len(responses) - hof_count,
            review_count=review_count
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples", tags=["General"])
async def get_examples():
    """Sample texts to test the API."""
    return {
        "examples": [
            {"text": "aaj ka din bahut acha tha yaar", "expected": "NOT"},
            {"text": "yaar tu bahut bura insaan hai", "expected": "HOF"},
            {"text": "Modi ji ne acha kaam kiya hai", "expected": "NOT"},
            {"text": "I love this beautiful day", "expected": "NOT"},
            {"text": "tujhe toh main dekhta hun", "expected": "HOF"},
        ]
    }