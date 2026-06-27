# api/model_loader.py
"""
Singleton model loader.
Why singleton? We don't want to reload 950MB model on every request.
Load once at startup, reuse forever.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
from functools import lru_cache


HF_REPO    = "sourabh5500/hate-speech-muril"
MAX_LENGTH = 128


class ModelLoader:
    """
    Singleton class that loads and holds the model in memory.
    Thread-safe for FastAPI's async workers.
    """
    _instance = None
    _model    = None
    _tokenizer= None
    _device   = None

    def __new__(cls):
        # Singleton pattern — only one instance ever created
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        """Load model from HuggingFace Hub."""
        if self._model is not None:
            logger.info("Model already loaded, skipping...")
            return

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Device: {self._device}")
        logger.info(f"Loading model from HF Hub: {HF_REPO}")

        self._tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
        self._model     = AutoModelForSequenceClassification.from_pretrained(
            HF_REPO
        )
        self._model     = self._model.to(self._device)
        self._model.eval()

        logger.info("✅ Model loaded successfully")

    def predict(self, texts: list[str], threshold: float = 0.5) -> list[dict]:
        """
        Run inference on a list of texts.
        Returns list of prediction dicts.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        encoding = self._tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encoding)
            probs   = torch.softmax(outputs.logits, dim=1)

        results = []
        for i, text in enumerate(texts):
            prob_not = probs[i][0].item()
            prob_hof = probs[i][1].item()

            # Apply threshold for HOF
            # Default 0.5, but can be lowered to catch more hate
            if prob_hof >= threshold:
                label      = "HOF"
                confidence = prob_hof
            else:
                label      = "NOT"
                confidence = prob_not

            # Routing decision based on confidence
            if confidence >= 0.65:
                routing = "auto_classify"
            elif confidence >= 0.60:
                routing = "classify_log"
            else:
                routing = "human_review"

            results.append({
                "text"      : text,
                "label"     : label,
                "confidence": round(confidence, 4),
                "scores"    : {
                    "NOT": round(prob_not, 4),
                    "HOF": round(prob_hof, 4)
                },
                "routing"   : routing,
                "text_length": len(text),
            })

        return results

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return str(self._device) if self._device else "not loaded"

    @property
    def model_name(self) -> str:
        return HF_REPO


# Global singleton instance
model_loader = ModelLoader()