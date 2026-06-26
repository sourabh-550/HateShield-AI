# src/config.py
"""
Central configuration for the entire project.
Why: Having all configs in one place means no magic numbers scattered
across files. Change one thing here, it updates everywhere.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
# Path(__file__) = this file's location = src/config.py
# .parent = src/
# .parent.parent = project root
ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Create dirs if they don't exist (safe to call multiple times)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Config ───────────────────────────────────────────────────────
# MuRIL = Multilingual Representations for Indian Languages (Google)
# IndicBERT = AI4Bharat's BERT for Indic languages
# We'll try both; MuRIL is usually better for Hinglish
MODEL_NAME = "google/muril-base-cased"
# MODEL_NAME = "ai4bharat/indic-bert"  # Alternative

MAX_LENGTH = 128      # Max tokens per input sequence
BATCH_SIZE = 16       # Reduce to 8 if you hit OOM
LEARNING_RATE = 2e-5  # Standard for BERT fine-tuning
NUM_EPOCHS = 5
WARMUP_STEPS = 100

# ── Label Config ───────────────────────────────────────────────────────
# HASOC dataset labels
LABEL2ID = {
    "NOT": 0,   # Not offensive
    "HOF": 1,   # Hate or Offensive
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ── Training Config ────────────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1