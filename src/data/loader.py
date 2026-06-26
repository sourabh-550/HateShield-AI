# src/data/loader.py
"""
Data loading and train/val/test splitting.

Design Decision: Keep loading separate from preprocessing.
This way we can swap datasets without touching preprocessing logic,
and swap preprocessors without touching loading logic.
(Single Responsibility Principle)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger

from src.config import (
    PROCESSED_DATA_DIR,
    SEED,
    TEST_SIZE,
    VAL_SIZE,
    LABEL2ID
)
from src.data.preprocessor import HinglishPreprocessor


def load_and_split(
    csv_path: str,
    text_col: str = 'text',
    label_col: str = 'label',
    run_preprocessing: bool = True,
    save: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CSV, preprocess text, split into train/val/test.
    
    Why stratified split? To maintain label ratio in each split.
    With 53/47 split, a random split could accidentally give
    test set 60/40 which would skew evaluation.
    
    Returns:
        train_df, val_df, test_df
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Label distribution:\n{df[label_col].value_counts()}")

    if run_preprocessing:
        logger.info("Running preprocessing pipeline...")
        preprocessor = HinglishPreprocessor()
        df['text_clean'] = preprocessor.clean_batch(df[text_col].tolist())

        # Drop rows where cleaning returned None
        before = len(df)
        df = df.dropna(subset=['text_clean'])
        logger.info(f"Dropped {before - len(df)} rows after preprocessing")

        # Use cleaned text as the main text column
        df['text'] = df['text_clean']
        df = df.drop(columns=['text_clean'])

    # ── Train / Test split ─────────────────────────────────
    # stratify=label ensures both splits have same class ratio
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,        # 20% for test
        random_state=SEED,
        stratify=df[label_col]
    )

    # ── Train / Val split ──────────────────────────────────
    # val_size relative to train_val size
    # If total=100, test=20, train_val=80
    # val = 10% of total = 10/80 = 0.125 of train_val
    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=SEED,
        stratify=train_val_df[label_col]
    )

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    if save:
        train_df.to_csv(PROCESSED_DATA_DIR / 'train.csv', index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / 'val.csv', index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / 'test.csv', index=False)
        logger.info(f"Saved splits to {PROCESSED_DATA_DIR}")

    return train_df, val_df, test_df