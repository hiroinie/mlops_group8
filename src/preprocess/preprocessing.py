"""
preprocessing.py

Performs all preprocessing on the opioid data for MLOps pipelines:
- Renames columns (configurable)
- Buckets and one-hot encodes 'rx_ds' (configurable bucket count/labels)
- Normalizes 'rx_ds' (minmax or zscore, from config)
- All options driven by config.yaml for reproducibility and instructional clarity
"""

import os
import logging
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """
    Rename columns as specified in the rename_map.
    """
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {rename_map}")
    return df


def bucketize_column(df: pd.DataFrame, col: str, n_buckets: int, bucket_labels, one_hot: bool) -> pd.DataFrame:
    df = df.copy()
    bucket_col = f"{col}_bucket"
    df[bucket_col] = pd.qcut(df[col], q=n_buckets, labels=bucket_labels)
    logger.info(
        f"Bucketed '{col}' into {n_buckets} quantiles: {bucket_labels}")
    if one_hot:
        # <--- Ensures int type
        dummies = pd.get_dummies(df[bucket_col], prefix=bucket_col, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        logger.info(f"One-hot encoded '{bucket_col}'")
        df.drop([bucket_col], axis=1, inplace=True)
    return df


def normalize_column(df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
    """
    Normalize a column using the specified method.
    """
    df = df.copy()
    new_col = f"{col}_norm"
    if method == 'minmax':
        min_val = df[col].min()
        max_val = df[col].max()
        df[new_col] = (df[col] - min_val) / (max_val - min_val)
        logger.info(f"Normalized '{col}' using min-max scaling")
    elif method == 'zscore':
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[new_col] = (df[col] - mean_val) / std_val
        logger.info(f"Normalized '{col}' using z-score scaling")
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return df


def run_preprocessing_pipeline(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Apply all preprocessing steps as specified in config.
    Saves the processed dataframe to processed_path from config.
    """
    pp_cfg = config.get("preprocessing", {}).get("rx_ds", {})

    # Column renaming
    rename_map = pp_cfg.get("rename_columns", {})
    df = rename_columns(df, rename_map)

    # Bucketize and one-hot encode if enabled
    if pp_cfg.get("bucketize", True):
        n_buckets = pp_cfg.get("n_buckets", 4)
        bucket_labels = pp_cfg.get(
            "bucket_labels", [f"Q{i+1}" for i in range(n_buckets)])
        one_hot = pp_cfg.get("one_hot_encode_buckets", True)
        df = bucketize_column(df, "rx_ds", n_buckets, bucket_labels, one_hot)

    # Normalize if enabled
    normalization = pp_cfg.get("normalization", "minmax")
    if normalization:
        df = normalize_column(df, "rx_ds", normalization)

    # Save to processed_path
    processed_path = config["data_source"].get("processed_path")
    if not processed_path:
        logger.error("No processed_path found in config['data_source']")
        raise ValueError("No processed_path in config['data_source']")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Saved processed data to {processed_path}")

    return df


# CLI for standalone use
if __name__ == "__main__":
    import sys
    import yaml
    import pandas as pd
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Check for correct usage
    if len(sys.argv) < 3:
        logging.error("Usage: python -m src.preprocess.preprocessing <raw_data.csv> <config.yaml>")
        logging.error("Example: python -m src.preprocess.preprocessing data/raw/opiod_raw_data.csv config.yaml")
        sys.exit(1)

    raw_data_path = sys.argv[1]
    config_path = sys.argv[2]

    df = pd.read_csv(raw_data_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_preprocessing_pipeline(df, config)
