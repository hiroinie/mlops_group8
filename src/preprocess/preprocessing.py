"""
preprocessing.py

Performs all preprocessing on the opioid data for MLOps pipelines:
- Renames columns (configurable, done before pipeline)
- Uses sklearn KBinsDiscretizer for quantile bucketing (out-of-the-box)
- Uses sklearn MinMaxScaler or StandardScaler for normalization (from config)
- Uses sklearn OneHotEncoder for dummy variables
- All options driven by config.yaml for reproducibility and instructional clarity
"""

import os
import logging
from typing import Dict
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """
    Rename columns as specified in the rename_map.
    """
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {rename_map}")
    return df


def run_preprocessing_pipeline(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Apply all preprocessing steps as specified in config.
    Saves the processed dataframe to processed_path from config.
    Uses only sklearn out-of-the-box transformers.
    """
    pp_cfg = config.get("preprocessing", {}).get("rx_ds", {})

    # --- 1. Column renaming (before pipeline)
    rename_map = pp_cfg.get("rename_columns", {})
    df = rename_columns(df, rename_map)

    # --- 2. KBinsDiscretizer for quantile buckets (out-of-the-box)
    if pp_cfg.get("bucketize", True):
        n_buckets = pp_cfg.get("n_buckets", 4)
        strategy = "quantile"
        kbd = KBinsDiscretizer(
            n_bins=n_buckets, encode='onehot-dense', strategy=strategy)
        bucketized = kbd.fit_transform(df[['rx_ds']])
        bucket_cols = [f"rx_ds_bucket_Q{i+1}" for i in range(n_buckets)]
        bucket_df = pd.DataFrame(
            bucketized, columns=bucket_cols, index=df.index)
        df = pd.concat([df, bucket_df], axis=1)
        logger.info(
            f"Bucketized 'rx_ds' into {n_buckets} quantile bins and one-hot encoded")

    # --- 3. Normalization
    normalization = pp_cfg.get("normalization", "minmax")
    if normalization:
        if normalization == "minmax":
            scaler = MinMaxScaler()
            df['rx_ds_norm'] = scaler.fit_transform(df[['rx_ds']])
            logger.info(f"Normalized 'rx_ds' using min-max scaling")
        elif normalization == "zscore":
            scaler = StandardScaler()
            df['rx_ds_norm'] = scaler.fit_transform(df[['rx_ds']])
            logger.info(f"Normalized 'rx_ds' using z-score scaling")
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

    # --- 4. Save to processed_path
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
        logging.error(
            "Usage: python -m src.preprocess.preprocessing <raw_data.csv> <config.yaml>")
        logging.error(
            "Example: python -m src.preprocess.preprocessing data/raw/opiod_raw_data.csv config.yaml")
        sys.exit(1)

    raw_data_path = sys.argv[1]
    config_path = sys.argv[2]

    df = pd.read_csv(raw_data_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_preprocessing_pipeline(df, config)
