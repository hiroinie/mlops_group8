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
from typing import Dict, List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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


def build_preprocessing_pipeline(config: Dict, features: List[str]) -> ColumnTransformer:
    """
    Build a full ColumnTransformer for all features.
    Only rx_ds is bucketized/normalized, others pass through.
    """
    pp_cfg = config.get("preprocessing", {}).get("rx_ds", {})
    n_buckets = pp_cfg.get("n_buckets", 4)
    normalization = pp_cfg.get("normalization", "minmax")
    rx_ds_pipeline = []
    # Quantile bucketing + one-hot encoding for rx_ds
    rx_ds_pipeline.append(
        ('bucketize', KBinsDiscretizer(n_bins=n_buckets,
         encode='onehot-dense', strategy='quantile'))
    )
    # Normalization after bucketing for rx_ds
    if normalization == "minmax":
        rx_ds_pipeline.append(('normalize', MinMaxScaler()))
    elif normalization == "zscore":
        rx_ds_pipeline.append(('normalize', StandardScaler()))
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    # All other features just pass through
    other_features = [col for col in features if col != 'rx_ds']

    # Compose ColumnTransformer
    ct = ColumnTransformer(
        transformers=[
            ('rx_ds_pipeline', Pipeline(rx_ds_pipeline), ['rx_ds']),
            ('passthrough', 'passthrough', other_features)
        ]
    )
    return ct


# CLI for standalone preprocessing and artifact saving (for manual/debug use)
if __name__ == "__main__":
    import sys
    import yaml
    import pickle

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    if len(sys.argv) < 4:
        logger.error(
            "Usage: python -m src.preprocess.preprocessing <raw_data.csv> <config.yaml> <out_pipeline.pkl>")
        sys.exit(1)

    raw_data_path, config_path, pipeline_path = sys.argv[1:4]
    df = pd.read_csv(raw_data_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features = config["features"]
    rename_map = config.get("preprocessing", {}).get(
        "rx_ds", {}).get("rename_columns", {})
    df = rename_columns(df, rename_map)

    # Fit pipeline on whole data (for demo—real pipeline fits only on train set)
    pipe = build_preprocessing_pipeline(config, features)
    pipe.fit(df[features])

    # Save fitted pipeline
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipe, f)
    logger.info(f"Fitted and saved pipeline to {pipeline_path}")
