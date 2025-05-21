"""
preprocessing.py

Leakage-proof, sklearn Pipeline-based preprocessing:
- Renames columns (configurable)
- Uses KBinsDiscretizer for quantile bucketing (out-of-the-box, onehot-dense)
- MinMaxScaler or StandardScaler for normalization (from config)
- All logic is modular and parameterized by config.yaml
"""


import logging
import pandas as pd
from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns as specified in the rename_map."""
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {rename_map}")
    return df


def build_preprocessing_pipeline(config: Dict, continuous: list[str] | None = None) -> Pipeline:
    """
    Create a sklearn Pipeline/ColumnTransformer for preprocessing.
    Only applies to 'rx_ds' feature as required.
    """
    if not continuous:
        continuous = ["rx_ds"]
    pp_cfg = config.get("preprocessing", {}).get("rx_ds", {})
    transformers = []
    # --- Bucketing (as one-hot)
    if pp_cfg.get("bucketize", True):
        n_buckets = pp_cfg.get("n_buckets", 4)
        transformers.append((
            "bucketize",
            KBinsDiscretizer(n_bins=n_buckets,
                             encode="onehot-dense", strategy="quantile"),
            continuous
        ))
    # --- Normalization
    normalization = pp_cfg.get("normalization", "minmax")
    if normalization:
        scaler = MinMaxScaler() if normalization == "minmax" else StandardScaler()
        transformers.append((
            "norm",
            scaler,
            continuous
        ))

    # Build ColumnTransformer, each step adds new columns
    col_transform = ColumnTransformer(
        transformers, remainder="passthrough", verbose_feature_names_out=False)
    pipeline = Pipeline([
        ("col_transform", col_transform)
    ])
    return pipeline


def get_output_feature_names(preprocessor: Pipeline, input_features: List[str], config: Dict) -> List[str]:
    """
    Returns output column names after transformation.
    Handles bucket names and normalization naming as needed.
    """
    pp_cfg = config.get("preprocessing", {}).get("rx_ds", {})
    output_names = []
    for name, trans, cols in preprocessor.named_steps["col_transform"].transformers_:
        if name == "bucketize" and pp_cfg.get("bucketize", True):
            # Buckets are one-hot, custom labels from config
            bucket_labels = pp_cfg.get(
                "bucket_labels", [f"Q{i+1}" for i in range(pp_cfg.get("n_buckets", 4))])
            output_names.extend(
                [f"rx_ds_bucket_{label}" for label in bucket_labels])
        elif name == "norm":
            output_names.append("rx_ds_norm")
        elif name == "remainder":
            # Passthrough other columns
            passthroughs = [f for f in input_features if f != "rx_ds"]
            output_names.extend(passthroughs)
    return output_names


def run_preprocessing_pipeline(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Apply all preprocessing steps as specified in config.
    Fit only on df (not train/test split, handled in model.py).
    """
    pp_cfg = config.get("preprocessing", {}).get("rx_ds", {})
    # Step 1: Rename columns
    rename_map = pp_cfg.get("rename_columns", {})
    df = rename_columns(df, rename_map)
    # No transformation here; just return df as-is
    return df


# CLI for standalone use
if __name__ == "__main__":
    import sys
    import yaml
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    if len(sys.argv) < 3:
        logging.error(
            "Usage: python -m src.preprocess.preprocessing <raw_data.csv> <config.yaml>")
        sys.exit(1)
    raw_data_path = sys.argv[1]
    config_path = sys.argv[2]
    df = pd.read_csv(raw_data_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    df = run_preprocessing_pipeline(df, config)
    print(df.head())  # for CLI test, can be removed
