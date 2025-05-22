"""
preprocessing.py

Leakage-proof, sklearn Pipeline-based preprocessing:
- Fully config-driven: features, categorical, continuous, and target columns
- Supports imputation, scaling, encoding, per-feature selection
- Modular, extensible, and parameterized by config.yaml
"""

import logging
from typing import Dict
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {rename_map}")
    return df


class ColumnRenamer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible column renamer"""

    def __init__(self, rename_map: dict | None = None):
        self.rename_map = rename_map or {}

    def fit(self, X, y=None):
        return self                      # no learning

    def transform(self, X):
        return X.rename(columns=self.rename_map, inplace=False)


def build_preprocessing_pipeline(config: Dict) -> Pipeline:
    pp_cfg = config.get("preprocessing", {})
    features_cfg = config.get("features", {})
    # Retrieve feature lists from config
    continuous = features_cfg.get("continuous", [])
    categorical = features_cfg.get("categorical", [])
    # Handle missing configuration
    if not continuous:
        logger.warning(
            "No continuous features specified in config. Defaulting to empty list")
    if not categorical:
        logger.info(
            "No categorical features specified in config. If this is expected, ignore this message")

    transformers = []

    # --- Continuous processing
    for col in continuous:
        steps = []
        col_cfg = pp_cfg.get(col, {})
        # Imputation
        if col_cfg.get("impute", True):
            imputer_strategy = col_cfg.get("imputer_strategy", "mean")
            steps.append(("imputer", SimpleImputer(strategy=imputer_strategy)))
        # Scaling
        scaler_type = col_cfg.get("scaler", "minmax")
        if scaler_type == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif scaler_type == "standard":
            steps.append(("scaler", StandardScaler()))
        # Bucketing (optional)
        if col_cfg.get("bucketize", False):
            n_buckets = col_cfg.get("n_buckets", 4)
            steps.append(("bucketize", KBinsDiscretizer(
                n_bins=n_buckets, encode="onehot-dense", strategy="quantile")))
        if steps:
            transformers.append((f"{col}_num", Pipeline(steps), [col]))

    # --- Categorical processing
    for col in categorical:
        steps = []
        col_cfg = pp_cfg.get(col, {})
        # Imputation
        if col_cfg.get("impute", True):
            imputer_strategy = col_cfg.get("imputer_strategy", "most_frequent")
            steps.append(("imputer", SimpleImputer(strategy=imputer_strategy)))
        # Encoding
        encoder_type = col_cfg.get("encoding", "onehot")
        if encoder_type == "onehot":
            steps.append(("encoder", OneHotEncoder(
                sparse_output=False, handle_unknown="ignore")))
        elif encoder_type == "ordinal":
            steps.append(("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1)))
        # Passthrough otherwise
        if steps:
            transformers.append((f"{col}_cat", Pipeline(steps), [col]))

    # Passthrough for all other columns (if any)
    all_specified = set(continuous + categorical)
    passthrough = [c for c in config.get(
        "raw_features", []) if c not in all_specified]
    if passthrough:
        transformers.append(("passthrough", "passthrough", passthrough))

    # Build column transformer
    col_transform = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )

    renamer = ColumnRenamer(pp_cfg.get("rename_columns", {}))

    pipeline = Pipeline(
        steps=[
            ("rename", renamer),          # new first step
            ("col_transform", col_transform),
        ]
    )
    return pipeline


def get_output_feature_names(preprocessor: Pipeline, input_features: list, config: dict) -> list:
    """
    Return the list of feature names after transformation, matching the output shape.
    Handles multiple scalers/encoders/buckets.
    """
    output_names = []
    ct = preprocessor.named_steps["col_transform"]
    for name, trans, cols in ct.transformers_:
        if hasattr(trans, "get_feature_names_out"):
            # for encoders or pipelines that expose this method
            try:
                output_names.extend(trans.get_feature_names_out(cols))
            except Exception:
                # Sometimes only OneHotEncoder supports this, fallback otherwise
                output_names.extend(cols)
        elif hasattr(trans, "named_steps"):
            # For Pipelines inside ColumnTransformer
            last_step = list(trans.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                try:
                    output_names.extend(last_step.get_feature_names_out(cols))
                except Exception:
                    output_names.extend(cols)
            else:
                output_names.extend(cols)
        elif trans == "passthrough":
            output_names.extend(cols)
        else:
            output_names.extend(cols)
    return output_names


def run_preprocessing_pipeline(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    # Optionally handle column renaming from config
    rename_map = config.get("preprocessing", {}).get("rename_columns", {})
    df = rename_columns(df, rename_map)
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
    print(df.head())
