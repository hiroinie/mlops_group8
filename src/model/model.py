"""
model.py

Leakage-proof, full MLOps pipeline:
- Splits raw data first
- Fits preprocessing (ColumnTransformer) on train set only, applies to val/test
- Trains model, evaluates, and saves both model and preprocessing artifacts
"""

import os
import logging
import pickle
from typing import List, Dict, Any
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from src.preprocess.preprocessing import build_preprocessing_pipeline, rename_columns

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "decision_tree": DecisionTreeClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def split_data(df: pd.DataFrame, features: List[str], target: str, split_cfg: Dict[str, Any]):
    X = df[features].values
    y = df[target].values
    test_size = split_cfg.get("test_size", 0.2)
    valid_size = split_cfg.get("valid_size", 0.2)
    random_state = split_cfg.get("random_state", 42)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + valid_size), random_state=random_state, stratify=y
    )
    rel_valid = valid_size / (test_size + valid_size)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_valid, random_state=random_state, stratify=y_temp
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train_model(X_train, y_train, model_type, params):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(**params)
    model.fit(X_train, y_train)
    logger.info(f"Trained model: {model_type}")
    return model


def evaluate_model(model, X, y, metrics):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(
        model, "predict_proba") else None
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    results = {}
    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower == "accuracy":
            results["Accuracy"] = accuracy_score(y, y_pred)
        elif metric_lower in ["precision", "precision (ppv)", "positive predictive value (ppv)"]:
            results["Precision (PPV)"] = precision_score(
                y, y_pred, zero_division=0)
        elif metric_lower in ["recall", "sensitivity"]:
            results["Recall (Sensitivity)"] = recall_score(
                y, y_pred, zero_division=0)
        elif metric_lower == "specificity":
            results["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        elif metric_lower == "f1 score":
            results["F1 Score"] = f1_score(y, y_pred, zero_division=0)
        elif metric_lower == "negative predictive value (npv)":
            results["Negative Predictive Value (NPV)"] = tn / \
                (tn + fn) if (tn + fn) > 0 else 0.0
        elif metric_lower == "roc auc":
            results["ROC AUC"] = roc_auc_score(
                y, y_prob) if y_prob is not None else float("nan")
    return results


def save_artifact(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved artifact to {path}")


def format_metrics(metrics: dict, ndigits: int = 2) -> dict:
    formatted = {}
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            formatted[k] = round(float(v), ndigits)
        elif hasattr(v, "item"):
            formatted[k] = round(float(v.item()), ndigits)
        else:
            formatted[k] = v
    return formatted


def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any]):
    # Rename columns before splitting (if needed)
    pp_cfg = config.get("preprocessing", {}).get("rx_ds", {})
    rename_map = pp_cfg.get("rename_columns", {})
    df = rename_columns(df, rename_map)

    features = config["features"]
    target = config["target"]
    split_cfg = config["data_split"]
    metrics = config["metrics"]

    missing = [col for col in features if col not in df.columns]
    if missing:
        logger.error(f"Missing features in processed data: {missing}")
        raise ValueError(f"Missing features: {missing}")

    # Split data before any fitting (no leakage)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        df, features, target, split_cfg)

    # Build and fit pipeline on training set only
    pipeline = build_preprocessing_pipeline(config, features)
    pipeline.fit(X_train)

    # Transform all splits
    X_train_prep = pipeline.transform(X_train)
    X_valid_prep = pipeline.transform(X_valid)
    X_test_prep = pipeline.transform(X_test)

    # Train model
    model_config = config["model"]
    active = model_config.get("active", "decision_tree")
    active_model_cfg = model_config[active]
    model_type = active
    params = active_model_cfg.get("params", {})
    save_path = active_model_cfg.get("save_path", "models/model.pkl")
    model = train_model(X_train_prep, y_train, model_type, params)

    # Evaluate
    results_valid = evaluate_model(model, X_valid_prep, y_valid, metrics)
    results_test = evaluate_model(model, X_test_prep, y_test, metrics)
    logger.info(f"Validation set metrics: {format_metrics(results_valid)}")
    logger.info(f"Test set metrics: {format_metrics(results_test)}")

    # Save model and pipeline artifacts
    save_artifact(model, save_path)
    pipe_path = config.get("artifacts", {}).get(
        "preprocessing_pipeline", "models/preprocessing_pipeline.pkl")
    save_artifact(pipeline, pipe_path)


# CLI for standalone training (optional)
if __name__ == "__main__":
    import sys
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python -m src.model.model <raw_data.csv> <config.yaml>")
        sys.exit(1)
    raw_data_path, config_path = sys.argv[1:3]
    df = pd.read_csv(raw_data_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    run_model_pipeline(df, config)
