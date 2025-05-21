"""
model.py

Leakage-proof, end-to-end MLOps pipeline:
- Splits raw data first
- Fits preprocessing pipeline ONLY on train split, applies to valid/test
- Trains model, evaluates, and saves model and preprocessing artifacts
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
from src.preprocess.preprocessing import build_preprocessing_pipeline, get_output_feature_names, run_preprocessing_pipeline

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
    logger.info(f"Artifact saved to {path}")


def format_metrics(metrics: dict, ndigits: int = 2) -> dict:
    return {k: round(float(v), ndigits) if isinstance(v, (float, int)) else v for k, v in metrics.items()}


def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any]):
    df = run_preprocessing_pipeline(df, config)
    assert "rx_ds" in df.columns, "rx_ds column not found in DataFrame after preprocessing"
    # Features here are all columns except target and those excluded by config
    # 2 – define feature list from config to avoid leakage
    raw_features = config.get("raw_features", [])
    input_features = [f for f in raw_features if f != config["target"]] \
        if raw_features else [c for c in df.columns if c != config["target"]]
    target = config["target"]
    split_cfg = config["data_split"]
    metrics = config["metrics"]
    model_config = config["model"]
    active = model_config.get("active", "decision_tree")
    active_model_cfg = model_config[active]
    model_type = active
    params = active_model_cfg.get("params", {})
    save_path = active_model_cfg.get("save_path", "models/model.pkl")
    preproc_path = "models/preprocessing.pkl"

    # Split the data first (to prevent leakage)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        df, input_features, target, split_cfg)
    # Build preprocessing pipeline
    # extend if you add more numeric features
    continuous_cols = ["rx_ds"]
    preprocessor = build_preprocessing_pipeline(config, continuous_cols)
    # Fit on train, transform all
    X_train_pp = preprocessor.fit_transform(
        pd.DataFrame(X_train, columns=input_features))
    X_valid_pp = preprocessor.transform(
        pd.DataFrame(X_valid, columns=input_features))
    X_test_pp = preprocessor.transform(
        pd.DataFrame(X_test, columns=input_features))
    # Feature names after transformation
    out_cols = get_output_feature_names(preprocessor, input_features, config)
    X_train_pp = pd.DataFrame(X_train_pp, columns=out_cols)
    X_valid_pp = pd.DataFrame(X_valid_pp, columns=out_cols)
    X_test_pp = pd.DataFrame(X_test_pp, columns=out_cols)

    # Save preprocessing pipeline artifact
    save_artifact(preprocessor, preproc_path)
    # Train model
    model = train_model(X_train_pp.values, y_train, model_type, params)
    # Evaluate
    results_valid = evaluate_model(model, X_valid_pp.values, y_valid, metrics)
    results_test = evaluate_model(model, X_test_pp.values, y_test, metrics)
    formatted_results_valid = format_metrics(results_valid)
    formatted_results_test = format_metrics(results_test)
    logger.info(f"Validation set metrics: {formatted_results_valid}")
    logger.info(f"Test set metrics: {formatted_results_test}")
    # Save model
    save_artifact(model, save_path)


# CLI for standalone training
if __name__ == "__main__":
    import sys
    import yaml
    import logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    try:
        from src.data_load.data_loader import get_data
        df = get_data(config_path=config_path, data_stage="raw")
    except ImportError:
        data_path = config["data_source"]["raw_path"]
        df = pd.read_csv(data_path)
    run_model_pipeline(df, config)
