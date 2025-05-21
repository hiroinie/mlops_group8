"""
model.py

Handles data splitting, dynamic model selection/training, evaluation, persistence, and saving splits for MLOps pipelines.
Supports DecisionTreeClassifier, LogisticRegression, and RandomForestClassifier via config.
"""

import os
import logging
import pickle
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "decision_tree": DecisionTreeClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def split_and_save_data(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    split_cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split DataFrame into train, validation, and test sets. Save splits as CSVs.
    """
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

    # Save splits as CSVs (with headers)
    os.makedirs(os.path.dirname(split_cfg["train_path"]), exist_ok=True)
    pd.DataFrame(X_train, columns=features).assign(
        **{target: y_train}).to_csv(split_cfg["train_path"], index=False)
    pd.DataFrame(X_valid, columns=features).assign(
        **{target: y_valid}).to_csv(split_cfg["valid_path"], index=False)
    pd.DataFrame(X_test, columns=features).assign(
        **{target: y_test}).to_csv(split_cfg["test_path"], index=False)

    logger.info(
        f"Data split and saved: train={X_train.shape[0]}, valid={X_valid.shape[0]}, test={X_test.shape[0]}"
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    params: Dict[str, Any]
):
    """
    Train specified model type using params.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        model_type (str): 'decision_tree', 'logistic_regression', or 'random_forest'.
        params (dict): Model hyperparameters.

    Returns:
        Trained model instance.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(**params)
    model.fit(X_train, y_train)
    logger.info(f"Trained model: {model_type}")
    return model


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    metrics: List[str]
) -> Dict[str, float]:
    """
    Evaluate model using specified metrics.

    Args:
        model: Trained model.
        X (np.ndarray): Features.
        y (np.ndarray): Ground truth.
        metrics (List[str]): Metrics to compute.

    Returns:
        Dict[str, float]: Metric results.
    """
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


def save_model(model, path: str):
    """
    Save model to disk using pickle.

    Args:
        model: Trained model.
        path (str): File path to save model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def run_model_pipeline(
    df: pd.DataFrame,
    config: Dict[str, Any],
):
    """
    Complete model pipeline: splits data, trains model, evaluates, saves splits and model.
    """
    features = config["features"]
    target = config["target"]
    split_cfg = config["data_split"]
    metrics = config["metrics"]

    model_config = config["model"]
    active = model_config.get("active", "decision_tree")
    active_model_cfg = model_config[active]

    model_type = active
    params = active_model_cfg.get("params", {})
    save_path = active_model_cfg.get("save_path", "models/model.pkl")

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_and_save_data(
        df, features, target, split_cfg
    )

    model = train_model(X_train, y_train, model_type, params)

    results_valid = evaluate_model(model, X_valid, y_valid, metrics)
    results_test = evaluate_model(model, X_test, y_test, metrics)

    logger.info(f"Validation set metrics: {results_valid}")
    logger.info(f"Test set metrics: {results_test}")

    print("Validation metrics:", results_valid)
    print("Test metrics:", results_test)

    save_model(model, save_path)
