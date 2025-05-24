"""evaluator.py

Evaluation utilities for MLOps pipelines.

The module originally handled only binary classification but has been extended
to support regression models as well.  Metrics to compute are driven by
``config['metrics']`` so the same interface works for both tasks.  The
functions return results as dictionaries and can optionally save them as JSON
artifacts for reproducibility.

Teaching note: Keep evaluation logic separate from training for clarity and
reuse.
"""

import logging
import os
import json
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def evaluate_classification(
    model,
    X,
    y,
    config: Dict[str, Any],
    save_path: Optional[str] = None,
    split: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a binary classifier on given features and labels.
    Metrics and reporting are driven by config['metrics'].

    Args:
        model: Trained estimator (must implement predict/predict_proba)
        X: Features (array or DataFrame)
        y: Ground truth labels
        config: Full config dict, expects 'metrics' key
        save_path: Optional JSON file to save results (default: None)
        split: Optional label for reporting (e.g., "validation", "test")

    Returns:
        Dictionary of metric names and values
    """
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = None

    # Calculate confusion matrix for specificity/NPV
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    results = {}
    for metric in config.get("metrics", []):
        m = metric.lower()
        if m == "accuracy":
            results["Accuracy"] = accuracy_score(y, y_pred)
        elif m in ["precision", "precision (ppv)", "positive predictive value (ppv)"]:
            results["Precision (PPV)"] = precision_score(
                y, y_pred, zero_division=0)
        elif m in ["recall", "sensitivity"]:
            results["Recall (Sensitivity)"] = recall_score(
                y, y_pred, zero_division=0)
        elif m == "specificity":
            results["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        elif m == "f1 score":
            results["F1 Score"] = f1_score(y, y_pred, zero_division=0)
        elif m == "negative predictive value (npv)":
            results["Negative Predictive Value (NPV)"] = tn / \
                (tn + fn) if (tn + fn) > 0 else 0.0
        elif m == "roc auc":
            results["ROC AUC"] = roc_auc_score(
                y, y_prob) if y_prob is not None else float("nan")
        # Add more custom metrics if needed

    # Optionally include the confusion matrix for teaching
    results["Confusion Matrix"] = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    # Optionally save to JSON
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {save_path}")

    def round_metrics(metrics_dict, ndigits=2):
        rounded = {}
        for k, v in metrics_dict.items():
            if isinstance(v, dict):
                rounded[k] = {ik: (round(iv, ndigits) if isinstance(
                    iv, float) else iv) for ik, iv in v.items()}
            elif isinstance(v, float):
                rounded[k] = round(v, ndigits)
            else:
                rounded[k] = v
        return rounded

    rounded_results = round_metrics(results)

    # Log metrics
    split_label = f" [{split}]" if split else ""
    logger.info(f"Evaluation metrics{split_label}: {rounded_results}")

    return results


def evaluate_regression(
    model,
    X,
    y,
    config: Dict[str, Any],
    save_path: Optional[str] = None,
    split: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a regression model according to ``config['metrics']``."""
    y_pred = model.predict(X)

    results = {}
    for metric in config.get("metrics", []):
        m = metric.lower()
        if m in ["mse", "mean squared error"]:
            results["MSE"] = mean_squared_error(y, y_pred)
        elif m in ["mae", "mean absolute error"]:
            results["MAE"] = mean_absolute_error(y, y_pred)
        elif m in ["r2", "r^2", "r2 score"]:
            results["R2"] = r2_score(y, y_pred)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {save_path}")

    rounded = {k: round(float(v), 2) for k, v in results.items()}
    split_label = f" [{split}]" if split else ""
    logger.info(f"Evaluation metrics{split_label}: {rounded}")

    return results
