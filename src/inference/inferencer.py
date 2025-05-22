"""
inference.py

Perform batch inference using saved preprocessing pipeline and trained model.
- Loads pipeline and model from config.yaml paths
- Processes raw input data (CSV)
- Outputs predictions (class + probability) to CSV
"""

import argparse
import logging
import os
import sys
import pandas as pd
import pickle
import yaml


def load_artifact(path, artifact_type="object"):
    """Load a pickled artifact (pipeline/model) from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{artifact_type.capitalize()} not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )


def run_inference(config_path, input_path, output_path):
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config.yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load artifacts
    pipeline_path = config.get("artifacts", {}).get(
        "preprocessing_pipeline", "models/preprocessing_pipeline.pkl")
    model_cfg = config.get("model", {})
    active = model_cfg.get("active", "decision_tree")
    model_path = model_cfg.get(active, {}).get("save_path", "models/model.pkl")

    logger.info(f"Loading preprocessing pipeline from {pipeline_path}")
    pipeline = load_artifact(pipeline_path, artifact_type="pipeline")

    logger.info(f"Loading trained model from {model_path}")
    model = load_artifact(model_path, artifact_type="model")

    # Load input data
    logger.info(f"Reading input data from {input_path}")
    input_df = pd.read_csv(input_path)
    logger.info(f"Input shape: {input_df.shape}")

    # Features: should match original raw_features in config
    raw_features = config.get("raw_features", [])
    if not raw_features:
        logger.error("raw_features not set in config.yaml")
        sys.exit(1)

    missing = [col for col in raw_features if col not in input_df.columns]
    if missing:
        logger.error(f"Input data missing required columns: {missing}")
        sys.exit(1)
    input_data = input_df[raw_features]

    # Run preprocessing
    logger.info("Applying preprocessing pipeline to input data")
    X_proc = pipeline.transform(input_data)

    # Run inference
    logger.info("Generating predictions")
    y_pred = model.predict(X_proc)
    # Predict probability if supported
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_proc)[:, 1]
        input_df["prediction_proba"] = y_prob
    input_df["prediction"] = y_pred

    # Write output
    logger.info(f"Writing predictions to {output_path}")
    input_df.to_csv(output_path, index=False)
    logger.info("Inference completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input", required=True,
                        help="CSV file with new raw data")
    parser.add_argument("--output", required=True,
                        help="CSV file to save predictions")
    args = parser.parse_args()

    run_inference(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
    )
