"""
main.py

Project entry point for the MLOps data pipeline.
- Loads configuration and environment variables
- Sets up logging
- Calls the data loading module
- Triggers the model pipeline (including split, preprocessing, training)
"""

import argparse
import sys
import logging
import os
import yaml
from src.data_load.data_loader import get_data
from src.model.model import run_model_pipeline

logger = logging.getLogger(__name__)


def setup_logging(logging_config: dict):
    """
    Set up logging to file and console using provided configuration.
    """
    log_file = logging_config.get("log_file", "logs/main.log")
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_format = logging_config.get(
        "format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    date_format = logging_config.get(
        "datefmt", "%Y-%m-%d %H:%M:%S"
    )
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, logging_config.get("level", "INFO")),
        format=log_format,
        datefmt=date_format,
        filemode="a"
    )
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, logging_config.get("level", "INFO")))
    formatter = logging.Formatter(log_format, date_format)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration settings from a YAML file.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main entry point for the MLOps project pipeline.
    - Loads config, sets up logging, loads raw data
    - Calls model pipeline (splits, preprocesses, trains, evaluates, saves artifacts)
    """
    parser = argparse.ArgumentParser(
        description="Main entry point for the MLOps project pipeline")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration YAML file")
    parser.add_argument("--env", type=str, default=".env",
                        help="Path to an optional .env file for secrets or environment variables")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.exception(f"Failed to load config: {e}")
        sys.exit(1)

    try:
        setup_logging(config.get("logging", {}))
    except Exception as e:
        logger.exception(f"Failed to set up logging: {e}")
        sys.exit(1)

    logger.info("Pipeline started")

    try:
        # Always load raw data (all splitting and preprocessing are handled inside the model pipeline)
        df_raw = get_data(config_path=args.config,
                          env_path=args.env, data_stage="raw")
        if df_raw is None or not hasattr(df_raw, "shape"):
            logger.error(
                "Data loading failed: get_data did not return a valid DataFrame")
            sys.exit(1)
        logger.info(f"Raw data loaded successfully. Shape: {df_raw.shape}")

        run_model_pipeline(df=df_raw, config=config)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
