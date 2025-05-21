"""
main.py

Project entry point for the MLOps data pipeline.
- Loads configuration and environment variables
- Sets up logging
- Calls the data loading module
- Provides a CLI interface for flexible configuration
"""

import argparse
import sys
import logging
import os
import yaml
import pandas as pd
from src.data_load.data_loader import get_data
from src.model.model import run_model_pipeline

logger = logging.getLogger(__name__)


def setup_logging(logging_config: dict):
    """
    Set up logging to file and console using provided configuration.

    Args:
        logging_config (dict): Logging configuration dictionary
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
    formatter = logging.Formatter(
        log_format,
        date_format)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration settings from a YAML file.

    Args:
        config_path (str): Path to YAML configuration file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If the YAML is invalid
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main entry point for the MLOps project pipeline.
    This function parses command-line arguments to configure the pipeline, loads configuration and environment variables,
    sets up logging, loads the specified data stage (raw or processed), and runs the model pipeline. It handles errors
    at each stage, logging failures and exiting gracefully if necessary.
    Command-line Arguments:
        --config: Path to the configuration YAML file (default: "config.yaml").
        --env: Path to an optional .env file for secrets or environment variables (default: ".env").
        --data_stage: Which data stage to load: "raw" or "processed" (default: "processed").
    Workflow:
        1. Parse arguments and load configuration.
        2. Set up logging as specified in the configuration.
        3. Load data according to the specified stage.
        4. Run the model pipeline using the loaded data and configuration.
        5. Log progress and handle exceptions at each step.
    Exits with status code 1 on failure.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Main entry point for the MLOps project pipeline")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration YAML file")
    parser.add_argument("--env", type=str, default=".env",
                        help="Path to an optional .env file for secrets or environment variables")
    parser.add_argument("--data_stage", type=str, default="processed",
                        help="Which data stage to load: raw or processed")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        setup_logging(config.get("logging", {}))
    except Exception as e:
        print(f"Failed to set up logging: {e}", file=sys.stderr)
        sys.exit(1)

    logger.info("Pipeline started")

    try:
        # By default, load processed data for modeling
        df = get_data(config_path=args.config, env_path=args.env,
                      data_stage=args.data_stage)
        if df is None or not hasattr(df, "shape"):
            logger.error(
                "Data loading failed: get_data did not return a valid DataFrame")
            print("Data loading failed: get_data did not return a valid DataFrame")
            sys.exit(1)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Data loaded. Shape: {df.shape}")

        # Run the model pipeline and save splits/model
        run_model_pipeline(df=df, config=config)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")
        sys.exit(1)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
