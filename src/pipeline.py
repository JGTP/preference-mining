import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Tuple, Optional
from src.preprocessing import DataPreprocessor
from src.classifier import cross_validate_RIPPER, analyse_rulesets


def setup_logging():
    """Configure logging for the pipeline"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_preprocessing_config(config_path: str = "config/preprocessing.yaml") -> dict:
    """
    Load preprocessing configuration from YAML file.

    Args:
        config_path: Path to preprocessing configuration file

    Returns:
        Dictionary containing preprocessing configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logging.warning(
            f"Config file {config_path} not found. Using default configuration."
        )
        return {}

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        raise


def execute_pipeline(
    data_path: str,
    target_column: str,
    test: bool = False,
    config_path: str = "config/preprocessing.yaml",
    cache_dir: str = "cache",
    n_splits: int = 3,
) -> dict:
    """
    Execute the pipeline with caching support.
    """
    setup_logging()
    logging.info("Starting pipeline execution")

    try:
        logging.info(f"Loading data from {data_path}")
        if test:
            data = pd.read_excel(data_path, nrows=100)
        else:
            data = pd.read_excel(data_path)
        logging.info(f"Loaded {len(data)} rows and {len(data.columns)} columns")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    try:
        logging.info(f"Loading configuration from {config_path}")
        config = load_preprocessing_config(config_path)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    preprocessor = DataPreprocessor(
        date_columns=config.get("date_columns", []),
        coordinate_columns=config.get("coordinate_columns", []),
        categorical_columns=config.get("categorical_columns", []),
        numeric_categorical_columns=config.get("numeric_categorical_columns", []),
        columns_to_exclude=config.get("columns_to_exclude", []),
        missing_value_codes=config.get("missing_value_codes", {}),
        cache_dir=cache_dir,
    )

    try:
        logging.info("Preprocessing data or loading from cache")
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
            data, target_column=target_column
        )
        logging.info(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

    try:
        logging.info("Performing cross-validation with RIPPER")
        all_rulesets = cross_validate_RIPPER(
            X_train, y_train, n_splits=n_splits, target_name=target_column
        )
        rule_analysis = analyse_rulesets(all_rulesets)
    except Exception as e:
        logging.error(f"Error during cross-validation: {e}")
        raise

    logging.info("Pipeline execution completed successfully")
    return rule_analysis
