import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Tuple, Optional
from src.preprocessing import DataPreprocessor
from src.classifier import train_RIPPER_classifier, evaluate_classifier

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

def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = "data/processed",
) -> None:
    """Save processed datasets to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)

def execute_pipeline(
    data_path: str,
    target_column: str,
    test: bool = False,
    config_path: str = "config/preprocessing.yaml",
    output_dir: Optional[str] = None,
    save_intermediate: bool = False,
) -> Tuple[object, list]:
    """
    Execute the complete data processing and classification pipeline.

    Args:
        data_path: Path to the input Excel file
        config_path: Path to preprocessing configuration file
        target_column: Name of the target variable column
        output_dir: Directory to save processed data (if save_intermediate is True)
        save_intermediate: Whether to save intermediate processed data

    Returns:
        Tuple of (trained classifier, rules)
    """
    setup_logging()
    logging.info("Starting pipeline execution")

    try:
        logging.info(f"Loading data from {data_path}")
        if test == True:
            data = pd.read_excel(data_path, nrows=10)
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
    )

    try:
        logging.info("Preprocessing data")
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
            data, target_column=target_column
        )
        logging.info(
            f"Preprocessed data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}"
        )

        if save_intermediate and output_dir:
            logging.info(f"Saving processed data to {output_dir}")
            save_processed_data(X_train, X_test, y_train, y_test, output_dir)

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

    try:
        logging.info("Training RIPPER classifier")
        classifier, rules = train_RIPPER_classifier(X_train, y_train, target_column)
        logging.info(f"Generated {len(rules)} rules")
    except Exception as e:
        logging.error(f"Error during classifier training: {e}")
        raise

    try:
        logging.info("Evaluating classifier")
        evaluate_classifier(classifier, X_test, y_test, target_column, rules)
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

    logging.info("Pipeline execution completed successfully")
    return classifier, rules
