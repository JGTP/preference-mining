import pandas as pd
import yaml
import logging
from pathlib import Path
from src.black_box import analyze_conditional_importances, train_and_evaluate_model
from src.ripper import (
    analyse_rulesets_globally,
    cross_validate_RIPPER,
    export_analysis_results,
)
from src.preprocessing import DataPreprocessor


def setup_logging():
    """Configure logging for the pipeline"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_preprocessing_config(config_path: str = "config/gtd.yaml") -> dict:
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
    test_size: int = None,
    config_path: str = "config/gtd.yaml",
    cache_dir: str = None,
    min_year=1997,
    n_splits: int = 3,
    output_dir: str = "results",
) -> dict:
    """
    Execute the pipeline with conditional feature importance analysis.
    """
    # Convert and create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different outputs
    feature_importance_dir = output_dir / "feature_importances"
    ruleset_dir = output_dir / "rulesets"

    feature_importance_dir.mkdir(parents=True, exist_ok=True)
    ruleset_dir.mkdir(parents=True, exist_ok=True)

    setup_logging()
    logging.info("Starting pipeline execution")

    try:
        logging.info(f"Loading data from {data_path}")
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
        min_year=min_year,
        year_column="iyear",
        test_size=test_size,
    )

    try:
        logging.info("Preprocessing data or loading from cache")
        X, y = preprocessor.preprocess_data(data, target_column=target_column)
        logging.info(f"Data shapes - X: {X.shape}, y: {y.shape}")

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

    try:
        logging.info("Performing cross-validation with RIPPER")
        all_rulesets = cross_validate_RIPPER(
            X, y, n_splits=n_splits, target_name=target_column
        )
        # Combine X and y for analysis
        data_with_target = X.copy()
        data_with_target[target_column] = y

        rule_analysis = analyse_rulesets_globally(
            all_rulesets, data_with_target, target_name=target_column
        )

        # Export the ruleset analysis
        logging.info("Exporting ruleset analysis to JSON")
        feature_names = X.columns.tolist()
        export_paths = export_analysis_results(
            rule_analysis=rule_analysis,
            feature_names=feature_names,
            output_dir=ruleset_dir,
        )
        ruleset_path = export_paths["ruleset_analysis_path"]

        # Train black box model
        logging.info("Training black box model")
        bb_model, model = train_and_evaluate_model(X, y)

        # Perform conditional feature importance analysis
        logging.info("Calculating conditional feature importances")
        conditional_results = analyze_conditional_importances(
            X, rule_analysis, model, output_dir=feature_importance_dir
        )
        logging.info(
            f"Conditional feature importances saved to: {conditional_results['output_path']}"
        )

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise

    logging.info("Pipeline execution completed successfully")

    pipeline_results = {
        "ruleset_analysis_path": ruleset_path,
        "conditional_importance_path": conditional_results["output_path"],
        "conditional_results": conditional_results["conditional_importances"],
    }

    final_export_paths = export_analysis_results(
        pipeline_results=pipeline_results, output_dir=output_dir
    )

    return {**pipeline_results, **final_export_paths}
