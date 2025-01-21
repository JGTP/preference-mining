import json
import pandas as pd
import yaml
import logging
from pathlib import Path
from datetime import datetime
from src.black_box import analyze_conditional_importances, train_and_evaluate_model
from src.ripper import (
    analyse_rulesets_globally,
    cross_validate_RIPPER,
    export_analysis_results,
)
from src.preprocessing import DataPreprocessor
from src.feature_set_analysis import analyze_feature_set_differences
import numpy as np


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
    feature_set_config: dict = None,
) -> dict:
    """
    Execute the pipeline with conditional feature importance and feature set analysis.

    Args:
        data_path: Path to input data file
        target_column: Name of target column
        test_size: Size of test set
        config_path: Path to preprocessing configuration
        cache_dir: Directory for caching preprocessing results
        min_year: Minimum year to include in analysis
        n_splits: Number of cross-validation splits
        output_dir: Directory for output files
        feature_set_config: Configuration for feature set analysis
            {
                "deltas": List[float],  # e.g., [0.1, 0.2, 0.3]
                "min_set_size": int,    # default: 1
                "max_set_size": int,    # optional
                "max_pairs_per_rule": int  # default: 1000
            }
    """
    # Set default feature set configuration if none provided
    if feature_set_config is None:
        feature_set_config = {
            "deltas": [0.1, 0.2, 0.3],
            "min_set_size": 1,
            "max_pairs_per_rule": 1000,
        }

    # Convert and create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different outputs
    feature_importance_dir = output_dir / "feature_importances"
    ruleset_dir = output_dir / "rulesets"
    feature_sets_dir = output_dir / "feature_sets"

    for directory in [feature_importance_dir, ruleset_dir, feature_sets_dir]:
        directory.mkdir(parents=True, exist_ok=True)

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

        # Perform feature set analysis
        logging.info("Analyzing feature set differences")
        feature_set_results = analyze_feature_set_differences(
            conditional_results=conditional_results,
            deltas=feature_set_config["deltas"],
            output_dir=feature_sets_dir,
            min_set_size=feature_set_config.get("min_set_size", 1),
            max_set_size=feature_set_config.get("max_set_size", None),
            max_pairs_per_rule=feature_set_config.get("max_pairs_per_rule", 1000),
        )
        logging.info(
            f"Feature set analysis saved to: {feature_set_results['output_path']}"
        )

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise

    logging.info("Writing pipeline results")
    pipeline_results = {
        "ruleset_analysis_path": ruleset_path,
        "conditional_importance_path": conditional_results["output_path"],
        "feature_set_analysis_path": feature_set_results["output_path"],
        "conditional_results": conditional_results["conditional_importances"],
        "feature_set_results": feature_set_results["results"],
    }

    # Export final combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_path = output_dir / f"combined_results_{timestamp}.json"

    serialised = convert_to_serializable(
        {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "pipeline_version": "1.1",
                "feature_set_config": feature_set_config,
            },
            "summary": {
                "conditional_importance_location": str(
                    conditional_results["output_path"]
                ),
                "feature_set_analysis_location": str(
                    feature_set_results["output_path"]
                ),
                "total_rules_analyzed": len(
                    conditional_results["conditional_importances"]
                ),
            },
            "results": pipeline_results,
        }
    )

    with open(final_results_path, "w") as f:
        json.dump(
            serialised,
            f,
            indent=2,
        )

    pipeline_results["final_results_path"] = str(final_results_path)
    logging.info("Pipeline execution completed successfully")
    return pipeline_results


def convert_to_serializable(obj):
    """
    Recursively convert objects in a dictionary to be JSON serializable.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    elif isinstance(obj, Path):
        return str(obj)  # Convert Path objects to strings
    else:
        return obj  # Return the object as is if no conversion is needed
