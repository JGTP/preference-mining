import logging
from pathlib import Path
from tqdm import tqdm
import yaml
from src.progress_logger import setup_progress_logging
from src.preprocessing import DataPreprocessor
from src.black_box import train_and_evaluate_model
from src.feature_set_analysis import EnhancedFeatureAnalyser
from src.ripper import cross_validate_RIPPER
from src.utils import save_json_results
from src.visualisation import create_plots, process_results


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_preprocessing_config(config_path: str = "config/gtd.yaml") -> dict:
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
    test_size: int = None,
    config_path: str = "config/gtd.yaml",
    cache_dir: str = None,
    min_year: int = 1997,
    n_splits: int = 5,
    output_dir: str = "results",
    analysis_config: dict = None,
) -> dict:
    if analysis_config is None:
        analysis_config = {
            "epsilons": [0.1, 0.2],
            "deltas": [i / 20 for i in range(1, 11)],
            "max_set_size": 10,
            "top_features": 20,
        }

    config = load_preprocessing_config(config_path)
    target_column = config.get("target").get("column")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_analysis_dir = output_dir / "feature_analysis"
    feature_analysis_dir.mkdir(parents=True, exist_ok=True)
    progress_logger = setup_progress_logging()

    try:
        logging.info(f"Loading data from {data_path}")
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
        preprocess_id = progress_logger.create_progress_bar(
            "preprocess", 100, "Preprocessing data"
        )
        X, y = preprocessor.preprocess_data(data_path, target_column=target_column)
        progress_logger.update_progress(preprocess_id, 100)
        progress_logger.close_progress_bar(preprocess_id)

        ripper_id = progress_logger.create_progress_bar(
            "ripper", n_splits, "Cross-validating RIPPER"
        )
        stable_rules = cross_validate_RIPPER(
            X, y, n_splits=n_splits, progress_logger=progress_logger
        )
        tqdm.write(f"\nNumber of stable rules found: {len(stable_rules)}")
        progress_logger.close_progress_bar(ripper_id)

        bb_id = progress_logger.create_progress_bar(
            "blackbox", 100, "Training black box model"
        )
        bb_results, model = train_and_evaluate_model(X, y)
        progress_logger.update_progress(bb_id, 100)
        progress_logger.close_progress_bar(bb_id)

        analyser = EnhancedFeatureAnalyser(
            model=model.model,
            X=X,
            epsilons=analysis_config.get("epsilons"),
            deltas=analysis_config.get("deltas"),
            max_set_size=analysis_config.get("max_set_size", 10),
            top_features=analysis_config.get("top_features", 20),
            enable_disk_cache=False,
            progress_logger=progress_logger,
        )

        feature_analysis = analyser.analyse_ruleset(
            ruleset=stable_rules, output_dir=feature_analysis_dir
        )

        df = process_results(
            feature_analysis,
            shap_values=analyser.shap_values,
            correlations=analyser.correlations,
        )
        create_plots(
            df,
            output_dir,
            n_rules=len(stable_rules),
            test_size=test_size,
            max_set_size=analysis_config.get("max_set_size", 10),
            top_features=analysis_config.get("top_features", 20),
            n_splits=n_splits,
            total_features=len(X.columns),
        )

        pipeline_results = {
            "metadata": {
                "pipeline_version": "2.0",
                "analysis_config": analysis_config,
            },
            "black_box_results": bb_results,
            "feature_analysis_results": feature_analysis,
        }

        output_path = save_json_results(
            pipeline_results, output_dir, "pipeline_results"
        )
        pipeline_results["output_path"] = output_path

        analyser.cleanup()
        progress_logger.shutdown()
        return pipeline_results

    except Exception as e:
        progress_logger.shutdown()
        logging.error(f"Error during pipeline execution: {e}")
        raise
