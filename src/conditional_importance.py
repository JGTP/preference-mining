from typing import Dict, List, Union, Tuple
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from src.preprocessing import DataPreprocessor


@dataclass
class ConditionalImportanceResult:
    """Stores the results of conditional feature importance analysis for a single rule"""

    rule_conditions: List[str]
    rule_prediction: str
    subset_size: int
    total_size: int
    feature_importance: Dict[str, float]
    subset_performance: Dict[str, float]
    analysis_timestamp: str


def apply_rule_conditions(data: pd.DataFrame, conditions: List[str]) -> pd.Series:
    """
    Applies RIPPER rule conditions to create a boolean mask for the dataset

    Args:
        data: DataFrame to apply conditions to
        conditions: List of conditions in format "feature = value" or "feature <= value" etc.

    Returns:
        Boolean mask indicating which rows match all conditions
    """
    mask = pd.Series(True, index=data.index)

    for condition in conditions:
        # Split condition into feature, operator, and value
        parts = condition.split()
        feature = parts[0]
        operator = parts[1]
        value = parts[2]

        # Apply the condition using string comparison for categorical features
        if operator == "=":
            mask &= data[feature].astype(str) == str(value)
        elif operator == "<=":
            mask &= pd.to_numeric(data[feature], errors="coerce") <= float(value)
        elif operator == ">=":
            mask &= pd.to_numeric(data[feature], errors="coerce") >= float(value)
        elif operator == "<":
            mask &= pd.to_numeric(data[feature], errors="coerce") < float(value)
        elif operator == ">":
            mask &= pd.to_numeric(data[feature], errors="coerce") > float(value)

    return mask


def calculate_subset_performance(
    model, X: pd.DataFrame, y: pd.Series
) -> Dict[str, float]:
    """Calculate performance metrics for the model on a given subset"""
    y_pred = model.predict(X)

    performance = {
        "accuracy": float((y_pred == y).mean()),
        "subset_positive_ratio": float(y.mean()),
        "prediction_positive_ratio": float(y_pred.mean()),
    }

    return performance


def preprocess_subset(
    X: pd.DataFrame, y: pd.Series, preprocessor: DataPreprocessor
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess a data subset using the provided preprocessor

    Args:
        X: Feature DataFrame
        y: Target series
        preprocessor: Initialized DataPreprocessor instance

    Returns:
        Tuple of (processed_features, processed_target)

    Raises:
        ValueError: If target series is None
    """
    if y is None:
        raise ValueError("Target series cannot be None")

    # Create a copy of the preprocessor to avoid modifying the original
    subset_preprocessor = DataPreprocessor(
        categorical_columns=preprocessor.categorical_columns,
        numeric_categorical_columns=preprocessor.numeric_categorical_columns,
        one_hot_encoding=preprocessor.one_hot_encoding,
    )

    X_processed, y_processed = subset_preprocessor.preprocess_data(X, y)
    return X_processed, y_processed


def analyze_conditional_importance(
    X: pd.DataFrame,
    y: pd.Series,
    ruleset: Dict,
    model,
    preprocessor: DataPreprocessor,
    random_state: int = 42,
    test_size: float = 0.2,
    min_subset_size: int = 100,
) -> List[ConditionalImportanceResult]:
    """
    Analyze feature importance conditioned on each rule in the ruleset

    Args:
        X: Original (non-preprocessed) feature DataFrame
        y: Target series
        ruleset: Dictionary containing RIPPER rules
        model: Trained model
        preprocessor: Initialized DataPreprocessor instance
        random_state: Random state for reproducibility
        test_size: Proportion of data to use for test set
        min_subset_size: Minimum size of subset to analyze

    Returns:
        List of ConditionalImportanceResult objects

    Raises:
        ValueError: If target series is None
    """
    if y is None:
        raise ValueError("Target series cannot be None")

    results = []
    explainer = shap.TreeExplainer(model)

    for rule in ruleset["rules"]:
        # Apply rule conditions to create subset from original data
        mask = apply_rule_conditions(X, rule["if_conditions"])
        subset_X_raw = X[mask]
        subset_y = y[mask]

        # Skip if subset is too small
        if len(subset_X_raw) < min_subset_size:
            logging.warning(
                f"Skipping rule {rule['if_conditions']} - subset size {len(subset_X_raw)} < {min_subset_size}"
            )
            continue

        try:
            # Preprocess the subset
            subset_X_processed, subset_y_processed = preprocess_subset(
                subset_X_raw, subset_y, preprocessor
            )

            # Split subset into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                subset_X_processed,
                subset_y_processed,
                test_size=test_size,
                random_state=random_state,
            )

            # Train a new model on the subset
            subset_model = model.__class__(**model.get_params())
            subset_model.fit(X_train, y_train)

            # Calculate SHAP values for test set
            shap_values = explainer.shap_values(X_test)

            # Calculate mean absolute SHAP values for each feature
            feature_importance = {
                feature: float(np.abs(shap_values[:, i]).mean())
                for i, feature in enumerate(X_test.columns)
            }

            # Calculate performance metrics
            subset_performance = calculate_subset_performance(
                subset_model, X_test, y_test
            )

            # Store results
            result = ConditionalImportanceResult(
                rule_conditions=rule["if_conditions"],
                rule_prediction=rule["then_prediction"],
                subset_size=len(subset_X_raw),
                total_size=len(X),
                feature_importance=feature_importance,
                subset_performance=subset_performance,
                analysis_timestamp=datetime.now().isoformat(),
            )

            results.append(result)

        except Exception as e:
            logging.error(f"Error processing rule {rule['if_conditions']}: {str(e)}")
            continue

    return results


def export_conditional_importance(
    results: List[ConditionalImportanceResult],
    output_dir: str = "results",
    filename: str = "conditional_importance.json",
) -> Path:
    """Export conditional importance results to JSON"""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to dictionary format
    results_dict = {
        "metadata": {
            "export_date": datetime.now().isoformat(),
            "number_of_rules_analyzed": len(results),
        },
        "conditional_importance": [asdict(result) for result in results],
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    return output_path
