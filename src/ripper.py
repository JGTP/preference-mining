from pathlib import Path
from typing import Dict, Any, List, Union
import wittgenstein as lw
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from colorama import Fore, Style, init
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from datetime import datetime

init(autoreset=True)


def train_RIPPER_classifier(X_train, y_train):
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train

    ripper = lw.RIPPER(k=2, prune_size=0.33)
    try:
        ripper.fit(trainset=X_train_array, y=y_train_array)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None

    ruleset = ripper.ruleset_
    for rule in ruleset.rules:
        tuple = rule.class_ns_
        rule.predicted_class = 0 if tuple[0] > tuple[1] else 1

    return ripper, ruleset


def analyse_rulesets_globally(all_rulesets, full_dataset, target_name):
    global_rule_summary = {}

    for fold, ruleset in enumerate(all_rulesets, start=1):
        ruleset_str = str(ruleset)

        if ruleset_str not in global_rule_summary:
            global_rule_summary[ruleset_str] = {
                "ruleset": ruleset,
                "folds": [],
                "fold_positions": [],
                "rules": [],
            }

        for rule in ruleset.rules:
            rule_text = str(rule)
            global_rule_summary[ruleset_str]["rules"].append(rule)

        global_rule_summary[ruleset_str]["folds"].append(fold)
        global_rule_summary[ruleset_str]["fold_positions"].append(
            len(global_rule_summary[ruleset_str]["fold_positions"]) + 1
        )

    total_folds = len(all_rulesets)
    for ruleset_str, summary in global_rule_summary.items():
        summary["stability"] = {
            "fold_frequency": len(summary["folds"]) / total_folds,
            "avg_position": (
                np.mean(summary["fold_positions"]) if summary["fold_positions"] else 0
            ),
            "position_std": (
                np.std(summary["fold_positions"])
                if len(summary["fold_positions"]) > 1
                else 0
            ),
        }

    return global_rule_summary


def process_fold(fold_data):
    train_index, test_index, X, y = fold_data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return train_RIPPER_classifier(X_train, y_train)


def cross_validate_RIPPER(X, y, n_splits=5, n_jobs=-1):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_data = [(train_idx, test_idx, X, y) for train_idx, test_idx in kf.split(X)]

    with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        results = list(executor.map(partial(process_fold), fold_data))

    all_rulesets = [ruleset for _, ruleset in results if ruleset is not None]

    print(
        Fore.GREEN
        + f"\nCross-Validation Complete ({len(all_rulesets)} successful folds)\n"
        + "=" * 50
    )
    return all_rulesets


def format_rule_for_json(rule, metrics, feature_names):
    def format_condition(cond):
        feature_name = feature_names[cond.feature]
        operator = determine_operator(cond)
        value = convert_to_serializable(cond.val)
        return {"feature": feature_name, "operator": operator, "value": value}

    return {
        "rule_structure": {
            "conditions": [format_condition(cond) for cond in rule.conds],
            "predicted_class": rule.predicted_class,
        },
        "raw_rule_str": str(rule),
        "stability": {
            "fold_frequency": round(float(metrics["stability"]["fold_frequency"]), 2),
            "average_position": round(float(metrics["stability"]["avg_position"]), 2),
            "position_std": round(float(metrics["stability"]["position_std"]), 2),
            "appeared_in_folds": [convert_to_serializable(x) for x in metrics["folds"]],
        },
    }


def format_rule_condition(condition, feature_names):
    """
    Format a single rule condition using feature names instead of indices.

    Args:
        condition: Rule condition from wittgenstein
        feature_names: List of feature names

    Returns:
        str: Formatted condition string
    """
    feature_name = feature_names[condition.feature]

    # Clean up the condition string and extract operator
    cond_str = str(condition)
    # Handle common operators
    if "<=" in cond_str:
        operator = "<="
    elif ">=" in cond_str:
        operator = ">="
    elif "=" in cond_str:
        operator = "="
    elif "<" in cond_str:
        operator = "<"
    elif ">" in cond_str:
        operator = ">"
    else:
        operator = "="  # default to equality if no operator found

    # Clean and format the value
    value = convert_to_serializable(condition.val)
    if isinstance(value, (int, float)):
        # Handle special cases for missing values
        if value == -99 or value == -999:
            formatted_value = "MISSING"
        else:
            formatted_value = f"{value:g}"  # Remove trailing zeros
    else:
        formatted_value = str(value).strip()

    # Build a clean, readable condition
    if formatted_value == "MISSING":
        return f"{feature_name} is MISSING"
    else:
        return f"{feature_name} {operator} {formatted_value}"


def convert_to_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Any object that might contain numpy types

    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle potential NaN and infinite values
        if np.isnan(obj):
            return "MISSING"
        elif np.isinf(obj):
            return "INFINITE"
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def determine_operator(cond):
    """Determine the operator for a condition."""
    cond_str = str(cond)
    if "<=" in cond_str:
        return "<="
    elif ">=" in cond_str:
        return ">="
    elif "=" in cond_str:
        return "=="
    elif "<" in cond_str:
        return "<"
    elif ">" in cond_str:
        return ">"
    return "=="


def _path_to_str(obj: Any) -> Any:
    """
    Recursively convert any Path objects to strings in a nested structure.

    Args:
        obj: Any Python object that might contain Path objects

    Returns:
        The same structure with all Path objects converted to strings
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _path_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_path_to_str(item) for item in obj]
    return obj


def _serialize_for_json(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable types.
    Handles:
    - Path objects -> str
    - numpy numeric types -> Python numeric types
    - nested dictionaries and lists

    Args:
        obj: Any Python object that might need conversion

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    return obj


def export_analysis_results(
    rule_analysis: Dict = None,
    feature_names: List[str] = None,
    pipeline_results: Dict[str, Any] = None,
    output_dir: Union[str, Path] = "results",
    include_conditional_results: bool = True,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    export_paths = {}

    if rule_analysis and feature_names:
        formatted_rulesets = []
        for ruleset_str, summary in rule_analysis.items():
            formatted_ruleset = {
                "raw_ruleset_str": ruleset_str,
                "rules": [
                    format_rule_for_json(
                        rule,
                        {
                            "stability": {
                                "fold_frequency": summary["stability"][
                                    "fold_frequency"
                                ],
                                "avg_position": summary["stability"]["avg_position"],
                                "position_std": summary["stability"]["position_std"],
                            },
                            "folds": summary["folds"],
                        },
                        feature_names,
                    )
                    for rule in summary["rules"]
                ],
                "stability": summary["stability"],
            }
            formatted_rulesets.append(formatted_ruleset)

        ruleset_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_rulesets": len(formatted_rulesets),
            },
            "rulesets": formatted_rulesets,
        }

        ruleset_filename = f"ruleset_analysis_{timestamp}.json"
        ruleset_filepath = output_dir / ruleset_filename

        with open(ruleset_filepath, "w", encoding="utf-8") as f:
            json.dump(
                _serialize_for_json(ruleset_data), f, indent=2, ensure_ascii=False
            )

        export_paths["ruleset_analysis_path"] = str(ruleset_filepath)

    return export_paths
