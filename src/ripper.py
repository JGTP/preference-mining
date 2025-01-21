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


def train_RIPPER_classifier(X_train, y_train, target_name="target", verbose=True):
    """
    Train a RIPPER classifier with optimized performance.

    Parameters:
    X_train: Features training data (numpy array or pandas DataFrame)
    y_train: Target training data (numpy array or pandas Series)
    verbose: Whether to print detailed output

    Returns:
    tuple: (trained ripper, ruleset)
    """
    # Convert DataFrame to numpy array if needed - faster operations
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train

    ripper = lw.RIPPER(k=2, prune_size=0.33, verbosity=1 if verbose else 0)
    try:
        ripper.fit(trainset=X_train_array, y=y_train_array)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None

    ruleset = ripper.ruleset_

    if verbose:
        print_ruleset_metrics(ruleset, X_train, y_train, target_name)

    return ripper, ruleset


def evaluate_rule_with_f1(rule, data, target_name):
    """
    Evaluate a single rule's performance using F1 score with named columns.

    Parameters:
    rule: Rule object from wittgenstein
    data: DataFrame containing features and target
    target_name: Name of the target column

    Returns:
    dict: Dictionary containing F1 and related metrics
    """
    # Create mask for instances that match rule conditions
    covered_mask = np.ones(len(data), dtype=bool)

    for condition in rule.conds:
        feature_idx = condition.feature  # This is a numeric index
        feature_name = data.columns[feature_idx]  # Get the actual column name

        # Extract operator and value from condition
        cond_str = str(condition)
        operation = "".join([c for c in cond_str if not c.isalnum() and c != "."])
        value = condition.val

        # Access column by name
        feature_values = data[feature_name]

        if operation == "<=":
            covered_mask &= feature_values <= value
        elif operation == ">=":
            covered_mask &= feature_values >= value
        elif operation == "=":
            covered_mask &= feature_values == value
        elif operation == "!=":
            covered_mask &= feature_values != value

    # Handle prediction class
    predicted_class = (
        rule.class_ns_[0] if isinstance(rule.class_ns_, tuple) else rule.class_ns_
    )

    # Get actual and predicted labels
    y_true = (data[target_name] == predicted_class).astype(int)
    y_pred = np.zeros_like(y_true)
    y_pred[covered_mask] = 1

    # Calculate metrics
    support = covered_mask.sum()
    if support > 0:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        precision = recall = f1 = 0

    return {
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "covered_instances": data[covered_mask],
    }


def print_ruleset_metrics(ruleset, X_train, y_train, target_name):
    """Calculate and print F1-based metrics for each rule with named columns"""
    df_train = pd.DataFrame(
        X_train, columns=X_train.columns if isinstance(X_train, pd.DataFrame) else None
    )
    df_train[target_name] = y_train

    print(Fore.CYAN + "\nDecision Rules:\n" + "=" * 50)

    # Evaluate entire ruleset
    ruleset_metrics = evaluate_ruleset_with_f1(ruleset, df_train, target_name)

    # Print individual rule metrics with column names
    for i, (rule, metrics) in enumerate(ruleset_metrics["rule_metrics"], start=1):
        print(Fore.YELLOW + f"\nRule {i}:")

        # Format conditions with column names
        for condition in rule.conds:
            feature_name = df_train.columns[condition.feature]
            # Format the condition value based on its type
            if isinstance(condition.val, (int, float)):
                formatted_value = f"{condition.val:g}"  # Remove trailing zeros
            else:
                formatted_value = str(condition.val)

            # Extract operator from condition string
            cond_str = str(condition)
            operation = "".join([c for c in cond_str if not c.isalnum() and c != "."])

            print(Fore.GREEN + f"  IF {feature_name} {operation} {formatted_value}")

        # Handle prediction class
        predicted_class = (
            rule.class_ns_[0] if isinstance(rule.class_ns_, tuple) else rule.class_ns_
        )
        print(Fore.MAGENTA + f"  THEN Class = {predicted_class}")

        print(
            Fore.BLUE + f"  [Support: {metrics['support']}, "
            f"Precision: {metrics['precision']:.3f}, "
            f"Recall: {metrics['recall']:.3f}, "
            f"F1: {metrics['f1_score']:.3f}]"
        )

    # Print overall ruleset metrics
    print(Fore.CYAN + "\nOverall Ruleset Metrics:")
    print(Fore.BLUE + f"  F1 Score: {ruleset_metrics['overall_f1']:.3f}")
    print(Fore.BLUE + f"  Precision: {ruleset_metrics['overall_precision']:.3f}")
    print(Fore.BLUE + f"  Recall: {ruleset_metrics['overall_recall']:.3f}")


def evaluate_ruleset_with_f1(ruleset, data, target_name):
    """
    Evaluate entire ruleset using F1 score.

    Parameters:
    ruleset: Ruleset from RIPPER
    data: DataFrame containing features and target
    target_name: Name of target column

    Returns:
    dict: Dictionary containing ruleset-level F1 metrics
    """
    all_predictions = np.zeros(len(data))
    all_true = np.zeros(len(data))
    remaining_data = data.copy()

    rule_metrics = []

    for rule in ruleset.rules:
        metrics = evaluate_rule_with_f1(rule, remaining_data, target_name)
        rule_metrics.append((rule, metrics))

        # Update remaining data
        if len(metrics["covered_instances"]) > 0:
            remaining_data = remaining_data.drop(metrics["covered_instances"].index)

        # Update predictions for overall F1 calculation
        covered_indices = metrics["covered_instances"].index
        all_predictions[covered_indices] = 1

        # Handle prediction class - if class_ns_ is a tuple, take the first element
        predicted_class = (
            rule.class_ns_[0] if isinstance(rule.class_ns_, tuple) else rule.class_ns_
        )
        all_true[covered_indices] = (
            data.loc[covered_indices, target_name] == predicted_class
        ).astype(int)

    # Calculate overall ruleset metrics
    overall_f1 = f1_score(all_true, all_predictions, zero_division=0)
    overall_precision = precision_score(all_true, all_predictions, zero_division=0)
    overall_recall = recall_score(all_true, all_predictions, zero_division=0)

    return {
        "rule_metrics": rule_metrics,
        "overall_f1": overall_f1,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
    }


def print_ruleset_metrics(ruleset, X_train, y_train, target_name):
    """Calculate and print F1-based metrics for each rule"""
    df_train = pd.DataFrame(
        X_train, columns=X_train.columns if isinstance(X_train, pd.DataFrame) else None
    )
    df_train[target_name] = y_train

    print(Fore.CYAN + "\nDecision Rules:\n" + "=" * 50)

    # Evaluate entire ruleset
    ruleset_metrics = evaluate_ruleset_with_f1(ruleset, df_train, target_name)

    # Print individual rule metrics
    for i, (rule, metrics) in enumerate(ruleset_metrics["rule_metrics"], start=1):
        print(Fore.YELLOW + f"\nRule {i}:")
        for condition in rule.conds:
            print(Fore.GREEN + f"  IF {condition}")

        # Handle prediction class - if class_ns_ is a tuple, take the first element
        predicted_class = (
            rule.class_ns_[0] if isinstance(rule.class_ns_, tuple) else rule.class_ns_
        )
        print(Fore.MAGENTA + f"  THEN Class = {predicted_class}")

        print(
            Fore.BLUE + f"  [Support: {metrics['support']}, "
            f"Precision: {metrics['precision']:.3f}, "
            f"Recall: {metrics['recall']:.3f}, "
            f"F1: {metrics['f1_score']:.3f}]"
        )

    # Print overall ruleset metrics
    print(Fore.CYAN + "\nOverall Ruleset Metrics:")
    print(Fore.BLUE + f"  F1 Score: {ruleset_metrics['overall_f1']:.3f}")
    print(Fore.BLUE + f"  Precision: {ruleset_metrics['overall_precision']:.3f}")
    print(Fore.BLUE + f"  Recall: {ruleset_metrics['overall_recall']:.3f}")


def analyse_rulesets_globally(all_rulesets, full_dataset, target_name):
    """
    Analyse rules across all rulesets with F1-based metrics.
    """
    global_rule_summary = {}

    # Analyse each unique rule across all rulesets
    for fold, ruleset in enumerate(all_rulesets, start=1):
        for rule in ruleset.rules:
            rule_text = str(rule)

            if rule_text not in global_rule_summary:
                # Evaluate rule globally
                metrics = evaluate_rule_with_f1(rule, full_dataset, target_name)
                global_rule_summary[rule_text] = {
                    "rule": rule,
                    "metrics": metrics,
                    "folds": [],
                    "fold_positions": [],
                }

            # Track fold appearances
            global_rule_summary[rule_text]["folds"].append(fold)
            global_rule_summary[rule_text]["fold_positions"].append(
                len(global_rule_summary[rule_text]["folds"])
            )

    # Calculate stability metrics
    total_folds = len(all_rulesets)
    for rule_text, summary in global_rule_summary.items():
        summary["stability"] = {
            "fold_frequency": len(summary["folds"]) / total_folds,
            "avg_position": np.mean(summary["fold_positions"]),
            "position_std": (
                np.std(summary["fold_positions"])
                if len(summary["fold_positions"]) > 1
                else 0
            ),
        }

    # Print comprehensive analysis
    print(Fore.YELLOW + "\nGlobal Rule Analysis:\n" + "=" * 50)
    for rule_text, summary in global_rule_summary.items():
        metrics = summary["metrics"]
        stability = summary["stability"]

        print(Fore.CYAN + f"\nRule: {rule_text}")
        print(Fore.GREEN + "Global Metrics:")
        print(f"  Support: {metrics['support']}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")

        print(Fore.GREEN + "Stability Metrics:")
        print(f"  Fold Frequency: {stability['fold_frequency']:.2f}")
        print(f"  Average Position: {stability['avg_position']:.2f}")
        print(f"  Position Std: {stability['position_std']:.2f}")

        print(Fore.BLUE + f"Appeared in folds: {summary['folds']}")

    return global_rule_summary


def process_fold(fold_data, target_name):
    """Process a single fold for parallel execution"""
    train_index, test_index, X, y = fold_data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return train_RIPPER_classifier(X_train, y_train, target_name, verbose=False)


def cross_validate_RIPPER(X, y, n_splits=5, target_name="target", n_jobs=-1):
    """
    Parallel cross-validation implementation.

    Parameters:
    X: Features data
    y: Target data
    n_splits: Number of folds
    target_name: Name of target column
    n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
    list: Aggregated rulesets
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Prepare fold data
    fold_data = [(train_idx, test_idx, X, y) for train_idx, test_idx in kf.split(X)]

    # Process folds in parallel
    with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        results = list(
            executor.map(partial(process_fold, target_name=target_name), fold_data)
        )

    # Filter out None results and extract rulesets
    all_rulesets = [ruleset for _, ruleset in results if ruleset is not None]

    print(
        Fore.GREEN
        + f"\nCross-Validation Complete ({len(all_rulesets)} successful folds)\n"
        + "=" * 50
    )
    return all_rulesets


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


def format_rule_for_json(rule, metrics, feature_names):
    """
    Format a single rule and its metrics for JSON export.
    Args:
        rule: Rule object from wittgenstein
        metrics: Dictionary containing rule metrics
        feature_names: List of feature names
    Returns:
        dict: Formatted rule data
    """
    conditions = []
    for cond in rule.conds:
        feature_name = feature_names[cond.feature]
        operator = determine_operator(cond)
        value = convert_to_serializable(cond.val)
        conditions.append(
            {"feature": feature_name, "operator": operator, "value": value}
        )

    predicted_class = (
        rule.class_ns_[0] if isinstance(rule.class_ns_, tuple) else rule.class_ns_
    )

    return {
        "antecedents": conditions,
        "consequent": str(predicted_class),
        "metrics": {
            "support": convert_to_serializable(metrics["support"]),
            "precision": round(float(metrics["precision"]), 3),
            "recall": round(float(metrics["recall"]), 3),
            "f1_score": round(float(metrics["f1_score"]), 3),
        },
        "stability": {
            "fold_frequency": round(float(metrics["stability"]["fold_frequency"]), 2),
            "average_position": round(float(metrics["stability"]["avg_position"]), 2),
            "position_std": round(float(metrics["stability"]["position_std"]), 2),
            "appeared_in_folds": [convert_to_serializable(x) for x in metrics["folds"]],
        },
    }


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
    """
    Export both ruleset analysis and pipeline results to JSON files.

    Args:
        rule_analysis: Dictionary containing rule analysis results
        feature_names: List of feature names
        pipeline_results: Dictionary containing pipeline results including conditional importances
        output_dir: Directory to save the JSON files
        include_conditional_results: Whether to include the full conditional results

    Returns:
        Dict[str, str]: Dictionary with paths to the exported files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    export_paths = {}

    # Export ruleset analysis if provided
    if rule_analysis and feature_names:
        formatted_rules = []
        for rule_text, summary in rule_analysis.items():
            formatted_rule = format_rule_for_json(
                summary["rule"],
                {
                    **summary["metrics"],
                    "stability": summary["stability"],
                    "folds": summary["folds"],
                },
                feature_names,
            )
            formatted_rules.append(formatted_rule)

        ruleset_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_rules": len(formatted_rules),
            },
            "rules": formatted_rules,
        }

        ruleset_filename = f"ruleset_analysis_{timestamp}.json"
        ruleset_filepath = output_dir / ruleset_filename

        with open(ruleset_filepath, "w", encoding="utf-8") as f:
            json.dump(
                _serialize_for_json(ruleset_data), f, indent=2, ensure_ascii=False
            )

        export_paths["ruleset_analysis_path"] = str(ruleset_filepath)

    # Export pipeline results if provided
    if pipeline_results:
        # Convert any non-serializable objects in pipeline_results
        pipeline_results = _serialize_for_json(pipeline_results)

        pipeline_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "pipeline_version": "1.0",
            },
            "summary": {},
        }

        # Include ruleset path if it was just created
        if "ruleset_analysis_path" in export_paths:
            pipeline_data["summary"]["ruleset_location"] = export_paths[
                "ruleset_analysis_path"
            ]

        # Add conditional importance information
        if "conditional_importance_path" in pipeline_results:
            pipeline_data["summary"]["conditional_importance_location"] = (
                pipeline_results["conditional_importance_path"]
            )

        if include_conditional_results and "conditional_results" in pipeline_results:
            pipeline_data["conditional_results"] = pipeline_results[
                "conditional_results"
            ]
            pipeline_data["summary"]["total_rules_analyzed"] = len(
                pipeline_results["conditional_results"]
            )

        pipeline_filename = f"pipeline_results_{timestamp}.json"
        pipeline_folder = output_dir / "intermediate"
        pipeline_folder.mkdir(parents=True, exist_ok=True)
        pipeline_filepath = pipeline_folder / pipeline_filename

        with open(pipeline_filepath, "w", encoding="utf-8") as f:
            json.dump(
                _serialize_for_json(pipeline_data), f, indent=2, ensure_ascii=False
            )

        export_paths["pipeline_results_path"] = str(pipeline_filepath)

    return export_paths
