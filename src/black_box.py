import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    make_scorer,
)
from pathlib import Path
import json
from datetime import datetime
import shap

from src.ripper import determine_operator


def safe_score(scorer_func):
    """Wrapper to handle edge cases in scoring functions."""

    def wrapped_score(*args, **kwargs):
        try:
            return scorer_func(*args, **kwargs)
        except Exception:
            return 0.0

    return wrapped_score


class ModelTrainer:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            class_weight="balanced",
            random_state=42,
        )
        self.scoring = self._define_scoring_metrics()
        self.feature_names = None

    def _define_scoring_metrics(self):
        """Define scoring metrics with safety wrapper."""
        return {
            "accuracy": make_scorer(safe_score(accuracy_score)),
            "f1": make_scorer(safe_score(f1_score)),
            "precision": make_scorer(safe_score(precision_score)),
            "recall": make_scorer(safe_score(recall_score)),
            "roc_auc": make_scorer(safe_score(roc_auc_score), needs_proba=True),
        }

    def train(self, X_train, y_train):
        """Train the model on the provided training data."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        self.model.fit(X_train, y_train)
        return self

    def evaluate_cv(self, X, y, cv=5):
        """
        Perform cross-validation evaluation.

        Args:
            X: Features matrix
            y: Target vector
            cv: Number of cross-validation folds

        Returns:
            dict: Metric names mapped to (mean, std) tuples
        """
        if len(X) == 0 or len(y) == 0:
            return {metric: (0.0, 0.0) for metric in self.scoring}

        cv_results = cross_validate(self.model, X, y, scoring=self.scoring, cv=cv)
        return {
            metric.replace("test_", ""): (np.mean(scores), np.std(scores))
            for metric, scores in cv_results.items()
        }

    def evaluate(self, X, y):
        """
        Evaluate model on test data.

        Args:
            X: Features matrix
            y: Target vector

        Returns:
            dict: Metric names mapped to scores
        """
        predictions = self.predict(X)
        probas = self.predict_proba(X)

        return {
            name: scorer._score_func(
                y, predictions if not scorer._kwargs.get("needs_proba") else probas
            )
            for name, scorer in self.scoring.items()
        }

    def predict(self, X):
        """Make predictions on input data."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Make probability predictions on input data."""
        return self.model.predict_proba(X)

    def get_shap_feature_importances(self, X):
        """
        Calculate SHAP feature importances for the trained model.

        Args:
            X: The input dataset (same format as training data).

        Returns:
            dict: Dictionary containing feature importances and metadata.
        """
        if not hasattr(self.model, "predict"):
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Create a SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X)

        # Calculate mean absolute SHAP values for each feature
        mean_shap_values = np.abs(shap_values.values).mean(axis=0)

        importance_dict = {
            "feature_importances": {
                name: float(importance)
                for name, importance in zip(self.feature_names, mean_shap_values)
            },
            "metadata": {
                "model_type": self.model.__class__.__name__,
                "n_features": len(self.feature_names),
                "feature_names": self.feature_names,
                "timestamp": datetime.now().isoformat(),
            },
        }

        return importance_dict

    def save_feature_importances(self, X, output_dir="results/feature_importances"):
        """
        Calculate feature importances and save them to a JSON file.

        Args:
            X: The input dataset to calculate SHAP values for
            output_dir: Directory where the JSON file will be saved

        Returns:
            Path: Path to the saved JSON file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        importance_dict = self.get_shap_feature_importances(X)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feature_importances_{timestamp}.json"
        filepath = output_dir / filename

        # Save to JSON file
        with open(filepath, "w") as f:
            json.dump(importance_dict, f, indent=2)

        return filepath


def train_and_evaluate_model(
    X,
    y,
    test_size=0.2,
    random_state=42,
    cv_folds=5,
    output_dir: str | Path = "results/feature_importances",
):
    """
    Comprehensive function that:
    1. Splits data into train/test sets
    2. Performs cross-validation on training data
    3. Trains final model on full training data
    4. Evaluates on held-out test set
    5. Calculates and saves feature importances

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Features matrix
    y : array-like of shape (n_samples,)
        Target vector
    test_size : float, default=0.2
        Proportion of dataset to include in the test split
    random_state : int, default=42
        Random state for reproducibility
    cv_folds : int, default=5
        Number of folds for cross-validation
    output_dir : str or Path, default="results"
        Directory where feature importances will be saved

    Returns:
    --------
    dict containing:
        - 'cv_metrics': Cross-validation results on training data
        - 'test_metrics': Final performance on test set
        - 'model': Trained ModelTrainer instance
        - 'feature_importance_path': Path to saved feature importances JSON
    """
    # Convert output_dir to Path and create it
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input data
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Input data is empty")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Initialize trainer
    trainer = ModelTrainer()

    # Get cross-validation metrics
    cv_metrics = trainer.evaluate_cv(X_train, y_train, cv=cv_folds)

    # Print cross-validation results
    print("\nCross-validation metrics:")
    print("-" * 50)
    for metric, (mean, std) in cv_metrics.items():
        print(f"{metric:10s}: {mean:.3f} (±{std:.3f})")

    # Train final model on full training data
    trainer.train(X_train, y_train)

    # Evaluate on test set
    test_metrics = trainer.evaluate(X_test, y_test)

    # Print test set results
    print("\nHeld-out test set metrics:")
    print("-" * 50)
    for metric, score in test_metrics.items():
        print(f"{metric:10s}: {score:.3f}")

    # Calculate and save feature importances
    feature_importance_path = trainer.save_feature_importances(X_train, output_dir)
    print(f"\nFeature importances saved to: {feature_importance_path}")

    return {
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "feature_importance_path": feature_importance_path,
    }, trainer


def analyze_conditional_importances(
    X, rule_analysis, trained_model, output_dir="results/feature_importances"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditional_importances = {}

    for ruleset_key, ruleset_data in rule_analysis.items():
        for rule_idx, rule_object in enumerate(ruleset_data["rules"]):
            mask = pd.Series(True, index=X.index)

            for condition in rule_object.conds:
                feature_idx = condition.feature
                feature_name = X.columns[feature_idx]
                operator = determine_operator(str(condition))
                value = convert_to_serializable(condition.val)

                # Apply condition to filter rows
                if operator == "==":
                    mask &= X[feature_name] == value
                elif operator == "!=":
                    mask &= X[feature_name] != value
                elif operator == ">":
                    mask &= X[feature_name] > value
                elif operator == "<":
                    mask &= X[feature_name] < value
                elif operator == ">=":
                    mask &= X[feature_name] >= value
                elif operator == "<=":
                    mask &= X[feature_name] <= value

            # Skip if too few samples match the rule
            if mask.sum() < 10:
                continue

            # Get subset of data matching the rule conditions
            X_subset = X[mask]

            # Calculate SHAP values for this subset
            importance_dict = trained_model.get_shap_feature_importances(X_subset)

            # Create a unique key for each rule
            rule_key = f"Ruleset {ruleset_key} - Rule {rule_idx}"
            conditional_importances[rule_key] = {
                "feature_importances": importance_dict["feature_importances"],
                "support": int(mask.sum()),
                "support_percentage": float(mask.sum() / len(X) * 100),
            }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"conditional_importances_{timestamp}.json"

    results = {
        "conditional_importances": conditional_importances,
        "metadata": {
            "n_rules_analyzed": len(conditional_importances),
            "total_samples": len(X),
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Convert results to a serializable format
    serializable_results = convert_to_serializable(results)

    # Write the converted dictionary to JSON
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    return {
        "conditional_importances": conditional_importances,
        "output_path": output_path,
    }


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
    else:
        return obj  # Return the object as is if no conversion is needed
