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

    def _define_scoring_metrics(self):
        return {
            "accuracy": make_scorer(safe_score(accuracy_score)),
            "f1": make_scorer(safe_score(f1_score)),
            "precision": make_scorer(safe_score(precision_score)),
            "recall": make_scorer(safe_score(recall_score)),
            "roc_auc": make_scorer(safe_score(roc_auc_score), needs_proba=True),
        }

    def train(self, X_train, y_train):
        """Train the model on the provided training data."""
        self.model.fit(X_train, y_train)
        return self

    def evaluate_cv(self, X_train, y_train, cv=5):
        """Perform cross-validation and return the results."""
        try:
            cv_results = cross_validate(
                self.model,
                X_train,
                y_train,
                cv=cv,
                scoring=self.scoring,
                return_train_score=True,
                n_jobs=-1,
            )

            metrics = {}
            for metric in self.scoring.keys():
                scores = cv_results[f"test_{metric}"]
                valid_scores = scores[~np.isnan(scores)]
                if len(valid_scores) > 0:
                    mean_score = np.mean(valid_scores)
                    std_score = np.std(valid_scores) if len(valid_scores) > 1 else 0
                else:
                    mean_score = 0.0
                    std_score = 0.0
                metrics[metric] = (mean_score, std_score)

            return metrics
        except Exception as e:
            print(f"Warning: Cross-validation failed with error: {str(e)}")
            return {metric: (0.0, 0.0) for metric in self.scoring.keys()}

    def evaluate(self, X_test, y_test):
        """Evaluate the model on the test set."""
        try:
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1]

            return {
                "accuracy": safe_score(accuracy_score)(y_test, y_pred),
                "f1": safe_score(f1_score)(y_test, y_pred),
                "precision": safe_score(precision_score)(y_test, y_pred),
                "recall": safe_score(recall_score)(y_test, y_pred),
                "roc_auc": safe_score(roc_auc_score)(y_test, y_proba),
            }
        except Exception as e:
            print(f"Warning: Evaluation failed with error: {str(e)}")
            return {metric: 0.0 for metric in self.scoring.keys()}

    def predict(self, X):
        """Make predictions on new data."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get probability predictions on new data."""
        return self.model.predict_proba(X)


def train_and_evaluate_model(X, y, test_size=0.2, random_state=42, cv_folds=5):
    """
    Comprehensive function that:
    1. Splits data into train/test sets
    2. Performs cross-validation on training data
    3. Trains final model on full training data
    4. Evaluates on held-out test set

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

    Returns:
    --------
    dict containing:
        - 'cv_metrics': Cross-validation results on training data
        - 'test_metrics': Final performance on test set
        - 'model': Trained ModelTrainer instance
        - 'train_indices': Indices of training data
        - 'test_indices': Indices of test data
    """
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

    return {
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "model": trainer,
    }
