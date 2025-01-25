import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate, train_test_split
import shap
from src.ripper import determine_operator
from src.utils import convert_to_serializable, save_json_results, evaluate_model


class ModelTrainer:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            class_weight="balanced",
            random_state=42,
        )
        self.feature_names = None

    def train(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        self.model.fit(X_train, y_train)
        return self

    def evaluate_cv(self, X, y, cv=5):
        if len(X) == 0 or len(y) == 0:
            return {
                metric: (0.0, 0.0)
                for metric in ["accuracy", "f1", "precision", "recall", "roc_auc"]
            }
        cv_results = cross_validate(self.model, X, y, cv=cv)
        return {
            metric.replace("test_", ""): (np.mean(scores), np.std(scores))
            for metric, scores in cv_results.items()
        }

    def evaluate(self, X, y):
        predictions = self.predict(X)
        probas = self.predict_proba(X)
        return evaluate_model(y, predictions, probas)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_shap_feature_importances(self, X):
        if not hasattr(self.model, "predict"):
            raise ValueError("Model has not been trained yet. Call train() first.")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X)
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
            },
        }
        return importance_dict

    def save_feature_importances(self, X, output_dir="results/feature_importances"):
        importance_dict = self.get_shap_feature_importances(X)
        return save_json_results(importance_dict, output_dir, "feature_importances")


def train_and_evaluate_model(
    X,
    y,
    test_size=0.2,
    random_state=42,
    cv_folds=5,
    output_dir="results/feature_importances",
):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Input data is empty")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    trainer = ModelTrainer()
    cv_metrics = trainer.evaluate_cv(X_train, y_train, cv=cv_folds)
    trainer.train(X_train, y_train)
    test_metrics = trainer.evaluate(X_test, y_test)
    feature_importance_path = trainer.save_feature_importances(X_train, output_dir)

    return {
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "feature_importance_path": feature_importance_path,
    }, trainer


def analyze_conditional_importances(
    X, rule_analysis, trained_model, output_dir="results/feature_importances"
):
    conditional_importances = {}

    for ruleset_key, ruleset_data in rule_analysis.items():
        for rule_idx, rule_object in enumerate(ruleset_data["rules"]):
            mask = pd.Series(True, index=X.index)

            for condition in rule_object.conds:
                feature_idx = condition.feature
                feature_name = X.columns[feature_idx]
                operator = determine_operator(str(condition))
                value = convert_to_serializable(condition.val)

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

            if mask.sum() < 10:
                continue

            X_subset = X[mask]
            importance_dict = trained_model.get_shap_feature_importances(X_subset)
            rule_key = f"Ruleset {ruleset_key} - Rule {rule_idx}"
            conditional_importances[rule_key] = {
                "feature_importances": importance_dict["feature_importances"],
                "support": int(mask.sum()),
                "support_percentage": float(mask.sum() / len(X) * 100),
            }

    results = {
        "conditional_importances": conditional_importances,
        "metadata": {
            "n_rules_analyzed": len(conditional_importances),
            "total_samples": len(X),
        },
    }

    output_path = save_json_results(results, output_dir, "conditional_importances")
    return {
        "conditional_importances": conditional_importances,
        "output_path": output_path,
    }
