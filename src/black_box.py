import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate, train_test_split
import shap
from src.utils import evaluate_model, save_json_results, convert_to_serialisable
from src.ripper import determine_operator


class ModelTrainer:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            class_weight="balanced",
            random_state=42,
        )
        self.feature_names = None

    def train(self, X, y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data is empty")
        self.feature_names = (
            X.columns.tolist()
            if isinstance(X, pd.DataFrame)
            else [f"feature_{i}" for i in range(X.shape[1])]
        )
        self.model.fit(X, y)
        return self

    def evaluate_cv(self, X, y, cv=5):
        cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "f1", "precision", "recall", "roc_auc"],
        )
        return {
            metric.replace("test_", ""): (np.mean(scores), np.std(scores))
            for metric, scores in cv_results.items()
        }

    def evaluate(self, X, y):
        return evaluate_model(y, self.predict(X), self.predict_proba(X))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ModelExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def get_shap_values(self, X):
        shap_values = self.explainer(X)
        squared_shap = np.square(shap_values.values)
        mean_shap_values = squared_shap.mean(axis=0)
        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, mean_shap_values)
        }

    def explain(self, X, output_dir=None):
        importance_dict = {
            "feature_importances": self.get_shap_values(X),
            "metadata": {
                "model_type": self.model.__class__.__name__,
                "n_features": len(self.feature_names),
                "feature_names": self.feature_names,
            },
        }
        if output_dir:
            return save_json_results(importance_dict, output_dir, "feature_importances")
        return importance_dict


def train_and_evaluate_model(
    X, y, test_size=0.2, random_state=42, cv_folds=5, output_dir=None
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = ModelTrainer().train(X_train, y_train)
    cv_metrics = model.evaluate_cv(X_train, y_train, cv=cv_folds)
    test_metrics = model.evaluate(X_test, y_test)
    explainer = ModelExplainer(model.model, model.feature_names)
    feature_importance_path = explainer.explain(X_train, output_dir)
    return {
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "feature_importance_path": feature_importance_path,
    }, model


def analyse_conditional_importances(
    X, rule_analysis, trained_model, output_dir="results/feature_importances"
):
    conditional_importances = {}
    explainer = ModelExplainer(trained_model.model, trained_model.feature_names)
    for ruleset_key, ruleset_data in rule_analysis.items():
        for rule_idx, rule_object in enumerate(ruleset_data["rules"]):
            mask = pd.Series(True, index=X.index)
            for condition in rule_object.conds:
                feature_idx = condition.feature
                feature_name = X.columns[feature_idx]
                operator = determine_operator(str(condition))
                value = convert_to_serialisable(condition.val)
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
            importance_dict = explainer.explain(X_subset)
            rule_key = f"Ruleset {ruleset_key} - Rule {rule_idx}"
            conditional_importances[rule_key] = {
                "feature_importances": importance_dict["feature_importances"],
                "support": int(mask.sum()),
                "support_percentage": float(mask.sum() / len(X) * 100),
            }
    results = {
        "conditional_importances": conditional_importances,
        "metadata": {
            "n_rules_analysed": len(conditional_importances),
            "total_samples": len(X),
        },
    }
    output_path = save_json_results(results, output_dir, "conditional_importances")
    return {
        "conditional_importances": conditional_importances,
        "output_path": output_path,
    }
