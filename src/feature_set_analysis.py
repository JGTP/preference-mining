from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import numpy as np
import pandas as pd
from itertools import combinations
import shap
from concurrent.futures import ThreadPoolExecutor
from src.utils import save_json_results


@dataclass
class FeatureSetResult:
    set1: Set[str]
    set2: Set[str]
    set1_importance: float
    set2_importance: float
    importance_difference: float
    max_correlation_set1: float
    max_correlation_set2: float


class EnhancedFeatureAnalyzer:
    def __init__(
        self,
        model,
        X: pd.DataFrame,
        epsilons: List[float] = None,
        deltas: List[float] = None,
        max_set_size: int = 10,
        progress_logger: Optional[object] = None,
    ):
        self.model = model
        self.X = X
        self.epsilons = epsilons or [i / 20 for i in range(1, 11)]
        self.deltas = deltas or [0.05, 0.10, 0.15, 0.20, 0.25]
        self.max_set_size = max_set_size
        self.feature_names = X.columns.tolist()
        self.correlation_matrix = X.corr()
        self.progress_logger = progress_logger
        self._shap_cache = {}

        # Precompute feature combinations
        self.feature_combinations = {}
        all_features = set(self.feature_names)
        for size in range(1, self.max_set_size + 1):
            self.feature_combinations[size] = [
                set(combo) for combo in combinations(all_features, size)
            ]

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X)
        self.shap_scale = np.abs(shap_values.values).mean()
        self.deltas = [p * self.shap_scale for p in self.deltas]

    def _calculate_shap_values(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate SHAP values with caching based on data shape and content hash."""
        # Create a cache key based on data shape and a simple hash of values
        cache_key = (data.shape[0], hash(str(data.values.tobytes())))

        if cache_key in self._shap_cache:
            return self._shap_cache[cache_key]

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(data)
        mean_shap = np.abs(shap_values.values).mean(axis=0)
        result = dict(zip(self.feature_names, mean_shap))

        self._shap_cache[cache_key] = result
        return result

    def _check_correlation_threshold(
        self, feature_set: Set[str], epsilon: float
    ) -> bool:
        if len(feature_set) < 2:
            return True
        feature_list = list(feature_set)
        corr_subset = self.correlation_matrix.loc[feature_list, feature_list]
        # Create upper triangular mask to avoid duplicate checks
        mask = np.triu(np.ones_like(corr_subset), k=1)
        return np.all(np.abs(corr_subset.values[mask.astype(bool)]) <= epsilon)

    def _aggregate_importance(
        self, feature_set: Set[str], importances: Dict[str, float]
    ) -> float:
        return sum(importances.get(feature, 0) for feature in feature_set)

    def analyze_rule(self, rule) -> Dict:
        conditional_data = self._get_conditional_data(rule)
        if len(conditional_data) < 10:
            return {}

        shap_values = self._calculate_shap_values(conditional_data)
        all_features = set(self.feature_names)
        results = {}

        for epsilon in self.epsilons:
            for delta in self.deltas:
                valid_pairs = []

                for size in range(1, self.max_set_size + 1):
                    feature_sets = [
                        combo_set
                        for combo_set in self.feature_combinations[size]
                        if self._check_correlation_threshold(combo_set, epsilon)
                    ]

                    for set1, set2 in combinations(feature_sets, 2):
                        if not set1.intersection(set2):
                            imp1 = self._aggregate_importance(set1, shap_values)
                            imp2 = self._aggregate_importance(set2, shap_values)
                            if imp1 - imp2 > delta:
                                valid_pairs.append(
                                    FeatureSetResult(
                                        set1=set1,
                                        set2=set2,
                                        set1_importance=imp1,
                                        set2_importance=imp2,
                                        importance_difference=imp1 - imp2,
                                        max_correlation_set1=(
                                            max(
                                                abs(self.correlation_matrix.loc[f1, f2])
                                                for f1, f2 in combinations(set1, 2)
                                            )
                                            if len(set1) > 1
                                            else 0
                                        ),
                                        max_correlation_set2=(
                                            max(
                                                abs(self.correlation_matrix.loc[f1, f2])
                                                for f1, f2 in combinations(set2, 2)
                                            )
                                            if len(set2) > 1
                                            else 0
                                        ),
                                    )
                                )

                if valid_pairs:
                    results[f"epsilon_{epsilon}_delta_{delta:.1f}%"] = [
                        {
                            "set1": list(pair.set1),
                            "set2": list(pair.set2),
                            "set1_importance": float(pair.set1_importance),
                            "set2_importance": float(pair.set2_importance),
                            "importance_difference": float(pair.importance_difference),
                            "max_correlation_set1": float(pair.max_correlation_set1),
                            "max_correlation_set2": float(pair.max_correlation_set2),
                        }
                        for pair in valid_pairs
                    ]

        return results

    def _get_conditional_data(self, rule) -> pd.DataFrame:
        mask = pd.Series(True, index=self.X.index)
        for condition in rule.conds:
            feature_idx = condition.feature
            feature_name = self.feature_names[feature_idx]
            cond_str = str(condition)

            if "-" in cond_str:
                try:
                    value_str = cond_str.split("=")[1]
                    lower, upper = map(float, value_str.split("-"))
                    mask &= (self.X[feature_name] >= lower) & (
                        self.X[feature_name] <= upper
                    )
                    continue
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse range condition: {cond_str}"
                    ) from e

            for op_str, operator in [
                ("=<", "<="),
                ("=>", ">="),
                ("=", "=="),
                ("<", "<"),
                (">", ">"),
            ]:
                if op_str in cond_str:
                    try:
                        value = float(cond_str.split(op_str)[1])
                        mask &= eval(f"self.X[feature_name] {operator} value")
                        break
                    except Exception as e:
                        raise ValueError(
                            f"Failed to parse condition: {cond_str}"
                        ) from e
            else:
                raise ValueError(f"Unrecognized condition format: {cond_str}")

        return self.X[mask]

    def analyze_ruleset(self, ruleset, output_dir=None) -> Dict:
        results = {
            "rule_analyses": {},
            "metadata": {
                "epsilons": self.epsilons,
                "deltas": self.deltas,
                "shap_scale": float(self.shap_scale),
                "max_set_size": self.max_set_size,
                "total_features": len(self.feature_names),
            },
        }

        with ThreadPoolExecutor() as executor:
            rule_futures = {
                executor.submit(self.analyze_rule, rule): i
                for i, rule in enumerate(ruleset)
            }

            for future in rule_futures:
                rule_results = future.result()
                rule_idx = rule_futures[future]

                if rule_results:
                    results["rule_analyses"][f"rule_{rule_idx}"] = {
                        "rule_string": str(ruleset[rule_idx]),
                        "analysis": rule_results,
                    }

                if self.progress_logger:
                    try:
                        self.progress_logger.update_progress("analysis", 1)
                    except Exception:
                        pass

        if output_dir:
            output_path = save_json_results(results, output_dir, "feature_analysis")
            results["output_path"] = output_path

        return results
