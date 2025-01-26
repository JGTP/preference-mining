from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import numpy as np
import pandas as pd
from itertools import combinations
import shap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from src.utils import save_json_results, convert_to_serialisable
import tempfile


@dataclass
class FeatureSetResult:
    set1: Set[str]
    set2: Set[str]
    set1_importance: float
    set2_importance: float
    importance_difference: float
    max_correlation_set1: float
    max_correlation_set2: float


class EnhancedFeatureAnalyser:
    def __init__(
        self,
        model,
        X: pd.DataFrame,
        epsilons: List[float] = None,
        deltas: List[float] = None,
        max_set_size: int = 10,
        progress_logger: Optional[object] = None,
        temp_dir: Optional[str] = None,
    ):
        self.model = model
        self.X = X
        self.epsilons = epsilons or [i / 20 for i in range(1, 11)]
        self.deltas = deltas or [0.05, 0.10, 0.15, 0.20, 0.25]
        self.max_set_size = max_set_size
        self.feature_names = X.columns.tolist()
        self.correlation_matrix = X.corr()
        self.progress_logger = progress_logger
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._calculate_and_store_shap()
            self._store_correlations()
        except Exception as e:
            self.cleanup()
            raise e

    def cleanup(self):
        """Explicitly cleanup temporary files when analysis is complete"""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            for file in self.temp_dir.glob("*.json"):
                try:
                    file.unlink()
                except:
                    pass
            try:
                self.temp_dir.rmdir()
            except:
                pass

    def _calculate_and_store_shap(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X)
        squared_shap = np.square(shap_values.values)
        mean_shap_values = squared_shap.mean(axis=0)
        shap_dict = {
            name: float(importance)
            for name, importance in zip(self.feature_names, mean_shap_values)
        }
        with open(self.temp_dir / "shap_values.json", "w") as f:
            json.dump(shap_dict, f)

    def _store_correlations(self):
        corr_dict = {}
        for size in range(2, self.max_set_size + 1):
            for feature_set in combinations(self.feature_names, size):
                max_corr = max(
                    abs(self.correlation_matrix.loc[f1, f2])
                    for f1, f2 in combinations(feature_set, 2)
                )
                corr_dict[str(sorted(list(feature_set)))] = float(max_corr)

        with open(self.temp_dir / "correlations.json", "w") as f:
            json.dump(corr_dict, f)

    def _get_shap_values(self):
        with open(self.temp_dir / "shap_values.json", "r") as f:
            return json.load(f)

    def _get_correlation(self, feature_set):
        if len(feature_set) < 2:
            return 0.0
        with open(self.temp_dir / "correlations.json", "r") as f:
            correlations = json.load(f)
            return correlations.get(str(sorted(list(feature_set))), 0.0)

    def _check_correlation_threshold(
        self, feature_set: Set[str], epsilon: float
    ) -> bool:
        if len(feature_set) < 2:
            return True

        try:
            with open(self.temp_dir / "correlations.json", "r") as f:
                correlations = json.load(f)
                key = str(sorted(list(feature_set)))
                return correlations.get(key, 0.0) <= epsilon
        except:
            return True  # Fallback if file read fails

    def _aggregate_importance(
        self, feature_set: Set[str], importances: Dict[str, float]
    ) -> float:
        return sum(importances.get(feature, 0) for feature in feature_set)

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
        return self.X[mask]

    def analyse_rule(self, rule) -> Dict:
        conditional_data = self._get_conditional_data(rule)
        if len(conditional_data) < 10:
            return {}

        shap_values = self._get_shap_values()
        results = {}

        batch_size = 1000
        feature_combinations = []
        for size in range(1, self.max_set_size + 1):
            feature_combinations.extend(combinations(self.feature_names, size))

        for epsilon in self.epsilons:
            for delta in self.deltas:
                valid_pairs = []

                for i in range(0, len(feature_combinations), batch_size):
                    batch = feature_combinations[i : i + batch_size]
                    batch_pairs = combinations(batch, 2)

                    for combo1, combo2 in batch_pairs:
                        set1, set2 = set(combo1), set(combo2)
                        if not set1.intersection(set2):
                            imp1 = self._aggregate_importance(set1, shap_values)
                            imp2 = self._aggregate_importance(set2, shap_values)

                            if imp2 > 0 and (imp1 - imp2) / imp2 > delta:
                                if self._check_correlation_threshold(set1, epsilon):
                                    valid_pairs.append(
                                        FeatureSetResult(
                                            set1=set1,
                                            set2=set2,
                                            set1_importance=imp1,
                                            set2_importance=imp2,
                                            importance_difference=imp1 - imp2,
                                            max_correlation_set1=self._get_correlation(
                                                set1
                                            ),
                                            max_correlation_set2=self._get_correlation(
                                                set2
                                            ),
                                        )
                                    )

                if valid_pairs:
                    key = f"epsilon_{epsilon}_delta_{delta:.1f}%"
                    results[key] = [
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

                    with open(
                        self.temp_dir / f"results_e{epsilon}_d{delta}.json", "w"
                    ) as f:
                        json.dump(results[key], f)

        return results

    def analyse_ruleset(self, ruleset, output_dir=None) -> Dict:
        results = {
            "rule_analyses": {},
            "metadata": {
                "epsilons": self.epsilons,
                "deltas": self.deltas,
                "max_set_size": self.max_set_size,
                "total_features": len(self.feature_names),
            },
        }

        with ThreadPoolExecutor() as executor:
            rule_futures = {
                executor.submit(self.analyse_rule, rule): i
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
