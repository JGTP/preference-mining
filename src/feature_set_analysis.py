from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any
import numpy as np
import pandas as pd
from itertools import combinations
import shap
import tempfile
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from src.utils import save_json_results, convert_to_serialisable
from src.ripper import determine_operator


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
        epsilons: Optional[List[float]] = None,
        deltas: Optional[List[float]] = None,
        max_set_size: int = 10,
        top_features: int = 20,
        enable_disk_cache: bool = False,
        temp_dir: Optional[str] = None,
        progress_logger: Optional[Any] = None,
    ):
        """
        Initialize the feature analyser with optional disk caching.

        Args:
            model: Trained model to analyze
            X: Input features DataFrame
            epsilons: List of correlation thresholds
            deltas: List of importance difference thresholds
            max_set_size: Maximum size of feature sets to consider
            top_features: Number of top features to consider for set1
            enable_disk_cache: If True, cache intermediate results to disk
            temp_dir: Directory for cached files (only used if enable_disk_cache=True)
            progress_logger: Optional progress logger
        """
        self.model = model
        self.X = X
        self.epsilons = epsilons or [i / 20 for i in range(1, 11)]
        self.deltas = deltas or [0.05, 0.10, 0.15, 0.20, 0.25]
        self.max_set_size = max_set_size
        self.top_features = top_features
        self.feature_names = X.columns.tolist()
        self.correlation_matrix = X.corr()
        self.enable_disk_cache = enable_disk_cache
        self.progress_logger = progress_logger

        logging.info(
            f"Initializing feature analysis with {len(self.feature_names)} features"
        )
        logging.info(
            f"Using {len(self.epsilons)} epsilon values and {len(self.deltas)} delta values"
        )

        # Initialize disk cache if enabled
        if enable_disk_cache:
            self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_dir = None

        # Initialize progress tracking
        if self.progress_logger:
            self.progress_logger.create_progress_bar(
                "init_calculations",
                3,  # Three main initialization steps
                "Initializing feature calculations",
            )

        try:
            # Pre-compute and cache in memory
            self.shap_values = self._calculate_shap_values()
            if self.progress_logger:
                self.progress_logger.update_progress("init_calculations", 1)

            self.correlations = self._calculate_correlations()
            if self.progress_logger:
                self.progress_logger.update_progress("init_calculations", 1)

            self.feature_combinations = self._precompute_combinations()
            if self.progress_logger:
                self.progress_logger.update_progress("init_calculations", 1)

            # Optionally cache to disk
            if self.enable_disk_cache:
                self._cache_to_disk()

        except Exception as e:
            self.cleanup()
            raise e

    def _calculate_shap_values(self) -> Dict[str, float]:
        """Calculate and return SHAP values"""
        logging.info("Calculating SHAP values")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X)
        squared_shap = np.square(shap_values.values)
        mean_shap_values = squared_shap.mean(axis=0)

        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, mean_shap_values)
        }

    def _calculate_correlations(self) -> Dict[str, float]:
        """Calculate and return correlations between feature sets"""
        logging.info("Calculating feature correlations")
        corr_dict = {}

        if self.progress_logger:
            total_combinations = sum(
                len(list(combinations(self.feature_names, size)))
                for size in range(2, self.max_set_size + 1)
            )
            self.progress_logger.create_progress_bar(
                "correlation_calc", total_combinations, "Calculating correlations"
            )

        for size in range(2, self.max_set_size + 1):
            for feature_set in combinations(self.feature_names, size):
                correlations = [
                    abs(self.correlation_matrix.loc[f1, f2])
                    for f1, f2 in combinations(feature_set, 2)
                ]
                valid_correlations = [c for c in correlations if not np.isnan(c)]
                max_corr = max(valid_correlations) if valid_correlations else 0.0
                corr_dict[str(sorted(list(feature_set)))] = float(max_corr)

                if self.progress_logger:
                    self.progress_logger.update_progress("correlation_calc", 1)

        return corr_dict

    def _cache_to_disk(self) -> None:
        """Cache computed values to disk if enabled"""
        if not self.enable_disk_cache:
            return

        logging.info("Caching computed values to disk")
        try:
            with open(self.temp_dir / "shap_values.json", "w") as f:
                json.dump(self.shap_values, f)
            with open(self.temp_dir / "correlations.json", "w") as f:
                json.dump(self.correlations, f)
            logging.info("Successfully cached values to disk")
        except Exception as e:
            logging.warning(f"Failed to cache values to disk: {e}")

    def cleanup(self) -> None:
        """Clean up temporary files if disk cache was enabled"""
        if not self.enable_disk_cache or not self.temp_dir:
            return

        logging.info("Cleaning up temporary files")
        try:
            for file in self.temp_dir.glob("*.json"):
                try:
                    file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete {file}: {e}")
            try:
                self.temp_dir.rmdir()
            except Exception as e:
                logging.warning(f"Failed to remove temp directory: {e}")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def _precompute_combinations(self) -> Dict[str, List[Dict]]:
        """Pre-compute all feature combinations with their sets and metrics"""
        combinations_dict = {
            "set1": [],  # from top features
            "set2": [],  # from all features
        }

        # Get top feature names based on SHAP values
        top_features = sorted(
            self.shap_values.items(), key=lambda x: x[1], reverse=True
        )[: self.top_features]
        top_feature_names = [f[0] for f in top_features]

        if self.progress_logger:
            total_combinations = sum(
                len(list(combinations(names, size)))
                for size in range(1, self.max_set_size + 1)
                for names in [top_feature_names, self.feature_names]
            )
            self.progress_logger.create_progress_bar(
                "combinations_calc",
                total_combinations,
                "Computing feature combinations",
            )

        # Pre-compute all combinations and their metrics
        for size in range(1, self.max_set_size + 1):
            # Set 1 combinations (from top features)
            for combo in combinations(top_feature_names, size):
                feature_set = frozenset(combo)
                importance = sum(self.shap_values[f] for f in feature_set)
                correlation = (
                    max(self.correlations.get(str(sorted(list(combo))), 0.0), 0.0)
                    if size > 1
                    else 0.0
                )
                combinations_dict["set1"].append(
                    {
                        "features": feature_set,
                        "importance": importance,
                        "correlation": correlation,
                    }
                )
                if self.progress_logger:
                    self.progress_logger.update_progress("combinations_calc", 1)

            # Set 2 combinations (from all features)
            for combo in combinations(self.feature_names, size):
                feature_set = frozenset(combo)
                importance = sum(self.shap_values[f] for f in feature_set)
                correlation = (
                    max(self.correlations.get(str(sorted(list(combo))), 0.0), 0.0)
                    if size > 1
                    else 0.0
                )
                combinations_dict["set2"].append(
                    {
                        "features": feature_set,
                        "importance": importance,
                        "correlation": correlation,
                    }
                )
                if self.progress_logger:
                    self.progress_logger.update_progress("combinations_calc", 1)

        return combinations_dict

    def analyse_rule(self, rule) -> Dict:
        """Analyze a single rule to find its specific preference relations."""
        results = {}
        rule_str = str(rule)

        # Extract rule conditions
        conditions = []
        for cond in rule.conds:
            feature_idx = cond.feature
            feature_name = self.feature_names[feature_idx]
            operator = determine_operator(str(cond))
            value = convert_to_serialisable(cond.val)
            conditions.append(
                {"feature": feature_name, "operator": operator, "value": value}
            )

        for epsilon in self.epsilons:
            for delta in self.deltas:
                valid_pairs = []
                logging.info(f"Analyzing rule with epsilon={epsilon}, delta={delta}")

                # Get features mentioned in this rule's conditions
                rule_features = {cond["feature"] for cond in conditions}

                # Filter set1 combinations based on correlation threshold
                # AND require at least one feature from the rule conditions
                valid_set1 = [
                    combo
                    for combo in self.feature_combinations["set1"]
                    if (
                        combo["correlation"] <= epsilon
                        and rule_features.intersection(combo["features"])
                    )
                ]

                # Find valid pairs
                for combo1 in valid_set1:
                    for combo2 in self.feature_combinations["set2"]:
                        if not combo1["features"].intersection(combo2["features"]):
                            imp1 = combo1["importance"]
                            imp2 = combo2["importance"]

                            if imp2 > 0 and (imp1 - imp2) / imp2 > delta:
                                valid_pairs.append(
                                    FeatureSetResult(
                                        set1=combo1["features"],
                                        set2=combo2["features"],
                                        set1_importance=imp1,
                                        set2_importance=imp2,
                                        importance_difference=imp1 - imp2,
                                        max_correlation_set1=combo1["correlation"],
                                        max_correlation_set2=combo2["correlation"],
                                    )
                                )

                if valid_pairs:
                    key = f"epsilon_{epsilon}_delta_{delta:.3f}%"
                    results[key] = {
                        "relations": [
                            self._format_result(pair) for pair in valid_pairs
                        ],
                        "conditions": conditions,
                    }

                if self.progress_logger:
                    self.progress_logger.update_progress(f"rule_analysis_{id(rule)}", 1)

        return results

    def analyse_ruleset(self, ruleset: List, output_dir: Optional[str] = None) -> Dict:
        """Analyze a full ruleset using parallel processing"""
        if self.progress_logger:
            self.progress_logger.create_progress_bar(
                "ruleset_analysis", len(ruleset), "Analyzing ruleset"
            )

        results = {
            "rule_analyses": {},
            "metadata": {
                "epsilons": self.epsilons,
                "deltas": self.deltas,
                "max_set_size": self.max_set_size,
                "total_features": len(self.feature_names),
            },
        }

        logging.info(f"Starting analysis of {len(ruleset)} rules")
        with ThreadPoolExecutor() as executor:
            rule_futures = {
                executor.submit(self.analyse_rule, rule): i
                for i, rule in enumerate(ruleset)
            }

            for future in rule_futures:
                try:
                    rule_results = future.result()
                    rule_idx = rule_futures[future]
                    if rule_results:
                        results["rule_analyses"][f"rule_{rule_idx}"] = {
                            "rule_string": str(ruleset[rule_idx]),
                            "analysis": rule_results,
                        }
                    if self.progress_logger:
                        self.progress_logger.update_progress("ruleset_analysis", 1)

                except Exception as e:
                    logging.error(f"Error analyzing rule: {e}")

        if output_dir:
            output_path = save_json_results(results, output_dir, "feature_analysis")
            results["output_path"] = output_path

        return results

    def _format_result(self, pair: FeatureSetResult) -> Dict:
        """Format a FeatureSetResult for output"""
        return {
            "set1": list(pair.set1),
            "set2": list(pair.set2),
            "set1_importance": float(pair.set1_importance),
            "set2_importance": float(pair.set2_importance),
            "importance_difference": float(pair.importance_difference),
            "max_correlation_set1": float(pair.max_correlation_set1),
            "max_correlation_set2": float(pair.max_correlation_set2),
        }
