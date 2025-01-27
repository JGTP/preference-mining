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
import logging

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
        top_features: int = 20,
        progress_logger: Optional[object] = None,
        temp_dir: Optional[str] = None,
    ):
        self.model = model
        self.X = X
        self.epsilons = epsilons or [i / 20 for i in range(1, 11)]
        self.deltas = deltas or [0.05, 0.10, 0.15, 0.20, 0.25]
        self.max_set_size = max_set_size
        self.top_features = top_features
        self.feature_names = X.columns.tolist()
        self.correlation_matrix = X.corr()
        self.progress_logger = progress_logger
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Log initialisation
        logging.info(f"Initialising feature analysis with {len(self.feature_names)} features")
        logging.info(f"Using {len(self.epsilons)} epsilon values and {len(self.deltas)} delta values")
        
        if self.progress_logger:
            self.progress_logger.create_progress_bar(
                "init_calculations", 
                2,  # Two main initialisation steps
                "Initialising feature calculations"
            )
        
        try:
            self._calculate_and_store_shap()
            if self.progress_logger:
                self.progress_logger.update_progress("init_calculations", 1)
            
            self._store_correlations()
            if self.progress_logger:
                self.progress_logger.update_progress("init_calculations", 1)
                
        except Exception as e:
            self.cleanup()
            raise e

    def cleanup(self):
        """Explicitly cleanup temporary files when analysis is complete"""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            logging.info("Cleaning up temporary files")
            for file in self.temp_dir.glob("*.json"):
                try:
                    file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete {file}: {e}")
            try:
                self.temp_dir.rmdir()
            except Exception as e:
                logging.warning(f"Failed to remove temp directory: {e}")

    def _calculate_and_store_shap(self):
        logging.info("Calculating SHAP values")
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
        logging.info("SHAP values calculated and stored")

    def _store_correlations(self):
        logging.info("Calculating feature correlations")
        if self.progress_logger:
            total_combinations = sum(
                len(list(combinations(self.feature_names, size)))
                for size in range(2, self.max_set_size + 1)
            )
            self.progress_logger.create_progress_bar(
                "correlation_calc",
                total_combinations,
                "Calculating correlations"
            )

        corr_dict = {}
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

        with open(self.temp_dir / "correlations.json", "w") as f:
            json.dump(corr_dict, f)
        logging.info("Correlations calculated and stored")

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
        except Exception as e:
            logging.warning(f"Error checking correlation threshold: {e}")
            return True

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

    def analyse_rule(self, rule):
        if self.progress_logger:
            # Create progress bars for this rule's analysis stages
            self.progress_logger.create_progress_bar(
                f"rule_setup_{id(rule)}", 
                3,  # Three main setup steps
                f"Setting up analysis for rule {str(rule)[:30]}..."
            )
            
        # Step 1: Get conditional data
        conditional_data = self._get_conditional_data(rule)
        if self.progress_logger:
            self.progress_logger.update_progress(f"rule_setup_{id(rule)}", 1)
            
        if len(conditional_data) < 10:
            logging.info(f"Skipping rule analysis - insufficient samples ({len(conditional_data)} < 10)")
            return {}

        # Step 2: Get SHAP values and prepare feature combinations
        shap_values = self._get_shap_values()
        if self.progress_logger:
            self.progress_logger.update_progress(f"rule_setup_{id(rule)}", 1)

        # Sort features and prepare combinations
        top_features = sorted(
            shap_values.items(), key=lambda x: x[1], reverse=True
        )[:self.top_features]
        top_feature_names = [f[0] for f in top_features]

        feature_combinations_set1 = []
        feature_combinations_set2 = []
        
        for size in range(1, self.max_set_size + 1):
            feature_combinations_set1.extend(combinations(top_feature_names, size))
            feature_combinations_set2.extend(combinations(self.feature_names, size))
            
        if self.progress_logger:
            self.progress_logger.update_progress(f"rule_setup_{id(rule)}", 1)
            
            # Create progress bar for epsilon-delta combinations
            total_ed_combinations = len(self.epsilons) * len(self.deltas)
            self.progress_logger.create_progress_bar(
                f"rule_combinations_{id(rule)}", 
                total_ed_combinations,
                f"Processing combinations for rule {str(rule)[:30]}..."
            )

        results = {}
        processed_combinations = 0
        total_ed_combinations = len(self.epsilons) * len(self.deltas)

        for epsilon in self.epsilons:
            for delta in self.deltas:
                valid_pairs = []
                logging.info(f"Analysing with epsilon={epsilon}, delta={delta}")

                for combo1 in feature_combinations_set1:
                    set1 = set(combo1)
                    if not self._check_correlation_threshold(set1, epsilon):
                        continue

                    for combo2 in feature_combinations_set2:
                        set2 = set(combo2)
                        if not set1.intersection(set2):
                            imp1 = self._aggregate_importance(set1, shap_values)
                            imp2 = self._aggregate_importance(set2, shap_values)

                            if imp2 > 0 and (imp1 - imp2) / imp2 > delta:
                                valid_pairs.append(
                                    FeatureSetResult(
                                        set1=set1,
                                        set2=set2,
                                        set1_importance=imp1,
                                        set2_importance=imp2,
                                        importance_difference=imp1 - imp2,
                                        max_correlation_set1=self._get_correlation(set1),
                                        max_correlation_set2=self._get_correlation(set2),
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
                
                if self.progress_logger:
                    self.progress_logger.update_progress(f"rule_combinations_{id(rule)}", 1)
                    
                processed_combinations += 1
                logging.info(f"Processed {processed_combinations}/{total_ed_combinations} combinations")

        logging.info(f"Completed analysis for rule: {str(rule)[:50]}...")
        return results

    def analyse_ruleset(self, ruleset, output_dir=None) -> Dict:
        # Initialize progress tracking
        if self.progress_logger:
            # Calculate total number of operations for setup
            total_setup_steps = len(ruleset)  # One step per rule for initial setup
            self.progress_logger.create_progress_bar(
                "feature_analysis_setup", 
                total_setup_steps, 
                "Setting up feature analysis"
            )
            
            # Calculate total combinations for detailed progress
            total_combinations = 0
            for epsilon in self.epsilons:
                for delta in self.deltas:
                    total_combinations += len(ruleset)
            
            self.progress_logger.create_progress_bar(
                "feature_analysis_combinations", 
                total_combinations,
                "Analysing feature combinations"
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
                        logging.info(f"Completed analysis for rule {rule_idx}")
                except Exception as e:
                    logging.error(f"Error analysing rule: {e}")

        if output_dir:
            output_path = save_json_results(results, output_dir, "feature_analysis")
            results["output_path"] = output_path
            logging.info(f"Results saved to {output_path}")

        return results