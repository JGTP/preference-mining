import datetime
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import json


@dataclass
class FeatureSetComparison:
    better_set: Set[str]
    worse_set: Set[str]
    better_importance: float
    worse_importance: float
    importance_difference: float


class FeatureSetAnalyzer:
    def __init__(self, min_set_size: int = 1, max_set_size: Optional[int] = 5):
        """
        Initialize the analyzer with constraints on feature set sizes.

        Args:
            min_set_size: Minimum number of features in a set (default: 1)
            max_set_size: Maximum number of features in a set (optional)
        """
        self.min_set_size = min_set_size
        self.max_set_size = max_set_size

    def _normalize_importances(self, importances: Dict[str, float]) -> Dict[str, float]:
        """Normalize SHAP values to sum to 1."""
        total = sum(importances.values())
        return (
            {k: v / total for k, v in importances.items()} if total > 0 else importances
        )

    def _calculate_set_importance(
        self, feature_set: Set[str], normalized_importances: Dict[str, float]
    ) -> float:
        """Calculate the aggregated importance for a set of features."""
        return sum(normalized_importances.get(feature, 0) for feature in feature_set)

    def _generate_valid_sets(self, features: List[str]) -> List[Set[str]]:
        """
        Generate valid feature sets efficiently using itertools.combinations.
        Only generates sets within the specified size constraints.
        """
        max_size = self.max_set_size or len(features)
        valid_sets = []

        for size in range(self.min_set_size, max_size + 1):
            valid_sets.extend(set(combo) for combo in combinations(features, size))

            # Early stopping if we have too many sets
            if len(valid_sets) > 10000:
                logging.warning(
                    "Large number of feature sets generated. Consider reducing max_set_size."
                )
                break

        return valid_sets

    def find_significant_pairs(
        self, importances: Dict[str, float], delta: float, max_pairs: int = None
    ) -> List[FeatureSetComparison]:
        """
        Find pairs of feature sets where one significantly outperforms the other.

        Args:
            importances: Dictionary of feature importance scores
            delta: Minimum difference in importance required
            max_pairs: Maximum number of pairs to return (None means no limit)

        Returns:
            List of FeatureSetComparison objects
        """
        normalized_importances = self._normalize_importances(importances)
        features = list(normalized_importances.keys())

        # Generate all valid feature sets
        feature_sets = self._generate_valid_sets(features)

        # Sort sets by their importance for early stopping optimization
        sets_with_importance = [
            (s, self._calculate_set_importance(s, normalized_importances))
            for s in feature_sets
        ]
        sets_with_importance.sort(key=lambda x: x[1], reverse=True)

        significant_pairs = []

        # Compare sets efficiently by starting with highest importance sets
        for i, (set_b, importance_b) in enumerate(sets_with_importance):
            # Early stopping if we have enough pairs
            if max_pairs is not None and len(significant_pairs) >= max_pairs:
                break

            for set_w, importance_w in sets_with_importance[i + 1 :]:

                # Skip if sets overlap
                if set_b & set_w:
                    continue

                # Skip if difference isn't large enough
                difference = importance_b - importance_w
                if difference <= delta:
                    continue

                significant_pairs.append(
                    FeatureSetComparison(
                        better_set=set_b,
                        worse_set=set_w,
                        better_importance=importance_b,
                        worse_importance=importance_w,
                        importance_difference=difference,
                    )
                )

                if len(significant_pairs) >= max_pairs:
                    break
                if max_pairs is not None and len(significant_pairs) >= max_pairs:
                    break

        return significant_pairs


def analyze_feature_set_differences(
    conditional_results: Dict,
    deltas: List[float],
    output_dir: Path,
    min_set_size: int = 1,
    max_set_size: Optional[int] = 5,
    max_pairs_per_rule: int = None,
) -> Dict:
    """
    Analyze feature set differences across multiple rules and delta values.

    Args:
        conditional_results: Dictionary containing rule-wise feature importances
        deltas: List of delta values to test
        output_dir: Directory to save results
        min_set_size: Minimum size of feature sets
        max_set_size: Maximum size of feature sets
        max_pairs_per_rule: Maximum number of pairs to find per rule

    Returns:
        Dictionary containing analysis results and export paths
    """
    analyzer = FeatureSetAnalyzer(min_set_size=min_set_size, max_set_size=max_set_size)
    results = {}

    for rule_name, rule_data in conditional_results["conditional_importances"].items():
        importances = rule_data["feature_importances"]
        rule_results = {}

        for delta in deltas:
            significant_pairs = analyzer.find_significant_pairs(
                importances=importances, delta=delta, max_pairs=max_pairs_per_rule
            )

            # Convert to serializable format
            rule_results[str(delta)] = [
                {
                    "better_set": list(pair.better_set),
                    "worse_set": list(pair.worse_set),
                    "better_importance": pair.better_importance,
                    "worse_importance": pair.worse_importance,
                    "importance_difference": pair.importance_difference,
                }
                for pair in significant_pairs
            ]

        results[rule_name] = rule_results

    # Export results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"feature_set_differences_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "export_date": datetime.datetime.now().isoformat(),
                    "deltas_analyzed": deltas,
                    "min_set_size": min_set_size,
                    "max_set_size": max_set_size,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    return {"output_path": str(output_path), "results": results}
