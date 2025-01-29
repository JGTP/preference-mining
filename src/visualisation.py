import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Union
from collections import defaultdict


def calculate_max_relations(N: int, max_set_size: int, top_features: int) -> int:
    """Calculate maximum possible number of feature set relations.

    Args:
        N: Total number of features
        max_set_size: Maximum size of feature sets
        top_features: Number of top features to consider (will be capped at N)

    Returns:
        int: Maximum possible number of relations
    """
    # Validate and cap parameters
    top_features = min(top_features, N)
    max_set_size = min(max_set_size, N)

    total = 0
    for set1_size in range(1, min(max_set_size, top_features) + 1):
        # Calculate number of possible set1 combinations from top features
        set1_count = 1
        for i in range(set1_size):
            set1_count = set1_count * (top_features - i) // (i + 1)

        # Calculate remaining features after selecting set1
        remaining_features = (N - top_features) + (top_features - set1_size)
        set2_total = 0

        # Calculate possible set2 combinations from remaining features
        for i in range(1, max_set_size + 1):
            if i <= remaining_features:
                set2_count = 1
                for j in range(i):
                    set2_count = set2_count * (remaining_features - j) // (j + 1)
                set2_total += set2_count

        total += set1_count * set2_total

    return total


def process_results(
    results: Dict[str, Any],
    shap_values: Optional[Dict[str, float]] = None,
    correlations: Optional[Dict[str, float]] = None,
    temp_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Process feature analysis results into a DataFrame for visualization.

    Args:
        results: Dictionary containing feature analysis results
        shap_values: Dictionary of SHAP values for each feature
        correlations: Dictionary of feature correlations
        temp_dir: Optional directory containing cached values

    Returns:
        DataFrame containing processed results
    """
    if shap_values is None or correlations is None:
        if temp_dir is None:
            raise ValueError(
                "Either shap_values and correlations or temp_dir must be provided"
            )
        temp_dir = Path(temp_dir)
        try:
            with open(temp_dir / "shap_values.json", "r") as f:
                shap_values = json.load(f)
            with open(temp_dir / "correlations.json", "r") as f:
                correlations = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load cached values from temp_dir: {e}")

    metrics_by_params = defaultdict(
        lambda: {
            "all_relations": [],
            "unique_pairs": set(),
            "W": [],
            "B": [],
            "size_dist": defaultdict(int),
        }
    )

    for rule_id, rule_info in results["rule_analyses"].items():
        rule_string = rule_info["rule_string"]
        for param_key, analyses in rule_info["analysis"].items():
            epsilon = float(param_key.split("_")[1])
            delta = float(param_key.split("delta_")[1].rstrip("%"))

            if analyses:
                for analysis in analyses["relations"]:
                    set1_frozen = frozenset(analysis["set1"])
                    set2_frozen = frozenset(analysis["set2"])
                    feature_pair = (set1_frozen, set2_frozen)

                    metrics_by_params[(epsilon, delta)]["all_relations"].append(
                        feature_pair
                    )
                    metrics_by_params[(epsilon, delta)]["unique_pairs"].add(
                        feature_pair
                    )

                    w_size = len(analysis["set1"])
                    b_size = len(analysis["set2"])
                    metrics_by_params[(epsilon, delta)]["W"].append(w_size)
                    metrics_by_params[(epsilon, delta)]["B"].append(b_size)

                    size_key = f"({w_size},{b_size})"
                    metrics_by_params[(epsilon, delta)]["size_dist"][size_key] += 1

    rows = []
    for (epsilon, delta), metrics in metrics_by_params.items():
        total_relations = len(metrics["all_relations"])
        unique_relations = len(metrics["unique_pairs"])

        size_dist = metrics["size_dist"]
        if total_relations > 0:
            size_dist_pct = {
                size: (count / total_relations * 100)
                for size, count in size_dist.items()
            }
        else:
            size_dist_pct = {}

        row = {
            "epsilon": epsilon,
            "delta": delta,
            "N_total": total_relations,
            "N_unique": unique_relations,
            "W": np.mean(metrics["W"]) if metrics["W"] else 0,
            "B": np.mean(metrics["B"]) if metrics["B"] else 0,
            "size_distribution": size_dist_pct,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def create_dimension_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    max_set_size: int,
    top_features: int,
    test_size: Optional[int] = None,
    n_splits: int = 3,
) -> None:
    """Create dimension distribution heatmap plots.

    Args:
        df: DataFrame containing processed results
        output_dir: Directory to save plots
        max_set_size: Maximum size of feature sets
        top_features: Number of top features considered
        test_size: Optional test set size
        n_splits: Number of cross-validation splits
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilons = sorted(df["epsilon"].unique())
    n_plots = len(epsilons)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False
    )

    size_combinations = [
        f"({i},{j})"
        for i in range(1, max_set_size + 1)
        for j in range(1, max_set_size + 1)
    ]

    for idx, epsilon in enumerate(epsilons):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        epsilon_data = df[df["epsilon"] == epsilon].sort_values("delta")
        deltas = epsilon_data["delta"].values

        heatmap_data = np.zeros((len(size_combinations), len(deltas)))

        for i, delta in enumerate(deltas):
            delta_row = epsilon_data[epsilon_data["delta"] == delta]
            if not delta_row.empty:
                size_dist = delta_row["size_distribution"].iloc[0]
                for j, size_combo in enumerate(size_combinations):
                    heatmap_data[j, i] = size_dist.get(size_combo, 0)

        im = ax.imshow(heatmap_data, aspect="auto", cmap="Blues")

        ax.set_xticks(range(len(deltas)))
        ax.set_xticklabels([f"{d:.2f}" for d in deltas], rotation=45)
        ax.set_yticks(range(len(size_combinations)))
        ax.set_yticklabels(size_combinations)

        ax.set_xlabel("δ")
        ax.set_ylabel("Set Sizes (|Dw|, |Db|)")
        ax.set_title(f"ε = {epsilon:.2f}")

        plt.colorbar(im, ax=ax, label="Percentage of Relations")

    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].remove()

    fig.tight_layout()

    params = [
        f"max{max_set_size}",
        f"top{top_features}",
        f"splits{n_splits}",
    ]
    if test_size is not None:
        params.append(f"test{test_size}")

    filename = f"dimension_distribution_{'_'.join(params)}.pdf"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_plots(
    df: pd.DataFrame,
    output_dir: Path,
    n_rules: int,
    test_size: Optional[int] = None,
    max_set_size: int = 10,
    top_features: int = 20,
    n_splits: int = 3,
    total_features: int = None,
) -> None:
    """Create all visualization plots.

    Args:
        df: DataFrame containing processed results
        output_dir: Directory to save plots
        n_rules: Number of rules analyzed
        test_size: Optional test set size
        max_set_size: Maximum size of feature sets
        top_features: Number of top features considered
        n_splits: Number of cross-validation splits
        total_features: Total number of features in dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if total_features is not None:
        max_relations = calculate_max_relations(
            total_features, max_set_size, top_features
        )
    else:
        max_relations = "Unknown"

    epsilons = sorted(df["epsilon"].unique())
    n_plots = len(epsilons)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create relations plot
    fig_relations, axes_relations = plt.subplots(
        n_rows + 1,
        n_cols,
        figsize=(5 * n_cols, 4 * (n_rows + 1)),
        gridspec_kw={"height_ratios": [0.2] + [1] * n_rows},
        squeeze=False,
    )

    summary_text = (
        f"Total Rules: {n_rules}\nMaximum Possible Unique Relations: {max_relations}"
    )
    axes_relations[0, 0].text(
        0.5,
        0.5,
        summary_text,
        ha="center",
        va="center",
        transform=axes_relations[0, 0].transAxes,
    )
    axes_relations[0, 0].axis("off")

    for j in range(1, n_cols):
        axes_relations[0, j].axis("off")

    # Create dimensions plot
    fig_dimensions, axes_dimensions = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    for idx, epsilon in enumerate(epsilons):
        row = (idx // n_cols) + 1
        col = idx % n_cols

        epsilon_data = df[df["epsilon"] == epsilon].sort_values("delta")

        # Plot relations
        ax1 = axes_relations[row, col]
        ax2 = ax1.twinx()

        l1 = ax1.plot(
            epsilon_data["delta"],
            epsilon_data["N_total"],
            "-o",
            color="#1f77b4",
            label="Total Relations",
            markersize=6,
            markerfacecolor="white",
        )

        l2 = ax2.plot(
            epsilon_data["delta"],
            epsilon_data["N_unique"],
            "--s",
            color="#ff7f0e",
            label="Unique Relations",
            markersize=6,
            markerfacecolor="white",
        )

        ax1.set_xlabel("δ")
        ax1.set_ylabel("Total Relations")
        ax2.set_ylabel("Unique Relations")
        ax1.set_title(f"ε = {epsilon:.2f}")
        ax1.grid(True, alpha=0.3)

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

        # Plot dimensions
        ax = axes_dimensions[idx // n_cols, col]

        ax.plot(
            epsilon_data["delta"],
            epsilon_data["W"],
            "-^",
            color="#2ca02c",
            label="$D_w$ size",
            markersize=6,
            markerfacecolor="white",
        )
        ax.plot(
            epsilon_data["delta"],
            epsilon_data["B"],
            "--v",
            color="#d62728",
            label="$D_b$ size",
            markersize=6,
            markerfacecolor="white",
        )

        ax.set_xlabel("δ")
        ax.set_ylabel("Average Dimensions")
        ax.set_title(f"ε = {epsilon:.2f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    # Remove empty subplots
    for idx in range(n_plots, n_rows * n_cols):
        row = (idx // n_cols) + 1
        col = idx % n_cols
        axes_relations[row, col].remove()

    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes_dimensions[row, col].remove()

    # Save plots
    for fig, name in [(fig_relations, "relations"), (fig_dimensions, "dimensions")]:
        fig.tight_layout()

        params = [
            f"max{max_set_size}",
            f"top{top_features}",
            f"splits{n_splits}",
        ]
        if test_size is not None:
            params.append(f"test{test_size}")

        filename = f"{name}_plot_{'_'.join(params)}.pdf"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Create dimension distribution plot
    create_dimension_distribution(
        df, output_dir, max_set_size, top_features, test_size, n_splits
    )
