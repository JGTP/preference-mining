import json

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from typing import Dict, Any, Optional, Union

from collections import defaultdict


def calculate_max_relations(N: int, max_set_size: int, top_features: int) -> int:
    """

    Calculate theoretical maximum number of preference relations.

    Args:

        N: Total number of features

        max_set_size: Maximum size of feature sets

        top_features: Number of top features to consider for set1

    Returns:

        int: Maximum possible number of preference relations

    """

    total = 0

    for set1_size in range(1, min(max_set_size, top_features) + 1):

        # Calculate number of ways to choose set1_size items from top_features

        set1_count = 1

        for i in range(set1_size):

            set1_count = set1_count * (top_features - i) // (i + 1)

        # For each set1, calculate valid set2 combinations

        # Available features for set2 are those not in top_features, plus unused top features

        remaining_features = (N - top_features) + (top_features - set1_size)

        set2_total = 0

        for i in range(1, max_set_size + 1):

            if i <= remaining_features:

                # Calculate number of ways to choose i items from remaining_features

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
    """
    Process the analysis results into a DataFrame suitable for plotting.
    Now correctly tracks total and unique relations.
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

    # Create a defaultdict for metrics
    metrics_by_params = defaultdict(
        lambda: {
            "all_relations": [],  # List to store ALL relations (including duplicates)
            "unique_pairs": set(),  # Set to store unique feature set pairs
            "W": [],
            "B": [],
        }
    )

    # First pass: collect all metrics for each epsilon-delta combination
    for rule_id, rule_info in results["rule_analyses"].items():
        rule_string = rule_info["rule_string"]
        for param_key, analyses in rule_info["analysis"].items():
            epsilon = float(param_key.split("_")[1])
            delta = float(param_key.split("delta_")[1].rstrip("%"))

            if analyses:
                for analysis in analyses["relations"]:
                    # Create hashable versions of the feature sets
                    set1_frozen = frozenset(analysis["set1"])
                    set2_frozen = frozenset(analysis["set2"])
                    feature_pair = (set1_frozen, set2_frozen)

                    # Store ALL relations (including duplicates)
                    metrics_by_params[(epsilon, delta)]["all_relations"].append(
                        feature_pair
                    )

                    # Store unique pairs
                    metrics_by_params[(epsilon, delta)]["unique_pairs"].add(
                        feature_pair
                    )

                    # Track dimensions
                    metrics_by_params[(epsilon, delta)]["W"].append(
                        len(analysis["set1"])
                    )
                    metrics_by_params[(epsilon, delta)]["B"].append(
                        len(analysis["set2"])
                    )

    # Second pass: compute metrics
    rows = []
    for (epsilon, delta), metrics in metrics_by_params.items():
        # Total relations is now the raw count of all relations found
        total_relations = len(metrics["all_relations"])
        # Unique relations remains the count of unique feature set pairs
        unique_relations = len(metrics["unique_pairs"])

        row = {
            "epsilon": epsilon,
            "delta": delta,
            "N_total": total_relations,  # Raw count including duplicates
            "N_unique": unique_relations,  # Count after deduplication
            "W": np.mean(metrics["W"]) if metrics["W"] else 0,
            "B": np.mean(metrics["B"]) if metrics["B"] else 0,
        }
        rows.append(row)

    return pd.DataFrame(rows)


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
    """

    Create two separate multi-panel figures with improved visualization.

    """

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate theoretical maximum number of relations

    max_relations = calculate_max_relations(total_features, max_set_size, top_features)

    # Sort epsilon values and calculate number of rows/columns for subplots

    epsilons = sorted(df["epsilon"].unique())

    n_plots = len(epsilons)

    n_cols = min(3, n_plots)

    n_rows = (n_plots + n_cols - 1) // n_cols

    # Figure 1: Relations (N_total and N_unique)

    fig_relations, axes_relations = plt.subplots(
        n_rows + 1,
        n_cols,
        figsize=(5 * n_cols, 4 * (n_rows + 1)),
        gridspec_kw={"height_ratios": [0.2] + [1] * n_rows},
        squeeze=False,
    )

    # Add summary text in the top row

    summary_text = (
        f"Total Rules: {n_rules}\nMaximum Possible Relations: {max_relations}"
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

    # Figure 2: Dimensions (W and B)

    fig_dimensions, axes_dimensions = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    # Plot for each epsilon value

    for idx, epsilon in enumerate(epsilons):

        row = (idx // n_cols) + 1  # +1 to account for summary row in relations plot

        col = idx % n_cols

        # Get data for this epsilon

        epsilon_data = df[df["epsilon"] == epsilon].sort_values("delta")

        # Plot 1: Relations

        ax1 = axes_relations[row, col]

        ax2 = ax1.twinx()

        # Plot N_total on primary y-axis

        l1 = ax1.plot(
            epsilon_data["delta"],
            epsilon_data["N_total"],
            "-o",
            color="#1f77b4",
            label="Total Relations",
            markersize=6,
            markerfacecolor="white",
        )

        # Plot N_unique on secondary y-axis

        l2 = ax2.plot(
            epsilon_data["delta"],
            epsilon_data["N_unique"],
            "--s",
            color="#ff7f0e",
            label="Unique Relations",
            markersize=6,
            markerfacecolor="white",
        )

        # Labels and title for relations plot

        ax1.set_xlabel("δ")

        ax1.set_ylabel("Total Relations")

        ax2.set_ylabel("Unique Relations")

        ax1.set_title(f"ε = {epsilon:.2f}")

        ax1.grid(True, alpha=0.3)

        # Add legend

        lines = l1 + l2

        labels = [l.get_label() for l in lines]

        ax1.legend(lines, labels, loc="upper right")

        # Plot 2: Dimensions

        ax = axes_dimensions[idx // n_cols, col]

        # Plot W and B on same y-axis

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

        # Labels and title for dimensions plot

        ax.set_xlabel("δ")

        ax.set_ylabel("Average Dimensions")

        ax.set_title(f"ε = {epsilon:.2f}")

        ax.grid(True, alpha=0.3)

        ax.legend(loc="upper right")

    # Remove empty subplots if any

    for idx in range(n_plots, n_rows * n_cols):

        row = (idx // n_cols) + 1  # +1 for summary row

        col = idx % n_cols

        axes_relations[row, col].remove()

    for idx in range(n_plots, n_rows * n_cols):

        row = idx // n_cols

        col = idx % n_cols

        axes_dimensions[row, col].remove()

    # Adjust layout and save figures

    for fig, name in [(fig_relations, "relations"), (fig_dimensions, "dimensions")]:

        fig.tight_layout()

        # Create filename with parameters

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
