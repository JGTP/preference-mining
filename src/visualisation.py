import json
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
        # Number of possible set1 combinations of this size
        set1_count = sum(1 for _ in range(1, top_features + 1))
        for i in range(set1_size):
            set1_count = set1_count * (top_features - i) // (i + 1)

        # For each set1 of this size, calculate possible set2 combinations
        remaining_features = N - set1_size
        set2_total = 0
        for i in range(1, max_set_size + 1):
            if i <= remaining_features:
                set2_count = sum(1 for _ in range(1, remaining_features + 1))
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

    Args:
        results: Analysis results dictionary
        shap_values: Dictionary of SHAP values
        correlations: Dictionary of correlations
        temp_dir: Directory containing cached values (if shap_values/correlations not provided)

    Returns:
        DataFrame with processed metrics
    """
    # Load cached values if not provided directly
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

    # Create a nested defaultdict to accumulate metrics
    metrics_by_params = defaultdict(lambda: defaultdict(list))

    # First pass: collect all metrics for each epsilon-delta combination
    for rule_info in results["rule_analyses"].values():
        for param_key, analyses in rule_info["analysis"].items():
            epsilon = float(param_key.split("_")[1])
            delta = float(param_key.split("delta_")[1].rstrip("%"))

            # Collect metrics for this epsilon-delta combination
            metrics_by_params[(epsilon, delta)]["N"].append(len(analyses))

            if analyses:  # Only process if we have analyses
                metrics_by_params[(epsilon, delta)]["W"].append(
                    sum(len(analysis["set1"]) for analysis in analyses) / len(analyses)
                )
                metrics_by_params[(epsilon, delta)]["B"].append(
                    sum(len(analysis["set2"]) for analysis in analyses) / len(analyses)
                )

    # Second pass: compute averages across rules
    rows = []
    for (epsilon, delta), metrics in metrics_by_params.items():
        row = {
            "epsilon": epsilon,
            "delta": delta,
            "N": sum(metrics["N"]),  # Total number of preference relations
            "W": (
                sum(metrics["W"]) / len(metrics["W"]) if metrics["W"] else 0
            ),  # Average dimensions
            "B": (
                sum(metrics["B"]) / len(metrics["B"]) if metrics["B"] else 0
            ),  # Average dimensions
        }
        rows.append(row)

    return pd.DataFrame(rows)


def create_plot(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: Optional[int] = None,
    max_set_size: int = 10,
    top_features: int = 20,
    n_splits: int = 3,
    total_features: int = None,
) -> None:
    """
    Create separate plots for each epsilon value.

    Args:
        df: DataFrame containing the metrics to plot
        output_dir: Directory where plots should be saved
        test_size: Size of test set used (optional)
        max_set_size: Maximum feature set size
        top_features: Number of top features considered
        n_splits: Number of CV splits
        total_features: Total number of features in dataset
    """
    # Define metrics with their labels and styles
    metrics = {
        "N": {
            "label": "Number of Preference Relations",
            "linestyle": "-",
            "marker": "o",
            "color": "#1f77b4",  # Blue
            "axis": "primary",
        },
        "W": {
            "label": "Average Dimensions in $D_w$",
            "linestyle": "--",
            "marker": "s",
            "color": "#2ca02c",  # Green
            "axis": "secondary",
        },
        "B": {
            "label": "Average Dimensions in $D_b$",
            "linestyle": ":",
            "marker": "^",
            "color": "#d62728",  # Red
            "axis": "secondary",
        },
    }

    # Calculate theoretical maximum if total_features is provided
    max_relations = None
    if total_features is not None:
        max_relations = calculate_max_relations(
            total_features, max_set_size, top_features
        )

    # Create separate plot for each epsilon value
    for epsilon in sorted(df["epsilon"].unique()):
        # Create figure and axes with more compact size
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()

        # Adjust margins to be tighter
        plt.subplots_adjust(right=0.85)

        # Get data for this epsilon
        epsilon_data = df[df["epsilon"] == epsilon].sort_values("delta")

        # Plot lines and collect for legend
        lines = []
        labels = []

        # Plot each metric
        for metric, style in metrics.items():
            ax = ax1 if style["axis"] == "primary" else ax2

            if metric == "N" and max_relations is not None:
                percentage_data = (epsilon_data[metric] / max_relations) * 100

            line = ax.plot(
                epsilon_data["delta"],
                epsilon_data[metric],
                label=style["label"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                color=style["color"],
                markersize=8,
                markerfacecolor="white",
                markeredgewidth=1.5,
            )[0]

            lines.append(line)
            labels.append(style["label"])

            # Add percentage annotations for N
            if metric == "N" and max_relations is not None:
                for x, y, p in zip(
                    epsilon_data["delta"], epsilon_data[metric], percentage_data
                ):
                    ax1.annotate(
                        f"{p:.1f}%",
                        (x, y),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )

        # Customize axes
        ax1.set_xlabel("δ")
        ax1.set_ylabel("Number of Preference Relations (N)")
        ax2.set_ylabel("Average Dimensions (W, B)")

        # Add grid
        ax1.grid(True, alpha=0.3, zorder=0)

        # Add title with theoretical maximum if available
        if max_relations is not None:
            plt.title(
                f"ε = {epsilon:.2f} (max: {max_relations})",
                pad=10,  # Reduce padding
                fontsize=10,  # Slightly smaller font
            )

        # Adjust layout
        fig.tight_layout()

        # Add legend in a more compact position
        fig.legend(
            lines,
            labels,
            bbox_to_anchor=(1.0, 1.0),
            loc="upper left",
            borderaxespad=0,
            frameon=True,
            edgecolor="black",
            fancybox=False,
            fontsize=8,  # Slightly smaller font for compactness
        )

        # Create filename with parameters
        params = [
            f"eps{epsilon:.2f}",
            f"max{max_set_size}",
            f"top{top_features}",
            f"splits{n_splits}",
        ]
        if test_size is not None:
            params.append(f"test{test_size}")

        filename = f"epsilon_plot_{'_'.join(params)}.pdf"

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plot
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
