import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Union
from collections import defaultdict


def process_results(
    results: Dict[str, Any],
    shap_values: Optional[Dict[str, float]] = None,
    correlations: Optional[Dict[str, float]] = None,
    temp_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Process the analysis results into a DataFrame suitable for plotting.
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
    """
    # Define metrics with their labels and styles
    metrics = {
        "N": {
            "label": "Number of Preference Relations",
            "linestyle": "-",
            "marker": "o",
            "color": "#1f77b4",  # Blue
        },
        "W": {
            "label": "Average Dimensions in $D_w$",
            "linestyle": "--",
            "marker": "s",
            "color": "#2ca02c",  # Green
        },
        "B": {
            "label": "Average Dimensions in $D_b$",
            "linestyle": ":",
            "marker": "^",
            "color": "#d62728",  # Red
        },
    }

    # Create separate plot for each epsilon value
    for epsilon in sorted(df["epsilon"].unique()):
        epsilon_data = df[df["epsilon"] == epsilon]

        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()

        # Sort by delta to ensure lines are connected in order
        epsilon_data = epsilon_data.sort_values("delta")

        # Plot each metric
        lines = []
        labels = []
        for metric, metric_style in metrics.items():
            ax = ax1 if metric == "N" else ax2
            line = ax.plot(
                epsilon_data["delta"],
                epsilon_data[metric],
                label=metric_style["label"],
                linestyle=metric_style["linestyle"],
                marker=metric_style["marker"],
                color=metric_style["color"],
                markersize=8,
                markerfacecolor="white",
                markeredgewidth=1.5,
            )[0]
            lines.append(line)
            labels.append(metric_style["label"])

        # Customize axes
        ax1.set_xlabel("δ")
        ax1.set_ylabel("Number of Preference Relations (N)")
        ax2.set_ylabel("Average Dimensions (W, B)")

        # Add grid
        ax1.grid(True, alpha=0.3, zorder=0)

        # Adjust layout
        fig.tight_layout()

        # Create legend
        fig.legend(
            lines,
            labels,
            bbox_to_anchor=(1.25, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=True,
            edgecolor="black",
            fancybox=False,
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
