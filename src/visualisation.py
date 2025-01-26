import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


def process_results(results: Dict[str, Any], temp_dir: Path) -> pd.DataFrame:
    rows = []

    # Load cached SHAP values and correlations
    with open(temp_dir / "shap_values.json", "r") as f:
        shap_values = json.load(f)
    with open(temp_dir / "correlations.json", "r") as f:
        correlations = json.load(f)

    for rule_info in results["rule_analyses"].values():
        for param_key, analyses in rule_info["analysis"].items():
            epsilon = float(param_key.split("_")[1])
            delta = float(param_key.split("delta_")[1].rstrip("%"))

            # Use stored values instead of recalculating
            num_preference_relations = len(analyses)
            total_mean_dimensions_Dw = (
                sum(len(analysis["set1"]) for analysis in analyses)
                / num_preference_relations
            )
            total_mean_dimensions_Db = (
                sum(len(analysis["set2"]) for analysis in analyses)
                / num_preference_relations
            )

            row = {
                "epsilon": epsilon,
                "delta": delta,
                "total_preference_relations": num_preference_relations,
                "mean_dimensions_Dw": total_mean_dimensions_Dw,
                "mean_dimensions_Db": total_mean_dimensions_Db,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def create_combined_plot(df: pd.DataFrame, output_path: Path, x_var: str) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    grouped = df.groupby(x_var)

    mean_relations = grouped["total_preference_relations"].mean()
    sem_relations = grouped["total_preference_relations"].sem()
    ln1 = ax1.errorbar(
        mean_relations.index,
        mean_relations.values,
        yerr=sem_relations.values,
        color="blue",
        label="N (Total Preference Relations)",
        marker="o",
        capsize=5,
        linestyle="-",
        markersize=8,
        elinewidth=1.5,
        capthick=1.5,
    )

    colors = ["red", "green"]
    markers = ["s", "^"]
    lines = [ln1]
    for metric, label, color, marker in zip(
        ["mean_dimensions_Dw", "mean_dimensions_Db"],
        ["W (Mean Dimensions in $D_w$)", "B (Mean Dimensions in $D_b$)"],
        colors,
        markers,
    ):
        mean_vals = grouped[metric].mean()
        sem_vals = grouped[metric].sem()
        ln = ax2.errorbar(
            mean_vals.index,
            mean_vals.values,
            yerr=sem_vals.values,
            color=color,
            label=label,
            marker=marker,
            capsize=5,
            linestyle="-",
            markersize=8,
            elinewidth=1.5,
            capthick=1.5,
        )
        lines.append(ln)

    x_label = "\u03B5" if x_var == "epsilon" else "\u03B4"
    ax1.set_xlabel(f"${x_label}$")
    ax1.set_ylabel("Number of Preference Relations")
    ax2.set_ylabel("Mean Number of Dimensions")

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(0.05, 1.15))

    plt.title(f"Preference Relation Metrics vs {x_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_epsilon_plot(df: pd.DataFrame, output_path: Path) -> None:
    create_combined_plot(df, output_path, "epsilon")


def create_delta_plot(df: pd.DataFrame, output_path: Path) -> None:
    create_combined_plot(df, output_path, "delta")
