import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


def process_results(results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for rule_info in results["rule_analyses"].values():
        for param_key, analyses in rule_info["analysis"].items():
            epsilon = float(param_key.split("_")[1])
            delta = float(param_key.split("delta_")[1].rstrip("%"))
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
    plt.figure(figsize=(10, 6))
    metrics = ["total_preference_relations", "mean_dimensions_Dw", "mean_dimensions_Db"]
    labels = [
        "Total Preference Relations",
        "Mean Dimensions in $D_w$",
        "Mean Dimensions in $D_b$",
    ]
    colours = ["blue", "red", "green"]
    markers = ["o", "s", "^"]
    grouped = df.groupby(x_var)
    plt.figure(figsize=(10, 6))
    for metric, label, colour, marker in zip(metrics, labels, colours, markers):
        mean_values = grouped[metric].mean()
        sem_values = grouped[metric].sem()
        plt.errorbar(
            mean_values.index,
            mean_values.values,
            yerr=sem_values.values,
            color=colour,
            label=label,
            marker=marker,
            capsize=5,
            linestyle="-",
            markersize=8,
            elinewidth=1.5,
            capthick=1.5,
        )
    plt.xlabel(f"${x_var}$")
    plt.ylabel("Metric Value")
    plt.title(f"Preference Relation Metrics vs {x_var.capitalize()}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_epsilon_plot(df: pd.DataFrame, output_path: Path) -> None:
    create_combined_plot(df, output_path, "epsilon")


def create_delta_plot(df: pd.DataFrame, output_path: Path) -> None:
    create_combined_plot(df, output_path, "delta")
