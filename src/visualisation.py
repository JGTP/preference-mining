import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

def process_results(results: Dict[str, Any], temp_dir: Path) -> pd.DataFrame:
    """
    Process the analysis results into a DataFrame suitable for plotting.
    
    Args:
        results: Dictionary containing the analysis results
        temp_dir: Path to directory containing cached SHAP values and correlations
        
    Returns:
        DataFrame with columns epsilon, delta, N, W, B containing processed metrics
    """
    # Load cached SHAP values and correlations
    with open(temp_dir / "shap_values.json", "r") as f:
        shap_values = json.load(f)
    with open(temp_dir / "correlations.json", "r") as f:
        correlations = json.load(f)

    # Create a nested defaultdict to accumulate metrics
    metrics_by_params = defaultdict(lambda: defaultdict(list))
    
    # First pass: collect all metrics for each epsilon-delta combination
    for rule_info in results["rule_analyses"].values():
        for param_key, analyses in rule_info["analysis"].items():
            epsilon = float(param_key.split("_")[1])
            delta = float(param_key.split("delta_")[1].rstrip("%"))
            
            # Collect metrics for this epsilon-delta combination
            metrics_by_params[(epsilon, delta)]['N'].append(len(analyses))
            
            if analyses:  # Only process if we have analyses
                metrics_by_params[(epsilon, delta)]['W'].append(
                    sum(len(analysis["set1"]) for analysis in analyses) / len(analyses)
                )
                metrics_by_params[(epsilon, delta)]['B'].append(
                    sum(len(analysis["set2"]) for analysis in analyses) / len(analyses)
                )

    # Second pass: compute averages across rules
    rows = []
    for (epsilon, delta), metrics in metrics_by_params.items():
        row = {
            "epsilon": epsilon,
            "delta": delta,
            "N": sum(metrics['N']),  # Total number of preference relations
            "W": sum(metrics['W']) / len(metrics['W']) if metrics['W'] else 0,  # Average dimensions
            "B": sum(metrics['B']) / len(metrics['B']) if metrics['B'] else 0,  # Average dimensions
        }
        rows.append(row)

    return pd.DataFrame(rows)

def create_plot(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a plot visualising the metrics across different epsilon and delta values.
    
    Args:
        df: DataFrame containing the metrics to plot
        output_path: Path where the plot should be saved
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define metrics with their labels and styles
    metrics = {
        'N': ('Number of Preference Relations', '-', 'o'),    # Circle
        'W': ('Average Dimensions in $D_w$', '--', 's'),     # Square
        'B': ('Average Dimensions in $D_b$', ':', '^')       # Triangle
    }
    
    # Define styles for each epsilon value
    epsilons = sorted(df['epsilon'].unique())
    styles_by_epsilon = {
        epsilons[0]: {
            'color': '#1f77b4',  # Blue
            'linewidth': 2.0,
            'alpha': 0.9,
            'zorder': 2,         # Higher zorder means drawn on top
            'markersize': 8      # Smaller markers for front layer
        },
        epsilons[1]: {
            'color': '#d62728',  # Red
            'linewidth': 1.5,
            'alpha': 0.7,
            'zorder': 1,
            'markersize': 12     # Larger markers for background layer
        }
    }
    
    # Plot each metric separately to control ordering
    for metric, (label, linestyle, marker) in metrics.items():
        # Plot epsilon values in reverse order (so first epsilon is on top)
        for epsilon in reversed(epsilons):
            epsilon_data = df[df['epsilon'] == epsilon]
            style = styles_by_epsilon[epsilon]
            
            # Sort by delta to ensure lines are connected in order
            epsilon_data = epsilon_data.sort_values('delta')
            
            ax.plot(epsilon_data['delta'], 
                   epsilon_data[metric],
                   label=f'{label} (ε={epsilon:.2f})',
                   color=style['color'],
                   linestyle=linestyle,
                   linewidth=style['linewidth'],
                   alpha=style['alpha'],
                   zorder=style['zorder'],
                   marker=marker,
                   markersize=style['markersize'],
                   markerfacecolor='white',
                   markeredgewidth=1.5,
                   markeredgecolor=style['color'])
    
    # Customize plot appearance
    ax.set_xlabel('δ')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3, zorder=0)  # Put grid behind everything
    
    # Adjust legend for better readability
    ax.legend(bbox_to_anchor=(1.05, 1), 
             loc='upper left',
             borderaxespad=0,
             frameon=True,
             edgecolor='black',
             fancybox=False)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()