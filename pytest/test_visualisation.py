import pytest
import os
import pandas as pd
import json
from pathlib import Path
from src.visualisation import process_results, create_epsilon_plot, create_delta_plot


@pytest.fixture
def mock_temp_dir(tmpdir):
    return Path(tmpdir)


@pytest.fixture
def sample_results():
    rule_analyses = {}
    for i in range(10):
        rule_key = f"rule_{i}"
        rule_analyses[rule_key] = {"rule_string": f"[test_rule_{i}]", "analysis": {}}
        for j in range(5):
            for k in range(5):
                epsilon = 0.1 * (j + 1)
                delta = 0.05 * (k + 1)
                param_key = f"epsilon_{epsilon}_delta_{delta}%"
                rule_analyses[rule_key]["analysis"][param_key] = [
                    {
                        "set1": ["feature" + str(f) for f in range(1 + i % 3)],
                        "set2": ["feature" + str(f) for f in range(2 + i % 4)],
                        "set1_importance": 0.6 + 0.1 * (i % 3),
                        "set2_importance": 0.1 * (i % 2),
                        "importance_difference": 0.5 + 0.1 * (i % 4),
                        "max_correlation_set1": 0.0,
                        "max_correlation_set2": 0.0,
                    }
                    for _ in range(3 + i % 5)
                ]
    return {"rule_analyses": rule_analyses}


@pytest.fixture
def setup_temp_files(mock_temp_dir):
    mock_temp_dir.mkdir(parents=True, exist_ok=True)
    shap_values = {f"feature{i}": 0.1 * i for i in range(10)}
    correlations = {
        f"[feature{i},feature{j}]": 0.1 for i in range(10) for j in range(i + 1, 10)
    }

    with open(mock_temp_dir / "shap_values.json", "w") as f:
        json.dump(shap_values, f)
    with open(mock_temp_dir / "correlations.json", "w") as f:
        json.dump(correlations, f)
    return mock_temp_dir


def test_process_results(sample_results, setup_temp_files):
    df = process_results(sample_results, setup_temp_files)
    assert isinstance(df, pd.DataFrame)
    assert all(
        col in df.columns
        for col in [
            "epsilon",
            "delta",
            "total_preference_relations",
            "mean_dimensions_Dw",
            "mean_dimensions_Db",
        ]
    )
    assert len(df) > 0
    assert df["total_preference_relations"].nunique() > 1
    assert df["mean_dimensions_Dw"].nunique() > 1
    assert df["mean_dimensions_Db"].nunique() > 1


def test_create_plots(sample_results, setup_temp_files):
    plots_dir = Path("pytest/test_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    df = process_results(sample_results, setup_temp_files)

    # Test epsilon plot
    epsilon_plot_path = plots_dir / "epsilon_plot.pdf"
    create_epsilon_plot(df, epsilon_plot_path)
    assert epsilon_plot_path.exists()
    assert epsilon_plot_path.stat().st_size > 0

    # Test delta plot
    delta_plot_path = plots_dir / "delta_plot.pdf"
    create_delta_plot(df, delta_plot_path)
    assert delta_plot_path.exists()
    assert delta_plot_path.stat().st_size > 0


def test_process_results_with_missing_files(sample_results, mock_temp_dir):
    with pytest.raises(FileNotFoundError):
        process_results(sample_results, mock_temp_dir)


def test_process_results_with_invalid_json(mock_temp_dir):
    mock_temp_dir.mkdir(parents=True, exist_ok=True)

    # Create invalid JSON files
    with open(mock_temp_dir / "shap_values.json", "w") as f:
        f.write("invalid json")
    with open(mock_temp_dir / "correlations.json", "w") as f:
        f.write("invalid json")

    with pytest.raises(json.JSONDecodeError):
        process_results({"rule_analyses": {}}, mock_temp_dir)
