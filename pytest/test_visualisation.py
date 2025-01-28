import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.visualisation import process_results, create_plot


@pytest.fixture
def sample_results_df():
    return pd.DataFrame(
        {
            "epsilon": [0.1, 0.1, 0.2, 0.2],
            "delta": [0.1, 0.2, 0.1, 0.2],
            "N": [10, 15, 12, 18],
            "W": [3.5, 4.0, 3.8, 4.2],
            "B": [2.5, 3.0, 2.8, 3.2],
        }
    )


def test_create_plot(tmp_path, sample_results_df):
    """Test plot creation with parameter-based filenames"""
    output_dir = tmp_path / "plots"
    create_plot(
        sample_results_df,
        output_dir,
        test_size=1000,
        max_set_size=10,
        top_features=20,
        n_splits=3,
    )

    # Check if separate plots were created for each epsilon
    expected_files = {
        "epsilon_plot_eps0.10_max10_top20_splits3_test1000.pdf",
        "epsilon_plot_eps0.20_max10_top20_splits3_test1000.pdf",
    }
    actual_files = {f.name for f in output_dir.glob("*.pdf")}
    assert actual_files == expected_files


def test_create_plot_without_test_size(tmp_path, sample_results_df):
    """Test plot creation without test_size parameter"""
    output_dir = tmp_path / "plots"
    create_plot(
        sample_results_df, output_dir, max_set_size=10, top_features=20, n_splits=3
    )

    # Check if files were created without test_size in name
    expected_files = {
        "epsilon_plot_eps0.10_max10_top20_splits3.pdf",
        "epsilon_plot_eps0.20_max10_top20_splits3.pdf",
    }
    actual_files = {f.name for f in output_dir.glob("*.pdf")}
    assert actual_files == expected_files


def test_process_results():
    """Test processing of analysis results"""
    # Create mock analysis results
    mock_results = {
        "rule_analyses": {
            "rule_1": {
                "analysis": {
                    "epsilon_0.1_delta_0.2%": [
                        {"set1": ["A", "B"], "set2": ["C", "D"]},
                        {"set1": ["E"], "set2": ["F", "G"]},
                    ]
                }
            }
        }
    }

    mock_shap_values = {
        "A": 0.5,
        "B": 0.3,
        "C": 0.2,
        "D": 0.4,
        "E": 0.6,
        "F": 0.1,
        "G": 0.3,
    }
    mock_correlations = {"[A, B]": 0.2, "[C, D]": 0.3, "[F, G]": 0.1}

    df = process_results(mock_results, mock_shap_values, mock_correlations)

    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ["epsilon", "delta", "N", "W", "B"])
    assert len(df) > 0


def test_empty_results(tmp_path):
    """Test handling of empty results"""
    empty_df = pd.DataFrame(columns=["epsilon", "delta", "N", "W", "B"])
    create_plot(empty_df, tmp_path)
    # Should create files even with empty data
    assert len(list(tmp_path.glob("*.pdf"))) == 0  # No plots created for empty data
