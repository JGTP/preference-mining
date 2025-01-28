import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.visualisation import calculate_max_relations, process_results, create_plots


@pytest.fixture
def sample_results():
    """Create sample analysis results that match the actual structure"""
    return {
        "rule_analyses": {
            "rule_1": {
                "rule_string": "Rule 1",
                "analysis": {
                    "epsilon_0.1_delta_0.2%": {
                        "relations": [
                            {
                                "set1": ["A", "B"],
                                "set2": ["C", "D"],
                                "set1_importance": 0.8,
                                "set2_importance": 0.3,
                            },
                            {
                                "set1": ["E"],
                                "set2": ["F", "G"],
                                "set1_importance": 0.7,
                                "set2_importance": 0.2,
                            },
                        ]
                    },
                    "epsilon_0.2_delta_0.2%": {
                        "relations": [
                            {
                                "set1": ["A", "B"],
                                "set2": ["C", "D"],
                                "set1_importance": 0.8,
                                "set2_importance": 0.3,
                            }
                        ]
                    },
                },
            },
            "rule_2": {
                "rule_string": "Rule 2",
                "analysis": {
                    "epsilon_0.1_delta_0.2%": {
                        "relations": [
                            {
                                "set1": ["A", "B"],
                                "set2": ["C", "D"],
                                "set1_importance": 0.8,
                                "set2_importance": 0.3,
                            }
                        ]
                    }
                },
            },
        }
    }


@pytest.fixture
def mock_shap_values():
    return {
        "A": 0.5,
        "B": 0.3,
        "C": 0.2,
        "D": 0.4,
        "E": 0.6,
        "F": 0.1,
        "G": 0.3,
    }


@pytest.fixture
def mock_correlations():
    return {
        str(sorted(["A", "B"])): 0.2,
        str(sorted(["C", "D"])): 0.3,
        str(sorted(["F", "G"])): 0.1,
    }


def test_process_results_basic(sample_results, mock_shap_values, mock_correlations):
    """Test basic processing of results with known inputs"""
    df = process_results(sample_results, mock_shap_values, mock_correlations)

    # Check DataFrame structure
    expected_columns = ["epsilon", "delta", "N_total", "N_unique", "W", "B"]
    assert all(col in df.columns for col in expected_columns)

    # For epsilon=0.1, delta=0.2:
    # - Rule 1 has 2 relations
    # - Rule 2 has 1 relation
    # - One relation appears in both rules
    # So N_total should be 3, N_unique should be 2
    epsilon_01_row = df[df["epsilon"] == 0.1].iloc[0]
    assert epsilon_01_row["N_total"] == 3  # Total relations including duplicates
    assert epsilon_01_row["N_unique"] == 2  # Unique relations after deduplication

    # For epsilon=0.2, delta=0.2:
    # - Only Rule 1 has 1 relation
    epsilon_02_row = df[df["epsilon"] == 0.2].iloc[0]
    assert epsilon_02_row["N_total"] == 1
    assert epsilon_02_row["N_unique"] == 1


def test_process_results_dimensions(
    sample_results, mock_shap_values, mock_correlations
):
    """Test that W and B calculations are correct"""
    df = process_results(sample_results, mock_shap_values, mock_correlations)

    # For epsilon=0.1, delta=0.2:
    # Rule 1: W=[2,1], B=[2,2]
    # Rule 2: W=[2], B=[2]
    epsilon_01_row = df[df["epsilon"] == 0.1].iloc[0]
    assert epsilon_01_row["W"] == (2 + 1 + 2) / 3  # Average W
    assert epsilon_01_row["B"] == 2.0  # Average B


def test_process_results_empty_input(mock_shap_values, mock_correlations):
    """Test handling of empty input"""
    empty_results = {"rule_analyses": {}}
    df = process_results(empty_results, mock_shap_values, mock_correlations)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_process_results_missing_values():
    """Test handling of missing shap_values and correlations"""
    with pytest.raises(ValueError):
        process_results({"rule_analyses": {}}, None, None)


def test_create_plots(tmp_path, sample_results, mock_shap_values, mock_correlations):
    """Test plot creation with parameter-based filenames"""
    output_dir = tmp_path / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = process_results(sample_results, mock_shap_values, mock_correlations)

    create_plots(
        df,
        output_dir,
        n_rules=2,
        test_size=1000,
        max_set_size=10,
        top_features=20,
        n_splits=3,
        total_features=7,
    )

    # Check if plots were created
    plot_files = list(output_dir.glob("*.pdf"))
    assert len(plot_files) == 2  # Should create both relations and dimensions plots

    file_names = {f.name for f in plot_files}
    expected_names = {
        "relations_plot_max10_top20_splits3_test1000.pdf",
        "dimensions_plot_max10_top20_splits3_test1000.pdf",
    }
    assert file_names == expected_names


def test_create_plots_without_test_size(
    tmp_path, sample_results, mock_shap_values, mock_correlations
):
    """Test plot creation without test_size parameter"""
    output_dir = tmp_path / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = process_results(sample_results, mock_shap_values, mock_correlations)

    create_plots(
        df,
        output_dir,
        n_rules=2,
        max_set_size=10,
        top_features=20,
        n_splits=3,
        total_features=7,
    )

    file_names = {f.name for f in output_dir.glob("*.pdf")}
    expected_names = {
        "relations_plot_max10_top20_splits3.pdf",
        "dimensions_plot_max10_top20_splits3.pdf",
    }
    assert file_names == expected_names


def test_calculate_max_relations():
    """Test calculation of maximum possible relations"""
    # Simple case: N=3, max_set_size=2, top_features=2
    assert calculate_max_relations(N=3, max_set_size=2, top_features=2) == 7

    # Edge case: minimal configuration
    assert calculate_max_relations(N=2, max_set_size=1, top_features=1) == 1

    # Test with larger values
    result = calculate_max_relations(N=5, max_set_size=2, top_features=3)
    assert result > 0
    assert isinstance(result, int)
