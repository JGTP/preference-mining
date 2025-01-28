import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.visualisation import (
    calculate_max_relations,
    process_results,
    create_plots,
    create_dimension_distribution,
)


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


@pytest.fixture
def mock_temp_dir(tmp_path):
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()

    with open(temp_dir / "shap_values.json", "w") as f:
        json.dump({"A": 0.5, "B": 0.3}, f)
    with open(temp_dir / "correlations.json", "w") as f:
        json.dump({str(sorted(["A", "B"])): 0.2}, f)

    return temp_dir


def test_process_results_basic(sample_results, mock_shap_values, mock_correlations):
    """Test basic processing of results with known inputs"""
    df = process_results(sample_results, mock_shap_values, mock_correlations)

    expected_columns = [
        "epsilon",
        "delta",
        "N_total",
        "N_unique",
        "W",
        "B",
        "size_distribution",
    ]
    assert all(col in df.columns for col in expected_columns)

    epsilon_01_row = df[df["epsilon"] == 0.1].iloc[0]
    assert epsilon_01_row["N_total"] == 3
    assert epsilon_01_row["N_unique"] == 2

    assert isinstance(epsilon_01_row["size_distribution"], dict)
    assert "(2,2)" in epsilon_01_row["size_distribution"]
    assert "(1,2)" in epsilon_01_row["size_distribution"]


def test_process_results_with_temp_dir(sample_results, mock_temp_dir):
    """Test processing results using cached values from temp directory"""
    df = process_results(sample_results, temp_dir=mock_temp_dir)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_process_results_missing_values():
    """Test handling of missing values"""
    with pytest.raises(ValueError):
        process_results({"rule_analyses": {}}, None, None)

    with pytest.raises(ValueError):
        process_results({"rule_analyses": {}}, None, None, temp_dir="nonexistent")


def test_create_dimension_distribution(
    tmp_path, sample_results, mock_shap_values, mock_correlations
):
    """Test creation of dimension distribution plot"""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    df = process_results(sample_results, mock_shap_values, mock_correlations)
    create_dimension_distribution(df, output_dir, max_set_size=3)

    distribution_plots = list(output_dir.glob("dimension_distribution*.pdf"))
    assert len(distribution_plots) == 1


def test_create_plots_basic(
    tmp_path, sample_results, mock_shap_values, mock_correlations
):
    """Test basic plot creation functionality"""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    df = process_results(sample_results, mock_shap_values, mock_correlations)
    create_plots(
        df,
        output_dir,
        n_rules=2,
        max_set_size=3,
        top_features=5,
        n_splits=3,
        total_features=7,
    )

    plot_files = list(output_dir.glob("*.pdf"))
    assert len(plot_files) == 3

    file_names = {f.name for f in plot_files}
    expected_names = {
        "relations_plot_max3_top5_splits3.pdf",
        "dimensions_plot_max3_top5_splits3.pdf",
        "dimension_distribution_max3_splits3.pdf",
    }
    assert file_names == expected_names


def test_create_plots_with_test_size(
    tmp_path, sample_results, mock_shap_values, mock_correlations
):
    """Test plot creation with test_size parameter"""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    df = process_results(sample_results, mock_shap_values, mock_correlations)
    create_plots(
        df,
        output_dir,
        n_rules=2,
        test_size=1000,
        max_set_size=3,
        top_features=5,
        n_splits=3,
        total_features=7,
    )

    file_names = {f.name for f in output_dir.glob("*.pdf")}
    expected_names = {
        "relations_plot_max3_top5_splits3_test1000.pdf",
        "dimensions_plot_max3_top5_splits3_test1000.pdf",
        "dimension_distribution_max3_splits3_test1000.pdf",
    }
    assert file_names == expected_names


def test_calculate_max_relations_basic():
    """Test basic cases for maximum relations calculation"""
    assert calculate_max_relations(N=3, max_set_size=2, top_features=2) == 7
    assert calculate_max_relations(N=2, max_set_size=1, top_features=1) == 1


def test_calculate_max_relations_larger_cases():
    """Test larger cases with manually verified results"""

    assert calculate_max_relations(N=4, max_set_size=2, top_features=2) == 15


def test_calculate_max_relations_parameter_relationships():
    """Test relationships between different parameter combinations"""

    n4_result = calculate_max_relations(N=4, max_set_size=2, top_features=2)
    n5_result = calculate_max_relations(N=5, max_set_size=2, top_features=2)
    assert n5_result > n4_result

    top2_result = calculate_max_relations(N=5, max_set_size=2, top_features=2)
    top3_result = calculate_max_relations(N=5, max_set_size=2, top_features=3)
    assert top3_result > top2_result

    size2_result = calculate_max_relations(N=5, max_set_size=2, top_features=3)
    size3_result = calculate_max_relations(N=5, max_set_size=3, top_features=3)
    assert size3_result > size2_result
