import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.visualisation import process_results, create_plot

@pytest.fixture
def mock_temp_dir(tmpdir):
    return Path(tmpdir)

@pytest.fixture
def sample_results():
    rule_analyses = {}
    for i in range(10):
        rule_key = f"rule_{i}"
        rule_analyses[rule_key] = {"rule_string": f"[test_rule_{i}]", "analysis": {}}
        for j in range(2):  # Only 2 epsilon values: 0.1 and 0.2
            for k in range(5):  # 5 delta values
                epsilon = 0.1 * (j + 1)  # Will give 0.1 and 0.2
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
    
    # Create sample SHAP values
    shap_values = {f"feature{i}": 0.1 * i for i in range(10)}
    
    # Create sample correlations
    correlations = {
        f"[feature{i},feature{j}]": 0.1 
        for i in range(10) 
        for j in range(i + 1, 10)
    }

    # Write the files
    with open(mock_temp_dir / "shap_values.json", "w") as f:
        json.dump(shap_values, f)
    with open(mock_temp_dir / "correlations.json", "w") as f:
        json.dump(correlations, f)
    
    return mock_temp_dir

def test_process_results(sample_results, setup_temp_files):
    # Process the results
    df = process_results(sample_results, setup_temp_files)
    
    # Basic DataFrame checks
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ["epsilon", "delta", "N", "W", "B"])
    assert len(df) > 0
    
    # Check data types
    assert df["epsilon"].dtype in [float, "float64"]
    assert df["delta"].dtype in [float, "float64"]
    assert df["N"].dtype in [float, "float64", int, "int64"]
    assert df["W"].dtype in [float, "float64"]
    assert df["B"].dtype in [float, "float64"]
    
    # Check value ranges
    assert all(0 <= x <= 1 for x in df["epsilon"])
    assert all(0 <= x <= 1 for x in df["delta"])
    assert all(x >= 0 for x in df["N"])
    assert all(x >= 0 for x in df["W"])
    assert all(x >= 0 for x in df["B"])
    
    # Check for expected number of unique values
    assert len(df["epsilon"].unique()) == 2  # As set in sample_results
    assert len(df["delta"].unique()) == 5    # As set in sample_results

def test_create_plot(sample_results, setup_temp_files):
    # Create test plots directory under pytest
    plots_dir = Path("pytest/test_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Process results and create plot
    df = process_results(sample_results, setup_temp_files)
    plot_path = plots_dir / "test_plot.pdf"
    
    # Test plot creation
    create_plot(df, plot_path)
    
    # Verify plot was created and has content
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    
    # Note: Not removing the files to allow for inspection

def test_process_results_with_missing_files(sample_results, mock_temp_dir):
    with pytest.raises(FileNotFoundError):
        process_results(sample_results, mock_temp_dir)

def test_process_results_with_invalid_json(mock_temp_dir):
    mock_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create files with invalid JSON
    with open(mock_temp_dir / "shap_values.json", "w") as f:
        f.write("invalid json")
    with open(mock_temp_dir / "correlations.json", "w") as f:
        f.write("invalid json")
    
    with pytest.raises(json.JSONDecodeError):
        process_results({"rule_analyses": {}}, mock_temp_dir)

def test_process_results_empty_input(setup_temp_files):
    empty_results = {"rule_analyses": {}}
    df = process_results(empty_results, setup_temp_files)
    
    # Check that we get an empty DataFrame
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    # The DataFrame's columns are created only when there's data to process
    # so we don't assert specific columns for empty input

def test_create_plot_minimal_data():
    # Create test plots directory under pytest
    plots_dir = Path("pytest/test_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal valid dataset (one data point per required epsilon)
    minimal_data = []
    for i, eps in enumerate([0.1, 0.2]):  # Only two epsilon values
        minimal_data.append({
            'epsilon': eps,
            'delta': 0.1,
            'N': 1,
            'W': 1,
            'B': 1
        })
    df = pd.DataFrame(minimal_data)
    plot_path = plots_dir / "minimal_plot.pdf"
    
    # Test plot creation with minimal data
    create_plot(df, plot_path)
    
    # Verify plot was created
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    
    # Note: Not removing the files to allow for inspection