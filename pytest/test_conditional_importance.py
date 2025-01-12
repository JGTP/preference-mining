import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from datetime import datetime
from src.conditional_importance import (
    apply_rule_conditions,
    analyze_conditional_importance,
    calculate_subset_performance,
    preprocess_subset,
    ConditionalImportanceResult,
    export_conditional_importance,
)
from src.preprocessing import DataPreprocessor
from pathlib import Path


@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance for testing"""
    return DataPreprocessor(
        categorical_columns=["feature2"],
        numeric_categorical_columns=["feature3"],
        one_hot_encoding=True,
    )


@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.choice(["A", "B", "C"], n_samples),
        "feature3": np.random.randint(0, 5, n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def processed_data(sample_data, preprocessor):
    """Create preprocessed sample data"""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]
    X_processed, y_processed = preprocessor.preprocess_data(X, y)
    return X_processed, y_processed, X, y


@pytest.fixture
def sample_ruleset():
    """Create sample ruleset for testing"""
    return {
        "metadata": {"export_date": datetime.now().isoformat(), "total_rules": 2},
        "rules": [
            {
                "if_conditions": ["feature2 = A"],
                "then_prediction": "1",
                "metrics": {"support": 100},
            },
            {
                "if_conditions": ["feature3 = 1", "feature1 <= 0.5"],
                "then_prediction": "0",
                "metrics": {"support": 50},
            },
        ],
    }


@pytest.fixture
def trained_model(processed_data):
    """Create trained model for testing"""
    X_processed, y_processed, _, _ = processed_data
    model = HistGradientBoostingClassifier(random_state=42, max_iter=10)
    model.fit(X_processed, y_processed)
    return model


def test_apply_rule_conditions_single_categorical():
    """Test applying a single categorical condition"""
    data = pd.DataFrame({"feature": ["A", "B", "A", "C"]})
    mask = apply_rule_conditions(data, ["feature = A"])
    expected = pd.Series([True, False, True, False])
    pd.testing.assert_series_equal(mask, expected)


def test_apply_rule_conditions_single_numeric():
    """Test applying a single numeric condition"""
    data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    mask = apply_rule_conditions(data, ["feature <= 2.0"])
    expected = pd.Series([True, True, False, False])
    pd.testing.assert_series_equal(mask, expected)


def test_preprocess_subset(sample_data, preprocessor):
    """Test preprocessing of data subset"""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]

    X_processed, y_processed = preprocess_subset(X, y, preprocessor)

    assert isinstance(X_processed, pd.DataFrame)
    assert isinstance(y_processed, pd.Series)
    assert len(X_processed) == len(y_processed)

    # Test that None target raises ValueError
    with pytest.raises(ValueError):
        preprocess_subset(X, None, preprocessor)


def test_calculate_subset_performance(trained_model, processed_data):
    """Test calculation of subset performance metrics"""
    X_processed, y_processed, _, _ = processed_data

    # Take a small subset for testing
    X_subset = X_processed.iloc[:100]
    y_subset = y_processed.iloc[:100]

    performance = calculate_subset_performance(trained_model, X_subset, y_subset)

    assert isinstance(performance, dict)
    assert "accuracy" in performance
    assert "subset_positive_ratio" in performance
    assert "prediction_positive_ratio" in performance
    assert all(0 <= v <= 1 for v in performance.values())


# def test_analyze_conditional_importance(
#     processed_data, sample_ruleset, trained_model, preprocessor
# ):
#     """Test complete conditional importance analysis"""
#     X_processed, y_processed, X_raw, y = processed_data

#     # Test with valid input
#     results = analyze_conditional_importance(
#         X=X_raw,
#         y=y,
#         ruleset=sample_ruleset,
#         model=trained_model,
#         preprocessor=preprocessor,
#         min_subset_size=10,  # Small for testing
#     )

#     assert isinstance(results, list)
#     assert len(results) > 0
#     assert all(isinstance(r, ConditionalImportanceResult) for r in results)

#     # Check first result structure
#     first_result = results[0]
#     assert isinstance(first_result.rule_conditions, list)
#     assert isinstance(first_result.feature_importance, dict)
#     assert all(isinstance(v, float) for v in first_result.feature_importance.values())

#     # Test that None target raises ValueError
#     with pytest.raises(ValueError):
#         analyze_conditional_importance(
#             X=X_raw,
#             y=None,
#             ruleset=sample_ruleset,
#             model=trained_model,
#             preprocessor=preprocessor,
#             min_subset_size=10,
#         )


def test_export_conditional_importance(
    tmp_path, processed_data, sample_ruleset, trained_model, preprocessor
):
    """Test exporting results to JSON"""
    X_processed, y_processed, X_raw, y = processed_data

    results = analyze_conditional_importance(
        X=X_raw,
        y=y,
        ruleset=sample_ruleset,
        model=trained_model,
        preprocessor=preprocessor,
        min_subset_size=10,
    )

    export_path = export_conditional_importance(
        results=results, output_dir=tmp_path, filename="test_output.json"
    )

    assert export_path.exists()
    assert export_path.suffix == ".json"

    # Verify JSON structure
    import json

    with open(export_path) as f:
        exported_data = json.load(f)

    assert "metadata" in exported_data
    assert "conditional_importance" in exported_data
    assert len(exported_data["conditional_importance"]) == len(results)
