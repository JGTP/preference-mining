import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.feature_set_analysis import EnhancedFeatureAnalyser
from sklearn.ensemble import HistGradientBoostingClassifier


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_a": np.random.randn(100),
            "feature_b": np.random.randn(100),
            "feature_c": np.random.randn(100),
            "feature_d": np.random.randn(100),
        }
    )


@pytest.fixture
def trained_model():
    model = HistGradientBoostingClassifier(random_state=42)
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    return model


@pytest.fixture
def mock_temp_dir(tmpdir):
    return Path(tmpdir)


@pytest.fixture
def analyser_with_disk_cache(sample_data, trained_model, mock_temp_dir):
    return EnhancedFeatureAnalyser(
        model=trained_model,
        X=sample_data,
        epsilons=[0.1, 0.2],
        deltas=[0.1, 0.2],
        max_set_size=3,
        enable_disk_cache=True,
        temp_dir=mock_temp_dir,
    )


@pytest.fixture
def analyser_without_disk_cache(sample_data, trained_model):
    return EnhancedFeatureAnalyser(
        model=trained_model,
        X=sample_data,
        epsilons=[0.1, 0.2],
        deltas=[0.1, 0.2],
        max_set_size=3,
        enable_disk_cache=False,
    )


def test_initialisation_with_disk_cache(analyser_with_disk_cache):
    assert len(analyser_with_disk_cache.epsilons) == 2
    assert len(analyser_with_disk_cache.deltas) == 2
    assert analyser_with_disk_cache.max_set_size == 3
    assert len(analyser_with_disk_cache.feature_names) == 4
    assert isinstance(analyser_with_disk_cache.correlation_matrix, pd.DataFrame)
    assert analyser_with_disk_cache.temp_dir is not None
    assert analyser_with_disk_cache.temp_dir.exists()
    assert (analyser_with_disk_cache.temp_dir / "shap_values.json").exists()
    assert (analyser_with_disk_cache.temp_dir / "correlations.json").exists()


def test_initialisation_without_disk_cache(analyser_without_disk_cache):
    assert len(analyser_without_disk_cache.epsilons) == 2
    assert len(analyser_without_disk_cache.deltas) == 2
    assert analyser_without_disk_cache.max_set_size == 3
    assert len(analyser_without_disk_cache.feature_names) == 4
    assert isinstance(analyser_without_disk_cache.correlation_matrix, pd.DataFrame)
    assert analyser_without_disk_cache.temp_dir is None
    assert analyser_without_disk_cache.shap_values is not None
    assert analyser_without_disk_cache.correlations is not None


def test_cleanup(analyser_with_disk_cache, mock_temp_dir):

    temp_dir_path = analyser_with_disk_cache.temp_dir

    assert temp_dir_path.exists()
    assert (temp_dir_path / "shap_values.json").exists()
    assert (temp_dir_path / "correlations.json").exists()

    analyser_with_disk_cache.cleanup()

    assert not temp_dir_path.exists()

    test_files = [
        temp_dir_path / "shap_values.json",
        temp_dir_path / "correlations.json",
    ]
    assert all(not f.exists() for f in test_files)


def test_no_cleanup_without_disk_cache(analyser_without_disk_cache):
    analyser_without_disk_cache.cleanup()


def test_shap_values_calculation(analyser_without_disk_cache):
    shap_values = analyser_without_disk_cache.shap_values
    assert isinstance(shap_values, dict)
    assert len(shap_values) == len(analyser_without_disk_cache.feature_names)
    assert all(isinstance(v, float) for v in shap_values.values())


def test_correlation_calculation(analyser_without_disk_cache):
    correlations = analyser_without_disk_cache.correlations
    assert isinstance(correlations, dict)
    assert all(isinstance(v, float) for v in correlations.values())


def test_feature_combinations(analyser_without_disk_cache):
    combinations = analyser_without_disk_cache.feature_combinations
    assert isinstance(combinations, dict)
    assert "set1" in combinations
    assert "set2" in combinations
    assert all(isinstance(combo, dict) for combo in combinations["set1"])
    assert all(isinstance(combo, dict) for combo in combinations["set2"])


def test_analyse_rule(analyser_without_disk_cache):

    class MockRule:
        def __init__(self):
            self.conds = []

        def __str__(self):
            return "mock_rule"

    rule = MockRule()
    result = analyser_without_disk_cache.analyse_rule(rule)
    assert isinstance(result, dict)


def test_analyse_ruleset(analyser_without_disk_cache):

    class MockRule:
        def __init__(self):
            self.conds = []

        def __str__(self):
            return "mock_rule"

    ruleset = [MockRule(), MockRule()]
    result = analyser_without_disk_cache.analyse_ruleset(ruleset)
    assert isinstance(result, dict)
    assert "rule_analyses" in result
    assert "metadata" in result
    assert "epsilons" in result["metadata"]
    assert "deltas" in result["metadata"]
    assert "max_set_size" in result["metadata"]


def test_empty_ruleset(analyser_without_disk_cache):
    result = analyser_without_disk_cache.analyse_ruleset([])
    assert result["rule_analyses"] == {}
    assert "metadata" in result


def test_invalid_max_set_size(sample_data, trained_model):
    analyser = EnhancedFeatureAnalyser(
        model=trained_model, X=sample_data, max_set_size=0, enable_disk_cache=False
    )
    assert analyser.max_set_size == 0


def test_error_handling_model_none(sample_data):
    with pytest.raises(Exception):
        EnhancedFeatureAnalyser(model=None, X=sample_data, enable_disk_cache=False)


def test_directory_creation_with_disk_cache(sample_data, trained_model, mock_temp_dir):
    nested_dir = mock_temp_dir / "nested" / "temp"
    analyser = EnhancedFeatureAnalyser(
        model=trained_model, X=sample_data, enable_disk_cache=True, temp_dir=nested_dir
    )
    assert nested_dir.exists()
    assert (nested_dir / "shap_values.json").exists()
    assert (nested_dir / "correlations.json").exists()
    analyser.cleanup()


def test_feature_importance_consistency(analyser_without_disk_cache):

    shap_values = analyser_without_disk_cache.shap_values
    combinations = analyser_without_disk_cache.feature_combinations

    for feature in analyser_without_disk_cache.feature_names:
        single_feature_sets = [
            combo
            for combo in combinations["set1"]
            if len(combo["features"]) == 1 and feature in combo["features"]
        ]
        if single_feature_sets:
            assert (
                abs(single_feature_sets[0]["importance"] - shap_values[feature]) < 1e-10
            )
