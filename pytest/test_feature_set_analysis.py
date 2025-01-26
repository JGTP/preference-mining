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
def analyser(sample_data, trained_model, mock_temp_dir):
    return EnhancedFeatureAnalyser(
        model=trained_model,
        X=sample_data,
        epsilons=[0.1, 0.2],
        deltas=[0.1, 0.2],
        max_set_size=3,
        temp_dir=mock_temp_dir,
    )


def test_initialisation(analyser):
    assert len(analyser.epsilons) == 2
    assert len(analyser.deltas) == 2
    assert analyser.max_set_size == 3
    assert len(analyser.feature_names) == 4
    assert isinstance(analyser.correlation_matrix, pd.DataFrame)


def test_temp_file_management(analyser):
    assert analyser.temp_dir.exists()
    assert (analyser.temp_dir / "shap_values.json").exists()
    assert (analyser.temp_dir / "correlations.json").exists()

    temp_dir = analyser.temp_dir
    analyser.cleanup()
    assert not temp_dir.exists()


def test_shap_values_persistence(analyser):
    with open(analyser.temp_dir / "shap_values.json", "r") as f:
        stored_values = json.load(f)
    assert isinstance(stored_values, dict)
    assert len(stored_values) == len(analyser.feature_names)
    assert all(isinstance(v, float) for v in stored_values.values())


def test_correlation_persistence(analyser):
    with open(analyser.temp_dir / "correlations.json", "r") as f:
        stored_correlations = json.load(f)
    assert isinstance(stored_correlations, dict)
    assert all(isinstance(v, float) for v in stored_correlations.values())


def test_analyser_with_temp_dir(sample_data, trained_model, mock_temp_dir):
    analyser = EnhancedFeatureAnalyser(
        model=trained_model, X=sample_data, temp_dir=mock_temp_dir
    )
    assert analyser.temp_dir == mock_temp_dir
    assert (mock_temp_dir / "shap_values.json").exists()
    analyser.cleanup()


def test_error_handling_temp_files(sample_data, trained_model, mock_temp_dir):
    with pytest.raises(Exception):
        EnhancedFeatureAnalyser(model=None, X=sample_data, temp_dir=mock_temp_dir)
    assert not (mock_temp_dir / "shap_values.json").exists()


def test_correlation_threshold(analyser):
    test_data = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [2, 4, 6, 8],
            "feature_c": [1, 3, 2, 4],
            "feature_d": [-1, -2, -3, -4],
        }
    )
    analyser.correlation_matrix = test_data.corr()
    analyser._store_correlations()

    assert not analyser._check_correlation_threshold(
        {"feature_a", "feature_b"}, epsilon=0.5
    )
    assert analyser._check_correlation_threshold(
        {"feature_a", "feature_c"}, epsilon=0.9
    )
    assert not analyser._check_correlation_threshold(
        {"feature_a", "feature_d"}, epsilon=0.5
    )
    assert not analyser._check_correlation_threshold(
        {"feature_a", "feature_b", "feature_c"}, epsilon=0.5
    )
    assert analyser._check_correlation_threshold({"feature_a"}, epsilon=0.5)
    assert analyser._check_correlation_threshold(set(), epsilon=0.5)


def test_aggregate_importance(analyser):
    importances = {
        "feature_a": 0.4,
        "feature_b": 0.3,
        "feature_c": 0.2,
        "feature_d": 0.1,
    }
    feature_set = {"feature_a", "feature_b"}
    result = analyser._aggregate_importance(feature_set, importances)
    assert isinstance(result, float)
    assert result == pytest.approx(0.7)


def test_get_conditional_data(analyser, sample_data):
    class MockCondition:
        def __init__(self):
            self.feature = 0
            self.val = 0.5

        def __str__(self):
            return "feature_a <= 0.5"

    class MockRule:
        def __init__(self):
            self.conds = [MockCondition()]

    rule = MockRule()
    result = analyser._get_conditional_data(rule)
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(sample_data)


def test_analyse_rule(analyser):
    class MockRule:
        def __init__(self):
            self.conds = []

    rule = MockRule()
    result = analyser.analyse_rule(rule)
    assert isinstance(result, dict)


def test_analyse_ruleset(analyser):
    class MockRule:
        def __init__(self):
            self.conds = []

        def __str__(self):
            return "mock_rule"

    ruleset = [MockRule(), MockRule()]
    result = analyser.analyse_ruleset(ruleset)
    assert isinstance(result, dict)
    assert "rule_analyses" in result
    assert "metadata" in result
    assert "epsilons" in result["metadata"]
    assert "deltas" in result["metadata"]
    assert "max_set_size" in result["metadata"]
    analyser.cleanup()


def test_empty_ruleset(analyser):
    result = analyser.analyse_ruleset([])
    assert result["rule_analyses"] == {}
    assert "metadata" in result
    analyser.cleanup()


def test_invalid_max_set_size(sample_data, trained_model, mock_temp_dir):
    analyser = EnhancedFeatureAnalyser(
        model=trained_model, X=sample_data, max_set_size=0, temp_dir=mock_temp_dir
    )
    assert analyser.max_set_size == 0
    analyser.cleanup()
