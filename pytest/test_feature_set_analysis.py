import pytest
import pandas as pd
import numpy as np
from src.feature_set_analysis import EnhancedFeatureAnalyzer
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
def analyzer(sample_data, trained_model):
    return EnhancedFeatureAnalyzer(
        model=trained_model,
        X=sample_data,
        epsilons=[0.1, 0.2],
        deltas=[0.1, 0.2],
        max_set_size=3,
    )


def test_initialization(analyzer):
    assert len(analyzer.epsilons) == 2
    assert len(analyzer.deltas) == 2
    assert analyzer.max_set_size == 3
    assert len(analyzer.feature_names) == 4
    assert isinstance(analyzer.correlation_matrix, pd.DataFrame)
    assert hasattr(analyzer, "shap_scale")


def test_correlation_threshold(analyzer):
    feature_set = {"feature_a", "feature_b"}
    result = analyzer._check_correlation_threshold(feature_set, epsilon=0.5)
    assert isinstance(result, bool)


def test_aggregate_importance(analyzer):
    importances = {
        "feature_a": 0.4,
        "feature_b": 0.3,
        "feature_c": 0.2,
        "feature_d": 0.1,
    }
    feature_set = {"feature_a", "feature_b"}
    result = analyzer._aggregate_importance(feature_set, importances)
    assert isinstance(result, float)
    assert result == pytest.approx(0.7)


def test_calculate_shap_values(analyzer, sample_data):
    result = analyzer._calculate_shap_values(sample_data)
    assert isinstance(result, dict)
    assert len(result) == len(sample_data.columns)
    assert all(isinstance(v, float) for v in result.values())
    assert all(v >= 0 for v in result.values())


def test_analyze_rule(analyzer):
    class MockRule:
        def __init__(self):
            self.conds = []

    rule = MockRule()
    result = analyzer.analyze_rule(rule)
    assert isinstance(result, dict)


def test_analyze_ruleset(analyzer):
    class MockRule:
        def __init__(self):
            self.conds = []

        def __str__(self):
            return "mock_rule"

    ruleset = [MockRule(), MockRule()]
    result = analyzer.analyze_ruleset(ruleset)

    assert isinstance(result, dict)
    assert "rule_analyses" in result
    assert "metadata" in result
    assert "epsilons" in result["metadata"]
    assert "deltas" in result["metadata"]
    assert "max_set_size" in result["metadata"]


def test_get_conditional_data(analyzer, sample_data):
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
    result = analyzer._get_conditional_data(rule)
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(sample_data)


def test_empty_ruleset(analyzer):
    result = analyzer.analyze_ruleset([])
    assert result["rule_analyses"] == {}
    assert "metadata" in result


def test_invalid_max_set_size(sample_data, trained_model):
    analyzer = EnhancedFeatureAnalyzer(
        model=trained_model, X=sample_data, max_set_size=0
    )
    assert analyzer.max_set_size == 0
