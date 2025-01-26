import pytest
import pandas as pd
import numpy as np
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
def analyser(sample_data, trained_model):
    return EnhancedFeatureAnalyser(
        model=trained_model,
        X=sample_data,
        epsilons=[0.1, 0.2],
        deltas=[0.1, 0.2],
        max_set_size=3,
    )


def test_initialisation(analyser):
    assert len(analyser.epsilons) == 2
    assert len(analyser.deltas) == 2
    assert analyser.max_set_size == 3
    assert len(analyser.feature_names) == 4
    assert isinstance(analyser.correlation_matrix, pd.DataFrame)
    assert hasattr(analyser, "shap_scale")


def test_shap_cache(analyser, sample_data):
    result1 = analyser._calculate_shap_values(sample_data)
    cache_size1 = len(analyser._shap_cache)
    assert cache_size1 > 0

    result2 = analyser._calculate_shap_values(sample_data)
    cache_size2 = len(analyser._shap_cache)
    assert cache_size2 == cache_size1
    assert result1 == result2


def test_feature_combinations_precomputation(analyser):
    assert hasattr(analyser, "feature_combinations")
    assert len(analyser.feature_combinations) == analyser.max_set_size
    for size in range(1, analyser.max_set_size + 1):
        assert isinstance(analyser.feature_combinations[size], list)
        assert all(
            isinstance(combo, set) for combo in analyser.feature_combinations[size]
        )


def test_parallel_rule_analysis(analyser):
    class MockRule:
        def __init__(self):
            self.conds = []

        def __str__(self):
            return "mock_rule"

    ruleset = [MockRule() for _ in range(5)]
    result = analyser.analyse_ruleset(ruleset)

    assert isinstance(result, dict)
    assert "rule_analyses" in result
    assert len(result["rule_analyses"]) <= len(ruleset)


def test_correlation_threshold(analyser):
    test_data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
            "c": [1, 3, 2, 4],
            "d": [-1, -2, -3, -4],
        }
    )
    analyser.correlation_matrix = test_data.corr()

    assert not analyser._check_correlation_threshold({"a", "b"}, epsilon=0.5)
    assert analyser._check_correlation_threshold({"a", "c"}, epsilon=0.9)
    assert not analyser._check_correlation_threshold({"a", "d"}, epsilon=0.5)
    assert not analyser._check_correlation_threshold({"a", "b", "c"}, epsilon=0.5)
    assert analyser._check_correlation_threshold({"a"}, epsilon=0.5)
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


def test_calculate_shap_values(analyser, sample_data):
    result = analyser._calculate_shap_values(sample_data)
    assert isinstance(result, dict)
    assert len(result) == len(sample_data.columns)
    assert all(isinstance(v, float) for v in result.values())
    assert all(v >= 0 for v in result.values())


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


def test_empty_ruleset(analyser):
    result = analyser.analyse_ruleset([])
    assert result["rule_analyses"] == {}
    assert "metadata" in result


def test_invalid_max_set_size(sample_data, trained_model):
    analyser = EnhancedFeatureAnalyser(
        model=trained_model, X=sample_data, max_set_size=0
    )
    assert analyser.max_set_size == 0
