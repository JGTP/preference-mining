import pytest
import numpy as np
from src.feature_set_analysis import FeatureSetAnalyzer, FeatureSetComparison


@pytest.fixture
def sample_importances():
    """Fixture providing sample feature importance scores"""
    return {"feature_a": 0.4, "feature_b": 0.3, "feature_c": 0.2, "feature_d": 0.1}


@pytest.fixture
def analyzer():
    """Fixture providing a FeatureSetAnalyzer instance"""
    return FeatureSetAnalyzer(min_set_size=1, max_set_size=3)


def test_normalize_importances(analyzer, sample_importances):
    """Test that importance scores are properly normalized"""
    normalized = analyzer._normalize_importances(sample_importances)

    # Check that values sum to 1
    assert abs(sum(normalized.values()) - 1.0) < 1e-10

    # Check relative proportions are maintained
    assert normalized["feature_a"] > normalized["feature_b"]
    assert normalized["feature_b"] > normalized["feature_c"]
    assert normalized["feature_c"] > normalized["feature_d"]


def test_normalize_importances_empty(analyzer):
    """Test normalization with empty dictionary"""
    empty_importances = {}
    normalized = analyzer._normalize_importances(empty_importances)
    assert normalized == {}


def test_calculate_set_importance(analyzer, sample_importances):
    """Test calculation of importance scores for feature sets"""
    normalized = analyzer._normalize_importances(sample_importances)

    # Test single feature
    single_set = {"feature_a"}
    assert (
        analyzer._calculate_set_importance(single_set, normalized)
        == normalized["feature_a"]
    )

    # Test multiple features
    multi_set = {"feature_a", "feature_b"}
    expected = normalized["feature_a"] + normalized["feature_b"]
    assert (
        abs(analyzer._calculate_set_importance(multi_set, normalized) - expected)
        < 1e-10
    )


def test_generate_valid_sets(analyzer):
    """Test generation of valid feature sets"""
    features = ["feature_a", "feature_b", "feature_c"]
    sets = analyzer._generate_valid_sets(features)

    # Check number of sets
    # For min_size=1, max_size=3, with 3 features:
    # 1-feature sets: 3
    # 2-feature sets: 3
    # 3-feature sets: 1
    # Total: 7 sets
    assert len(sets) == 7

    # Check set sizes respect constraints
    for s in sets:
        assert len(s) >= analyzer.min_set_size
        assert len(s) <= analyzer.max_set_size


def test_find_significant_pairs(analyzer, sample_importances):
    """Test finding of significant feature set pairs"""
    pairs = analyzer.find_significant_pairs(
        importances=sample_importances, delta=0.2, max_pairs=10
    )

    # Check that returned objects are of correct type
    assert all(isinstance(pair, FeatureSetComparison) for pair in pairs)

    # Check that pairs have no overlapping features
    for pair in pairs:
        assert len(pair.better_set & pair.worse_set) == 0

    # Check that differences exceed delta
    for pair in pairs:
        assert pair.importance_difference >= 0.2


def test_find_significant_pairs_no_results(analyzer):
    """Test finding significant pairs with high delta threshold"""
    pairs = analyzer.find_significant_pairs(
        importances={"feature_a": 0.5, "feature_b": 0.5},
        delta=0.6,  # No pairs should have this large a difference
    )
    assert len(pairs) == 0


def test_max_set_size_constraint():
    """Test that max_set_size constraint is respected"""
    analyzer = FeatureSetAnalyzer(min_set_size=1, max_set_size=2)
    features = ["a", "b", "c", "d"]
    sets = analyzer._generate_valid_sets(features)

    # With max_size=2, no sets should have more than 2 features
    assert all(len(s) <= 2 for s in sets)

    # Should include all 1-feature and 2-feature combinations
    # 1-feature sets: 4
    # 2-feature sets: 6
    # Total: 10 sets
    assert len(sets) == 10


def test_empty_importances(analyzer):
    """Test behavior with empty importance dictionary"""
    pairs = analyzer.find_significant_pairs(importances={}, delta=0.1)
    assert len(pairs) == 0


import pytest
from pathlib import Path
import json
import datetime
from src.feature_set_analysis import analyze_feature_set_differences


@pytest.fixture
def sample_conditional_results():
    """Fixture providing sample conditional importance results"""
    return {
        "conditional_importances": {
            "rule_1": {
                "feature_importances": {
                    "feature_a": 0.4,
                    "feature_b": 0.3,
                    "feature_c": 0.2,
                    "feature_d": 0.1,
                }
            },
            "rule_2": {
                "feature_importances": {
                    "feature_x": 0.5,
                    "feature_y": 0.3,
                    "feature_z": 0.2,
                }
            },
        }
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture providing temporary directory for output files"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


def test_analyze_feature_set_differences_basic(
    sample_conditional_results, temp_output_dir
):
    """Test basic functionality of analyze_feature_set_differences"""
    deltas = [0.2, 0.3]
    result = analyze_feature_set_differences(
        conditional_results=sample_conditional_results,
        deltas=deltas,
        output_dir=temp_output_dir,
        min_set_size=1,
        max_set_size=2,
        max_pairs_per_rule=5,
    )

    # Check that output contains expected keys
    assert "output_path" in result
    assert "results" in result

    # Check that results contain both rules
    assert "rule_1" in result["results"]
    assert "rule_2" in result["results"]

    # Check that each rule has results for both deltas
    for rule in result["results"].values():
        assert str(deltas[0]) in rule
        assert str(deltas[1]) in rule


def test_output_file_structure(sample_conditional_results, temp_output_dir):
    """Test structure and content of output JSON file"""
    deltas = [0.2]
    result = analyze_feature_set_differences(
        conditional_results=sample_conditional_results,
        deltas=deltas,
        output_dir=temp_output_dir,
        max_pairs_per_rule=10,  # Add explicit max_pairs value
    )

    # Check that output file exists
    output_path = Path(result["output_path"])
    assert output_path.exists()

    # Read and validate output file content
    with open(output_path) as f:
        output_data = json.load(f)

    # Check metadata structure
    assert "metadata" in output_data
    assert "export_date" in output_data["metadata"]
    assert "deltas_analyzed" in output_data["metadata"]
    assert "min_set_size" in output_data["metadata"]
    assert "max_set_size" in output_data["metadata"]

    # Check results structure
    assert "results" in output_data
    assert isinstance(output_data["results"], dict)


def test_feature_set_pair_structure(sample_conditional_results, temp_output_dir):
    """Test structure of individual feature set pairs in results"""
    result = analyze_feature_set_differences(
        conditional_results=sample_conditional_results,
        deltas=[0.2],
        output_dir=temp_output_dir,
        max_pairs_per_rule=1,
    )

    # Get pairs for first rule and delta
    rule_name = list(result["results"].keys())[0]
    pairs = result["results"][rule_name]["0.2"]

    # Check pair structure if any pairs were found
    if pairs:
        pair = pairs[0]
        assert "better_set" in pair
        assert "worse_set" in pair
        assert "better_importance" in pair
        assert "worse_importance" in pair
        assert "importance_difference" in pair
        assert isinstance(pair["better_set"], list)
        assert isinstance(pair["worse_set"], list)
        assert isinstance(pair["better_importance"], float)
        assert isinstance(pair["worse_importance"], float)
        assert isinstance(pair["importance_difference"], float)


def test_empty_conditional_results(temp_output_dir):
    """Test behavior with empty conditional results"""
    empty_results = {"conditional_importances": {}}
    result = analyze_feature_set_differences(
        conditional_results=empty_results, deltas=[0.2], output_dir=temp_output_dir
    )

    assert result["results"] == {}
    assert Path(result["output_path"]).exists()


def test_max_pairs_constraint(sample_conditional_results, temp_output_dir):
    """Test that max_pairs_per_rule constraint is respected"""
    max_pairs = 2
    result = analyze_feature_set_differences(
        conditional_results=sample_conditional_results,
        deltas=[0.2],
        output_dir=temp_output_dir,
        max_pairs_per_rule=max_pairs,
    )

    # Check that no rule has more than max_pairs pairs
    for rule in result["results"].values():
        assert len(rule["0.2"]) <= max_pairs


def test_invalid_output_directory():
    """Test behavior with invalid output directory"""
    with pytest.raises(Exception):
        analyze_feature_set_differences(
            conditional_results={"conditional_importances": {}},
            deltas=[0.2],
            output_dir=Path("/nonexistent/directory"),
        )
