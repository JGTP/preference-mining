import pytest
from src.visualisation import calculate_max_relations


def test_calculate_max_relations_basic():
    """Test basic cases with manually calculated expected values

    Case N=3, max_set_size=2, top_features=2:
    set1 possibilities (from top 2 features):
    - size 1: (2 choose 1) = 2 combinations
    - size 2: (2 choose 2) = 1 combination

    For each set1 of size 1 (2 of these):
    - can have set2 of size 1: (2 remaining features choose 1) = 2 possibilities
    - can have set2 of size 2: (2 remaining features choose 2) = 1 possibility
    Total for size 1: 2 * (2 + 1) = 6

    For set1 of size 2 (1 of these):
    - can have set2 of size 1: (1 remaining feature choose 1) = 1 possibility
    Total for size 2: 1 * 1 = 1

    Grand total: 6 + 1 = 7 relations
    """
    assert calculate_max_relations(N=3, max_set_size=2, top_features=2) == 7

    """
    Case N=4, max_set_size=2, top_features=2:
    set1 possibilities (from top 2 features):
    - size 1: (2 choose 1) = 2 combinations
    - size 2: (2 choose 2) = 1 combination
    
    For each set1 of size 1 (2 of these):
    - can have set2 of size 1: (3 remaining features choose 1) = 3 possibilities
    - can have set2 of size 2: (3 remaining features choose 2) = 3 possibilities
    Total for size 1: 2 * (3 + 3) = 12
    
    For set1 of size 2 (1 of these):
    - can have set2 of size 1: (2 remaining features choose 1) = 2 possibilities
    - can have set2 of size 2: (2 remaining features choose 2) = 1 possibility
    Total for size 2: 1 * (2 + 1) = 3
    
    Grand total: 12 + 3 = 15 relations
    """
    assert calculate_max_relations(N=4, max_set_size=2, top_features=2) == 15


def test_calculate_max_relations_simple_cases():
    """Test very simple cases with easily verifiable results

    Case N=2, max_set_size=1, top_features=1:
    set1 possibilities: only 1 feature can be chosen
    set2 possibilities: only 1 remaining feature can be chosen
    Total: 1 relation
    """
    assert calculate_max_relations(N=2, max_set_size=1, top_features=1) == 1

    """
    Case N=3, max_set_size=1, top_features=2:
    set1 possibilities: can choose either of top 2 features = 2 possibilities
    For each set1: can choose 1 from remaining 2 features = 2 possibilities
    Total: 2 * 2 = 4 relations
    """
    assert calculate_max_relations(N=3, max_set_size=1, top_features=2) == 4


def test_calculate_max_relations_parameter_relationships():
    """Test relationships between different parameter combinations with exact values"""
    # Calculate specific values for N=5, max_set_size=2, top_features=3:
    # set1 possibilities:
    # - size 1: (3 choose 1) = 3 combinations
    # - size 2: (3 choose 2) = 3 combinations
    # For each size 1 set1 (3 of these):
    # - can have set2 of size 1: (4 choose 1) = 4 possibilities
    # - can have set2 of size 2: (4 choose 2) = 6 possibilities
    # For each size 2 set1 (3 of these):
    # - can have set2 of size 1: (3 choose 1) = 3 possibilities
    # - can have set2 of size 2: (3 choose 2) = 3 possibilities
    # Total: 3 * (4 + 6) + 3 * (3 + 3) = 30 + 18 = 48
    result1 = calculate_max_relations(N=5, max_set_size=2, top_features=3)
    assert result1 == 48

    # Calculate for N=5, max_set_size=2, top_features=4
    # Similar calculation but with more top features
    result2 = calculate_max_relations(N=5, max_set_size=2, top_features=4)
    assert result2 > result1  # Should have more relations with more top features


def test_calculate_max_relations_edge_cases():
    """Test edge cases with calculated expected values"""
    # When max_set_size=1, top_features=1, only one relation possible per remaining feature
    assert (
        calculate_max_relations(N=3, max_set_size=1, top_features=1) == 2
    )  # Can pair with either of 2 remaining features

    # When N=2, can only form minimal relations
    assert (
        calculate_max_relations(N=2, max_set_size=2, top_features=2) == 2
    )  # Only one possible pairing


def test_calculate_max_relations_larger_cases():
    """Test larger cases where we can still manually verify the results

    Case N=6, max_set_size=2, top_features=3:
    set1 possibilities:
    - size 1: (3 choose 1) = 3 combinations
    - size 2: (3 choose 2) = 3 combinations

    For each size 1 set1 (3 of these):
    - can have set2 of size 1: (5 choose 1) = 5 possibilities
    - can have set2 of size 2: (5 choose 2) = 10 possibilities
    Total for size 1: 3 * (5 + 10) = 45

    For each size 2 set1 (3 of these):
    - can have set2 of size 1: (4 choose 1) = 4 possibilities
    - can have set2 of size 2: (4 choose 2) = 6 possibilities
    Total for size 2: 3 * (4 + 6) = 30

    Grand total: 45 + 30 = 75 relations
    """
    assert calculate_max_relations(N=6, max_set_size=2, top_features=3) == 75
