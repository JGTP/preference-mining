import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.black_box import (
    ModelTrainer,
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic dataset with numerical features only."""
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],
        random_state=42,
    )

    # Convert to DataFrame with meaningful feature names
    feature_names = [f"feature_{i}" for i in range(20)]
    X = pd.DataFrame(X, columns=feature_names)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def edge_case_data():
    """Generate edge case datasets for testing."""
    # Single class dataset
    X_single = pd.DataFrame(np.random.randn(100, 5))
    y_single = np.zeros(100)

    # Empty dataset
    X_empty = pd.DataFrame(columns=[f"feature_{i}" for i in range(5)])
    y_empty = pd.Series(dtype=float)

    # Dataset with missing values
    X_missing = pd.DataFrame(np.random.randn(100, 5))
    X_missing.iloc[10:20, 0] = np.nan
    y_missing = np.random.randint(0, 2, 100)

    return {
        "single_class": (X_single, y_single),
        "empty": (X_empty, y_empty),
        "missing": (X_missing, y_missing),
    }


def test_basic_training(synthetic_data):
    """Test the complete model training and evaluation pipeline."""
    X_train, X_test, y_train, y_test = synthetic_data

    trainer = ModelTrainer()

    # Test cross-validation
    cv_metrics = trainer.evaluate_cv(X_train, y_train)
    for metric, (mean_score, std_score) in cv_metrics.items():
        assert isinstance(mean_score, float)
        assert isinstance(std_score, float)
        assert 0 <= mean_score <= 1
        assert std_score >= 0

    # Test model training
    trainer.train(X_train, y_train)
    importance_dict = trainer.get_shap_feature_importances(X_train)
    assert "feature_importances" in importance_dict
    assert len(importance_dict["feature_importances"]) == X_train.shape[1]

    # Test evaluation
    test_metrics = trainer.evaluate(X_test, y_test)
    for metric, score in test_metrics.items():
        assert isinstance(score, float)
        assert 0 <= score <= 1


def test_empty_data(edge_case_data):
    """Test handling of empty datasets."""
    X_empty, y_empty = edge_case_data["empty"]
    trainer = ModelTrainer()

    # Should handle empty data gracefully in cross-validation
    cv_metrics = trainer.evaluate_cv(X_empty, y_empty)
    for metric, (mean_score, std_score) in cv_metrics.items():
        assert mean_score == 0.0
        assert std_score == 0.0

    # Should raise ValueError when trying to train
    with pytest.raises(ValueError):
        trainer.train(X_empty, y_empty)


def test_single_class(edge_case_data):
    """Test handling of single-class datasets."""
    X_single, y_single = edge_case_data["single_class"]
    trainer = ModelTrainer()

    # Should handle single class data appropriately
    cv_metrics = trainer.evaluate_cv(X_single, y_single)
    for metric, (mean_score, std_score) in cv_metrics.items():
        assert isinstance(mean_score, float)
        assert isinstance(std_score, float)

    # Should be able to train and predict
    trainer.train(X_single, y_single)
    predictions = trainer.predict(X_single)
    assert all(predictions == 0)


def test_predictions(synthetic_data):
    """Test model predictions."""
    X_train, X_test, y_train, y_test = synthetic_data

    trainer = ModelTrainer()
    trainer.train(X_train, y_train)

    # Test predict
    predictions = trainer.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)
    assert set(predictions).issubset({0, 1})

    # Test predict_proba
    probabilities = trainer.predict_proba(X_test)
    assert probabilities.shape == (len(y_test), 2)
    assert np.allclose(np.sum(probabilities, axis=1), 1)
    assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_missing_value_predictions(edge_case_data):
    """Test predictions with missing values."""
    X_missing, y_missing = edge_case_data["missing"]

    trainer = ModelTrainer()
    trainer.train(X_missing, y_missing)

    # Should handle missing values in prediction
    predictions = trainer.predict(X_missing)
    assert len(predictions) == len(y_missing)

    probabilities = trainer.predict_proba(X_missing)
    assert probabilities.shape == (len(y_missing), 2)


def test_importance_calculation(synthetic_data):
    """Test feature importance calculation and storage."""
    X_train, _, y_train, _ = synthetic_data
    # Train model and get importances
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    importance_dict = trainer.get_shap_feature_importances(X_train)

    # Validate importance structure and values
    assert "feature_importances" in importance_dict
    assert "metadata" in importance_dict
    assert len(importance_dict["feature_importances"]) == X_train.shape[1]

    importances = importance_dict["feature_importances"]
    assert all(isinstance(v, float) for v in importances.values())
    assert all(v >= 0 for v in importances.values())

    # Test saving importances with X parameter
    output_path = trainer.save_feature_importances(X_train)
    assert output_path.exists()
