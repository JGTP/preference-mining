import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.black_box import ModelTrainer
import pandas as pd


@pytest.fixture
def synthetic_data():
    """Generate synthetic dataset with categorical features."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],
        random_state=42,
    )

    # Convert some features to categorical
    X = make_some_features_categorical(X, n_categorical_features=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def edge_case_data():
    """Generate edge case dataset with only one class."""
    X = np.random.randn(100, 5)
    y = np.zeros(100)  # All samples belong to class 0
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Discretise the first feature (example)
def make_some_features_categorical(X, n_categorical_features=5):
    X_df = pd.DataFrame(X)

    for i in range(n_categorical_features):
        X_df[i] = pd.cut(X_df[i], bins=4, labels=["low", "medium", "high", "very high"])

    return X_df


def test_model_training(synthetic_data):
    """Test the complete model training and evaluation pipeline."""
    X_train, X_test, y_train, y_test = synthetic_data

    # Initialize trainer with all numerical features
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

    # Test evaluation
    test_metrics = trainer.evaluate(X_test, y_test)
    for metric, score in test_metrics.items():
        assert isinstance(score, float)
        assert 0 <= score <= 1


def test_predictions(synthetic_data):
    """Test model predictions."""
    X_train, X_test, y_train, y_test = synthetic_data

    trainer = ModelTrainer()
    trainer.train(X_train, y_train)

    # Test predict
    predictions = trainer.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)

    # Test predict_proba
    probabilities = trainer.predict_proba(X_test)
    assert probabilities.shape == (len(y_test), 2)
    assert np.allclose(np.sum(probabilities, axis=1), 1)


def test_empty_data():
    """Test handling of empty data."""
    X = np.array([]).reshape(0, 5)
    y = np.array([])

    trainer = ModelTrainer()

    # Should handle empty data gracefully
    cv_metrics = trainer.evaluate_cv(X, y)
    for metric, (mean_score, std_score) in cv_metrics.items():
        assert isinstance(mean_score, float)
        assert isinstance(std_score, float)
        assert mean_score == 0
        assert std_score == 0
