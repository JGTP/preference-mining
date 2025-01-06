import pandas as pd
from src.classifier import train_RIPPER_classifier


def test_train_RIPPER_classifier():
    # Dummy dataset
    X_train = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    y_train = [0, 1, 0, 1]

    # Prepare data as DataFrame
    df_train = pd.DataFrame(X_train, columns=["feature1", "feature2"])
    df_train["target"] = y_train

    classifier, rules = train_RIPPER_classifier(
        df_train[["feature1", "feature2"]], df_train["target"]
    )

    assert classifier is not None  # Ensure classifier is trained
    assert len(rules) > 0  # Ensure rules are extracted
