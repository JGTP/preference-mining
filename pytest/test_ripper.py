import pandas as pd
from src.ripper import train_RIPPER_classifier


def test_train_RIPPER_classifier():
    X_train = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    y_train = [0, 1, 0, 1]
    df_train = pd.DataFrame(X_train, columns=["feature1", "feature2"])
    df_train["target"] = y_train
    ripper, rules = train_RIPPER_classifier(
        df_train[["feature1", "feature2"]], df_train["target"]
    )
    assert ripper is not None
    assert len(rules) > 0
