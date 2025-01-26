import pandas as pd
import numpy as np
import pytest
from src.preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "temperature": [25.5, -99, 27.8, 26.2, -99],
            "humidity": [60, 55, -9, 65, 70],
            "order_date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "shipping_date": [
                "2024-01-02",
                "2024-01-03",
                np.nan,
                "2024-01-06",
                "2024-01-07",
            ],
            "latitude": [40.7128, 34.0522, 41.8781, 29.7604, 45.5155],
            "longitude": [-74.0060, -118.2437, -87.6298, -95.3698, -122.6789],
            "product_category": ["A", "B", "A", "C", "B"],
            "region": ["North", "South", "North", "East", "West"],
            "status_code": [1, 2, 1, 3, 2],
            "notes": ["good", "-", "excellent", "NA", ""],
            "target": [0, 1, 0, 1, 0],
        }
    )


def test_initialisation():
    preprocessor = DataPreprocessor(
        date_columns=["order_date", "shipping_date"],
        coordinate_columns=["latitude", "longitude"],
        categorical_columns=["product_category", "region"],
        numeric_categorical_columns=["status_code"],
        columns_to_exclude=["notes"],
        missing_value_codes={"temperature": [-99], "humidity": [-9]},
    )
    assert preprocessor.date_columns == ["order_date", "shipping_date"]
    assert preprocessor.missing_value_codes["temperature"] == [-99]


def test_missing_value_handling(sample_data):
    preprocessor = DataPreprocessor(
        missing_value_codes={"-99": ["temperature"], "-9": ["humidity"]}
    )
    processed_data = preprocessor._handle_missing_values(sample_data)
    assert pd.isna(processed_data.loc[1, "temperature"])
    assert pd.isna(processed_data.loc[2, "humidity"])
    assert pd.isna(processed_data.loc[1, "notes"])


def test_categorical_encoding():
    test_data = pd.DataFrame(
        {
            "product_category": ["A", "B", "A", "C", "B"],
            "region": ["North", "South", "North", "East", "West"],
            "status_code": [1, 2, 1, 3, 2],
        }
    )
    preprocessor = DataPreprocessor(
        categorical_columns=["product_category", "region"],
        numeric_categorical_columns=["status_code"],
        one_hot_encoding=True,
    )
    processed_data = preprocessor._encode_categoricals(test_data)
    assert "product_category_A" in processed_data.columns
    assert "region_North" in processed_data.columns
    assert "status_code_1" in processed_data.columns
    assert processed_data["product_category_A"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert processed_data["region_North"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert processed_data["status_code_1"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
