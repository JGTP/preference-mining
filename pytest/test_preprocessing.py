import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample dataset with various data types and scenarios"""
    return pd.DataFrame(
        {
            # Numerical columns
            "temperature": [25.5, -99, 27.8, 26.2, -99],
            "humidity": [60, 55, -9, 65, 70],
            # Date columns
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
            # Coordinate columns
            "latitude": [40.7128, 34.0522, 41.8781, 29.7604, 45.5155],
            "longitude": [-74.0060, -118.2437, -87.6298, -95.3698, -122.6789],
            # String categorical columns
            "product_category": ["A", "B", "A", "C", "B"],
            "region": ["North", "South", "North", "East", "West"],
            # Numeric categorical columns
            "status_code": [1, 2, 1, 3, 2],
            # Various missing value formats
            "notes": ["good", "-", "excellent", "NA", ""],
            # Target column
            "target": [0, 1, 0, 1, 0],
        }
    )


def test_initialization():
    """Test preprocessor initialization"""
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
    """Test handling of various missing value formats"""
    preprocessor = DataPreprocessor(
        missing_value_codes={"temperature": [-99], "humidity": [-9]}
    )

    processed_data = preprocessor._handle_missing_values(sample_data)

    # Check if -99 was converted to NaN
    assert pd.isna(processed_data.loc[1, "temperature"])
    # Check if -9 was converted to NaN
    assert pd.isna(processed_data.loc[2, "humidity"])
    # Check if dash was converted to NaN
    assert pd.isna(processed_data.loc[1, "notes"])


def test_date_processing(sample_data):
    """Test date feature extraction"""
    preprocessor = DataPreprocessor(date_columns=["order_date", "shipping_date"])

    processed_data = preprocessor._process_dates(sample_data)

    # Check if new date features were created
    assert "order_date_year" in processed_data.columns
    assert "order_date_month" in processed_data.columns
    assert "order_date_day" in processed_data.columns

    # Check if original date columns were dropped
    assert "order_date" not in processed_data.columns

    # Verify extracted values
    assert processed_data.loc[0, "order_date_year"] == 2024
    assert processed_data.loc[0, "order_date_month"] == 1
    assert processed_data.loc[0, "order_date_day"] == 1


def test_categorical_encoding():
    """Test categorical encoding with a simple dataset"""
    # Create a simple test dataset
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
    )

    processed_data = preprocessor._encode_categoricals(test_data)

    # Print the columns for debugging
    print("\nProcessed columns:", processed_data.columns.tolist())

    # Check specific encoded columns exist
    assert "product_category_A" in processed_data.columns
    assert "region_North" in processed_data.columns
    assert "status_code_1" in processed_data.columns

    # Verify the values for a specific category
    assert processed_data["product_category_A"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert processed_data["region_North"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert processed_data["status_code_1"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]


def test_full_pipeline(sample_data):
    """Test the complete preprocessing pipeline"""
    preprocessor = DataPreprocessor(
        date_columns=["order_date", "shipping_date"],
        coordinate_columns=["latitude", "longitude"],
        categorical_columns=["product_category", "region"],
        numeric_categorical_columns=["status_code"],
        columns_to_exclude=["notes"],
        missing_value_codes={"temperature": [-99], "humidity": [-9]},
    )

    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
        sample_data, target_column="target"
    )

    # Check basic shapes
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)

    # Check if excluded columns are removed
    assert "notes" not in X_train.columns

    # Check if all column types are properly transformed
    assert all(X_train.dtypes != "object")  # No untransformed object columns
    assert not X_train.isnull().any().any()  # No missing values

    # Check if categorical columns are properly encoded
    assert any(col.startswith("product_category_") for col in X_train.columns)
    assert any(col.startswith("region_") for col in X_train.columns)
    assert any(col.startswith("status_code_") for col in X_train.columns)

    # Check if date features exist
    assert "order_date_year" in X_train.columns
    assert "shipping_date_month" in X_train.columns


def test_preprocessing_without_target(sample_data):
    """Test preprocessing without splitting into train/test"""
    preprocessor = DataPreprocessor(
        date_columns=["order_date"], categorical_columns=["product_category"]
    )

    processed_data = preprocessor.preprocess_data(sample_data)

    # Check if it returns a single DataFrame
    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) == len(sample_data)
    assert "order_date_year" in processed_data.columns
    assert "product_category_A" in processed_data.columns
