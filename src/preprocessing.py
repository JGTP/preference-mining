import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime


class DataPreprocessor:
    def __init__(
        self,
        date_columns=None,
        coordinate_columns=None,
        categorical_columns=None,
        numeric_categorical_columns=None,
        columns_to_exclude=None,
        missing_value_codes=None,
    ):
        """
        Initialize the preprocessor with column specifications.

        Args:
            date_columns (list): Columns containing date information
            coordinate_columns (list): Columns containing geographic coordinates
            categorical_columns (list): Columns containing categorical data as strings
            numeric_categorical_columns (list): Columns containing categorical data as numbers
            columns_to_exclude (list): Columns to exclude from processing
            missing_value_codes (dict): Dictionary mapping columns to their missing value codes
                                      (e.g., {'column_name': [-9, -99]})
        """
        self.date_columns = date_columns or []
        self.coordinate_columns = coordinate_columns or []
        self.categorical_columns = categorical_columns or []
        self.numeric_categorical_columns = numeric_categorical_columns or []
        self.columns_to_exclude = columns_to_exclude or []
        self.missing_value_codes = missing_value_codes or {}

        self.label_encoders = {}
        self.onehot_encoder = None
        self.scaler = StandardScaler()

    def _handle_missing_values(self, data):
        """Replace missing values and special codes with NaN."""
        df = data.copy()

        # Replace common missing value indicators
        df.replace(["", "-", "NA", "N/A", "unknown"], np.nan, inplace=True)

        # Replace missing value codes specific to columns
        for column, codes in self.missing_value_codes.items():
            if column in df.columns:
                df[column].replace(codes, np.nan, inplace=True)

        return df

    def _process_dates(self, data):
        """Extract useful features from date columns."""
        df = data.copy()

        for col in self.date_columns:
            if col in df.columns:
                # Convert to datetime
                df[col] = pd.to_datetime(df[col], errors="coerce")

                # Extract useful components
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek

                # Drop original date column
                df.drop(columns=[col], inplace=True)

        return df

    def _process_coordinates(self, data):
        """Process geographic coordinate columns."""
        df = data.copy()

        # Assume coordinates come in pairs (lat, lon)
        for i in range(0, len(self.coordinate_columns), 2):
            if i + 1 < len(self.coordinate_columns):
                lat_col = self.coordinate_columns[i]
                lon_col = self.coordinate_columns[i + 1]

                if lat_col in df.columns and lon_col in df.columns:
                    # Convert to float if needed
                    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
                    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

                    # Could add derived features like distance from reference point
                    # or clustering-based location categories

        return df

    def _encode_categoricals(self, data):
        """Encode categorical variables using Label and OneHot encoding."""
        df = data.copy()

        # Handle string categoricals
        for col in self.categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))

        # Handle numeric categoricals
        for col in self.numeric_categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Apply one-hot encoding to all categorical columns
        categorical_features = (
            self.categorical_columns + self.numeric_categorical_columns
        )
        if categorical_features:
            self.onehot_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            encoded_features = self.onehot_encoder.fit_transform(
                df[categorical_features]
            )

            # Create DataFrame with encoded features
            feature_names = self.onehot_encoder.get_feature_names_out(
                categorical_features
            )
            encoded_df = pd.DataFrame(
                encoded_features, columns=feature_names, index=df.index
            )

            # Drop original categorical columns and concatenate encoded features
            df = df.drop(columns=categorical_features)
            df = pd.concat([df, encoded_df], axis=1)

        return df

    def preprocess_data(self, data, target_column):
        """
        Main preprocessing pipeline.

        Args:
            data (pd.DataFrame): Input data
            target_column (str): Name of the target variable column

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) if target_column is provided,
            otherwise returns preprocessed features X
        """
        df = data.copy()
        y = target_column
        if target_column and target_column in df.columns:
            y = df[target_column].copy()
            df = df.drop(columns=[target_column])

        # Remove excluded columns
        df = df.drop(columns=self.columns_to_exclude, errors="ignore")

        # Apply preprocessing steps
        df = self._handle_missing_values(df)
        df = self._process_dates(df)
        df = self._process_coordinates(df)
        df = self._encode_categoricals(df)

        # Scale numerical features
        numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
        if len(numerical_columns) > 0:
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])

        if target_column is None:
            return df

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test
