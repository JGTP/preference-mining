from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import hashlib
import json


class DataPreprocessor:
    def __init__(
        self,
        date_columns=None,
        coordinate_columns=None,
        categorical_columns=None,
        numeric_categorical_columns=None,
        columns_to_exclude=None,
        missing_value_codes=None,
        cache_dir=None,
        min_year=None,
        year_column=None,
        test_size=None,
        one_hot_encoding=True,
    ):
        self.date_columns = date_columns or []
        self.coordinate_columns = coordinate_columns or []
        self.categorical_columns = categorical_columns or []
        self.numeric_categorical_columns = numeric_categorical_columns or []
        self.columns_to_exclude = columns_to_exclude or []
        self.missing_value_codes = missing_value_codes or {}
        self.min_year = min_year
        self.year_column = year_column
        self.test_size = test_size
        self.one_hot_encoding = one_hot_encoding
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        self.label_encoders = {}
        self.onehot_encoder = None
        self.scaler = StandardScaler()
        self.target_processor = None
        self.target_type = None

    def _preprocess_target(self, y):
        """
        Preprocess the target variable based on its type.

        Args:
            y (pd.Series): Target variable

        Returns:
            pd.Series: Processed target variable
        """
        if self.target_type is None:
            if y.dtype == "object" or y.dtype.name == "category":
                self.target_type = "categorical"
            else:
                self.target_type = "numerical"

        if self.target_type == "categorical":
            if self.target_processor is None:
                self.target_processor = LabelEncoder()
                return pd.Series(self.target_processor.fit_transform(y), index=y.index)
            return pd.Series(self.target_processor.transform(y), index=y.index)
        else:
            y = pd.to_numeric(y, errors="coerce")
            if self.missing_value_codes:
                for code, columns in self.missing_value_codes.items():
                    try:
                        numeric_code = float(code)
                        y = y.replace(numeric_code, np.nan)
                    except ValueError:
                        continue

        return y.map(bool)

    def _generate_cache_key(self, data, target_column):
        """Generate a unique cache key based on data content and preprocessing parameters"""

        config = {
            "date_columns": sorted(self.date_columns),
            "coordinate_columns": sorted(self.coordinate_columns),
            "categorical_columns": sorted(self.categorical_columns),
            "numeric_categorical_columns": sorted(self.numeric_categorical_columns),
            "columns_to_exclude": sorted(self.columns_to_exclude),
            "missing_value_codes": self.missing_value_codes,
            "target_column": target_column,
            "data_shape": data.shape,
            "columns": sorted(data.columns.tolist()),
            "data_hash": hashlib.md5(
                pd.util.hash_pandas_object(data).values
            ).hexdigest(),
        }

        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _save_preprocessed_data(self, X, y, cache_key):
        """Save preprocessed data and preprocessing objects to cache"""
        cache_path = self.cache_dir / f"{cache_key}"

        joblib.dump(
            {
                "X": X,
                "y": y,
                "onehot_encoder": self.onehot_encoder,
                "scaler": self.scaler,
                "target_processor": self.target_processor,
                "target_type": self.target_type,
            },
            cache_path,
        )

    def _load_preprocessed_data(self, cache_key):
        """Load preprocessed data and preprocessing objects from cache"""
        cache_path = self.cache_dir / f"{cache_key}"

        if not cache_path.exists():
            return None

        try:
            cached_data = joblib.load(cache_path)
            self.onehot_encoder = cached_data["onehot_encoder"]
            self.scaler = cached_data["scaler"]
            self.target_processor = cached_data["target_processor"]
            self.target_type = cached_data["target_type"]

            return (
                cached_data["X"],
                cached_data["y"],
            )
        except Exception as e:
            print(f"Error loading cached data: {e}")
            return None

    def _handle_missing_values(self, data):
        """Replace missing values and special codes with NaN."""
        df = data.copy()
        df.replace(
            ["", "-", "NA", "N/A", "unknown", "NaN", "nan"], np.nan, inplace=True
        )

        for code, affected_columns in self.missing_value_codes.items():
            try:
                numeric_code = float(code)
            except ValueError:
                numeric_code = code

            for column in affected_columns:
                if column in df.columns:
                    df[column] = df[column].replace(numeric_code, np.nan)

        return df

    def _process_coordinates(self, data):
        """Process geographic coordinate columns."""
        df = data.copy()

        for i in range(0, len(self.coordinate_columns), 2):
            if i + 1 < len(self.coordinate_columns):
                lat_col = self.coordinate_columns[i]
                lon_col = self.coordinate_columns[i + 1]

                if lat_col in df.columns and lon_col in df.columns:
                    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
                    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

        return df

    def _encode_categoricals(self, data):
        """Encode categorical variables using OneHot encoding directly."""
        df = data.copy()

        categorical_features = (
            self.categorical_columns + self.numeric_categorical_columns
        )

        for col in self.numeric_categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype("str")

        if categorical_features and self.one_hot_encoding:
            self.onehot_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            encoded_features = self.onehot_encoder.fit_transform(
                df[categorical_features]
            )

            feature_names = self.onehot_encoder.get_feature_names_out(
                categorical_features
            )
            encoded_df = pd.DataFrame(
                encoded_features, columns=feature_names, index=df.index
            )

            df = df.drop(columns=categorical_features)
            df = pd.concat([df, encoded_df], axis=1)

        return df

    def _filter_by_year(self, data):
        """
        Filter data to include only records from or after the specified minimum year.

        Args:
            data (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if self.min_year is None or self.year_column is None:
            return data

        if self.year_column not in data.columns:
            raise ValueError(f"Year column '{self.year_column}' not found in data")

        data[self.year_column] = pd.to_numeric(data[self.year_column], errors="coerce")

        filtered_data = data[data[self.year_column] >= self.min_year].copy()

        print(
            f"Filtered data from {self.min_year} onwards. "
            f"Rows before: {len(data)}, Rows after: {len(filtered_data)}"
        )

        return filtered_data

    def preprocess_data(self, data, target_column):
        """
        Main preprocessing pipeline with caching support.
        """
        if self.cache_dir:
            cache_key = self._generate_cache_key(data, target_column)
            cached_result = self._load_preprocessed_data(cache_key)
            if cached_result is not None:
                print("Using cached preprocessed data")
                return cached_result

        print("Cache missing, preprocessing data")

        df = data.copy()

        df = self._filter_by_year(df)

        if target_column is not None:
            if isinstance(target_column, pd.Series):
                # It's already a pandas series.
                y = target_column
            elif target_column and target_column in df.columns:
                y = self._preprocess_target(df[target_column].copy())
                df = df.drop(columns=[target_column])

        df = df.drop(columns=self.columns_to_exclude, errors="ignore")

        df = self._handle_missing_values(df)
        df = self._encode_categoricals(df)
        df = self._process_coordinates(df)

        if self.test_size:

            indices = np.random.choice(len(df), size=self.test_size, replace=False)
            df = df.iloc[indices]
            y = y.iloc[indices]

        if self.cache_dir:
            self._save_preprocessed_data(df, y, cache_key)

        return df, y

    def inverse_transform_target(self, y):
        """
        Convert preprocessed target values back to original scale/categories.

        Args:
            y: Preprocessed target values

        Returns:
            Target values in original scale/categories
        """
        if self.target_processor is None:
            return y

        if self.target_type == "categorical":
            return self.target_processor.inverse_transform(y)
        else:
            return self.target_processor.inverse_transform(y.reshape(-1, 1)).ravel()
