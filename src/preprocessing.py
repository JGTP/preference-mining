import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Tuple, Optional, Dict, List, Union
from src.utils import CacheManager


class DataPreprocessor:
    def __init__(
        self,
        date_columns: List[str] = None,
        coordinate_columns: List[str] = None,
        categorical_columns: List[str] = None,
        numeric_categorical_columns: List[str] = None,
        columns_to_exclude: List[str] = None,
        missing_value_codes: Dict[str, List] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        min_year: Optional[int] = None,
        year_column: Optional[str] = None,
        test_size: Optional[int] = None,
        one_hot_encoding: bool = True,
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
        self.cache_manager = CacheManager(cache_dir)
        self.label_encoders = {}
        self.onehot_encoder = None
        self.scaler = StandardScaler()
        self.target_processor = None
        self.target_type = None

    def _get_config(self):
        return {
            "date_columns": self.date_columns,
            "coordinate_columns": self.coordinate_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_categorical_columns": self.numeric_categorical_columns,
            "columns_to_exclude": self.columns_to_exclude,
            "missing_value_codes": self.missing_value_codes,
            "min_year": self.min_year,
            "year_column": self.year_column,
            "one_hot_encoding": self.one_hot_encoding,
            "test_size": self.test_size,
        }

    def _preprocess_target(self, y: pd.Series) -> pd.Series:
        if self.target_type is None:
            self.target_type = (
                "categorical"
                if y.dtype == "object" or y.dtype.name == "category"
                else "numerical"
            )
        if self.target_type == "categorical":
            if self.target_processor is None:
                self.target_processor = LabelEncoder()
                return pd.Series(self.target_processor.fit_transform(y), index=y.index)
            return pd.Series(self.target_processor.transform(y), index=y.index)
        y = pd.to_numeric(y, errors="coerce")
        if self.missing_value_codes:
            for code in self.missing_value_codes:
                try:
                    y = y.replace(float(code), np.nan)
                except ValueError:
                    continue
        return y.map(bool)

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def _process_coordinates(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for i in range(0, len(self.coordinate_columns), 2):
            if i + 1 < len(self.coordinate_columns):
                lat_col = self.coordinate_columns[i]
                lon_col = self.coordinate_columns[i + 1]
                if lat_col in df.columns and lon_col in df.columns:
                    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
                    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        return df

    def _encode_categoricals(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def _filter_by_year(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.min_year is None or self.year_column is None:
            return data
        if self.year_column not in data.columns:
            raise ValueError(f"Year column '{self.year_column}' not found in data")
        data[self.year_column] = pd.to_numeric(data[self.year_column], errors="coerce")
        return data[data[self.year_column] >= self.min_year].copy()

    def preprocess_data(
        self, data_path: Union[str, Path], target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.cache_manager.cache_dir:
            return self._process_full_dataset(pd.read_excel(data_path), target_column)
        cache_key = self.cache_manager.get_cache_key(data_path, self._get_config())
        cached_result = self.cache_manager.load_data(cache_key)
        if cached_result is not None:
            print("\nUsing cached preprocessed data")
            return cached_result
        print("\nCache missing - preprocessing full dataset")
        X, y = self._process_full_dataset(pd.read_excel(data_path), target_column)
        self.cache_manager.save_data((X, y), cache_key)
        return X, y

    def _process_full_dataset(
        self, data: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = data.copy()
        if self.columns_to_exclude:
            df = df.drop(
                columns=[col for col in self.columns_to_exclude if col in df.columns]
            )
        df = self._filter_by_year(df)
        if isinstance(target_column, str) and target_column in df.columns:
            y = self._preprocess_target(df[target_column].copy())
            df = df.drop(columns=[target_column])
        else:
            raise ValueError("Target column not found in dataset")
        df = self._handle_missing_values(df)
        df = self._encode_categoricals(df)
        df = self._process_coordinates(df)
        if self.test_size:
            indices = np.random.choice(len(df), size=self.test_size, replace=False)
            df = df.iloc[indices]
            y = y.iloc[indices]
        return df, y

    def inverse_transform_target(
        self, y: Union[pd.Series, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        if self.target_processor is None:
            return y
        if self.target_type == "categorical":
            return self.target_processor.inverse_transform(y)
        return self.target_processor.inverse_transform(y.reshape(-1, 1)).ravel()
