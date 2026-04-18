from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor(X_train: pd.DataFrame) -> Tuple[ColumnTransformer, Dict[str, list]]:
    categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    schema = {"categorical_cols": categorical_cols, "numeric_cols": numeric_cols}
    return preprocessor, schema

