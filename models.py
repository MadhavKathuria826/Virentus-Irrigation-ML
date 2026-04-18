from typing import Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def build_models(preprocessor) -> Dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)),
            ]
        ),
    }
