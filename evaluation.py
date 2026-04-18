from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_models(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows: List[Dict[str, float]] = []
    predictions: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        metrics = regression_metrics(y_test.to_numpy(), y_pred)
        rows.append({"model": name, **metrics})

    metrics_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return metrics_df, predictions


def plot_predicted_vs_actual(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=16, edgecolor="none")
    min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Actual Water Requirement")
    plt.ylabel("Predicted Water Requirement")
    plt.title("Predicted vs Actual Water Requirement")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _extract_rf_feature_importance(rf_pipeline, X_columns: List[str]) -> pd.DataFrame:
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    model = rf_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out(X_columns)
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(rf_pipeline, X_columns: List[str], output_path: Path, top_n: int = 15) -> pd.DataFrame:
    fi = _extract_rf_feature_importance(rf_pipeline, X_columns)
    top = fi.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Random Forest Feature Importance (Top {top_n})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return fi

