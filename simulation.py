from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _strategy_metrics(actual: np.ndarray, applied: np.ndarray) -> Dict[str, float]:
    diff = applied - actual
    total_used = float(np.sum(applied))
    overwatering = float(np.sum(np.clip(diff, a_min=0.0, a_max=None)))
    underwatering = float(np.sum(np.clip(-diff, a_min=0.0, a_max=None)))
    return {
        "total_water_usage": total_used,
        "overwatering_total": overwatering,
        "underwatering_total": underwatering,
    }


def run_simulation(y_true: pd.Series, y_pred: np.ndarray, fixed_amount: float) -> Dict[str, object]:
    actual = y_true.to_numpy(dtype=float)
    naive_applied = np.full_like(actual, fill_value=fixed_amount, dtype=float)
    ml_applied = np.clip(y_pred.astype(float), a_min=0.0, a_max=None)

    naive = _strategy_metrics(actual, naive_applied)
    ml = _strategy_metrics(actual, ml_applied)

    total_saved = naive["total_water_usage"] - ml["total_water_usage"]
    water_saved_pct = (total_saved / naive["total_water_usage"] * 100.0) if naive["total_water_usage"] else 0.0
    efficiency_improvement_pct = water_saved_pct

    return {
        "naive_strategy": naive,
        "ml_strategy": ml,
        "water_saved_total": float(total_saved),
        "water_saved_pct": float(water_saved_pct),
        "efficiency_improvement_pct": float(efficiency_improvement_pct),
        "error_distribution": {
            "naive_mean_error": float(np.mean(naive_applied - actual)),
            "ml_mean_error": float(np.mean(ml_applied - actual)),
            "naive_std_error": float(np.std(naive_applied - actual)),
            "ml_std_error": float(np.std(ml_applied - actual)),
        },
    }


def plot_water_usage_comparison(simulation_result: Dict[str, object], output_path: Path) -> None:
    naive_total = simulation_result["naive_strategy"]["total_water_usage"]
    ml_total = simulation_result["ml_strategy"]["total_water_usage"]

    plt.figure(figsize=(7, 5))
    labels = ["Naive", "ML-Based"]
    values = [naive_total, ml_total]
    bars = plt.bar(labels, values)
    plt.title("Total Water Usage Comparison")
    plt.ylabel("Water Used")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:,.0f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def simulation_metrics_row(simulation_result: Dict[str, object], best_model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": f"{best_model_name}_simulation",
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "water_saved_pct": simulation_result["water_saved_pct"],
                "efficiency_improvement_pct": simulation_result["efficiency_improvement_pct"],
                "naive_total_water_usage": simulation_result["naive_strategy"]["total_water_usage"],
                "ml_total_water_usage": simulation_result["ml_strategy"]["total_water_usage"],
                "naive_overwatering_total": simulation_result["naive_strategy"]["overwatering_total"],
                "ml_overwatering_total": simulation_result["ml_strategy"]["overwatering_total"],
                "naive_underwatering_total": simulation_result["naive_strategy"]["underwatering_total"],
                "ml_underwatering_total": simulation_result["ml_strategy"]["underwatering_total"],
            }
        ]
    )

