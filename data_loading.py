from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_COLUMN_ALIASES: Dict[str, str] = {
    "Temperature_C": "temperature",
    "Humidity": "humidity",
    "Rainfall_mm": "rainfall",
    "Soil_Type": "soil_type",
    "Crop_Type": "crop_type",
    "Season": "season",
    "Irrigation_Need": "water_requirement_raw",
    "evapotranspiration": "evapotranspiration",
    "Evapotranspiration": "evapotranspiration",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    remap = {c: DEFAULT_COLUMN_ALIASES[c] for c in df.columns if c in DEFAULT_COLUMN_ALIASES}
    return df.rename(columns=remap)


def _to_continuous_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    cleaned = series.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().mean() > 0.95:
        return numeric.fillna(numeric.median())

    ranking = {"low": 25.0, "medium": 45.0, "high": 70.0}
    mapped = cleaned.map(ranking)
    if mapped.notna().all():
        return mapped

    categories = sorted(cleaned.unique().tolist())
    cat_scale = np.linspace(20.0, 80.0, num=max(len(categories), 2))
    auto_map = {cat: cat_scale[idx] for idx, cat in enumerate(categories)}
    return cleaned.map(auto_map).astype(float)


def load_irrigation_data(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    target_candidates = [
        "water_requirement",
        "water_requirement_raw",
        "target",
    ]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError("No supported target column found. Expected one of water_requirement / Irrigation_Need / target.")

    if target_col != "water_requirement":
        df["water_requirement"] = _to_continuous_target(df[target_col])
    else:
        df["water_requirement"] = _to_continuous_target(df["water_requirement"])

    leakage_cols = {"water_requirement_raw", "target_raw"}
    for col in leakage_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df, "water_requirement"
