from typing import Tuple

import numpy as np
import pandas as pd

from utils import safe_divide


def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return ""


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    temp_col = _find_col(data, ("temperature", "Temperature", "temperature_c"))
    hum_col = _find_col(data, ("humidity", "Humidity"))
    rain_col = _find_col(data, ("rainfall", "Rainfall", "rainfall_mm"))
    et_col = _find_col(data, ("evapotranspiration", "ET", "et"))

    if temp_col and hum_col:
        data["temp_x_humidity"] = data[temp_col] * data[hum_col]

    if et_col and rain_col:
        data["evapotranspiration_ratio"] = safe_divide(
            data[et_col].to_numpy(dtype=float), data[rain_col].to_numpy(dtype=float)
        )
    elif temp_col and hum_col and rain_col:
        et_proxy = (0.35 * data[temp_col] + 0.25 * (100 - data[hum_col]) + 0.1).clip(lower=0.1)
        data["evapotranspiration_ratio"] = safe_divide(
            et_proxy.to_numpy(dtype=float), data[rain_col].to_numpy(dtype=float) + 1.0
        )

    if "season" in data.columns:
        season_norm = data["season"].astype(str).str.strip().str.lower()
        data["is_kharif"] = (season_norm == "kharif").astype(int)
        data["is_rabi"] = (season_norm == "rabi").astype(int)
        data["is_zaid"] = (season_norm == "zaid").astype(int)

    return data

