"""
Trend-Aware Feature Engineering Module for Battery Safety.

This module derives physical degradation trends and temporal features
from preprocessed NASA battery cycles, capturing capacity fade, internal
resistance growth, and thermal/voltage dynamics over rolling operational windows.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Ensure config can be imported from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def _polyfit_slope(y: np.ndarray) -> float:
    """
    Calculate the first-degree polynomial slope (linear rate of change) of a window.

    Parameters
    ----------
    y : np.ndarray
        Array of values in the window.

    Returns
    -------
    float
        Slope value or np.nan if calculation is not possible.
    """
    if len(y) < 2:
        return np.nan

    valid = ~np.isnan(y)
    if np.sum(valid) < 2:
        return np.nan

    x = np.arange(len(y))[valid]
    y_val = y[valid]

    try:
        slope, _ = np.polyfit(x, y_val, 1)
        return float(slope)
    except Exception:
        return np.nan


def compute_trend(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the linear slope of a series over a rolling window using np.polyfit.

    Parameters
    ----------
    series : pd.Series
        Input time-series or sequential cycle data.
    window : int
        Size of the rolling window in cycles. Must be >= 2.

    Returns
    -------
    pd.Series
        Series of linear slope values with same index as input (NaN for insufficient data).
    """
    if window < 2:
        raise ValueError(f"Window must be at least 2 for trend slope calculation, got {window}")

    if series.empty:
        return pd.Series(dtype=float, index=series.index)

    trend = series.rolling(window=window, min_periods=window).apply(_polyfit_slope, raw=True)
    trend.name = f"{series.name}_trend" if series.name else "trend"
    return trend


def compute_change_rate(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rate of change over a rolling window: (current - window_start) / window.

    Parameters
    ----------
    series : pd.Series
        Input sequential data.
    window : int
        Size of the window in number of cycles. Must be >= 1.

    Returns
    -------
    pd.Series
        Series of change rates with same index as input.
    """
    if window < 1:
        raise ValueError(f"Window must be >= 1 for change rate calculation, got {window}")

    if series.empty:
        return pd.Series(dtype=float, index=series.index)

    rate = (series - series.shift(window)) / float(window)
    rate.name = f"{series.name}_change_rate" if series.name else "change_rate"
    return rate


def compute_deviation(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the absolute deviation from the rolling average: |current - rolling_mean|.

    Parameters
    ----------
    series : pd.Series
        Input sequential data.
    window : int
        Size of the rolling window for the moving average. Must be >= 1.

    Returns
    -------
    pd.Series
        Series of absolute deviations with same index as input.
    """
    if window < 1:
        raise ValueError(f"Window must be >= 1 for deviation calculation, got {window}")

    if series.empty:
        return pd.Series(dtype=float, index=series.index)

    rolling_avg = series.rolling(window=window, min_periods=window).mean()
    deviation = (series - rolling_avg).abs()
    deviation.name = f"{series.name}_deviation" if series.name else "deviation"
    return deviation


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trend-aware degradation features separately for each battery cell.

    Features engineered per cell:
    - temp_trend: slope of temp_max over TREND_WINDOW_MEDIUM cycles
    - temp_change_rate: rate of temp_max change over TREND_WINDOW_SHORT cycles
    - voltage_trend: slope of voltage_drop over TREND_WINDOW_MEDIUM cycles
    - voltage_deviation: deviation of voltage_drop from TREND_WINDOW_LONG rolling avg
    - capacity_fade_pct: percentage capacity lost from cell initial capacity
    - capacity_fade_rate: slope of capacity over TREND_WINDOW_MEDIUM cycles
    - re_trend: slope of re over TREND_WINDOW_MEDIUM cycles
    - rct_trend: slope of rct over TREND_WINDOW_MEDIUM cycles
    - impedance_rise_pct: percentage rise of (re + rct) from cell initial value
    - soc_estimated: estimated State of Charge (capacity / initial_capacity * 100)

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed battery dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with all newly engineered features appended.
    """
    if df.empty:
        return df.copy()

    w_short = config.TREND_WINDOW_SHORT
    w_med = config.TREND_WINDOW_MEDIUM
    w_long = config.TREND_WINDOW_LONG

    def _engineer_cell_group(cell_group: pd.DataFrame) -> pd.DataFrame:
        grp = cell_group.sort_values("cycle_index").copy() if "cycle_index" in cell_group.columns else cell_group.copy()

        # 1. Thermal trends
        if "temp_max" in grp.columns:
            grp["temp_trend"] = compute_trend(grp["temp_max"], w_med)
            grp["temp_change_rate"] = compute_change_rate(grp["temp_max"], w_short)
        else:
            grp["temp_trend"] = np.nan
            grp["temp_change_rate"] = np.nan

        # 2. Voltage trends & deviation
        if "voltage_drop" in grp.columns:
            grp["voltage_trend"] = compute_trend(grp["voltage_drop"], w_med)
            grp["voltage_deviation"] = compute_deviation(grp["voltage_drop"], w_long)
        else:
            grp["voltage_trend"] = np.nan
            grp["voltage_deviation"] = np.nan

        # 3. Capacity fade & degradation rate
        if "capacity" in grp.columns:
            valid_caps = grp["capacity"].dropna()
            initial_cap = float(valid_caps.iloc[0]) if len(valid_caps) > 0 else float(config.NOMINAL_CAPACITY_AH)
            if initial_cap > 0:
                grp["capacity_fade_pct"] = ((initial_cap - grp["capacity"]) / initial_cap) * 100.0
                grp["soc_estimated"] = (grp["capacity"] / initial_cap) * 100.0
            else:
                grp["capacity_fade_pct"] = np.nan
                grp["soc_estimated"] = np.nan

            grp["capacity_fade_rate"] = compute_trend(grp["capacity"], w_med)
        else:
            grp["capacity_fade_pct"] = np.nan
            grp["soc_estimated"] = np.nan
            grp["capacity_fade_rate"] = np.nan

        # 4. Impedance trends and rise
        has_re = "re" in grp.columns
        has_rct = "rct" in grp.columns

        if has_re:
            grp["re_trend"] = compute_trend(grp["re"], w_med)
        else:
            grp["re_trend"] = np.nan

        if has_rct:
            grp["rct_trend"] = compute_trend(grp["rct"], w_med)
        else:
            grp["rct_trend"] = np.nan

        if has_re and has_rct:
            r_total = grp["re"] + grp["rct"]
            valid_r = r_total.dropna()
            initial_r = float(valid_r.iloc[0]) if len(valid_r) > 0 else np.nan
            if not pd.isna(initial_r) and initial_r > 0:
                grp["impedance_rise_pct"] = ((r_total - initial_r) / initial_r) * 100.0
            else:
                grp["impedance_rise_pct"] = np.nan
        else:
            grp["impedance_rise_pct"] = np.nan

        return grp

    if "cell_name" in df.columns:
        processed_groups = [
            _engineer_cell_group(group)
            for _, group in df.groupby("cell_name", sort=False)
        ]
        result_df = pd.concat(processed_groups, ignore_index=True)
        if "cycle_index" in result_df.columns:
            result_df = result_df.sort_values(["cell_name", "cycle_index"]).reset_index(drop=True)
    else:
        result_df = _engineer_cell_group(df).reset_index(drop=True)

    return result_df


def get_raw_feature_names() -> List[str]:
    """
    Return list of raw (non-temporal) measurement feature names.

    Returns
    -------
    List[str]
        List of raw feature column names.
    """
    return [
        "temp_max",
        "temp_mean",
        "temp_min",
        "voltage_start",
        "voltage_end",
        "voltage_drop",
        "voltage_std",
        "current_mean",
        "current_std",
        "capacity",
        "duration",
        "re",
        "rct",
    ]


def get_trend_feature_names() -> List[str]:
    """
    Return list of trend-aware temporal feature names.

    Returns
    -------
    List[str]
        List of trend feature column names.
    """
    return [
        "temp_trend",
        "temp_change_rate",
        "voltage_trend",
        "voltage_deviation",
        "capacity_fade_pct",
        "capacity_fade_rate",
        "re_trend",
        "rct_trend",
        "impedance_rise_pct",
        "soc_estimated",
    ]


def get_all_feature_names() -> List[str]:
    """
    Return combined list of all feature names (raw + trend-aware).

    Returns
    -------
    List[str]
        Complete list of modeling feature names.
    """
    return get_raw_feature_names() + get_trend_feature_names()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute feature engineering and filter initial cycles with insufficient history.

    Computes all temporal and physical degradation features, then drops rows
    where cycle history is less than config.MIN_CYCLES_FOR_TRENDS so that
    subsequent modeling does not encounter undefined window slopes.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed battery dataset.

    Returns
    -------
    pd.DataFrame
        Final feature-engineered DataFrame.
    """
    if df.empty:
        return df.copy()

    logger.info("Computing temporal degradation features for %d rows...", len(df))
    df_engineered = engineer_temporal_features(df)

    min_cycles = int(config.MIN_CYCLES_FOR_TRENDS)

    if "cycle_index" in df_engineered.columns:
        df_filtered = df_engineered[df_engineered["cycle_index"] >= min_cycles].copy()
    else:
        # Fallback to group row index if cycle_index is missing
        if "cell_name" in df_engineered.columns:
            df_filtered = df_engineered.groupby("cell_name", group_keys=False).apply(
                lambda g: g.iloc[min_cycles - 1 :] if len(g) >= min_cycles else pd.DataFrame(columns=g.columns)
            ).copy()
        else:
            df_filtered = df_engineered.iloc[min_cycles - 1 :].copy()

    df_filtered = df_filtered.reset_index(drop=True)
    logger.info(
        "Filtered dataset to cycles >= %d: %d rows remaining (dropped %d rows with insufficient trend history)",
        min_cycles,
        len(df_filtered),
        len(df_engineered) - len(df_filtered),
    )

    return df_filtered
