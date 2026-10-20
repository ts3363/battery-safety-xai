"""
Degradation-Aware Battery Risk Labeling Module.

This module synthesizes multi-factor degradation indicators (capacity fade,
impedance rise, thermal escalation, and voltage volatility) into a continuous
composite risk score, binary safety alarm labels, and categorical hazard levels
(LOW, MEDIUM, HIGH, CRITICAL).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure config can be imported from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def compute_capacity_fade_score(capacity_fade_pct: pd.Series) -> pd.Series:
    """
    Map capacity fade percentage to a normalized [0, 1] risk score.

    Mapping:
    - 0% fade (or less) -> 0.0
    - 20%+ fade -> 1.0 (industry EOL threshold)
    - Linear interpolation between 0% and 20%

    Parameters
    ----------
    capacity_fade_pct : pd.Series
        Percentage of nominal/initial capacity lost.

    Returns
    -------
    pd.Series
        Normalized capacity fade risk scores in [0, 1].
    """
    if capacity_fade_pct.empty:
        return pd.Series(dtype=float, index=capacity_fade_pct.index)

    score = (capacity_fade_pct / 20.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    score.name = "capacity_fade_score"
    return score


def compute_impedance_rise_score(impedance_rise_pct: pd.Series) -> pd.Series:
    """
    Map impedance rise percentage to a normalized [0, 1] risk score.

    Mapping:
    - 0% rise (or less) -> 0.0
    - 80%+ rise -> 1.0 (severe internal degradation)
    - Linear interpolation between 0% and 80%

    Parameters
    ----------
    impedance_rise_pct : pd.Series
        Percentage rise of total internal resistance (Re + Rct) from initial.

    Returns
    -------
    pd.Series
        Normalized impedance rise risk scores in [0, 1].
    """
    if impedance_rise_pct.empty:
        return pd.Series(dtype=float, index=impedance_rise_pct.index)

    score = (impedance_rise_pct / 80.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    score.name = "impedance_rise_score"
    return score


def compute_temperature_anomaly_score(
    temp_max: pd.Series, temp_trend: pd.Series
) -> pd.Series:
    """
    Compute thermal anomaly score combining absolute temperature and rate of rise.

    Components:
    - temp_level_score: 0 below 35°C, 1 above 55°C (linear between)
    - temp_trend_score: 0 below 0.1°C/cycle, 1 above 1.0°C/cycle (linear between)
    - Combined score: 0.6 * temp_level_score + 0.4 * temp_trend_score

    Parameters
    ----------
    temp_max : pd.Series
        Peak discharge temperature in °C.
    temp_trend : pd.Series
        Temperature slope over recent cycles in °C/cycle.

    Returns
    -------
    pd.Series
        Normalized temperature anomaly risk scores in [0, 1].
    """
    if temp_max.empty:
        return pd.Series(dtype=float, index=temp_max.index)

    temp_level_score = (
        ((temp_max - 35.0) / (55.0 - 35.0))
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )

    temp_trend_score = (
        ((temp_trend - 0.1) / (1.0 - 0.1))
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )

    combined = (0.6 * temp_level_score + 0.4 * temp_trend_score).clip(
        lower=0.0, upper=1.0
    ).fillna(0.0)
    combined.name = "temperature_anomaly_score"
    return combined


def compute_voltage_anomaly_score(
    voltage_deviation: pd.Series, voltage_trend: pd.Series
) -> pd.Series:
    """
    Compute voltage anomaly score combining deviation from baseline and trend.

    Components:
    - voltage_dev_score: 0 below 0.05V, 1 above 0.3V (linear between)
    - voltage_trend_score: 0 below 0.005V/cycle, 1 above 0.02V/cycle (linear between)
    - Combined score: 0.6 * voltage_dev_score + 0.4 * voltage_trend_score

    Parameters
    ----------
    voltage_deviation : pd.Series
        Absolute deviation of voltage drop from rolling average in V.
    voltage_trend : pd.Series
        Slope of voltage drop over recent cycles in V/cycle.

    Returns
    -------
    pd.Series
        Normalized voltage anomaly risk scores in [0, 1].
    """
    if voltage_deviation.empty:
        return pd.Series(dtype=float, index=voltage_deviation.index)

    voltage_dev_score = (
        ((voltage_deviation - 0.05) / (0.3 - 0.05))
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )

    voltage_trend_score = (
        ((voltage_trend.abs() - 0.005) / (0.02 - 0.005))
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )

    combined = (0.6 * voltage_dev_score + 0.4 * voltage_trend_score).clip(
        lower=0.0, upper=1.0
    ).fillna(0.0)
    combined.name = "voltage_anomaly_score"
    return combined


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute multi-factor composite risk score in [0, 1] using config.RISK_WEIGHTS.

    Formula:
        risk_score = (
            w_cap * capacity_score +
            w_imp * impedance_score +
            w_temp * temperature_score +
            w_volt * voltage_score
        )

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered battery dataset containing degradation and trend features.

    Returns
    -------
    pd.Series
        Continuous risk score in [0, 1].
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    weights = config.RISK_WEIGHTS
    w_cap = float(weights.get("capacity_fade", 0.40))
    w_imp = float(weights.get("impedance_rise", 0.30))
    w_temp = float(weights.get("temperature_anomaly", 0.20))
    w_volt = float(weights.get("voltage_anomaly", 0.10))

    # 1. Capacity fade score
    if "capacity_fade_pct" in df.columns:
        cap_score = compute_capacity_fade_score(df["capacity_fade_pct"])
    else:
        cap_score = pd.Series(0.0, index=df.index)

    # 2. Impedance rise score
    if "impedance_rise_pct" in df.columns:
        imp_score = compute_impedance_rise_score(df["impedance_rise_pct"])
    else:
        imp_score = pd.Series(0.0, index=df.index)

    # 3. Temperature anomaly score
    t_max = df["temp_max"] if "temp_max" in df.columns else pd.Series(0.0, index=df.index)
    t_trend = df["temp_trend"] if "temp_trend" in df.columns else pd.Series(0.0, index=df.index)
    temp_score = compute_temperature_anomaly_score(t_max, t_trend)

    # 4. Voltage anomaly score
    v_dev = df["voltage_deviation"] if "voltage_deviation" in df.columns else pd.Series(0.0, index=df.index)
    v_trend = df["voltage_trend"] if "voltage_trend" in df.columns else pd.Series(0.0, index=df.index)
    volt_score = compute_voltage_anomaly_score(v_dev, v_trend)

    composite_score = (
        w_cap * cap_score
        + w_imp * imp_score
        + w_temp * temp_score
        + w_volt * volt_score
    )

    composite_score = composite_score.clip(lower=0.0, upper=1.0).fillna(0.0)
    composite_score.name = "risk_score"
    return composite_score


def assign_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign risk labels (continuous, binary, and categorical) to battery cycle dataset.

    Labels added:
    - risk_score: continuous score in [0, 1]
    - risk_binary: 0 (safe) or 1 (risk) using config.RISK_THRESHOLD
    - risk_level: categorical hazard quartile (LOW, MEDIUM, HIGH, CRITICAL)

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered battery dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with risk_score, risk_binary, and risk_level appended.
    """
    df_labeled = df.copy()
    if df_labeled.empty:
        df_labeled["risk_score"] = pd.Series(dtype=float)
        df_labeled["risk_binary"] = pd.Series(dtype=int)
        df_labeled["risk_level"] = pd.Series(dtype=str)
        return df_labeled

    # 1. Continuous risk score
    risk_score = compute_risk_score(df_labeled)
    df_labeled["risk_score"] = risk_score

    # 2. Binary risk classification using threshold from config
    risk_threshold = float(config.RISK_THRESHOLD)
    df_labeled["risk_binary"] = (risk_score >= risk_threshold).astype(int)

    # 3. Categorical risk level based on quartiles
    # Try empirical quartile binning via pd.qcut; fallback to fixed interval quartiles
    try:
        q_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        risk_level = pd.qcut(risk_score, q=4, labels=q_labels, duplicates="raise").astype(str)
    except (ValueError, IndexError):
        # Fallback to fixed interval quartiles [0, 0.25, 0.50, 0.75, 1.0]
        risk_level = pd.cut(
            risk_score,
            bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            right=True,
        ).astype(str)

    df_labeled["risk_level"] = risk_level

    # Print summary distribution
    total_samples = len(df_labeled)
    print("=" * 60)
    print("BATTERY RISK LABEL DISTRIBUTION")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print("\nCategorical Risk Level Distribution:")
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = int((df_labeled["risk_level"] == level).sum())
        pct = (count / total_samples * 100) if total_samples > 0 else 0.0
        print(f"  {level:<10}: {count:>5} ({pct:>5.1f}%)")

    print(f"\nBinary Risk Distribution (Threshold = {risk_threshold}):")
    cnt_safe = int((df_labeled["risk_binary"] == 0).sum())
    cnt_risk = int((df_labeled["risk_binary"] == 1).sum())
    pct_safe = (cnt_safe / total_samples * 100) if total_samples > 0 else 0.0
    pct_risk = (cnt_risk / total_samples * 100) if total_samples > 0 else 0.0
    print(f"  Safe (0)   : {cnt_safe:>5} ({pct_safe:>5.1f}%)")
    print(f"  Risk (1)   : {cnt_risk:>5} ({pct_risk:>5.1f}%)")
    print("=" * 60)

    return df_labeled
