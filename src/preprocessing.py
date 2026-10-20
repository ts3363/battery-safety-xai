"""
Data Preprocessing and Validation Module for Battery Aging.

This module provides data cleaning, missing value handling with gap-aware
interpolation, IQR-based outlier detection, and safety validation across
battery cell measurements.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure config can be imported from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def _interpolate_small_gaps_series(series: pd.Series, max_gap: int = 2) -> tuple[pd.Series, int]:
    """
    Linearly interpolate missing values only for small consecutive NaN blocks.

    Parameters
    ----------
    series : pd.Series
        Input numeric series.
    max_gap : int, default=2
        Maximum consecutive NaNs to interpolate (< 3 consecutive NaNs means max_gap=2).

    Returns
    -------
    tuple[pd.Series, int]
        Tuple of (interpolated series, count of interpolated values).
    """
    if not series.isna().any():
        return series.copy(), 0

    is_na = series.isna()
    # Group contiguous blocks of True / False
    block_id = (is_na != is_na.shift()).cumsum()
    # Count the size of each block where is_na is True
    gap_sizes = is_na.groupby(block_id).transform("sum")

    # Only interpolate NaNs in blocks of size <= max_gap
    small_gap_mask = is_na & (gap_sizes <= max_gap) & (gap_sizes > 0)

    if not small_gap_mask.any():
        return series.copy(), 0

    interpolated_all = series.interpolate(method="linear")
    result = series.copy()
    result[small_gap_mask] = interpolated_all[small_gap_mask]

    n_interpolated = int(small_gap_mask.sum())
    return result, n_interpolated


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing values in battery cycle data.

    - Drops rows where critical features (capacity, voltage, temperature) are ALL missing.
    - For small gaps (< 3 consecutive NaNs): applies linear interpolation within
      each battery cell group.
    - For larger gaps: leaves NaNs untouched to avoid fabricating trend data.
    - Adds a boolean 'has_missing' column flagging rows that contain any remaining NaNs.
    - Logs interpolation and drop counts.

    Parameters
    ----------
    df : pd.DataFrame
        Input combined or single-cell battery dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with 'has_missing' flag.
    """
    if df.empty:
        cleaned = df.copy()
        if "has_missing" not in cleaned.columns:
            cleaned["has_missing"] = False
        return cleaned

    df_clean = df.copy()
    initial_rows = len(df_clean)

    # 1. Identify critical feature columns
    cap_cols = [c for c in df_clean.columns if "capacity" in c.lower()]
    volt_cols = [c for c in df_clean.columns if "voltage" in c.lower()]
    temp_cols = [c for c in df_clean.columns if "temp" in c.lower()]

    critical_present = cap_cols + volt_cols + temp_cols
    if critical_present:
        # A row is dropped if ALL present capacity, voltage, AND temperature columns are NaN
        cap_all_na = df_clean[cap_cols].isna().all(axis=1) if cap_cols else pd.Series(True, index=df_clean.index)
        volt_all_na = df_clean[volt_cols].isna().all(axis=1) if volt_cols else pd.Series(True, index=df_clean.index)
        temp_all_na = df_clean[temp_cols].isna().all(axis=1) if temp_cols else pd.Series(True, index=df_clean.index)

        all_critical_missing = cap_all_na & volt_all_na & temp_all_na
        df_clean = df_clean[~all_critical_missing].copy()

    rows_dropped = initial_rows - len(df_clean)

    # 2. Linear interpolation for small gaps (< 3 consecutive NaNs) per cell
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifiers and metadata
    cols_to_interpolate = [c for c in numeric_cols if c not in ["cycle_index", "has_missing"]]

    total_interpolated = 0

    if "cell_name" in df_clean.columns:
        processed_groups = []
        for _, group in df_clean.groupby("cell_name", sort=False):
            grp = group.sort_values("cycle_index") if "cycle_index" in group.columns else group
            grp_interpolated = grp.copy()
            for col in cols_to_interpolate:
                interp_s, count = _interpolate_small_gaps_series(grp[col], max_gap=2)
                grp_interpolated[col] = interp_s
                total_interpolated += count
            processed_groups.append(grp_interpolated)
        df_clean = pd.concat(processed_groups, ignore_index=True)
    else:
        for col in cols_to_interpolate:
            interp_s, count = _interpolate_small_gaps_series(df_clean[col], max_gap=2)
            df_clean[col] = interp_s
            total_interpolated += count

    # 3. Add 'has_missing' flag for any remaining un-interpolated missing values
    active_numeric = [c for c in cols_to_interpolate if c in df_clean.columns]
    df_clean["has_missing"] = df_clean[active_numeric].isna().any(axis=1)

    logger.info(
        "handle_missing_values: dropped %d rows with all critical features missing; "
        "interpolated %d values in small gaps (<3 consecutive NaNs); "
        "%d rows retain missing values",
        rows_dropped,
        total_interpolated,
        int(df_clean["has_missing"].sum()),
    )

    return df_clean


def detect_outliers(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """
    Detect outliers using the Interquartile Range (IQR) method (1.5 * IQR).

    Computed per cell group so that natural aging differences between cells
    do not distort detection bounds. Adds boolean '{feature}_outlier' columns.
    Outlier rows are flagged but not removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input battery dataset.
    features : list, optional
        List of numeric feature column names to test. If None, defaults to all
        numeric measurement columns (excluding cycle_index, has_missing, outlier flags).

    Returns
    -------
    pd.DataFrame
        DataFrame with '{feature}_outlier' columns appended.
    """
    df_result = df.copy()
    if df_result.empty:
        return df_result

    if features is None:
        exclude = ["cycle_index", "has_missing"]
        numeric_cols = df_result.select_dtypes(include=[np.number]).columns
        features = [
            c for c in numeric_cols
            if c not in exclude and not c.endswith("_outlier")
        ]

    # Initialize outlier columns to False
    for feat in features:
        df_result[f"{feat}_outlier"] = False

    group_col = "cell_name" if "cell_name" in df_result.columns else None

    if group_col:
        for feat in features:
            if feat not in df_result.columns:
                continue

            def _flag_group_outliers(group_series: pd.Series) -> pd.Series:
                clean_s = group_series.dropna()
                if len(clean_s) < 4:
                    return pd.Series(False, index=group_series.index)
                q1 = clean_s.quantile(0.25)
                q3 = clean_s.quantile(0.75)
                iqr = q3 - q1
                if iqr <= 1e-12:
                    return pd.Series(False, index=group_series.index)
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                return (group_series < lower_bound) | (group_series > upper_bound)

            outlier_series = df_result.groupby(group_col, group_keys=False)[feat].apply(
                _flag_group_outliers
            )
            df_result[f"{feat}_outlier"] = outlier_series.fillna(False)
    else:
        for feat in features:
            if feat not in df_result.columns:
                continue
            clean_s = df_result[feat].dropna()
            if len(clean_s) < 4:
                df_result[f"{feat}_outlier"] = False
                continue
            q1 = clean_s.quantile(0.25)
            q3 = clean_s.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 1e-12:
                df_result[f"{feat}_outlier"] = False
                continue
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_result[f"{feat}_outlier"] = (
                (df_result[feat] < lower_bound) | (df_result[feat] > upper_bound)
            ).fillna(False)

    return df_result


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate dataset integrity, monotonicity, and expected physical ranges.

    Checks:
    - Presence of expected NASA cells (config.BATTERY_CELLS)
    - Monotonically increasing cycle indices within each cell
    - Physical engineering ranges:
        - Temperature: 15°C to 80°C
        - Voltage: 2.0V to 4.5V
        - Capacity: 0.5 Ah to 2.5 Ah

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Returns
    -------
    dict
        Validation results containing:
        - 'all_cells_present': bool
        - 'missing_cells': list
        - 'cells_found': list
        - 'monotonic_cycles': dict[cell_name, bool]
        - 'value_ranges': dict
        - 'warnings': list[str]
        - 'is_valid': bool
    """
    warnings: List[str] = []

    # 1. Cell presence check
    expected_cells = list(config.BATTERY_CELLS)
    if "cell_name" in df.columns:
        cells_found = list(df["cell_name"].dropna().unique())
    else:
        cells_found = []
    missing_cells = [c for c in expected_cells if c not in cells_found]

    if missing_cells:
        warnings.append(
            f"Missing expected battery cells: {missing_cells} (found: {cells_found})"
        )
    all_cells_present = len(missing_cells) == 0

    # 2. Monotonic cycle indices
    monotonic_cycles: Dict[str, bool] = {}
    if "cell_name" in df.columns and "cycle_index" in df.columns:
        for cell, group in df.groupby("cell_name"):
            is_monotonic = bool(group["cycle_index"].is_monotonic_increasing)
            monotonic_cycles[str(cell)] = is_monotonic
            if not is_monotonic:
                warnings.append(f"Cell {cell} has non-monotonic cycle_index sequence.")
    elif "cycle_index" in df.columns:
        is_monotonic = bool(df["cycle_index"].is_monotonic_increasing)
        monotonic_cycles["dataset"] = is_monotonic
        if not is_monotonic:
            warnings.append("Dataset cycle_index sequence is non-monotonic.")

    # 3. Value ranges validation
    range_specs = {
        "temperature": {
            "columns": ["temp_max", "temp_mean", "temp_min"],
            "min": 15.0,
            "max": 80.0,
            "unit": "°C",
        },
        "voltage": {
            "columns": ["voltage_start", "voltage_end"],
            "min": 2.0,
            "max": 4.5,
            "unit": "V",
        },
        "capacity": {
            "columns": ["capacity"],
            "min": 0.5,
            "max": 2.5,
            "unit": "Ah",
        },
    }

    value_ranges: Dict[str, Any] = {}

    for spec_name, spec in range_specs.items():
        present_cols = [c for c in spec["columns"] if c in df.columns]
        if not present_cols:
            continue

        spec_min = spec["min"]
        spec_max = spec["max"]
        out_of_range_total = 0
        observed_min = float("inf")
        observed_max = float("-inf")

        for col in present_cols:
            clean_vals = df[col].dropna()
            if clean_vals.empty:
                continue
            c_min = float(clean_vals.min())
            c_max = float(clean_vals.max())
            observed_min = min(observed_min, c_min)
            observed_max = max(observed_max, c_max)

            viol_count = int(((clean_vals < spec_min) | (clean_vals > spec_max)).sum())
            out_of_range_total += viol_count
            if viol_count > 0:
                warnings.append(
                    f"Feature '{col}' has {viol_count} values outside expected range "
                    f"[{spec_min}, {spec_max}] {spec['unit']} (range observed: [{c_min:.2f}, {c_max:.2f}])"
                )

        value_ranges[spec_name] = {
            "expected_range": [spec_min, spec_max],
            "observed_min": None if observed_min == float("inf") else observed_min,
            "observed_max": None if observed_max == float("-inf") else observed_max,
            "out_of_range_count": out_of_range_total,
            "in_range": out_of_range_total == 0,
        }

    is_valid = len(warnings) == 0

    return {
        "all_cells_present": all_cells_present,
        "missing_cells": missing_cells,
        "cells_found": cells_found,
        "monotonic_cycles": monotonic_cycles,
        "value_ranges": value_ranges,
        "warnings": warnings,
        "is_valid": is_valid,
    }


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute end-to-end preprocessing pipeline on combined dataset.

    Steps:
    1. Handle missing values (critical drop + gap-limited interpolation)
    2. Detect outliers using IQR method
    3. Validate ranges, monotonicity, and completeness

    Parameters
    ----------
    df : pd.DataFrame
        Raw combined battery dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned, outlier-flagged DataFrame ready for feature engineering.
    """
    logger.info("Starting preprocessing pipeline on %d rows...", len(df))

    # Step 1: Missing values handling
    df_imputed = handle_missing_values(df)

    # Step 2: Outlier detection
    df_flagged = detect_outliers(df_imputed)

    # Step 3: Data validation
    validation = validate_data(df_flagged)

    # Summary reporting
    outlier_cols = [c for c in df_flagged.columns if c.endswith("_outlier")]
    total_outlier_flags = int(df_flagged[outlier_cols].sum().sum()) if outlier_cols else 0
    missing_rows = int(df_flagged["has_missing"].sum()) if "has_missing" in df_flagged.columns else 0

    print("=" * 60)
    print("BATTERY PREPROCESSING PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Initial rows:                {len(df)}")
    print(f"Cleaned rows:                {len(df_flagged)}")
    print(f"Rows with remaining missing: {missing_rows}")
    print(f"Total outlier flags set:     {total_outlier_flags}")
    print(f"Validation status:           {'PASSED' if validation['is_valid'] else 'WARNINGS DETECTED'}")

    if validation["warnings"]:
        print("\nValidation Warnings:")
        for w in validation["warnings"]:
            print(f"  - {w}")
    print("=" * 60)

    return df_flagged
