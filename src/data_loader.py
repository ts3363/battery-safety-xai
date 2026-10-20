"""
NASA Li-ion Battery Aging Dataset Loader.

This module parses MATLAB (.mat) files from the NASA Prognostics Center of
Excellence (PCoE) battery aging dataset (cells B0005, B0006, B0007, B0018),
extracts operational cycles (charge, discharge, impedance), computes cycle-level
features, and produces a structured, unified pandas DataFrame aligned across
measurements.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat

# Ensure config can be imported from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def _extract_field(data_struct: Any, field_name: str) -> Optional[np.ndarray]:
    """
    Safely extract a numeric field from a MATLAB data structure.

    Parameters
    ----------
    data_struct : Any
        The cycle 'data' object extracted from the MATLAB structure.
    field_name : str
        Name of the field to extract (e.g., 'Voltage_measured').

    Returns
    -------
    Optional[np.ndarray]
        1D numpy array of values, a scalar float, or None if extraction fails.
    """
    if data_struct is None:
        return None
    try:
        val = None
        if isinstance(data_struct, dict):
            val = data_struct.get(field_name, None)
        elif hasattr(data_struct, "dtype") and data_struct.dtype.names and field_name in data_struct.dtype.names:
            val = data_struct[field_name]
        elif hasattr(data_struct, "__getitem__"):
            try:
                val = data_struct[field_name]
            except (IndexError, KeyError, TypeError):
                val = None

        if val is None:
            return None

        arr = np.squeeze(val)
        if hasattr(arr, "ndim") and arr.ndim == 0:
            return float(arr.item()) if hasattr(arr, "item") else float(arr)

        if isinstance(arr, np.ndarray):
            if np.issubdtype(arr.dtype, np.number):
                return arr.astype(float)
            return arr
        return np.asarray(arr, dtype=float)
    except Exception as exc:
        logger.debug("Failed extracting field %s: %s", field_name, exc)
        return None


def _extract_cycle_type(cycle_struct: Any) -> str:
    """
    Extract and normalize cycle type string ('charge', 'discharge', 'impedance').

    Parameters
    ----------
    cycle_struct : Any
        Cycle object from MATLAB cycle array.

    Returns
    -------
    str
        Normalized lowercase cycle type string.
    """
    try:
        val = None
        if isinstance(cycle_struct, dict):
            val = cycle_struct.get("type", "")
        elif hasattr(cycle_struct, "dtype") and cycle_struct.dtype.names and "type" in cycle_struct.dtype.names:
            val = cycle_struct["type"]
        else:
            val = cycle_struct["type"]

        val = np.squeeze(val)
        if hasattr(val, "item"):
            val = val.item()
        val_str = str(val).lower().strip()

        if "discharge" in val_str:
            return "discharge"
        elif "charge" in val_str:
            return "charge"
        elif "impedance" in val_str:
            return "impedance"
        return val_str
    except Exception:
        return ""


def load_single_battery(mat_file_path: str, cell_name: str) -> dict:
    """
    Load a single NASA battery .mat file and extract all cycles.

    Navigates the deeply nested MATLAB struct hierarchy:
    mat_data[cell_name][0][0]['cycle'][0] -> list of cycles.

    Parameters
    ----------
    mat_file_path : str
        Path to the battery cell .mat file.
    cell_name : str
        Battery cell identifier, e.g. 'B0005'.

    Returns
    -------
    dict
        Dictionary containing:
        - 'cell_name': str
        - 'charge_cycles': list of dicts with charge cycle measurements
        - 'discharge_cycles': list of dicts with discharge cycle measurements
        - 'impedance_cycles': list of dicts with impedance cycle measurements
    """
    path = Path(mat_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"MAT file not found at: {mat_file_path}")

    mat_data = loadmat(str(path))
    if cell_name not in mat_data:
        # Check case-insensitive match for key
        matched_keys = [k for k in mat_data.keys() if k.lower() == cell_name.lower()]
        if matched_keys:
            cell_name_in_mat = matched_keys[0]
        else:
            available = [k for k in mat_data.keys() if not k.startswith("__")]
            raise KeyError(f"Cell '{cell_name}' not found in {mat_file_path}. Keys: {available}")
    else:
        cell_name_in_mat = cell_name

    cell_obj = mat_data[cell_name_in_mat]
    # Handle nested [0][0] indexing
    if isinstance(cell_obj, np.ndarray) and cell_obj.size > 0:
        cell_struct = cell_obj[0, 0] if cell_obj.ndim >= 2 else cell_obj[0]
    else:
        cell_struct = cell_obj

    # Extract cycle array
    try:
        if isinstance(cell_struct, dict):
            raw_cycles = cell_struct["cycle"]
        elif hasattr(cell_struct, "dtype") and cell_struct.dtype.names and "cycle" in cell_struct.dtype.names:
            raw_cycles = cell_struct["cycle"]
        else:
            raw_cycles = cell_struct["cycle"]

        raw_cycles = np.squeeze(raw_cycles)
        if raw_cycles.ndim == 0:
            cycles_list = [raw_cycles.item()]
        else:
            cycles_list = list(raw_cycles)
    except Exception as exc:
        raise ValueError(f"Failed to access 'cycle' array in {cell_name}: {exc}") from exc

    charge_cycles: List[Dict[str, Any]] = []
    discharge_cycles: List[Dict[str, Any]] = []
    impedance_cycles: List[Dict[str, Any]] = []

    discharge_count = 0
    charge_count = 0
    impedance_count = 0

    for idx, cycle in enumerate(cycles_list):
        cycle_type = _extract_cycle_type(cycle)

        # Ambient temperature
        ambient_temp = None
        try:
            if isinstance(cycle, dict):
                amb = cycle.get("ambient_temperature", None)
            else:
                amb = cycle["ambient_temperature"]
            amb = np.squeeze(amb)
            ambient_temp = float(amb.item() if hasattr(amb, "item") else amb)
        except Exception:
            ambient_temp = None

        # Cycle timestamp / date vector
        cycle_time = None
        try:
            if isinstance(cycle, dict):
                t_val = cycle.get("time", None)
            else:
                t_val = cycle["time"]
            cycle_time = np.squeeze(t_val)
        except Exception:
            cycle_time = None

        # Cycle data struct
        data_struct = None
        try:
            if isinstance(cycle, dict):
                d_raw = cycle.get("data", None)
            else:
                d_raw = cycle["data"]
            if isinstance(d_raw, np.ndarray) and d_raw.size > 0:
                data_struct = d_raw[0, 0] if d_raw.ndim >= 2 else d_raw[0]
            else:
                data_struct = d_raw
        except Exception:
            data_struct = None

        if cycle_type == "discharge":
            discharge_count += 1
            discharge_entry: Dict[str, Any] = {
                "cycle_index": discharge_count,
                "global_cycle_index": idx + 1,
                "ambient_temperature": ambient_temp,
                "time": cycle_time,
                "Voltage_measured": _extract_field(data_struct, "Voltage_measured"),
                "Current_measured": _extract_field(data_struct, "Current_measured"),
                "Temperature_measured": _extract_field(data_struct, "Temperature_measured"),
                "Current_load": _extract_field(data_struct, "Current_load"),
                "Voltage_load": _extract_field(data_struct, "Voltage_load"),
                "Capacity": _extract_field(data_struct, "Capacity"),
                "Time": _extract_field(data_struct, "Time"),
            }
            discharge_cycles.append(discharge_entry)

        elif cycle_type == "charge":
            charge_count += 1
            charge_entry: Dict[str, Any] = {
                "cycle_index": charge_count,
                "global_cycle_index": idx + 1,
                "ambient_temperature": ambient_temp,
                "time": cycle_time,
                "Voltage_measured": _extract_field(data_struct, "Voltage_measured"),
                "Current_measured": _extract_field(data_struct, "Current_measured"),
                "Temperature_measured": _extract_field(data_struct, "Temperature_measured"),
                "Current_charge": _extract_field(data_struct, "Current_charge"),
                "Voltage_charge": _extract_field(data_struct, "Voltage_charge"),
                "Time": _extract_field(data_struct, "Time"),
            }
            charge_cycles.append(charge_entry)

        elif cycle_type == "impedance":
            impedance_count += 1
            # Associate impedance with the current discharge cycle count
            associated_discharge_cycle = discharge_count if discharge_count > 0 else 1
            impedance_entry: Dict[str, Any] = {
                "cycle_index": associated_discharge_cycle,
                "impedance_cycle_index": impedance_count,
                "global_cycle_index": idx + 1,
                "ambient_temperature": ambient_temp,
                "time": cycle_time,
                "Sense_current": _extract_field(data_struct, "Sense_current"),
                "Battery_current": _extract_field(data_struct, "Battery_current"),
                "Current_ratio": _extract_field(data_struct, "Current_ratio"),
                "Battery_impedance": _extract_field(data_struct, "Battery_impedance"),
                "Rectified_Impedance": _extract_field(data_struct, "Rectified_Impedance"),
                "Re": _extract_field(data_struct, "Re"),
                "Rct": _extract_field(data_struct, "Rct"),
            }
            impedance_cycles.append(impedance_entry)

    logger.info(
        "Loaded %s: %d discharge, %d charge, %d impedance cycles",
        cell_name,
        len(discharge_cycles),
        len(charge_cycles),
        len(impedance_cycles),
    )

    return {
        "cell_name": cell_name,
        "charge_cycles": charge_cycles,
        "discharge_cycles": discharge_cycles,
        "impedance_cycles": impedance_cycles,
    }


def extract_discharge_features(discharge_cycles: list) -> pd.DataFrame:
    """
    Extract aggregated cycle-level features from discharge cycles.

    Features extracted per cycle:
    - cycle_index: int
    - temp_max: float
    - temp_mean: float
    - temp_min: float
    - voltage_start: float
    - voltage_end: float
    - voltage_drop: float
    - voltage_std: float
    - current_mean: float
    - current_std: float
    - capacity: float (last value of Capacity array)
    - duration: float (total discharge duration from Time)

    Parameters
    ----------
    discharge_cycles : list
        List of discharge cycle dictionaries.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per discharge cycle.
    """
    records: List[Dict[str, Any]] = []

    for idx, cycle in enumerate(discharge_cycles):
        cycle_idx = cycle.get("cycle_index", idx + 1)

        voltages = cycle.get("Voltage_measured", None)
        currents = cycle.get("Current_measured", None)
        temps = cycle.get("Temperature_measured", None)
        cap = cycle.get("Capacity", None)
        time_arr = cycle.get("Time", None)

        # Temperature statistics
        if temps is not None and isinstance(temps, np.ndarray) and temps.size > 0:
            temp_max = float(np.nanmax(temps))
            temp_mean = float(np.nanmean(temps))
            temp_min = float(np.nanmin(temps))
        else:
            temp_max = np.nan
            temp_mean = np.nan
            temp_min = np.nan

        # Voltage statistics
        if voltages is not None and isinstance(voltages, np.ndarray) and voltages.size > 0:
            voltage_start = float(voltages[0])
            voltage_end = float(voltages[-1])
            voltage_drop = float(voltage_start - voltage_end)
            voltage_std = float(np.nanstd(voltages))
        else:
            voltage_start = np.nan
            voltage_end = np.nan
            voltage_drop = np.nan
            voltage_std = np.nan

        # Current statistics
        if currents is not None and isinstance(currents, np.ndarray) and currents.size > 0:
            current_mean = float(np.nanmean(currents))
            current_std = float(np.nanstd(currents))
        else:
            current_mean = np.nan
            current_std = np.nan

        # Capacity (last value of Capacity array or scalar)
        if cap is not None:
            if isinstance(cap, np.ndarray):
                capacity = float(cap[-1]) if cap.size > 0 else np.nan
            elif isinstance(cap, (int, float, np.floating, np.integer)):
                capacity = float(cap)
            else:
                capacity = np.nan
        else:
            capacity = np.nan

        # Duration
        if time_arr is not None and isinstance(time_arr, np.ndarray) and time_arr.size > 0:
            if time_arr.size > 1:
                duration = float(time_arr[-1] - time_arr[0])
            else:
                duration = float(time_arr[0])
        else:
            duration = np.nan

        record = {
            "cycle_index": int(cycle_idx),
            "temp_max": temp_max,
            "temp_mean": temp_mean,
            "temp_min": temp_min,
            "voltage_start": voltage_start,
            "voltage_end": voltage_end,
            "voltage_drop": voltage_drop,
            "voltage_std": voltage_std,
            "current_mean": current_mean,
            "current_std": current_std,
            "capacity": capacity,
            "duration": duration,
        }
        records.append(record)

    columns = [
        "cycle_index",
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
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records)[columns]


def extract_impedance_features(impedance_cycles: list) -> pd.DataFrame:
    """
    Extract scalar impedance parameters (Re, Rct) from impedance cycles.

    Parameters
    ----------
    impedance_cycles : list
        List of impedance cycle dictionaries.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per impedance cycle and columns:
        ['cycle_index', 're', 'rct'].
    """
    records: List[Dict[str, Any]] = []

    for idx, cycle in enumerate(impedance_cycles):
        cycle_idx = cycle.get("cycle_index", idx + 1)

        # Electrolyte resistance Re
        re_val = cycle.get("Re", cycle.get("re", np.nan))
        if isinstance(re_val, np.ndarray):
            re_scalar = float(re_val.flat[0]) if re_val.size > 0 else np.nan
        elif re_val is not None and not pd.isna(re_val):
            re_scalar = float(re_val)
        else:
            re_scalar = np.nan

        # Charge transfer resistance Rct
        rct_val = cycle.get("Rct", cycle.get("rct", np.nan))
        if isinstance(rct_val, np.ndarray):
            rct_scalar = float(rct_val.flat[0]) if rct_val.size > 0 else np.nan
        elif rct_val is not None and not pd.isna(rct_val):
            rct_scalar = float(rct_val)
        else:
            rct_scalar = np.nan

        records.append({
            "cycle_index": int(cycle_idx),
            "re": re_scalar,
            "rct": rct_scalar,
        })

    columns = ["cycle_index", "re", "rct"]
    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records)[columns]


def load_all_batteries(data_dir: str = None) -> dict:
    """
    Load all battery cells defined in config.BATTERY_CELLS.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing .mat files. Defaults to config.DATA_RAW_DIR.

    Returns
    -------
    dict
        Dictionary mapping cell_name -> {
            'discharge_features': DataFrame,
            'impedance_features': DataFrame,
            'raw_cycles': dict
        }
    """
    if data_dir is None:
        directory = config.DATA_RAW_DIR
    else:
        directory = Path(data_dir)

    results: Dict[str, Dict[str, Any]] = {}

    for cell_name in config.BATTERY_CELLS:
        mat_file = directory / f"{cell_name}.mat"
        if not mat_file.is_file():
            # Search case-insensitively
            candidates = list(directory.glob(f"*{cell_name}*.mat"))
            if candidates:
                mat_file = candidates[0]
            else:
                raise FileNotFoundError(
                    f"MAT file for battery cell '{cell_name}' not found in directory: {directory}"
                )

        raw = load_single_battery(str(mat_file), cell_name)
        discharge_df = extract_discharge_features(raw["discharge_cycles"])
        impedance_df = extract_impedance_features(raw["impedance_cycles"])

        results[cell_name] = {
            "discharge_features": discharge_df,
            "impedance_features": impedance_df,
            "raw_cycles": raw,
        }

    return results


def create_combined_dataset(battery_data: dict) -> pd.DataFrame:
    """
    Merge discharge and impedance features for each cell and combine into one dataset.

    Impedance measurements are less frequent than discharge measurements;
    forward-fill (and backward-fill for initial cycles) aligns resistance
    values across discharge cycles.

    Parameters
    ----------
    battery_data : dict
        Mapping cell_name -> {
            'discharge_features': DataFrame,
            'impedance_features': DataFrame,
            ...
        }

    Returns
    -------
    pd.DataFrame
        Combined DataFrame sorted by cell_name and cycle_index with 'cell_name'
        column added.
    """
    cell_dfs: List[pd.DataFrame] = []

    for cell_name, data in battery_data.items():
        discharge_df = data["discharge_features"].copy()
        impedance_df = data.get("impedance_features", pd.DataFrame()).copy()

        if discharge_df.empty:
            continue

        discharge_df = discharge_df.sort_values("cycle_index").reset_index(drop=True)

        if not impedance_df.empty:
            # Clean duplicate measurements on same cycle index
            impedance_df = (
                impedance_df.sort_values("cycle_index")
                .drop_duplicates(subset=["cycle_index"], keep="last")
                .reset_index(drop=True)
            )

            # Left merge on cycle_index
            merged = pd.merge(discharge_df, impedance_df, on="cycle_index", how="left")

            # If all impedance rows were unmatched (e.g. impedance measured at offset cycles),
            # use backward merge_asof so each discharge cycle gets the latest measured impedance
            if merged["re"].isna().all() and not impedance_df["re"].isna().all():
                merged = pd.merge_asof(
                    discharge_df,
                    impedance_df[["cycle_index", "re", "rct"]],
                    on="cycle_index",
                    direction="backward",
                )

            # Forward fill then backward fill impedance features
            merged["re"] = merged["re"].ffill().bfill()
            merged["rct"] = merged["rct"].ffill().bfill()
        else:
            merged = discharge_df.copy()
            merged["re"] = np.nan
            merged["rct"] = np.nan

        merged["cell_name"] = str(cell_name)
        cell_dfs.append(merged)

    if not cell_dfs:
        columns = [
            "cell_name",
            "cycle_index",
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
        return pd.DataFrame(columns=columns)

    combined = pd.concat(cell_dfs, ignore_index=True)

    # Reorder columns with cell_name and cycle_index first
    ordered_cols = [
        "cell_name",
        "cycle_index",
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
    # Retain any extra columns if present
    remaining = [c for c in combined.columns if c not in ordered_cols]
    final_cols = ordered_cols + remaining

    combined = combined[final_cols].sort_values(["cell_name", "cycle_index"]).reset_index(drop=True)
    return combined
