"""
Global configuration for the XAI Battery Safety Framework.

All paths, hyperparameters, engineering thresholds, and experimental
settings are defined here for reproducibility.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
WARNINGS_DIR = RESULTS_DIR / "warnings"

# Create directories if they don't exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, WARNINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Battery cells in the NASA dataset
# ──────────────────────────────────────────────────────────────────────
BATTERY_CELLS = ["B0005", "B0006", "B0007", "B0018"]

# Nominal capacity (Ah) for these 18650 cells (rated ~2 Ah)
NOMINAL_CAPACITY_AH = 2.0

# End-of-life threshold: 80% of nominal capacity (industry standard)
EOL_CAPACITY_THRESHOLD = 0.80  # fraction of nominal

# ──────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────
# Sliding window sizes for trend features (in number of cycles)
TREND_WINDOW_SHORT = 3
TREND_WINDOW_MEDIUM = 5
TREND_WINDOW_LONG = 10

# Minimum number of cycles required before computing trend features
MIN_CYCLES_FOR_TRENDS = 5

# ──────────────────────────────────────────────────────────────────────
# Risk labeling
# ──────────────────────────────────────────────────────────────────────
# Weights for multi-factor risk score
RISK_WEIGHTS = {
    "capacity_fade": 0.40,
    "impedance_rise": 0.30,
    "temperature_anomaly": 0.20,
    "voltage_anomaly": 0.10,
}

# Binary risk threshold (on 0-1 risk score)
RISK_THRESHOLD = 0.5

# ──────────────────────────────────────────────────────────────────────
# Engineering safety thresholds
# ──────────────────────────────────────────────────────────────────────
SAFETY_THRESHOLDS = {
    "temperature": {
        "normal_max": 40.0,       # °C
        "warning_max": 50.0,      # °C
        "critical_max": 60.0,     # °C
    },
    "capacity_fade_pct": {
        "normal_max": 10.0,       # %
        "warning_max": 20.0,      # %
        "critical_max": 30.0,     # %
    },
    "impedance_rise_pct": {
        "normal_max": 30.0,       # %
        "warning_max": 50.0,      # %
        "critical_max": 80.0,     # %
    },
    "voltage_deviation": {
        "normal_max": 0.10,       # V
        "warning_max": 0.20,      # V
        "critical_max": 0.35,     # V
    },
    "temp_trend": {
        "normal_max": 0.3,        # °C/cycle
        "warning_max": 0.6,       # °C/cycle
        "critical_max": 1.0,      # °C/cycle
    },
    "capacity_fade_rate": {
        "normal_max": 0.005,      # Ah/cycle
        "warning_max": 0.010,     # Ah/cycle
        "critical_max": 0.020,    # Ah/cycle
    },
    "voltage_start": {
        "normal_max": 4.25,       # V (fully charged 18650)
        "warning_max": 4.35,      # V
        "critical_max": 4.50,     # V
    },
    "voltage_end": {
        "normal_max": 3.0,        # V (cutoff voltage)
        "warning_max": 2.7,       # V — inverted: LOWER is worse
        "critical_max": 2.5,      # V
    },
    "re": {
        "normal_max": 0.05,       # Ω
        "warning_max": 0.08,      # Ω
        "critical_max": 0.12,     # Ω
    },
    "rct": {
        "normal_max": 0.04,       # Ω
        "warning_max": 0.07,      # Ω
        "critical_max": 0.10,     # Ω
    },
    "soc_estimated": {
        "normal_max": 100.0,      # % — high SoC is normal
        "warning_max": 120.0,     # % — anomalous reading
        "critical_max": 150.0,    # %
    },
    "duration": {
        "normal_max": 3600.0,     # s (1 hour)
        "warning_max": 5400.0,    # s (1.5 hours)
        "critical_max": 7200.0,   # s (2 hours)
    },
    "current_mean": {
        "normal_max": 2.5,        # A
        "warning_max": 3.5,       # A
        "critical_max": 5.0,      # A
    },
    "temp_change_rate": {
        "normal_max": 0.5,        # °C/cycle
        "warning_max": 1.0,       # °C/cycle
        "critical_max": 2.0,      # °C/cycle
    },
    "re_trend": {
        "normal_max": 0.002,      # Ω/cycle
        "warning_max": 0.005,     # Ω/cycle
        "critical_max": 0.010,    # Ω/cycle
    },
    "rct_trend": {
        "normal_max": 0.002,      # Ω/cycle
        "warning_max": 0.005,     # Ω/cycle
        "critical_max": 0.010,    # Ω/cycle
    },
}

# Mapping from feature names to human-readable descriptions
FEATURE_DESCRIPTIONS = {
    "temp_max": "Maximum temperature during discharge",
    "temp_mean": "Mean temperature during discharge",
    "temp_min": "Minimum temperature during discharge",
    "temp_trend": "Temperature trend over recent cycles",
    "temp_change_rate": "Rate of temperature change",
    "voltage_start": "Voltage at start of discharge",
    "voltage_end": "Voltage at end of discharge",
    "voltage_drop": "Voltage drop during discharge",
    "voltage_std": "Voltage standard deviation during discharge",
    "voltage_trend": "Voltage drop trend over recent cycles",
    "voltage_deviation": "Voltage deviation from rolling average",
    "current_mean": "Mean discharge current",
    "current_std": "Discharge current standard deviation",
    "capacity": "Discharge capacity",
    "capacity_fade_pct": "Capacity fade percentage from initial",
    "capacity_fade_rate": "Rate of capacity loss per cycle",
    "re": "Electrolyte resistance",
    "rct": "Charge transfer resistance",
    "re_trend": "Electrolyte resistance trend",
    "rct_trend": "Charge transfer resistance trend",
    "impedance_rise_pct": "Impedance rise percentage from initial",
    "cycle_count": "Total number of cycles completed",
    "cycle_index": "Cycle index in sequence",
    "duration": "Discharge duration",
    "soc_estimated": "Estimated state of charge",
}

# Mapping from feature names to units
FEATURE_UNITS = {
    "temp_max": "°C",
    "temp_mean": "°C",
    "temp_min": "°C",
    "temp_trend": "°C/cycle",
    "temp_change_rate": "°C/cycle",
    "voltage_start": "V",
    "voltage_end": "V",
    "voltage_drop": "V",
    "voltage_std": "V",
    "voltage_trend": "V/cycle",
    "voltage_deviation": "V",
    "current_mean": "A",
    "current_std": "A",
    "capacity": "Ah",
    "capacity_fade_pct": "%",
    "capacity_fade_rate": "Ah/cycle",
    "re": "Ω",
    "rct": "Ω",
    "re_trend": "Ω/cycle",
    "rct_trend": "Ω/cycle",
    "impedance_rise_pct": "%",
    "cycle_count": "cycles",
    "cycle_index": "cycles",
    "duration": "s",
    "soc_estimated": "%",
}

# ──────────────────────────────────────────────────────────────────────
# Model hyperparameters
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

MODEL_PARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
    },
    "decision_tree": {
        "max_depth": 5,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "random_state": RANDOM_SEED,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 3,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
    },
}

# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
# Number of top features to report in explanations
TOP_K_FEATURES = 5

# SHAP/LIME comparison: number of test samples to explain
N_EXPLAIN_SAMPLES = 100

# Warning module: number of representative examples to generate
N_WARNING_EXAMPLES = 20

# Test split ratio (for within-battery evaluation)
TEST_SIZE = 0.2

# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_STYLE = "seaborn-v0_8-whitegrid"

# Color palette for battery cells
CELL_COLORS = {
    "B0005": "#1f77b4",
    "B0006": "#ff7f0e",
    "B0007": "#2ca02c",
    "B0018": "#d62728",
}

# Color palette for risk levels
RISK_COLORS = {
    "LOW": "#2ca02c",
    "MEDIUM": "#ff7f0e",
    "HIGH": "#d62728",
    "CRITICAL": "#7f0000",
}
