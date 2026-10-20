"""
Experiment 1: Model Comparison across Battery Cells.

Evaluates and compares the predictive performance of four machine learning models
(Logistic Regression, Decision Tree, Random Forest, XGBoost) using
Leave-One-Battery-Out Cross-Validation (LOBO-CV).

Generates:
- Comprehensive performance comparison table (Precision, Recall, F1, ROC-AUC)
- ROC curves with AUC annotations
- Precision-Recall curves
- Confusion matrices
- Grouped bar chart comparing model metrics
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project root import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

# Import modules from src
from src.evaluation import (
    compute_classification_metrics,
    evaluate_all_models,
    save_evaluation_results,
)
from src.feature_engineering import get_all_feature_names
from src.models import get_best_model, leave_one_battery_out_cv, train_all_models
from src.visualization import (
    plot_confusion_matrices,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
)


def run_experiment_1(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    target: str = "risk_binary",
) -> Dict[str, Any]:
    """
    Run Experiment 1: Compare predictive performance of all 4 models using LOBO-CV.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed, feature-engineered, and risk-labeled battery dataset.
    feature_names : Optional[List[str]], optional
        List of features to use. If None, retrieves all feature names from feature_engineering.
    target : str, default='risk_binary'
        Target risk label column name.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model_results': Detailed predictions, probabilities, and fold data per model
        - 'metrics_df': Consolidated performance metrics DataFrame
        - 'best_model': Name of the top-performing model
        - 'figures': Generated matplotlib Figure objects
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: MODEL PERFORMANCE COMPARISON (LOBO-CV)")
    print("=" * 70)

    if feature_names is None:
        try:
            feature_names = get_all_feature_names()
            # Ensure features exist in dataframe
            feature_names = [f for f in feature_names if f in df.columns]
        except Exception:
            feature_names = [
                c
                for c in df.columns
                if c not in ["cell_name", "cycle_count", "risk_binary", "risk_score", target]
                and np.issubdtype(df[c].dtype, np.number)
            ]

    print(f"Dataset shape: {df.shape}")
    print(f"Features count: {len(feature_names)}")
    print(f"Target column: {target}")
    if target in df.columns:
        print(f"Class distribution:\n{df[target].value_counts(normalize=True).round(3)}")

    # Fill NaN in features to avoid model errors
    numeric_feats = [f for f in feature_names if f in df.columns]
    df = df.copy()
    df[numeric_feats] = df[numeric_feats].fillna(0.0)

    # 1. Train models using Leave-One-Battery-Out Cross-Validation
    print("\nRunning Leave-One-Battery-Out Cross-Validation (LOBO-CV)...")
    model_results = train_all_models(df, feature_names=feature_names, target=target)

    # 2. Compute classification metrics for each model
    metrics_records = []
    for model_name, res in model_results.items():
        y_true = res["all_y_true"]
        y_pred = res["all_y_pred"]
        y_prob = res.get("all_y_prob", None)
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        metrics["model"] = model_name
        metrics_records.append(metrics)
        # Store metrics back into result dictionary
        res["metrics"] = metrics
        # Also store flat y_true/y_pred for compatibility
        res["y_true"] = y_true
        res["y_pred"] = y_pred
        # Store 1D positive-class probabilities for visualization compatibility
        _yp = np.asarray(y_prob)
        res["y_prob"] = _yp[:, 1] if _yp.ndim == 2 else _yp

    metrics_df = pd.DataFrame(metrics_records)
    # Reorder columns with model first
    cols = ["model"] + [c for c in metrics_df.columns if c != "model"]
    metrics_df = metrics_df[cols]

    print("\nLOBO-CV Performance Summary:")
    print(metrics_df.to_string(index=False))

    # 3. Determine best model
    try:
        best_model_name, _ = get_best_model(model_results)
    except Exception:
        # Fallback to model with highest F1 score
        best_model_name = metrics_df.sort_values("f1", ascending=False).iloc[0]["model"]

    print(f"\nBest performing model based on LOBO-CV: {best_model_name}")

    # 4. Save metrics table to config.TABLES_DIR
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = config.TABLES_DIR / "e1_model_comparison.csv"
    metrics_df.to_csv(table_path, index=False)
    print(f"Saved metrics table to: {table_path}")

    # 5. Generate and save publication visualizations
    print("\nGenerating visualizations...")
    figures = {}

    fig_comp = plot_model_comparison(metrics_df)
    figures["model_comparison"] = fig_comp

    fig_roc = plot_roc_curves(model_results)
    figures["roc_curves"] = fig_roc

    fig_pr = plot_pr_curves(model_results)
    figures["pr_curves"] = fig_pr

    fig_cm = plot_confusion_matrices(model_results)
    figures["confusion_matrices"] = fig_cm

    print(f"Saved all figures to: {config.FIGURES_DIR}")
    print("=" * 70 + "\n")

    return {
        "model_results": model_results,
        "metrics_df": metrics_df,
        "best_model": best_model_name,
        "figures": figures,
    }


def _load_data_safely() -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """Helper to load real or synthetic data for standalone testing."""
    # Check processed CSV
    proc_csv = config.DATA_PROCESSED_DIR / "battery_features.csv"
    if proc_csv.exists():
        print(f"Loading processed data from {proc_csv}...")
        df = pd.read_csv(proc_csv)
        feature_names = get_all_feature_names()
        return df, [f for f in feature_names if f in df.columns]

    # Check raw .mat files
    mat_files = list(config.DATA_RAW_DIR.glob("*.mat"))
    if mat_files:
        print(f"Loading and processing {len(mat_files)} raw NASA .mat files...")
        from src.data_loader import create_combined_dataset, load_all_batteries
        from src.feature_engineering import engineer_features
        from src.preprocessing import preprocess_pipeline
        from src.risk_labeling import assign_risk_labels

        raw_dict = load_all_batteries(config.DATA_RAW_DIR)
        df_raw = create_combined_dataset(raw_dict)
        df_clean = preprocess_pipeline(df_raw)
        df_feat = engineer_features(df_clean)
        df_labeled = assign_risk_labels(df_feat)
        return df_labeled, get_all_feature_names()

    # If raw data does not exist, notify user clearly
    print("=" * 70)
    print("NOTICE: NASA .mat files not found in data/raw.")
    print("Please download B0005.mat, B0006.mat, B0007.mat, B0018.mat from:")
    print("https://data.nasa.gov/dataset/li-ion-battery-aging-datasets")
    print("and place them into: data/raw/")
    print("=" * 70)
    return None, None


if __name__ == "__main__":
    df, feature_names = _load_data_safely()
    if df is not None:
        run_experiment_1(df, feature_names)
    else:
        print("Cannot run Experiment 1 without dataset.")
