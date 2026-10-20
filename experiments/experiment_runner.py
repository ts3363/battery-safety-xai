"""
Master Experiment Runner for the XAI Battery Safety Framework.

Orchestrates the entire research pipeline:
1. Loads and validates the NASA Li-ion Battery Aging Dataset (B0005, B0006, B0007, B0018)
2. Executes the data preprocessing and quality pipeline
3. Generates domain-aware raw and temporal trend features
4. Assigns multi-factor degradation risk labels
5. Executes Experiments 1 through 6:
   - E1: Predictive model performance comparison under LOBO-CV
   - E2: Raw vs. trend-aware feature importance and SHAP analysis
   - E3: Local explanations (SHAP + LIME) and warning generation
   - E4: Cross-battery explanation stability analysis
   - E5: Warning quality, factor correctness, and consistency evaluation
   - E6: Operational value comparison (XAI warnings vs. black-box predictions)
6. Compiles publication figures and tables into results/
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Project root import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

# Import modules from src/
try:
    from src.data_loader import create_combined_dataset, load_all_batteries
    from src.evaluation import evaluate_all_models, save_evaluation_results
    from src.explainability import explain_all_models
    from src.feature_engineering import (
        engineer_features,
        get_all_feature_names,
        get_raw_feature_names,
        get_trend_feature_names,
    )
    from src.models import get_best_model, leave_one_battery_out_cv, train_all_models
    from src.preprocessing import preprocess_pipeline
    from src.risk_labeling import assign_risk_labels
    from src.visualization import (
        plot_calibration_curves,
        plot_degradation_curves,
        plot_feature_correlation,
        plot_feature_distributions,
    )
    from src.warning_module import (
        format_warning_text,
        generate_batch_warnings,
        save_warning_examples,
    )
except ImportError as err:
    print(f"Warning during module import from src: {err}")

# Import experiment modules
from experiments.e1_model_comparison import run_experiment_1
from experiments.e2_feature_importance import run_experiment_2
from experiments.e3_local_explanations import run_experiment_3
from experiments.e4_explanation_stability import run_experiment_4
from experiments.e5_warning_evaluation import run_experiment_5
from experiments.e6_warning_vs_prediction import run_experiment_6


def load_and_prepare_dataset() -> Tuple[Optional[pd.DataFrame], Optional[list]]:
    """
    Load raw NASA .mat files or load preprocessed dataset from disk.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[list]]
        The processed DataFrame and list of feature names, or (None, None) if files are missing.
    """
    processed_path = config.DATA_PROCESSED_DIR / "battery_features.csv"
    if processed_path.exists():
        print(f"\n[Step 1-3] Loading cached processed dataset from: {processed_path}")
        df = pd.read_csv(processed_path)
        feature_names = [f for f in get_all_feature_names() if f in df.columns]
        return df, feature_names

    # Check for raw NASA .mat files in config.DATA_RAW_DIR
    mat_files = list(config.DATA_RAW_DIR.glob("*.mat"))
    required_cells = [f"{cell}.mat" for cell in config.BATTERY_CELLS]
    missing_cells = [c for c in required_cells if not (config.DATA_RAW_DIR / c).exists()]

    if missing_cells:
        print("\n" + "=" * 78)
        print("ERROR: NASA Li-ion Battery Aging dataset files not found!")
        print("=" * 78)
        print(f"Directory checked: {config.DATA_RAW_DIR.resolve()}")
        print(f"Missing required cell file(s): {', '.join(missing_cells)}")
        print("\nPlease download the official NASA Li-ion Battery dataset (.mat files):")
        print("  Source: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets")
        print("  Cells required: B0005.mat, B0006.mat, B0007.mat, B0018.mat")
        print(f"Place the downloaded .mat files directly into:\n  {config.DATA_RAW_DIR.resolve()}")
        print("=" * 78 + "\n")
        return None, None

    print(f"\n[Step 1] Loading raw NASA .mat files from {config.DATA_RAW_DIR}...")
    battery_dict = load_all_batteries(config.DATA_RAW_DIR)
    df_raw = create_combined_dataset(battery_dict)
    print(f"Combined raw cycles: {len(df_raw)}")

    print("\n[Step 2] Executing preprocessing pipeline...")
    df_clean = preprocess_pipeline(df_raw)
    print(f"Cleaned dataset cycles: {len(df_clean)}")

    print("\n[Step 3] Computing domain-aware feature engineering...")
    df_features = engineer_features(df_clean)

    print("\n[Step 3b] Computing multi-factor degradation risk labels...")
    df_final = assign_risk_labels(df_features)

    # Fill remaining NaN values in numeric features to avoid model errors
    numeric_cols = df_final.select_dtypes(include="number").columns
    nan_before = df_final[numeric_cols].isna().sum().sum()
    if nan_before > 0:
        print(f"  Filling {nan_before} remaining NaN values in numeric features...")
        df_final[numeric_cols] = df_final[numeric_cols].fillna(0.0)

    # Cache processed dataset for future instant execution
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(processed_path, index=False)
    print(f"Saved processed dataset to: {processed_path}")

    feature_names = [f for f in get_all_feature_names() if f in df_final.columns]
    return df_final, feature_names


def run_all_experiments() -> Dict[str, Any]:
    """
    Master orchestrator: executes complete pipeline from data preparation to E1-E6.

    Returns
    -------
    Dict[str, Any]
        Dictionary of experiment results, execution status, and runtimes.
    """
    total_start = time.time()
    print("=" * 78)
    print("   XAI BATTERY SAFETY FRAMEWORK - FULL EXPERIMENTAL SUITE   ")
    print("=" * 78)

    # 1. Load and prepare dataset
    df, feature_names = load_and_prepare_dataset()
    if df is None:
        print("\nPipeline terminated early because input dataset was not found.")
        return {"status": "ABORTED_DATA_MISSING"}

    experiment_status = {}
    experiment_results = {}

    # Helper runner
    def _run_step(exp_id: str, exp_name: str, runner_fn):
        print(f"\n>>> Running {exp_id}: {exp_name}...")
        t0 = time.time()
        try:
            res = runner_fn()
            dt = round(time.time() - t0, 2)
            experiment_status[exp_id] = {"name": exp_name, "status": "SUCCESS", "time_sec": dt}
            experiment_results[exp_id] = res
            print(f">>> {exp_id} completed successfully in {dt}s.")
        except Exception as e:
            dt = round(time.time() - t0, 2)
            experiment_status[exp_id] = {"name": exp_name, "status": f"FAILED ({str(e)})", "time_sec": dt}
            print(f">>> {exp_id} failed with error: {e}")

    # Step 4: Run E1-E6
    # E1: Model Comparison
    _run_step(
        "E1",
        "Predictive Model Comparison (LOBO-CV)",
        lambda: run_experiment_1(df, feature_names=feature_names),
    )

    e1_results = experiment_results.get("E1", {})
    model_results = e1_results.get("model_results", None)

    # E2: Feature Importance (Raw vs Trend)
    _run_step(
        "E2",
        "Raw vs. Trend-Aware Feature Importance",
        lambda: run_experiment_2(df),
    )

    # E3: Local Explanations & Warnings
    _run_step(
        "E3",
        "Local Explanations (SHAP + LIME) and Warnings",
        lambda: run_experiment_3(df, feature_names=feature_names, model_results=model_results),
    )

    # E4: Explanation Stability across Batteries
    _run_step(
        "E4",
        "Cross-Battery Explanation Stability (LOBO-CV)",
        lambda: run_experiment_4(df, feature_names=feature_names, model_results=model_results),
    )

    # E5: Safety Warning Quality Evaluation
    _run_step(
        "E5",
        "Safety Warning Quality Evaluation",
        lambda: run_experiment_5(df, feature_names=feature_names, model_results=model_results),
    )

    e5_results = experiment_results.get("E5", {})
    all_warnings = e5_results.get("warnings", None)

    # E6: Full Framework vs. Prediction-Only Baseline
    _run_step(
        "E6",
        "Full XAI Framework vs. Prediction-Only Baseline",
        lambda: run_experiment_6(
            df,
            feature_names=feature_names,
            model_results=model_results,
            warnings=all_warnings,
        ),
    )

    # Step 5: Generate remaining global visualizations
    print("\n[Step 5] Generating global dataset visualizations...")
    try:
        plot_degradation_curves(df)
        print("  - Generated degradation_curves.png")

        top_feats = [f for f in feature_names[:9] if f in df.columns]
        plot_feature_distributions(df, top_feats)
        print("  - Generated feature_distributions.png")

        corr_feats = [f for f in feature_names[:12] if f in df.columns]
        plot_feature_correlation(df, corr_feats)
        print("  - Generated feature_correlation.png")

        if model_results:
            calib_data = {}
            for m_name, m_res in model_results.items():
                if "y_true" in m_res and "y_prob" in m_res and m_res["y_prob"] is not None:
                    calib_data[m_name] = {"y_true": m_res["y_true"], "y_prob": m_res["y_prob"]}
            if calib_data:
                plot_calibration_curves(calib_data)
                print("  - Generated calibration_curves.png")
    except Exception as e:
        print(f"Notice during global visualization: {e}")

    # Step 6: Save consolidated summary
    print("\n[Step 6] Compiling and saving final experimental summary...")
    total_time = round(time.time() - total_start, 2)
    summary_data = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": total_time,
        "experiments": experiment_status,
    }

    summary_json_path = config.RESULTS_DIR / "experiment_execution_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved run summary to: {summary_json_path}")

    # Print Final Summary Table
    print("\n" + "=" * 78)
    print("                      EXPERIMENTAL RUN SUMMARY                      ")
    print("=" * 78)
    print(f"{'Experiment':<6} {'Name':<44} {'Status':<16} {'Runtime':<10}")
    print("-" * 78)
    for exp_id, info in experiment_status.items():
        print(f"{exp_id:<6} {info['name'][:42]:<44} {info['status']:<16} {info['time_sec']:>6.1f}s")
    print("-" * 78)
    print(f"Total Pipeline Runtime: {total_time:.2f} seconds")
    print(f"Output Figures: {config.FIGURES_DIR.resolve()}")
    print(f"Output Tables:  {config.TABLES_DIR.resolve()}")
    print(f"Output Warnings:{config.WARNINGS_DIR.resolve()}")
    print("=" * 78 + "\n")

    return {
        "summary": summary_data,
        "experiment_results": experiment_results,
    }


if __name__ == "__main__":
    run_all_experiments()
