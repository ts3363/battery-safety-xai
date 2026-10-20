"""
Experiment 2: Trend-Aware vs. Raw Feature Importance.

Analyzes the value of incorporating temporal/trend-aware features by evaluating:
1. Model predictive performance under three feature configurations:
   - Raw features only (instantaneous measurements)
   - Trend features only (rates of change, moving window slopes)
   - All features combined (raw + trend features)
2. Global feature importance comparison via SHAP values
3. Visual comparison of top risk drivers between raw and trend-aware sets
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Project root import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

from src.evaluation import compute_classification_metrics
from src.explainability import (
    compute_shap_values,
    get_global_feature_importance,
)
from src.feature_engineering import (
    get_all_feature_names,
    get_raw_feature_names,
    get_trend_feature_names,
)
from src.models import get_best_model, leave_one_battery_out_cv, train_all_models
from src.visualization import (
    plot_feature_comparison_trend_vs_raw,
    plot_shap_bar,
)


def run_experiment_2(
    df: pd.DataFrame,
    target: str = "risk_binary",
    model_choice: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Experiment 2: Compare raw features, trend features, and all features.

    Parameters
    ----------
    df : pd.DataFrame
        Battery dataset with engineered features and risk labels.
    target : str, default='risk_binary'
        Target column name.
    model_choice : Optional[str], optional
        Model architecture to evaluate (e.g. 'random_forest' or 'xgboost').
        If None, automatically selects best performing model on all features.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing results for each feature set:
        - 'metrics_df': Comparative performance table across feature sets
        - 'importance_raw': Global feature importance for raw features
        - 'importance_trend': Global feature importance for trend features
        - 'importance_all': Global feature importance for all features
        - 'results_by_set': Detailed predictions and metrics per feature set
        - 'figures': Generated matplotlib figures
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: RAW VS. TREND-AWARE FEATURE IMPORTANCE")
    print("=" * 70)

    # 1. Retrieve feature groups
    raw_features = [f for f in get_raw_feature_names() if f in df.columns]
    trend_features = [f for f in get_trend_feature_names() if f in df.columns]
    all_features = [f for f in get_all_feature_names() if f in df.columns]

    # Fallback if lists are empty or mismatch
    if not raw_features:
        raw_features = [c for c in df.columns if not any(x in c for x in ["trend", "rate", "fade", "deviation"])
                        and c not in ["cell_name", "cycle_count", target, "risk_score"]
                        and np.issubdtype(df[c].dtype, np.number)]
    if not trend_features:
        trend_features = [c for c in df.columns if any(x in c for x in ["trend", "rate", "fade", "deviation"])
                          and c not in ["cell_name", "cycle_count", target, "risk_score"]
                          and np.issubdtype(df[c].dtype, np.number)]
    if not all_features:
        all_features = list(dict.fromkeys(raw_features + trend_features))

    print(f"Total Raw Features:   {len(raw_features)}")
    print(f"Total Trend Features: {len(trend_features)}")
    print(f"Total All Features:   {len(all_features)}")

    feature_sets = {
        "raw_only": raw_features,
        "trend_only": trend_features,
        "all_features": all_features,
    }

    # 2. Evaluate LOBO-CV on each feature set
    results_by_set = {}
    metrics_records = []
    importance_by_set = {}

    for set_name, feats in feature_sets.items():
        print(f"\n--- Evaluating Feature Set: {set_name} ({len(feats)} features) ---")
        cv_results = leave_one_battery_out_cv(df, feature_names=feats, target=target)

        # leave_one_battery_out_cv returns a single-model result dict directly
        selected_model = cv_results.get("model_name", "random_forest")
        if model_choice is not None:
            selected_model = model_choice

        # Add backward-compatible keys
        cv_results["y_true"] = cv_results["all_y_true"]
        cv_results["y_pred"] = cv_results["all_y_pred"]
        cv_results["y_prob"] = cv_results["all_y_prob"]
        cv_results["models"] = [fr["model"] for fr in cv_results.get("results", [])]
        cv_results["fold_results"] = cv_results.get("results", [])

        # Use result directly
        model_res = cv_results
        y_true = model_res["all_y_true"]
        y_pred = model_res["all_y_pred"]
        y_prob = model_res.get("all_y_prob", None)

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        metrics["feature_set"] = set_name
        metrics["model"] = selected_model
        metrics["n_features"] = len(feats)
        metrics_records.append(metrics)

        # Compute SHAP values for global feature importance
        print(f"Computing SHAP values for {set_name} using {selected_model}...")
        X_sample = df[feats].dropna().sample(
            n=min(len(df), config.N_EXPLAIN_SAMPLES),
            random_state=config.RANDOM_SEED,
        )

        # Get fitted estimator from the first fold or retrain
        fitted_model = None
        fold_list = model_res.get("fold_results", model_res.get("results", []))
        if fold_list and len(fold_list) > 0:
            fitted_model = fold_list[0].get("model", None)
        elif "models" in model_res and model_res["models"]:
            fitted_model = model_res["models"][0]

        try:
            shap_vals = compute_shap_values(fitted_model, X_sample, model_name=selected_model)
            global_imp = get_global_feature_importance(shap_vals, feats)
        except Exception as e:
            print(f"Warning: SHAP computation fallback for {set_name}: {e}")
            # Fallback importance from feature importances or variance
            if hasattr(fitted_model, "feature_importances_"):
                imp_vals = fitted_model.feature_importances_
            else:
                imp_vals = np.ones(len(feats)) / len(feats)
            global_imp = pd.DataFrame({"feature": feats, "mean_abs_shap": imp_vals}).sort_values(
                "mean_abs_shap", ascending=False
            )

        importance_by_set[set_name] = global_imp
        results_by_set[set_name] = {
            "cv_results": cv_results,
            "metrics": metrics,
            "importance": global_imp,
            "features": feats,
        }

    # 3. Create comparison metrics DataFrame
    metrics_df = pd.DataFrame(metrics_records)
    cols = ["feature_set", "n_features", "precision", "recall", "f1", "roc_auc"]
    cols = [c for c in cols if c in metrics_df.columns]
    metrics_df = metrics_df[cols]

    print("\nFeature Set Comparison Summary:")
    print(metrics_df.to_string(index=False))

    # Save metrics table
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    comp_table_path = config.TABLES_DIR / "e2_feature_set_comparison.csv"
    metrics_df.to_csv(comp_table_path, index=False)
    print(f"Saved feature set comparison table to: {comp_table_path}")

    # Save feature importance tables
    for s_name, imp_df in importance_by_set.items():
        imp_path = config.TABLES_DIR / f"e2_importance_{s_name}.csv"
        imp_df.to_csv(imp_path, index=False)

    # 4. Generate comparison plots
    print("\nGenerating visualization comparisons...")
    figures = {}

    importance_raw = importance_by_set.get("raw_only", pd.DataFrame())
    importance_all = importance_by_set.get("all_features", pd.DataFrame())

    if not importance_raw.empty and not importance_all.empty:
        fig_comp = plot_feature_comparison_trend_vs_raw(importance_raw, importance_all)
        figures["trend_vs_raw"] = fig_comp

    if not importance_all.empty:
        fig_bar = plot_shap_bar(importance_all, model_name=f"{selected_model}_all_features")
        figures["shap_bar_all"] = fig_bar

    # Feature Set Performance Bar Chart
    fig_perf, ax = plt.subplots(figsize=(8, 5))
    metrics_melted = metrics_df.melt(
        id_vars=["feature_set"],
        value_vars=[c for c in ["f1", "roc_auc", "recall"] if c in metrics_df.columns],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(data=metrics_melted, x="feature_set", y="Score", hue="Metric", ax=ax, palette="Set1")
    ax.set_title("Performance by Feature Engineering Strategy", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    perf_fig_path = config.FIGURES_DIR / "e2_feature_set_performance.png"
    fig_perf.savefig(perf_fig_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    figures["feature_set_performance"] = fig_perf

    print(f"Saved figures to: {config.FIGURES_DIR}")
    print("=" * 70 + "\n")

    return {
        "metrics_df": metrics_df,
        "importance_raw": importance_raw,
        "importance_trend": importance_by_set.get("trend_only", pd.DataFrame()),
        "importance_all": importance_all,
        "results_by_set": results_by_set,
        "figures": figures,
    }


def _load_data_safely() -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """Helper to load real processed data for standalone execution."""
    proc_csv = config.DATA_PROCESSED_DIR / "battery_features.csv"
    if proc_csv.exists():
        print(f"Loading processed data from {proc_csv}...")
        df = pd.read_csv(proc_csv)
        return df, get_all_feature_names()

    mat_files = list(config.DATA_RAW_DIR.glob("*.mat"))
    if mat_files:
        print(f"Loading raw NASA files...")
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

    print("=" * 70)
    print("NOTICE: NASA .mat files not found in data/raw.")
    print("Please download B0005.mat, B0006.mat, B0007.mat, B0018.mat from:")
    print("https://data.nasa.gov/dataset/li-ion-battery-aging-datasets")
    print("and place them into: data/raw/")
    print("=" * 70)
    return None, None


if __name__ == "__main__":
    df, _ = _load_data_safely()
    if df is not None:
        run_experiment_2(df)
    else:
        print("Cannot run Experiment 2 without dataset.")
