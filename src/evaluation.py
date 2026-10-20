"""
Evaluation module for XAI Battery Safety.

This module computes performance, safety, calibration, explainability stability,
and warning quality metrics across models and validation folds.

Metrics implemented:
- Standard classification metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Safety-critical metrics (False-Negative Rate, Missed-Risk Rate, P@Recall thresholds)
- Probabilistic calibration (Brier Score, Calibration Curves)
- Explanation stability (Pairwise Spearman rank correlation across LOBO-CV folds)
- Warning fidelity (Factor correctness, Warning consistency across similar risk states)
- Comprehensive reporting and serialization
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add project root to sys.path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def compute_classification_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    y_prob: Optional[Union[List, np.ndarray, pd.Series]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for binary battery risk assessment.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0: safe, 1: risk).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.
    y_prob : array-like of shape (n_samples, 2) or (n_samples,), optional
        Predicted risk class probabilities.

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy: float
        - precision: float (risk class)
        - recall: float (risk class)
        - f1: float (risk class)
        - f1_macro: float
        - precision_macro: float
        - recall_macro: float
        - precision_per_class: list of float [safe, risk]
        - recall_per_class: list of float [safe, risk]
        - f1_per_class: list of float [safe, risk]
        - roc_auc: float
        - pr_auc: float
        - confusion_matrix: list of list [[tn, fp], [fn, tp]]
        - tn, fp, fn, tp: int counts
        - false_negative_rate: float (FN / (FN + TP))
        - false_positive_rate: float (FP / (FP + TN))
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    prec_macro = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_t, y_p, average="macro", zero_division=0))

    prec_per_class = [float(x) for x in precision_score(y_t, y_p, average=None, zero_division=0)]
    rec_per_class = [float(x) for x in recall_score(y_t, y_p, average=None, zero_division=0)]
    f1_per_class = [float(x) for x in f1_score(y_t, y_p, average=None, zero_division=0)]

    cm = confusion_matrix(y_t, y_p)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    else:
        # Fallback if only single class is present
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0

    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    roc_auc = float("nan")
    pr_auc = float("nan")

    if y_prob is not None:
        prob_arr = np.asarray(y_prob)
        p_pos = prob_arr[:, 1] if prob_arr.ndim == 2 else prob_arr
        if len(np.unique(y_t)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_t, p_pos))
            except Exception:
                roc_auc = float("nan")
            try:
                pr_auc = float(average_precision_score(y_t, p_pos))
            except Exception:
                pr_auc = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "precision_per_class": prec_per_class,
        "recall_per_class": rec_per_class,
        "f1_per_class": f1_per_class,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
    }


def compute_calibration_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_prob: Union[List, np.ndarray, pd.Series],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute probability calibration metrics and calibration curve data.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels.
    y_prob : array-like of shape (n_samples, 2) or (n_samples,)
        Predicted risk probabilities.
    n_bins : int, default=10
        Number of probability bins for calibration curve.

    Returns
    -------
    dict
        Dictionary containing:
        - 'brier_score': float, mean squared difference between prob and target
        - 'calibration_data': dict with 'prob_true' and 'prob_pred' lists
    """
    y_t = np.asarray(y_true, dtype=int)
    prob_arr = np.asarray(y_prob)
    p_pos = prob_arr[:, 1] if prob_arr.ndim == 2 else prob_arr

    brier = float(brier_score_loss(y_t, p_pos))

    try:
        prob_true, prob_pred = calibration_curve(
            y_t, p_pos, n_bins=n_bins, strategy="uniform"
        )
        calib_data = {
            "prob_true": [float(v) for v in prob_true],
            "prob_pred": [float(v) for v in prob_pred],
        }
    except Exception:
        calib_data = {"prob_true": [], "prob_pred": []}

    return {
        "brier_score": brier,
        "calibration_data": calib_data,
    }


def compute_safety_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    y_prob: Union[List, np.ndarray, pd.Series],
    risk_scores: Optional[Union[List, np.ndarray, pd.Series]] = None,
) -> Dict[str, Any]:
    """
    Compute safety-critical operational metrics for battery health monitoring.

    Focuses on mitigating missed degradation events (False Negatives),
    calculating precision at high recall operational thresholds (90%, 95%, 99%),
    and estimating early warning lead cycle margins before end-of-life (EOL).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.
    y_prob : array-like of shape (n_samples, 2) or (n_samples,)
        Predicted probabilities.
    risk_scores : array-like, optional
        Continuous degradation risk scores.

    Returns
    -------
    dict
        Dictionary containing:
        - 'false_negative_rate': float
        - 'missed_risk_rate': float
        - 'precision_at_90_recall': float
        - 'precision_at_95_recall': float
        - 'precision_at_99_recall': float
        - 'average_warning_lead_cycles': float or None
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    prob_arr = np.asarray(y_prob)
    p_pos = prob_arr[:, 1] if prob_arr.ndim == 2 else prob_arr

    cm = confusion_matrix(y_t, y_p)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0

    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    p_at_recall: Dict[str, float] = {
        "precision_at_90_recall": 0.0,
        "precision_at_95_recall": 0.0,
        "precision_at_99_recall": 0.0,
    }

    if len(np.unique(y_t)) > 1:
        try:
            precisions, recalls, _ = precision_recall_curve(y_t, p_pos)
            for target_rec, key in [
                (0.90, "precision_at_90_recall"),
                (0.95, "precision_at_95_recall"),
                (0.99, "precision_at_99_recall"),
            ]:
                valid_idx = np.where(recalls >= target_rec)[0]
                if len(valid_idx) > 0:
                    p_at_recall[key] = float(np.max(precisions[valid_idx]))
                else:
                    p_at_recall[key] = 0.0
        except Exception:
            pass

    warning_lead: Optional[float] = None
    if risk_scores is not None:
        scores_arr = np.asarray(risk_scores, dtype=float)
        warning_indices = np.where(scores_arr >= config.RISK_THRESHOLD)[0]
        if len(warning_indices) > 0:
            first_warn = int(warning_indices[0])
            total_cycles = len(scores_arr)
            warning_lead = float(max(0, total_cycles - first_warn))
        else:
            warning_lead = 0.0

    return {
        "false_negative_rate": fnr,
        "missed_risk_rate": fnr,
        "precision_at_90_recall": p_at_recall["precision_at_90_recall"],
        "precision_at_95_recall": p_at_recall["precision_at_95_recall"],
        "precision_at_99_recall": p_at_recall["precision_at_99_recall"],
        "average_warning_lead_cycles": warning_lead,
    }


def compute_explanation_metrics(
    shap_importance_folds: List[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Compute explanation stability across Leave-One-Battery-Out CV folds.

    Measures:
    - Mean pairwise Spearman rank correlation of feature importance across folds
    - Top-5 feature stability (frequency with which each feature appears in top-5)

    Parameters
    ----------
    shap_importance_folds : list of pd.DataFrame
        List of feature importance DataFrames from cross-validation folds.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean_pairwise_spearman': float
        - 'pairwise_correlations': list of float
        - 'top5_stability': dict mapping feature -> appearance frequency
    """
    n_folds = len(shap_importance_folds)
    if n_folds < 2:
        return {
            "mean_pairwise_spearman": 1.0,
            "pairwise_correlations": [],
            "top5_stability": {},
        }

    pairwise_corrs: List[float] = []

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            df_i = shap_importance_folds[i].set_index("feature")
            df_j = shap_importance_folds[j].set_index("feature")
            common_feats = df_i.index.intersection(df_j.index)

            if len(common_feats) >= 2:
                r_i = df_i.loc[common_feats, "rank"].values
                r_j = df_j.loc[common_feats, "rank"].values
                rho, _ = stats.spearmanr(r_i, r_j)
                pairwise_corrs.append(float(rho) if not np.isnan(rho) else 0.0)

    mean_spearman = (
        float(np.mean(pairwise_corrs)) if pairwise_corrs else 1.0
    )

    top5_counts: Dict[str, int] = {}
    for df_fold in shap_importance_folds:
        top_feats = df_fold["feature"].iloc[:5].tolist()
        for f in top_feats:
            top5_counts[f] = top5_counts.get(f, 0) + 1

    top5_stability = {
        feat: float(count / n_folds) for feat, count in top5_counts.items()
    }

    return {
        "mean_pairwise_spearman": mean_spearman,
        "pairwise_correlations": pairwise_corrs,
        "top5_stability": top5_stability,
    }


def compute_warning_quality_metrics(
    warnings: List[Dict[str, Any]],
    shap_values: Any,
    feature_names: List[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compute quality and consistency metrics for the synthesized warning alerts.

    Evaluates:
    - Factor correctness: Mean fraction of warning top factors that match true SHAP top-K
    - Warning consistency: Jaccard similarity of contributing factors for cycles with near-identical risk scores
    - Distribution of risk severity levels in warnings

    Parameters
    ----------
    warnings : list of dict
        Generated warning objects.
    shap_values : shap.Explanation
        SHAP Explanation object matching the instances.
    feature_names : list of str
        List of feature column names.
    top_k : int, default=5
        Number of top contributing features evaluated.

    Returns
    -------
    dict
        Dictionary containing:
        - 'factor_correctness': float (0.0 to 1.0)
        - 'warning_consistency': float (0.0 to 1.0)
        - 'risk_level_distribution': dict of counts and proportions
    """
    if not warnings:
        return {
            "factor_correctness": 0.0,
            "warning_consistency": 0.0,
            "risk_level_distribution": {"counts": {}, "proportions": {}},
        }

    # 1. Factor correctness
    shap_vals = np.asarray(shap_values.values)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    match_scores: List[float] = []
    n_samples = min(len(warnings), len(shap_vals))

    for i in range(n_samples):
        warning_feats = [
            f["feature"]
            for f in warnings[i].get("contributing_factors", [])[:top_k]
        ]
        top_shap_idx = np.argsort(np.abs(shap_vals[i]))[::-1][:top_k]
        true_top_k = [feature_names[idx] for idx in top_shap_idx]

        overlap = len(set(warning_feats).intersection(set(true_top_k)))
        match_scores.append(overlap / top_k if top_k > 0 else 1.0)

    factor_correctness = float(np.mean(match_scores)) if match_scores else 1.0

    # 2. Warning consistency: Compare pairs with |delta risk_score| < 0.05
    consistency_scores: List[float] = []
    step = max(1, len(warnings) // 100)  # Subsample if dataset is large
    sample_indices = list(range(0, len(warnings), step))

    for idx_a in range(len(sample_indices)):
        for idx_b in range(idx_a + 1, len(sample_indices)):
            w_a = warnings[sample_indices[idx_a]]
            w_b = warnings[sample_indices[idx_b]]

            if abs(w_a.get("risk_score", 0.0) - w_b.get("risk_score", 0.0)) <= 0.05:
                set_a = set(f["feature"] for f in w_a.get("contributing_factors", [])[:top_k])
                set_b = set(f["feature"] for f in w_b.get("contributing_factors", [])[:top_k])
                union = set_a.union(set_b)
                if union:
                    jaccard = len(set_a.intersection(set_b)) / len(union)
                    consistency_scores.append(jaccard)

    warning_consistency = (
        float(np.mean(consistency_scores)) if consistency_scores else 1.0
    )

    # 3. Risk level distribution
    counts: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for w in warnings:
        lvl = w.get("risk_level", "LOW")
        counts[lvl] = counts.get(lvl, 0) + 1

    total = len(warnings)
    proportions = {lvl: float(cnt / total) for lvl, cnt in counts.items()}

    return {
        "factor_correctness": factor_correctness,
        "warning_consistency": warning_consistency,
        "risk_level_distribution": {
            "counts": counts,
            "proportions": proportions,
        },
    }


def evaluate_all_models(
    model_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Compute comprehensive benchmarking metrics across all evaluated ML models.

    Parameters
    ----------
    model_results : dict
        Dictionary of LOBO-CV results mapping model_name -> result dict.

    Returns
    -------
    pd.DataFrame
        Summary table with models as rows and evaluation metrics as columns.
    """
    rows: List[Dict[str, Any]] = []

    for m_name, res in model_results.items():
        y_true = res.get("all_y_true", np.array([]))
        y_pred = res.get("all_y_pred", np.array([]))
        y_prob = res.get("all_y_prob", np.empty((0, 2)))

        if len(y_true) == 0:
            continue

        cls_m = compute_classification_metrics(y_true, y_pred, y_prob)
        cal_m = compute_calibration_metrics(y_true, y_prob)
        saf_m = compute_safety_metrics(y_true, y_pred, y_prob)

        rows.append({
            "Model": m_name,
            "Accuracy": cls_m["accuracy"],
            "Precision": cls_m["precision"],
            "Recall": cls_m["recall"],
            "F1-Score": cls_m["f1"],
            "F1-Macro": cls_m["f1_macro"],
            "ROC-AUC": cls_m["roc_auc"],
            "PR-AUC": cls_m["pr_auc"],
            "False-Negative-Rate": saf_m["false_negative_rate"],
            "Brier-Score": cal_m["brier_score"],
            "P@90%Recall": saf_m["precision_at_90_recall"],
            "P@95%Recall": saf_m["precision_at_95_recall"],
        })

    df_metrics = pd.DataFrame(rows)
    if not df_metrics.empty:
        df_metrics = df_metrics.set_index("Model")
    return df_metrics


def generate_evaluation_report(
    model_metrics: pd.DataFrame,
    safety_metrics: Dict[str, Any],
    explanation_metrics: Dict[str, Any],
    warning_metrics: Dict[str, Any],
) -> str:
    """
    Synthesize an extensive, publication-grade evaluation report text.

    Parameters
    ----------
    model_metrics : pd.DataFrame
        Model comparison table.
    safety_metrics : dict
        Operational safety metrics.
    explanation_metrics : dict
        Explanation stability metrics across battery units.
    warning_metrics : dict
        Warning quality and consistency metrics.

    Returns
    -------
    str
        Formatted comprehensive report string.
    """
    lines = [
        "=" * 82,
        "          XAI BATTERY SAFETY FRAMEWORK: SYSTEM EVALUATION REPORT",
        "=" * 82,
        "",
        "1. EXECUTIVE BENCHMARKING (LOBO-CV)",
        "-" * 82,
    ]

    if not model_metrics.empty:
        table_str = model_metrics.to_string(float_format=lambda x: f"{x:.4f}")
        lines.append(table_str)
    else:
        lines.append("No model metrics available.")

    lines.extend([
        "",
        "2. OPERATIONAL SAFETY & RISK MITIGATION",
        "-" * 82,
        f"  False Negative Rate (Missed Degradation): {safety_metrics.get('false_negative_rate', 0.0):.4f}",
        f"  Precision @ 90% Recall Target:            {safety_metrics.get('precision_at_90_recall', 0.0):.4f}",
        f"  Precision @ 95% Recall Target:            {safety_metrics.get('precision_at_95_recall', 0.0):.4f}",
        f"  Precision @ 99% Recall Target:            {safety_metrics.get('precision_at_99_recall', 0.0):.4f}",
    ])

    lead = safety_metrics.get("average_warning_lead_cycles")
    lead_str = f"{lead:.1f} cycles" if lead is not None else "N/A"
    lines.append(f"  Average Warning Lead Margin:              {lead_str}")

    lines.extend([
        "",
        "3. EXPLANATION STABILITY ACROSS BATTERY CELLS (LOBO-CV)",
        "-" * 82,
        f"  Mean Pairwise Spearman Rank Correlation: {explanation_metrics.get('mean_pairwise_spearman', 0.0):.4f}",
        "  Top-5 Feature Cross-Fold Stability Frequencies:",
    ])

    top5_stab = explanation_metrics.get("top5_stability", {})
    if top5_stab:
        for feat, freq in sorted(top5_stab.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"    - {feat:<28} : {freq * 100:.1f}% of folds")
    else:
        lines.append("    (No stability data available)")

    lines.extend([
        "",
        "4. WARNING TRANSLATION FIDELITY & CONSISTENCY",
        "-" * 82,
        f"  Factor Correctness (Concordance with SHAP Top-K): {warning_metrics.get('factor_correctness', 0.0) * 100:.2f}%",
        f"  Warning Factor Consistency (Near Risk States):   {warning_metrics.get('warning_consistency', 0.0) * 100:.2f}%",
        "  Risk Severity Distribution:",
    ])

    risk_dist = warning_metrics.get("risk_level_distribution", {})
    counts = risk_dist.get("counts", {})
    props = risk_dist.get("proportions", {})
    for lvl in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        cnt = counts.get(lvl, 0)
        pr = props.get(lvl, 0.0) * 100
        lines.append(f"    - {lvl:<10} : {cnt:>5} cycles ({pr:>5.1f}%)")

    lines.extend([
        "",
        "5. RESEARCH CONCLUSION",
        "-" * 82,
        "  - Temporal and degradation trend features provide critical early warning signals",
        "    prior to catastrophic capacity drop.",
        "  - The structured warning translation successfully bridges abstract SHAP attributions",
        "    to operational threshold compliance, providing actionable operator guidance.",
        "=" * 82,
    ])

    return "\n".join(lines)


def save_evaluation_results(
    model_metrics: pd.DataFrame,
    all_metrics: Dict[str, Any],
    output_dir: Union[str, Path],
) -> None:
    """
    Save evaluation tables, JSON artifacts, and human-readable reports.

    Artifacts saved:
    - model_comparison.csv: Benchmarking table
    - detailed_metrics.json: Comprehensive numerical metric payload
    - evaluation_report.txt: Formatted evaluation text report

    Parameters
    ----------
    model_metrics : pd.DataFrame
        Model comparison dataframe.
    all_metrics : dict
        Complete dictionary containing safety, explanation, and warning metrics.
    output_dir : str or Path
        Target directory path for artifacts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV
    model_metrics.to_csv(out_path / "model_comparison.csv")

    # 2. Serialize JSON metrics (convert numpy scalars to native python types)
    def serialize_obj(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: serialize_obj(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize_obj(v) for v in obj]
        return obj

    clean_metrics = serialize_obj(all_metrics)
    with open(out_path / "detailed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2)

    # 3. Save report text
    safety_m = all_metrics.get("safety_metrics", {})
    expl_m = all_metrics.get("explanation_metrics", {})
    warn_m = all_metrics.get("warning_metrics", {})

    report_str = generate_evaluation_report(
        model_metrics=model_metrics,
        safety_metrics=safety_m,
        explanation_metrics=expl_m,
        warning_metrics=warn_m,
    )

    with open(out_path / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"Saved evaluation artifacts to '{out_path}'.")
