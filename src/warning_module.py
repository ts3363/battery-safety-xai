"""
Warning module for XAI Battery Safety.

This module is the core innovation of the research framework: translating
abstract model predictions and SHAP feature attributions into structured,
interpretable safety warnings with actionable engineering guidance.

Features:
- Risk level classification (LOW, MEDIUM, HIGH, CRITICAL)
- Physical safety threshold benchmarking
- Inter-model uncertainty and confidence estimation
- Natural language explanation and recommendation synthesis
- Full structured warning generation
- Batch processing and serialization
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Add project root to sys.path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.explainability import get_local_explanation


def classify_risk_level(risk_score: float) -> str:
    """
    Classify a continuous risk score into a standardized safety risk level.

    Threshold intervals:
    - LOW:      risk_score < 0.30
    - MEDIUM:   0.30 <= risk_score < 0.50
    - HIGH:     0.50 <= risk_score <= 0.75
    - CRITICAL: risk_score > 0.75

    Parameters
    ----------
    risk_score : float
        Scalar risk score between 0.0 and 1.0.

    Returns
    -------
    str
        One of 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.
    """
    score = float(risk_score)
    if score < 0.30:
        return "LOW"
    elif score < 0.50:
        return "MEDIUM"
    elif score <= 0.75:
        return "HIGH"
    else:
        return "CRITICAL"


def assess_feature_status(feature_name: str, feature_value: float) -> str:
    """
    Compare a feature measurement against engineering safety thresholds.

    Maps features to safety threshold configurations in config.SAFETY_THRESHOLDS.

    Parameters
    ----------
    feature_name : str
        Name of the feature being evaluated.
    feature_value : float
        Current numerical value of the feature.

    Returns
    -------
    str
        Status label: 'NORMAL', 'WARNING', 'CRITICAL', or 'UNKNOWN'.
    """
    if np.isnan(feature_value):
        return "UNKNOWN"

    # Map aliases / sub-metrics to threshold categories
    category_map = {
        "temp_max": "temperature",
        "temp_mean": "temperature",
        "temp_min": "temperature",
        "temperature": "temperature",
        "capacity_fade_pct": "capacity_fade_pct",
        "impedance_rise_pct": "impedance_rise_pct",
        "voltage_deviation": "voltage_deviation",
        "temp_trend": "temp_trend",
        "temp_change_rate": "temp_change_rate",
        "capacity_fade_rate": "capacity_fade_rate",
        "voltage_start": "voltage_start",
        "voltage_end": "voltage_end",
        "re": "re",
        "rct": "rct",
        "re_trend": "re_trend",
        "rct_trend": "rct_trend",
        "soc_estimated": "soc_estimated",
        "duration": "duration",
        "current_mean": "current_mean",
    }

    threshold_key = category_map.get(feature_name)
    if threshold_key is None or threshold_key not in config.SAFETY_THRESHOLDS:
        return "UNKNOWN"

    thresh = config.SAFETY_THRESHOLDS[threshold_key]
    val = abs(float(feature_value)) if "pct" in threshold_key or "deviation" in threshold_key else float(feature_value)

    if val <= thresh["normal_max"]:
        return "NORMAL"
    elif val <= thresh["warning_max"]:
        return "WARNING"
    else:
        return "CRITICAL"


def estimate_prediction_uncertainty(
    model_results: Dict[str, Dict[str, Any]],
    instance_index: int,
) -> Dict[str, Any]:
    """
    Estimate prediction uncertainty by assessing consensus across models.

    Compares binary predictions for a given cycle instance across all
    evaluated models to determine consensus level and confidence rating.

    Parameters
    ----------
    model_results : dict
        Dictionary of LOBO-CV results mapping model_name -> result dict.
    instance_index : int
        Row index of the instance in concatenated CV predictions.

    Returns
    -------
    dict
        Dictionary containing:
        - 'model_agreement': float, fraction of models agreeing with majority
        - 'confidence': str ('HIGH', 'MODERATE', 'LOW')
        - 'predictions': dict mapping model_name -> predicted class (0 or 1)
    """
    predictions: Dict[str, int] = {}

    for m_name, m_res in model_results.items():
        preds = m_res.get("all_y_pred")
        if preds is not None and len(preds) > instance_index:
            predictions[m_name] = int(preds[instance_index])

    if not predictions:
        return {
            "model_agreement": 1.0,
            "confidence": "HIGH",
            "predictions": {},
        }

    pred_vals = list(predictions.values())
    majority_count = max(pred_vals.count(0), pred_vals.count(1))
    agreement = float(majority_count / len(pred_vals))

    if agreement >= 0.85:
        confidence = "HIGH"
    elif agreement >= 0.70:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    return {
        "model_agreement": agreement,
        "confidence": confidence,
        "predictions": predictions,
    }


def generate_recommended_action(
    risk_level: str,
    top_factors: List[Dict[str, Any]],
) -> str:
    """
    Synthesize operational safety recommendations based on risk and abnormal factors.

    Parameters
    ----------
    risk_level : str
        Assessed risk level ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
    top_factors : list of dict
        List of contributing factors with their assessed safety status.

    Returns
    -------
    str
        Actionable operational guidance string.
    """
    # Identify abnormal factors exceeding normal operating limits
    abnormal = [
        f["feature"]
        for f in top_factors
        if f.get("status") in ["WARNING", "CRITICAL"]
    ]

    if abnormal:
        factor_str = ", ".join(abnormal)
    elif top_factors:
        factor_str = ", ".join(f["feature"] for f in top_factors[:2])
    else:
        factor_str = "battery parameters"

    level = risk_level.upper()
    if level == "LOW":
        return "Continue normal operation. Monitor battery parameters."
    elif level == "MEDIUM":
        return f"Increase monitoring frequency. Check {factor_str}."
    elif level == "HIGH":
        return (
            f"Reduce charging load. Initiate safety inspection. "
            f"Monitor {factor_str} continuously."
        )
    elif level == "CRITICAL":
        return (
            "Immediately reduce load. Initiate emergency inspection per safety "
            "protocol. Do not continue charging if safety limits are exceeded."
        )
    else:
        return f"Maintain monitoring. Verify status of {factor_str}."


def generate_interpretable_message(
    risk_level: str,
    top_factors: List[Dict[str, Any]],
    confidence: str,
) -> str:
    """
    Synthesize a coherent, human-readable natural language safety warning message.

    Parameters
    ----------
    risk_level : str
        Risk severity level.
    top_factors : list of dict
        Top contributing factors with actual values and units.
    confidence : str
        Model confidence rating ('HIGH', 'MODERATE', 'LOW').

    Returns
    -------
    str
        Interpretable explanation narrative.
    """
    if len(top_factors) >= 2:
        f1, f2 = top_factors[0], top_factors[1]
        u1 = f" {f1.get('unit', '')}".rstrip()
        u2 = f" {f2.get('unit', '')}".rstrip()
        factor_desc = f"{f1['feature']} ({f1['actual_value']:.2f}{u1}) and {f2['feature']} ({f2['actual_value']:.2f}{u2})"
    elif len(top_factors) == 1:
        f1 = top_factors[0]
        u1 = f" {f1.get('unit', '')}".rstrip()
        factor_desc = f"{f1['feature']} ({f1['actual_value']:.2f}{u1})"
    else:
        factor_desc = "monitored degradation parameters"

    risk_label = risk_level.capitalize()
    msg = f"{risk_label} battery risk detected. The prediction is primarily driven by {factor_desc}."

    if confidence == "LOW":
        msg += " Caveat: Inter-model consensus is LOW (model-based assessment, not a validated physical diagnosis)."
    elif confidence == "MODERATE":
        msg += " Note: Inter-model consensus is MODERATE."

    action = generate_recommended_action(risk_level, top_factors)
    msg += f" Recommended action: {action}"
    return msg


def generate_warning(
    risk_score: float,
    shap_contributions: Dict[str, Any],
    feature_values: Dict[str, float],
    uncertainty: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a complete, structured battery safety warning object.

    Translation pipeline:
    1. Classifies overall risk level from risk score
    2. Extracts top-K contributing features from local SHAP explanations
    3. Benchmarks each factor against physical engineering thresholds
    4. Enriches with human-readable descriptions and measurement units
    5. Formulates actionable recommendation and natural language message
    6. Attaches uncertainty flags and diagnostic caveats

    Parameters
    ----------
    risk_score : float
        Degradation risk score between 0.0 and 1.0.
    shap_contributions : dict
        Local SHAP explanation dict (from get_local_explanation).
    feature_values : dict
        Dictionary of raw and trend feature values for the evaluated cycle.
    uncertainty : dict, optional
        Model consensus and uncertainty dict (from estimate_prediction_uncertainty).

    Returns
    -------
    dict
        Structured warning record containing:
        - risk_level: str
        - risk_score: float
        - confidence: str
        - uncertainty_flag: bool (True if confidence is LOW)
        - contributing_factors: list of factor dicts
        - interpretable_message: str
        - recommended_action: str
    """
    risk_level = classify_risk_level(risk_score)

    raw_contributions = shap_contributions.get("feature_contributions", [])
    top_k_items = raw_contributions[: config.TOP_K_FEATURES]

    contributing_factors: List[Dict[str, Any]] = []
    for idx, item in enumerate(top_k_items, start=1):
        feat_name = item["feature"]
        shap_val = float(item["shap_value"])
        act_val = float(feature_values.get(feat_name, item.get("feature_value", 0.0)))
        unit = config.FEATURE_UNITS.get(feat_name, "")
        status = assess_feature_status(feat_name, act_val)
        desc = config.FEATURE_DESCRIPTIONS.get(feat_name, feat_name)

        contributing_factors.append({
            "feature": feat_name,
            "shap_contribution": shap_val,
            "rank": idx,
            "actual_value": act_val,
            "unit": unit,
            "status": status,
            "description": desc,
        })

    if uncertainty is not None:
        confidence = uncertainty.get("confidence", "HIGH")
        uncertainty_flag = bool(confidence == "LOW")
    else:
        confidence = "HIGH"
        uncertainty_flag = False

    action = generate_recommended_action(risk_level, contributing_factors)
    message = generate_interpretable_message(risk_level, contributing_factors, confidence)

    return {
        "risk_level": risk_level,
        "risk_score": float(risk_score),
        "confidence": confidence,
        "uncertainty_flag": uncertainty_flag,
        "uncertainty_info": uncertainty,
        "contributing_factors": contributing_factors,
        "interpretable_message": message,
        "recommended_action": action,
    }


def generate_batch_warnings(
    df: pd.DataFrame,
    shap_values: Any,
    feature_names: List[str],
    model_results: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate structured warnings for all cycles in a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Battery dataset containing feature columns and labels.
    shap_values : shap.Explanation
        SHAP Explanation object matching rows in df.
    feature_names : list of str
        List of feature column names.
    model_results : dict, optional
        Cross-model results dict for uncertainty quantification.

    Returns
    -------
    list of dict
        List of warning dictionaries for all instances.
    """
    n_samples = len(df)
    warnings: List[Dict[str, Any]] = []

    for i in range(n_samples):
        row = df.iloc[i]

        # Extract risk score
        if "risk_score" in row:
            risk_score = float(row["risk_score"])
        elif "risk_binary" in row:
            risk_score = float(row["risk_binary"])
        else:
            risk_score = 0.5

        # Extract local SHAP explanation
        local_shap = get_local_explanation(shap_values, i, feature_names)

        # Feature measurements
        feat_dict = {f: float(row[f]) for f in feature_names if f in row}

        # Estimate prediction uncertainty if multi-model results provided
        unc = (
            estimate_prediction_uncertainty(model_results, i)
            if model_results is not None
            else None
        )

        warning = generate_warning(
            risk_score=risk_score,
            shap_contributions=local_shap,
            feature_values=feat_dict,
            uncertainty=unc,
        )

        # Attach metadata
        warning["row_index"] = i
        if "cell_name" in row:
            warning["cell_name"] = str(row["cell_name"])
        if "cycle_index" in row:
            warning["cycle_index"] = int(row["cycle_index"])

        warnings.append(warning)

    return warnings


def format_warning_text(warning: Dict[str, Any]) -> str:
    """
    Format a warning dictionary into a human-readable multi-line report.

    Parameters
    ----------
    warning : dict
        Structured warning record.

    Returns
    -------
    str
        Formatted text block suitable for CLI display or system logs.
    """
    risk_level = warning.get("risk_level", "UNKNOWN")
    risk_score = warning.get("risk_score", 0.0)
    confidence = warning.get("confidence", "UNKNOWN")
    unc_flag = warning.get("uncertainty_flag", False)
    cell = warning.get("cell_name", "N/A")
    cycle = warning.get("cycle_index", "N/A")

    lines = [
        "=" * 80,
        f"BATTERY SAFETY WARNING [LEVEL: {risk_level} | RISK SCORE: {risk_score:.3f}]",
        f"Cell: {cell} | Cycle: {cycle} | Confidence: {confidence} | Uncertainty Flag: {unc_flag}",
        "=" * 80,
        "SUMMARY:",
        f"  {warning.get('interpretable_message', '')}",
        "",
        "RECOMMENDED ACTION:",
        f"  {warning.get('recommended_action', '')}",
        "",
        "KEY CONTRIBUTING FACTORS (SHAP Impact & Engineering Status):",
    ]

    for factor in warning.get("contributing_factors", []):
        rank = factor.get("rank", "-")
        feat = factor.get("feature", "-")
        desc = factor.get("description", "")
        val = factor.get("actual_value", 0.0)
        unit = factor.get("unit", "")
        status = factor.get("status", "UNKNOWN")
        shap_c = factor.get("shap_contribution", 0.0)
        sign = "+" if shap_c >= 0 else ""
        # Adaptive formatting: use scientific notation for very small values
        if abs(shap_c) >= 0.001:
            shap_str = f"{sign}{shap_c:.4f}"
        elif abs(shap_c) > 0:
            shap_str = f"{sign}{shap_c:.2e}"
        else:
            shap_str = "0.0000"

        lines.append(
            f"  {rank}. {feat} ({desc})"
        )
        lines.append(
            f"     Value: {val:.3f} {unit} | Status: {status} | SHAP Impact: {shap_str}"
        )

    lines.extend([
        "=" * 80,
        "* DISCLAIMER: Model-based safety assessment; not a substitute for physical diagnostic protocols.",
        "=" * 80,
    ])

    return "\n".join(lines)


def save_warning_examples(
    warnings: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    n_examples: int = 20,
) -> None:
    """
    Save representative warning examples across all risk categories.

    Exports:
    - Individual JSON files for selected representative warnings
    - A consolidated summary CSV (warning_summary.csv)
    - A consolidated human-readable text log (warning_examples.txt)

    Parameters
    ----------
    warnings : list of dict
        Complete list of warning dictionaries.
    output_dir : str or Path
        Directory where artifacts will be saved.
    n_examples : int, default=20
        Total number of representative examples to export.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not warnings:
        print("No warnings provided to save.")
        return

    # Group warnings by risk level to ensure balanced representation
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    by_level: Dict[str, List[Dict[str, Any]]] = {lvl: [] for lvl in levels}

    for w in warnings:
        lvl = w.get("risk_level", "LOW")
        if lvl in by_level:
            by_level[lvl].append(w)
        else:
            by_level.setdefault(lvl, []).append(w)

    per_level_target = max(1, n_examples // len(levels))
    selected: List[Dict[str, Any]] = []

    for lvl in levels:
        group = by_level[lvl]
        if group:
            step = max(1, len(group) // per_level_target)
            selected.extend(group[::step][:per_level_target])

    # If shortfall, pad from remaining warnings
    if len(selected) < n_examples and len(warnings) > len(selected):
        remaining = [w for w in warnings if w not in selected]
        selected.extend(remaining[: n_examples - len(selected)])

    summary_records: List[Dict[str, Any]] = []
    text_blocks: List[str] = []

    for idx, w in enumerate(selected, start=1):
        lvl = w.get("risk_level", "UNKNOWN").lower()
        json_filename = f"warning_{idx:03d}_{lvl}.json"

        # Serialize JSON example
        with open(out_path / json_filename, "w", encoding="utf-8") as f:
            json.dump(w, f, indent=2)

        # Prepare summary row
        factors = w.get("contributing_factors", [])
        f1 = factors[0]["feature"] if len(factors) > 0 else "None"
        s1 = factors[0]["status"] if len(factors) > 0 else "None"
        f2 = factors[1]["feature"] if len(factors) > 1 else "None"
        s2 = factors[1]["status"] if len(factors) > 1 else "None"

        summary_records.append({
            "warning_id": idx,
            "filename": json_filename,
            "cell_name": w.get("cell_name", "N/A"),
            "cycle_index": w.get("cycle_index", "N/A"),
            "risk_score": w.get("risk_score", 0.0),
            "risk_level": w.get("risk_level", "UNKNOWN"),
            "confidence": w.get("confidence", "UNKNOWN"),
            "uncertainty_flag": w.get("uncertainty_flag", False),
            "top_factor_1": f1,
            "status_1": s1,
            "top_factor_2": f2,
            "status_2": s2,
            "recommended_action": w.get("recommended_action", ""),
        })

        text_blocks.append(format_warning_text(w))

    # Save summary CSV
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(out_path / "warning_summary.csv", index=False)

    # Save text log
    with open(out_path / "warning_examples.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_blocks))

    print(f"Saved {len(selected)} representative warning examples to '{out_path}'.")
