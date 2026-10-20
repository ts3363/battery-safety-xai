"""
Explainability module for XAI Battery Safety.

This module implements SHAP (SHapley Additive exPlanations) and LIME (Local
Interpretable Model-agnostic Explanations) analysis, including:
- TreeExplainer and LinearExplainer computation
- Global feature importance extraction
- Local sample-level explanations with base values and contributions
- LIME local surrogate explanations
- Concordance comparison between SHAP and LIME
- Cross-fold explanation stability analysis across battery cells
- Multi-model explanation pipeline
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import shap
import lime
import lime.lime_tabular
from sklearn.pipeline import Pipeline

# Add project root to sys.path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.models import predict_proba_safe


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
) -> shap.Explanation:
    """
    Compute SHAP values for a fitted model on feature matrix X.

    Automatically handles scikit-learn Pipelines (extracting the estimator
    and applying StandardScaler preprocessing if present), selects the
    appropriate explainer (TreeExplainer for tree models, LinearExplainer
    for logistic regression), and returns a unified 2D SHAP Explanation
    oriented toward the risk-positive class (class 1).

    Parameters
    ----------
    model : Any
        Fitted model or Pipeline.
    X : pd.DataFrame
        Feature matrix to explain.
    model_name : str
        Model architecture identifier ('logistic_regression', 'decision_tree',
        'random_forest', 'xgboost').

    Returns
    -------
    shap.Explanation
        SHAP Explanation object containing:
        - values: (n_samples, n_features) array for the risk-positive class
        - base_values: (n_samples,) array or scalar baseline
        - data: (n_samples, n_features) original feature values
        - feature_names: list of feature names

    Raises
    ------
    ValueError
        If model_name is unrecognized or explainer fails.
    """
    norm_name = model_name.lower().strip()

    # Extract classifier and handle Pipeline scaling if applicable
    if isinstance(model, Pipeline) or hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier", model.steps[-1][1])
        scaler = model.named_steps.get("scaler", None)
        if scaler is not None:
            X_scaled = scaler.transform(X)
            X_eval = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_eval = X.copy()
    else:
        classifier = model
        X_eval = X.copy()

    # Select appropriate explainer based on model architecture
    if norm_name == "logistic_regression":
        explainer = shap.LinearExplainer(classifier, X_eval)
        raw_exp = explainer(X_eval)
    elif norm_name in ["decision_tree", "random_forest", "xgboost"]:
        explainer = shap.TreeExplainer(classifier)
        raw_exp = explainer(X_eval)
    else:
        # Fallback to general Explainer
        explainer = shap.Explainer(classifier, X_eval)
        raw_exp = explainer(X_eval)

    # Standardize Explanation output for risk class (class 1)
    if isinstance(raw_exp, shap.Explanation):
        vals = raw_exp.values
        base_vals = raw_exp.base_values

        # If values has 3 dimensions (n_samples, n_features, n_classes)
        if vals.ndim == 3 and vals.shape[-1] >= 2:
            vals = vals[:, :, 1]
            if isinstance(base_vals, np.ndarray) and base_vals.ndim >= 2:
                base_vals = base_vals[:, 1]
            elif isinstance(base_vals, (list, np.ndarray)) and len(base_vals) >= 2:
                base_vals = base_vals[1]

        explanation = shap.Explanation(
            values=vals,
            base_values=base_vals,
            data=np.asarray(X),
            feature_names=list(X.columns),
        )
        return explanation

    # Fallback if explainer returned raw arrays
    if hasattr(explainer, "shap_values"):
        shap_arr = explainer.shap_values(X_eval)
        if isinstance(shap_arr, list):
            vals = shap_arr[1] if len(shap_arr) > 1 else shap_arr[0]
        elif isinstance(shap_arr, np.ndarray) and shap_arr.ndim == 3:
            vals = shap_arr[:, :, 1]
        else:
            vals = np.asarray(shap_arr)

        base_vals = getattr(explainer, "expected_value", 0.0)
        if isinstance(base_vals, (list, np.ndarray)) and len(base_vals) > 1:
            base_vals = base_vals[1]

        return shap.Explanation(
            values=vals,
            base_values=base_vals,
            data=np.asarray(X),
            feature_names=list(X.columns),
        )

    raise RuntimeError(f"Unable to extract SHAP explanation for model '{model_name}'.")


def get_global_feature_importance(
    shap_values: shap.Explanation,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute global feature importance as the mean absolute SHAP value.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP Explanation object.
    feature_names : list of str
        Ordered list of feature column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'mean_abs_shap', 'rank']
        sorted descending by mean_abs_shap.
    """
    vals = np.asarray(shap_values.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_abs = np.mean(np.abs(vals), axis=0)

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    })

    df_importance = df_importance.sort_values(
        by="mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    df_importance["rank"] = df_importance.index + 1
    return df_importance


def get_local_explanation(
    shap_values: shap.Explanation,
    sample_index: int,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Extract a local explanation for a specific sample instance.

    Computes base value, additive prediction, and per-feature contributions
    ranked by magnitude of impact.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP Explanation object.
    sample_index : int
        Index of the target sample in shap_values.
    feature_names : list of str
        List of feature column names.

    Returns
    -------
    dict
        Dictionary containing:
        - 'base_value': float, baseline expected model output
        - 'prediction': float, predicted output (base_value + sum of SHAP values)
        - 'feature_contributions': list of dicts with keys:
          ['feature', 'shap_value', 'feature_value', 'rank']
          sorted descending by absolute SHAP value.
    """
    vals = np.asarray(shap_values.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    sample_shap = vals[sample_index]

    # Handle data array
    if hasattr(shap_values, "data") and shap_values.data is not None:
        sample_data = np.asarray(shap_values.data)[sample_index]
    else:
        sample_data = np.zeros(len(feature_names))

    # Handle base value
    base_val_raw = shap_values.base_values
    if isinstance(base_val_raw, (np.ndarray, list)):
        base_arr = np.asarray(base_val_raw)
        if base_arr.ndim == 2:
            base_val = float(base_arr[sample_index, 1] if base_arr.shape[1] > 1 else base_arr[sample_index, 0])
        elif base_arr.ndim == 1 and len(base_arr) > sample_index:
            base_val = float(base_arr[sample_index])
        elif len(base_arr) > 0:
            base_val = float(base_arr[0])
        else:
            base_val = 0.0
    else:
        base_val = float(base_val_raw)

    prediction = base_val + float(np.sum(sample_shap))

    contributions: List[Dict[str, Any]] = []
    for i, feat in enumerate(feature_names):
        sv = float(sample_shap[i])
        fv = float(sample_data[i])
        contributions.append({
            "feature": feat,
            "shap_value": sv,
            "feature_value": fv,
            "_abs_shap": abs(sv),
        })

    # Sort descending by absolute SHAP contribution
    contributions.sort(key=lambda x: x["_abs_shap"], reverse=True)

    # Assign rank and clean temporary field
    for rank_idx, item in enumerate(contributions, start=1):
        item["rank"] = rank_idx
        del item["_abs_shap"]

    return {
        "base_value": base_val,
        "prediction": prediction,
        "feature_contributions": contributions,
    }


def compute_lime_explanation(
    model: Any,
    X_train: pd.DataFrame,
    instance: Union[np.ndarray, pd.Series],
    feature_names: List[str],
    num_features: int = 5,
) -> Dict[str, Any]:
    """
    Compute a LIME local tabular explanation for a single battery cycle instance.

    Uses a local sparse linear surrogate model fitted around the perturbed instance.

    Parameters
    ----------
    model : Any
        Fitted model or Pipeline.
    X_train : pd.DataFrame
        Training feature matrix used to compute feature statistics.
    instance : np.ndarray or pd.Series
        Single feature vector to explain.
    feature_names : list of str
        List of feature column names.
    num_features : int, default=5
        Maximum number of features in the local explanation.

    Returns
    -------
    dict
        Dictionary containing:
        - 'feature_weights': list of (feature_name, weight) tuples
        - 'prediction': float, predicted risk probability for the instance
        - 'score': float, R^2 goodness of fit of the local surrogate
    """
    if isinstance(instance, pd.Series):
        inst_arr = instance.to_numpy(dtype=float)
    else:
        inst_arr = np.asarray(instance, dtype=float).flatten()

    def predict_fn(x_matrix: np.ndarray) -> np.ndarray:
        df_chunk = pd.DataFrame(x_matrix, columns=feature_names)
        return predict_proba_safe(model, df_chunk)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=feature_names,
        class_names=["Safe", "Risk"],
        mode="classification",
        random_state=config.RANDOM_SEED,
    )

    exp = explainer.explain_instance(
        data_row=inst_arr,
        predict_fn=predict_fn,
        num_features=num_features,
        labels=(1,),
    )

    # Extract weights and associate with feature names
    raw_weights = exp.as_list(label=1)
    parsed_weights: List[Tuple[str, float]] = []

    for rule_str, weight in raw_weights:
        # Resolve feature name from rule string
        matched_feat = None
        for fn in sorted(feature_names, key=len, reverse=True):
            if fn in rule_str:
                matched_feat = fn
                break
        feature_key = matched_feat if matched_feat else rule_str
        parsed_weights.append((feature_key, float(weight)))

    pred_prob = float(predict_fn(inst_arr.reshape(1, -1))[0, 1])
    score = float(exp.score) if hasattr(exp, "score") else 0.0

    return {
        "feature_weights": parsed_weights,
        "prediction": pred_prob,
        "score": score,
    }


def compare_shap_lime(
    shap_contributions: Dict[str, Any],
    lime_explanation: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compare top-K features between SHAP and LIME local explanations.

    Evaluates explanation agreement using:
    - Overlap count in top-K features
    - Spearman rank correlation across union of top features

    Parameters
    ----------
    shap_contributions : dict
        Local SHAP explanation dict (from get_local_explanation).
    lime_explanation : dict
        Local LIME explanation dict (from compute_lime_explanation).
    top_k : int, default=5
        Number of top features to evaluate.

    Returns
    -------
    dict
        Dictionary containing:
        - 'shap_top_k': list of top-K feature names from SHAP
        - 'lime_top_k': list of top-K feature names from LIME
        - 'overlap_count': int, number of features common to both sets
        - 'rank_correlation': float, Spearman rank correlation
    """
    shap_top_k = [
        item["feature"]
        for item in shap_contributions.get("feature_contributions", [])[:top_k]
    ]

    lime_top_k = [
        item[0]
        for item in lime_explanation.get("feature_weights", [])[:top_k]
    ]

    overlap_count = len(set(shap_top_k).intersection(set(lime_top_k)))

    # Compute rank correlation over the union of top features
    all_features = list(dict.fromkeys(shap_top_k + lime_top_k))

    if len(all_features) >= 2:
        shap_ranks = [
            shap_top_k.index(f) + 1 if f in shap_top_k else top_k + 1
            for f in all_features
        ]
        lime_ranks = [
            lime_top_k.index(f) + 1 if f in lime_top_k else top_k + 1
            for f in all_features
        ]
        rho, _ = stats.spearmanr(shap_ranks, lime_ranks)
        rank_correlation = float(rho) if not np.isnan(rho) else 0.0
    else:
        rank_correlation = 1.0 if overlap_count == len(all_features) else 0.0

    return {
        "shap_top_k": shap_top_k,
        "lime_top_k": lime_top_k,
        "overlap_count": overlap_count,
        "rank_correlation": rank_correlation,
    }


def compute_explanation_stability(
    model: Any,
    X_folds: List[pd.DataFrame],
    model_name: str,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Evaluate explanation stability across LOBO-CV folds.

    Computes SHAP feature importance independently for each held-out battery fold
    and evaluates the pairwise Spearman rank correlation between all folds.

    Parameters
    ----------
    model : Any
        Fitted model or Pipeline.
    X_folds : list of pd.DataFrame
        List of feature matrices for each cross-validation fold.
    model_name : str
        Model architecture name.
    feature_names : list of str
        List of feature column names.

    Returns
    -------
    dict
        Dictionary containing:
        - 'fold_importances': list of global importance DataFrames per fold
        - 'pairwise_rank_correlations': list of pairwise Spearman correlations
        - 'mean_rank_correlation': float, average pairwise correlation
    """
    fold_importances: List[pd.DataFrame] = []

    for fold_X in X_folds:
        if len(fold_X) == 0:
            continue
        shap_exp = compute_shap_values(model, fold_X, model_name)
        imp_df = get_global_feature_importance(shap_exp, feature_names)
        fold_importances.append(imp_df)

    pairwise_correlations: List[float] = []
    n_folds = len(fold_importances)

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            df_i = fold_importances[i].set_index("feature").loc[feature_names, "rank"]
            df_j = fold_importances[j].set_index("feature").loc[feature_names, "rank"]
            rho, _ = stats.spearmanr(df_i.values, df_j.values)
            pairwise_correlations.append(float(rho) if not np.isnan(rho) else 0.0)

    mean_rank_corr = (
        float(np.mean(pairwise_correlations)) if pairwise_correlations else 1.0
    )

    return {
        "fold_importances": fold_importances,
        "pairwise_rank_correlations": pairwise_correlations,
        "mean_rank_correlation": mean_rank_corr,
    }


def explain_all_models(
    model_results: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Run comprehensive SHAP explainability and stability analysis across all models.

    Parameters
    ----------
    model_results : dict
        Dictionary of LOBO-CV results mapping model_name -> result dict.
    df : pd.DataFrame
        Dataset containing feature columns and cell_name.
    feature_names : list of str
        List of feature column names.

    Returns
    -------
    dict
        Dictionary mapping model_name -> dict with keys:
        - 'shap_values': shap.Explanation on full dataset
        - 'global_importance': pd.DataFrame of feature importance rankings
        - 'stability': dict of cross-cell stability metrics
    """
    explanations: Dict[str, Dict[str, Any]] = {}
    X_full = df[feature_names].copy()

    for m_name, m_res in model_results.items():
        folds = m_res.get("results", [])
        if folds:
            # Use fitted model from primary fold or reference
            representative_model = folds[0]["model"]
            X_folds = [f["X_test"] for f in folds if "X_test" in f and len(f["X_test"]) > 0]
        else:
            representative_model = m_res.get("model")
            X_folds = [X_full]

        # Compute full-data SHAP values
        shap_vals = compute_shap_values(representative_model, X_full, m_name)

        # Global feature importance
        global_imp = get_global_feature_importance(shap_vals, feature_names)

        # Cross-fold stability
        stability = compute_explanation_stability(
            model=representative_model,
            X_folds=X_folds,
            model_name=m_name,
            feature_names=feature_names,
        )

        explanations[m_name] = {
            "shap_values": shap_vals,
            "global_importance": global_imp,
            "stability": stability,
        }

    return explanations
