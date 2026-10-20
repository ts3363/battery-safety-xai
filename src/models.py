"""
Model training and evaluation module for XAI Battery Safety.

This module implements:
- Model instantiation for standard and gradient-boosted estimators
- Pipeline wrapping with StandardScaler for linear models
- Safe probability estimation handling edge cases
- Leave-One-Battery-Out Cross-Validation (LOBO-CV)
- Multi-model benchmarking and best-model selection
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Add project root to sys.path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def get_model(model_name: str) -> BaseEstimator:
    """
    Initialize and return a machine learning model based on model_name.

    Supported model architectures:
    - 'logistic_regression': Linear model with L2 regularization
    - 'decision_tree': Single decision tree classifier
    - 'random_forest': Ensemble of decision trees
    - 'xgboost': Gradient-boosted decision trees

    Parameters
    ----------
    model_name : str
        Name of the model architecture to instantiate.

    Returns
    -------
    BaseEstimator
        An un-fitted scikit-learn or XGBoost classifier estimator.

    Raises
    ------
    ValueError
        If model_name is not one of the supported architectures.
    """
    norm_name = model_name.lower().strip()
    if norm_name not in config.MODEL_PARAMS:
        valid_models = list(config.MODEL_PARAMS.keys())
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Supported models are: {valid_models}"
        )

    params = dict(config.MODEL_PARAMS[norm_name])

    if norm_name == "logistic_regression":
        return LogisticRegression(**params)

    elif norm_name == "decision_tree":
        return DecisionTreeClassifier(**params)

    elif norm_name == "random_forest":
        return RandomForestClassifier(**params)

    elif norm_name == "xgboost":
        # Pass use_label_encoder=False as required; handle newer xgboost gracefully
        try:
            return XGBClassifier(**params, use_label_encoder=False)
        except TypeError:
            return XGBClassifier(**params)

    else:
        raise ValueError(f"Unhandled model name: {model_name}")


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
) -> BaseEstimator:
    """
    Fit a machine learning model on training data.

    For logistic regression, standard feature scaling is applied via a
    scikit-learn Pipeline (StandardScaler + LogisticRegression). Tree-based
    models (decision_tree, random_forest, xgboost) are invariant to monotonic
    feature scaling and are fit directly.

    Parameters
    ----------
    model : BaseEstimator
        Unfitted model estimator.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training target labels.

    Returns
    -------
    BaseEstimator
        Fitted model or Pipeline.
    """
    if isinstance(model, LogisticRegression):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    if isinstance(model, Pipeline):
        model.fit(X_train, y_train)
        return model

    model.fit(X_train, y_train)
    return model


def predict_proba_safe(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Safely predict class probabilities for binary classification.

    Handles differences between standalone models, scikit-learn Pipelines,
    and single-class probability predictions. Always returns a 2D array of
    shape (n_samples, 2), where column 0 is P(y=0) and column 1 is P(y=1).

    Parameters
    ----------
    model : Any
        Fitted model or Pipeline.
    X : pd.DataFrame
        Feature matrix for prediction.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 2) containing binary probabilities.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 1:
            probs = np.column_stack([1.0 - probs, probs])
        elif probs.shape[1] == 1:
            # Single class observed during training
            if hasattr(model, "classes_") and model.classes_[0] == 1:
                probs = np.column_stack([np.zeros_like(probs[:, 0]), probs[:, 0]])
            else:
                probs = np.column_stack([probs[:, 0], np.zeros_like(probs[:, 0])])
        elif probs.shape[1] > 2:
            probs = probs[:, :2]
        return np.clip(probs, 0.0, 1.0)

    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        prob_pos = 1.0 / (1.0 + np.exp(-decision))
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

    elif hasattr(model, "predict"):
        preds = model.predict(X).astype(float)
        return np.column_stack([1.0 - preds, preds])

    else:
        raise AttributeError(
            f"The provided model {type(model)} does not implement "
            "predict_proba, decision_function, or predict."
        )


def leave_one_battery_out_cv(
    df: pd.DataFrame,
    feature_names: List[str],
    target: str = "risk_binary",
    model_name: str = "random_forest",
) -> Dict[str, Any]:
    """
    Perform Leave-One-Battery-Out Cross-Validation (LOBO-CV).

    For each unique battery cell in config.BATTERY_CELLS:
    - Trains on all remaining battery cells
    - Tests on the held-out battery cell
    - Records predictions, probabilities, and ground truth labels

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset containing battery cycles, features, and cell_name.
    feature_names : list of str
        List of feature column names to use as predictors.
    target : str, default='risk_binary'
        Target column name for classification.
    model_name : str, default='random_forest'
        Name of model architecture to evaluate.

    Returns
    -------
    dict
        Dictionary containing:
        - 'results': List of per-fold dicts:
            {'test_cell': str, 'y_true': np.ndarray, 'y_pred': np.ndarray,
             'y_prob': np.ndarray, 'model': fitted_model, 'X_test': pd.DataFrame,
             'X_train': pd.DataFrame, 'test_indices': np.ndarray}
        - 'all_y_true': Concatenated true labels across all folds
        - 'all_y_pred': Concatenated binary predictions across all folds
        - 'all_y_prob': Concatenated predicted probabilities (N, 2)
        - 'model_name': Name of evaluated model
    """
    if "cell_name" not in df.columns:
        raise KeyError("DataFrame must contain a 'cell_name' column for LOBO-CV.")

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    # Determine unique cells to iterate over
    available_cells = [c for c in config.BATTERY_CELLS if c in df["cell_name"].unique()]
    if not available_cells:
        available_cells = list(df["cell_name"].unique())

    fold_results: List[Dict[str, Any]] = []
    all_y_true_list: List[np.ndarray] = []
    all_y_pred_list: List[np.ndarray] = []
    all_y_prob_list: List[np.ndarray] = []

    for test_cell in available_cells:
        train_mask = df["cell_name"] != test_cell
        test_mask = df["cell_name"] == test_cell

        if not test_mask.any():
            continue

        X_train = df.loc[train_mask, feature_names].copy()
        y_train = df.loc[train_mask, target].copy()
        X_test = df.loc[test_mask, feature_names].copy()
        y_test = df.loc[test_mask, target].copy()

        # Initialize and fit model
        raw_model = get_model(model_name)
        fitted_model = train_model(raw_model, X_train, y_train)

        # Predict probabilities and binary labels using threshold
        y_prob = predict_proba_safe(fitted_model, X_test)
        y_pred = (y_prob[:, 1] >= config.RISK_THRESHOLD).astype(int)
        y_true_arr = y_test.to_numpy()

        fold_dict = {
            "test_cell": test_cell,
            "y_true": y_true_arr,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "model": fitted_model,
            "X_test": X_test,
            "X_train": X_train,
            "test_indices": df.index[test_mask].to_numpy(),
        }
        fold_results.append(fold_dict)

        all_y_true_list.append(y_true_arr)
        all_y_pred_list.append(y_pred)
        all_y_prob_list.append(y_prob)

    all_y_true = (
        np.concatenate(all_y_true_list) if all_y_true_list else np.array([])
    )
    all_y_pred = (
        np.concatenate(all_y_pred_list) if all_y_pred_list else np.array([])
    )
    all_y_prob = (
        np.concatenate(all_y_prob_list, axis=0)
        if all_y_prob_list
        else np.empty((0, 2))
    )

    return {
        "results": fold_results,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
        "all_y_prob": all_y_prob,
        "model_name": model_name,
    }


def train_all_models(
    df: pd.DataFrame,
    feature_names: List[str],
    target: str = "risk_binary",
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all four baseline models using LOBO-CV.

    Models trained:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost

    Prints a performance summary table comparing models across:
    Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset with features and target.
    feature_names : list of str
        List of feature column names.
    target : str, default='risk_binary'
        Target label column.

    Returns
    -------
    dict
        Dictionary mapping model_name -> LOBO-CV results dictionary.
    """
    models_to_evaluate = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    ]

    all_results: Dict[str, Dict[str, Any]] = {}

    print("\n" + "=" * 78)
    print("Leave-One-Battery-Out Cross-Validation (LOBO-CV) Benchmarking")
    print("=" * 78)
    print(
        f"{'Model':<22} {'Accuracy':<10} {'Precision':<10} "
        f"{'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}"
    )
    print("-" * 78)

    for m_name in models_to_evaluate:
        res = leave_one_battery_out_cv(
            df=df,
            feature_names=feature_names,
            target=target,
            model_name=m_name,
        )
        # Add backward-compatible keys for experiment code
        res["y_true"] = res["all_y_true"]
        res["y_pred"] = res["all_y_pred"]
        res["y_prob"] = res["all_y_prob"]
        res["models"] = [fr["model"] for fr in res.get("results", [])]
        res["fold_results"] = res.get("results", [])
        all_results[m_name] = res

        y_true = res["all_y_true"]
        y_pred = res["all_y_pred"]
        y_prob = res["all_y_prob"]

        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            except Exception:
                auc = float("nan")

            print(
                f"{m_name:<22} {acc:<10.4f} {prec:<10.4f} "
                f"{rec:<10.4f} {f1:<10.4f} {auc:<10.4f}"
            )
        else:
            print(f"{m_name:<22} No evaluation data available.")

    print("=" * 78 + "\n")
    return all_results


def get_best_model(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Identify the best-performing model based on cross-validated F1-score.

    Parameters
    ----------
    results : dict
        Dictionary mapping model_name -> LOBO-CV results dict.

    Returns
    -------
    tuple of (str, dict)
        Tuple containing (best_model_name, best_model_results_dict).

    Raises
    ------
    ValueError
        If results dictionary is empty.
    """
    if not results:
        raise ValueError("Results dictionary is empty. No models to compare.")

    best_name: Optional[str] = None
    best_f1: float = -1.0

    for m_name, m_res in results.items():
        y_true = m_res.get("all_y_true", np.array([]))
        y_pred = m_res.get("all_y_pred", np.array([]))

        if len(y_true) == 0:
            current_f1 = 0.0
        else:
            current_f1 = f1_score(y_true, y_pred, zero_division=0)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_name = m_name

    if best_name is None:
        best_name = list(results.keys())[0]

    return best_name, results[best_name]
