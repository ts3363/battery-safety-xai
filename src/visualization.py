"""
Visualization module for the XAI Battery Safety Framework.

Provides publication-quality plotting functions for:
- Battery degradation trajectories
- Feature distributions and correlations
- Model evaluation metrics, ROC, PR, and calibration curves
- SHAP global and local explanations
- Cross-battery explanation stability heatmaps
- Warning system distribution and contributing factors
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.calibration import calibration_curve

# Project root import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Set plot style safely
try:
    plt.style.use(config.FIGURE_STYLE)
except Exception:
    sns.set_theme(style="whitegrid")


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_degradation_curves(df: pd.DataFrame) -> plt.Figure:
    """
    Plot capacity vs cycle_index for each cell, colored by config.CELL_COLORS.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'cell_name', 'cycle_index', and 'capacity'.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    cells = df["cell_name"].unique() if "cell_name" in df.columns else config.BATTERY_CELLS

    for cell in cells:
        cell_df = df[df["cell_name"] == cell].sort_values("cycle_index")
        color = config.CELL_COLORS.get(cell, "#333333")
        ax.plot(
            cell_df["cycle_index"],
            cell_df["capacity"],
            label=f"Cell {cell}",
            color=color,
            linewidth=2.0,
            alpha=0.85,
        )

    # 80% EOL threshold line
    eol_capacity = config.NOMINAL_CAPACITY_AH * config.EOL_CAPACITY_THRESHOLD
    ax.axhline(
        y=eol_capacity,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"80% EOL Threshold ({eol_capacity:.2f} Ah)",
    )

    ax.set_title("Battery Degradation Curves", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Cycle Count", fontsize=12)
    ax.set_ylabel("Discharge Capacity (Ah)", fontsize=12)
    ax.set_ylim(bottom=0.0, top=config.NOMINAL_CAPACITY_AH * 1.15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "degradation_curves.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_feature_distributions(df: pd.DataFrame, features: List[str]) -> plt.Figure:
    """
    Plot subplots grid showing distribution of each feature, colored by cell_name.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing feature columns and 'cell_name'.
    features : List[str]
        List of feature column names to plot.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    n_features = len(features)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, feature in enumerate(features):
        ax = axes[i]
        if feature in df.columns:
            if "cell_name" in df.columns:
                sns.kdeplot(
                    data=df,
                    x=feature,
                    hue="cell_name",
                    palette=config.CELL_COLORS,
                    ax=ax,
                    common_norm=False,
                    fill=True,
                    alpha=0.25,
                    linewidth=1.5,
                )
            else:
                sns.histplot(data=df, x=feature, kde=True, ax=ax, color="#1f77b4")
            ax.set_title(feature, fontsize=11, fontweight="bold")
            ax.set_xlabel(config.FEATURE_UNITS.get(feature, ""), fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"Feature '{feature}'\nnot in DataFrame", ha="center", va="center")
            ax.set_axis_off()

    # Turn off unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle("Feature Distributions Across Battery Cells", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "feature_distributions.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_feature_correlation(df: pd.DataFrame, features: List[str]) -> plt.Figure:
    """
    Generate correlation heatmap of specified features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing feature columns.
    features : List[str]
        List of feature column names to include in correlation analysis.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    valid_features = [f for f in features if f in df.columns]
    corr = df[valid_features].corr()

    fig_size = max(8, min(14, len(valid_features) * 0.7))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=len(valid_features) <= 12,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "feature_correlation.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_model_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Plot grouped bar chart of F1, Precision, Recall, and ROC-AUC for each model.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with columns ['model', 'precision', 'recall', 'f1', 'roc_auc']
        or indexed by model name.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    df_plot = metrics_df.copy()
    if "model" not in df_plot.columns and df_plot.index.name == "model":
        df_plot = df_plot.reset_index()
    elif "model" not in df_plot.columns:
        df_plot["model"] = df_plot.index

    metric_cols = [c for c in ["f1", "precision", "recall", "roc_auc"] if c in df_plot.columns]
    if not metric_cols:
        # Fallback to all numeric columns
        metric_cols = [c for c in df_plot.select_dtypes(include=[np.number]).columns]

    df_melted = df_plot.melt(
        id_vars=["model"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )
    df_melted["Metric"] = df_melted["Metric"].str.upper()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = sns.color_palette("muted", n_colors=len(metric_cols))

    sns.barplot(
        data=df_melted,
        x="model",
        y="Score",
        hue="Metric",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(title="Metric", frameon=True, loc="lower right")

    # Annotate bar values
    for p in ax.patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            ax.annotate(
                f"{val:.2f}",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "model_comparison.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_roc_curves(model_results: Dict[str, Any]) -> plt.Figure:
    """
    Plot ROC curves for all models on the same figure with AUC in the legend.

    Parameters
    ----------
    model_results : Dict[str, Any]
        Dictionary keyed by model name, containing 'y_true' and 'y_prob'
        (or precomputed 'fpr', 'tpr', 'roc_auc').

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = sns.color_palette("tab10", n_colors=max(len(model_results), 1))

    for idx, (name, data) in enumerate(model_results.items()):
        color = colors[idx % len(colors)]
        if "fpr" in data and "tpr" in data:
            fpr, tpr = data["fpr"], data["tpr"]
            roc_auc = data.get("roc_auc", auc(fpr, tpr))
        elif "y_true" in data and "y_prob" in data:
            _yp = np.asarray(data["y_prob"])
            _yp1d = _yp[:, 1] if _yp.ndim == 2 else _yp
            fpr, tpr, _ = roc_curve(data["y_true"], _yp1d)
            roc_auc = auc(fpr, tpr)
        else:
            continue

        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})",
            color=color,
            linewidth=2.2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Chance (AUC = 0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="lower right", fontsize=10)

    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "roc_curves.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_pr_curves(model_results: Dict[str, Any]) -> plt.Figure:
    """
    Plot Precision-Recall curves for all models on the same figure.

    Parameters
    ----------
    model_results : Dict[str, Any]
        Dictionary keyed by model name, containing 'y_true' and 'y_prob'.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("tab10", n_colors=max(len(model_results), 1))

    baseline_set = False
    for idx, (name, data) in enumerate(model_results.items()):
        color = colors[idx % len(colors)]
        if "y_true" in data and "y_prob" in data:
            _yp = np.asarray(data["y_prob"])
            _yp1d = _yp[:, 1] if _yp.ndim == 2 else _yp
            precision, recall, _ = precision_recall_curve(data["y_true"], _yp1d)
            pr_auc = auc(recall, precision)
            ax.plot(
                recall,
                precision,
                label=f"{name} (PR-AUC = {pr_auc:.3f})",
                color=color,
                linewidth=2.2,
            )
            if not baseline_set:
                pos_ratio = np.mean(data["y_true"])
                ax.axhline(
                    y=pos_ratio,
                    color="gray",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Baseline ({pos_ratio:.2f})",
                )
                baseline_set = True
        elif "precision" in data and "recall" in data:
            precision, recall = data["precision"], data["recall"]
            pr_auc = data.get("pr_auc", auc(recall, precision))
            ax.plot(
                recall,
                precision,
                label=f"{name} (PR-AUC = {pr_auc:.3f})",
                color=color,
                linewidth=2.2,
            )

    ax.set_title("Precision-Recall (PR) Curves", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="lower left", fontsize=10)

    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "pr_curves.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_confusion_matrices(model_results: Dict[str, Any]) -> plt.Figure:
    """
    Plot a 2x2 subplot grid of confusion matrices for up to four models.

    Parameters
    ----------
    model_results : Dict[str, Any]
        Dictionary keyed by model name, containing 'y_true' and 'y_pred'.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    model_names = list(model_results.keys())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        data = model_results[name]
        if "cm" in data:
            cm = np.array(data["cm"])
        elif "y_true" in data and "y_pred" in data:
            cm = confusion_matrix(data["y_true"], data["y_pred"])
        else:
            ax.set_title(name)
            ax.text(0.5, 0.5, "No prediction data", ha="center", va="center")
            continue

        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Normal", "Risk"],
            yticklabels=["Normal", "Risk"],
            annot_kws={"fontsize": 11, "fontweight": "bold"},
        )
        ax.set_title(f"{name}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    # Blank out any remaining unused subplots
    for j in range(len(model_names), 4):
        axes[j].set_axis_off()

    fig.suptitle("Confusion Matrices across Models", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "confusion_matrices.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_shap_summary(shap_values: Any, feature_names: List[str], model_name: str) -> plt.Figure:
    """
    Plot SHAP summary / beeswarm plot for the specified model.

    Parameters
    ----------
    shap_values : Any
        SHAP values array or shap.Explanation object.
    feature_names : List[str]
        List of feature names.
    model_name : str
        Name of the model.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        import shap
        # Check if shap_values is a shap.Explanation or numpy array
        if isinstance(shap_values, np.ndarray):
            vals = shap_values
            if vals.ndim == 3:  # Multiclass or binary with 2 outputs
                vals = vals[:, :, 1]
            shap.summary_plot(vals, feature_names=feature_names, show=False, plot_size=None)
        else:
            shap.summary_plot(shap_values, feature_names=feature_names, show=False, plot_size=None)
        fig = plt.gcf()
    except Exception:
        # Fallback to bar-like summary if beeswarm fails
        if isinstance(shap_values, np.ndarray):
            vals = shap_values
            if vals.ndim == 3:
                vals = vals[:, :, 1]
            mean_abs = np.mean(np.abs(vals), axis=0)
        else:
            mean_abs = np.zeros(len(feature_names))

        imp_df = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        imp_df = imp_df.sort_values("importance", ascending=True).tail(15)

        ax.barh(imp_df["feature"], imp_df["importance"], color="#1f77b4", edgecolor="black")
        ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=11)
        ax.set_title(f"SHAP Feature Impact - {model_name}", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / f"shap_summary_{model_name}.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_shap_bar(global_importance: pd.DataFrame, model_name: str) -> plt.Figure:
    """
    Horizontal bar chart of mean |SHAP| values.

    Parameters
    ----------
    global_importance : pd.DataFrame
        DataFrame containing 'feature' and 'importance' columns.
    model_name : str
        Name of the model.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    df = global_importance.copy()
    if "importance" not in df.columns:
        # Check if importance is in another column
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df["importance"] = df[num_cols[0]]
    if "feature" not in df.columns:
        df["feature"] = df.index

    df = df.sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Highlight trend features
    colors = ["#ff7f0e" if "trend" in str(f) or "rate" in str(f) else "#1f77b4" for f in df["feature"]]

    bars = ax.barh(df["feature"], df["importance"], color=colors, edgecolor="black", linewidth=0.7)

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.002,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_title(f"Global Feature Importance (SHAP) - {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    # Custom legend for trend vs raw
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ff7f0e", edgecolor="black", label="Trend-Aware Feature"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Raw Feature"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / f"shap_bar_{model_name}.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_shap_waterfall(
    shap_values: Any,
    sample_index: int,
    model_name: str,
    risk_level: str,
    feature_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Generate waterfall plot for an individual prediction.

    Parameters
    ----------
    shap_values : Any
        SHAP values object (shap.Explanation or array/dict).
    sample_index : int
        Index of the sample to plot.
    model_name : str
        Name of the model.
    risk_level : str
        Categorical risk level ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
    feature_names : Optional[List[str]]
        List of feature names if not embedded in shap_values.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    plotted_via_shap = False
    try:
        import shap
        if hasattr(shap.plots, "waterfall") and hasattr(shap_values, "__getitem__"):
            expl = shap_values[sample_index]
            shap.plots.waterfall(expl, max_display=10, show=False)
            fig = plt.gcf()
            plotted_via_shap = True
    except Exception:
        plotted_via_shap = False

    if not plotted_via_shap:
        # Fallback custom waterfall / contribution plot
        if isinstance(shap_values, np.ndarray):
            vals = shap_values[sample_index]
            if vals.ndim == 2:
                vals = vals[:, 1]
        elif isinstance(shap_values, dict):
            vals = np.array(list(shap_values.values()))
            feature_names = list(shap_values.keys())
        else:
            vals = np.array(shap_values)

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(vals))]

        order = np.argsort(np.abs(vals))[-10:]
        top_names = [feature_names[i] for i in order]
        top_vals = vals[order]

        colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_vals]
        ax.barh(top_names, top_vals, color=colors, edgecolor="black", linewidth=0.7)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("SHAP Contribution to Risk Score", fontsize=11)
        ax.set_title(
            f"Local Explanation: Case #{sample_index} ({model_name} - {risk_level} Risk)",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    risk_col = config.RISK_COLORS.get(risk_level.upper(), "#333333")
    fig.patch.set_edgecolor(risk_col)

    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / f"shap_waterfall_{model_name}_{risk_level.lower()}.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_calibration_curves(calibration_data: Dict[str, Any]) -> plt.Figure:
    """
    Plot calibration curves for all models along with the perfect calibration line.

    Parameters
    ----------
    calibration_data : Dict[str, Any]
        Dictionary where keys are model names and values are either:
        - (prob_true, prob_pred) tuple
        - dict with 'y_true' and 'y_prob'
        - dict with 'prob_true' and 'prob_pred'

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    colors = sns.color_palette("tab10", n_colors=max(len(calibration_data), 1))

    for idx, (name, data) in enumerate(calibration_data.items()):
        color = colors[idx % len(colors)]
        if isinstance(data, tuple) and len(data) == 2:
            prob_true, prob_pred = data
        elif isinstance(data, dict):
            if "prob_true" in data and "prob_pred" in data:
                prob_true, prob_pred = data["prob_true"], data["prob_pred"]
            elif "y_true" in data and "y_prob" in data:
                prob_true, prob_pred = calibration_curve(data["y_true"], np.asarray(data["y_prob"])[:, 1] if np.asarray(data["y_prob"]).ndim == 2 else data["y_prob"], n_bins=10)
            else:
                continue
        else:
            continue

        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{name}",
        )

    ax.set_title("Probability Calibration Curves", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left", fontsize=10)

    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "calibration_curves.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_explanation_stability(stability_data: Union[pd.DataFrame, Dict[str, Any], np.ndarray]) -> plt.Figure:
    """
    Heatmap of pairwise rank correlations across LOBO-CV folds.

    Parameters
    ----------
    stability_data : Union[pd.DataFrame, Dict[str, Any], np.ndarray]
        Pairwise correlation matrix or dictionary containing 'correlation_matrix'.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    if isinstance(stability_data, dict):
        if "correlation_matrix" in stability_data:
            mat = stability_data["correlation_matrix"]
        elif "pairwise_correlations" in stability_data:
            mat = stability_data["pairwise_correlations"]
        else:
            mat = list(stability_data.values())[0]
    else:
        mat = stability_data

    if not isinstance(mat, pd.DataFrame):
        cell_labels = [f"Cell {c}" for c in config.BATTERY_CELLS[: len(mat)]]
        df_corr = pd.DataFrame(mat, index=cell_labels, columns=cell_labels)
    else:
        df_corr = mat

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        df_corr,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=1.0,
        cbar_kws={"label": "Spearman Rank Correlation (ρ)"},
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )

    ax.set_title("Explanation Stability across LOBO-CV Folds", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "explanation_stability.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_feature_comparison_trend_vs_raw(
    importance_raw: pd.DataFrame,
    importance_all: pd.DataFrame,
) -> plt.Figure:
    """
    Side-by-side comparison of feature importance with raw-only vs all features.

    Parameters
    ----------
    importance_raw : pd.DataFrame
        DataFrame with ['feature', 'importance'] for raw-only model.
    importance_all : pd.DataFrame
        DataFrame with ['feature', 'importance'] for all-features model.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "importance" not in d.columns:
            num_cols = d.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                d["importance"] = d[num_cols[0]]
        if "feature" not in d.columns:
            d["feature"] = d.index
        return d.sort_values("importance", ascending=True).tail(10)

    raw_sub = _prep_df(importance_raw)
    all_sub = _prep_df(importance_all)

    # Subplot 1: Raw-only features
    ax1.barh(raw_sub["feature"], raw_sub["importance"], color="#1f77b4", edgecolor="black")
    ax1.set_title("Model with Raw Features Only", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax1.grid(True, axis="x", linestyle=":", alpha=0.6)

    # Subplot 2: All features (color code trend features)
    colors = ["#ff7f0e" if "trend" in str(f) or "rate" in str(f) else "#1f77b4" for f in all_sub["feature"]]
    ax2.barh(all_sub["feature"], all_sub["importance"], color=colors, edgecolor="black")
    ax2.set_title("Model with Raw + Trend-Aware Features", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ff7f0e", edgecolor="black", label="Trend Feature"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Raw Feature"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", frameon=True)

    fig.suptitle("Feature Importance Comparison: Raw vs. Trend-Aware", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "trend_vs_raw_importance.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig


def plot_warning_summary(warnings: List[Dict[str, Any]]) -> plt.Figure:
    """
    Plot bar chart of risk level distribution and pie chart of most common contributing factors.

    Parameters
    ----------
    warnings : List[Dict[str, Any]]
        List of generated warning dictionaries.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # 1. Risk level distribution
    risk_levels = [w.get("risk_level", "LOW").upper() for w in warnings]
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = {lvl: risk_levels.count(lvl) for lvl in order}

    bar_colors = [config.RISK_COLORS.get(lvl, "#999999") for lvl in order]
    bars = ax1.bar(order, [counts[lvl] for lvl in order], color=bar_colors, edgecolor="black", linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.5,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax1.set_title("Safety Warning Risk Level Distribution", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Risk Level", fontsize=11)
    ax1.set_ylabel("Count of Warnings", fontsize=11)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    # 2. Contributing factors frequency
    factors = []
    for w in warnings:
        for f in w.get("contributing_factors", []):
            if isinstance(f, dict):
                factors.append(f.get("feature", "unknown"))
            elif isinstance(f, str):
                factors.append(f)

    if factors:
        factor_counts = pd.Series(factors).value_counts().head(6)
        colors = sns.color_palette("Set2", n_colors=len(factor_counts))
        ax2.pie(
            factor_counts.values,
            labels=factor_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "black", "linewidth": 0.8},
        )
        ax2.set_title("Top Primary Contributing Factors", fontsize=13, fontweight="bold", pad=10)
    else:
        ax2.text(0.5, 0.5, "No contributing factors found", ha="center", va="center")
        ax2.set_axis_off()

    fig.suptitle("Safety Warning System Summary Overview", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    _ensure_dir(config.FIGURES_DIR)
    save_path = config.FIGURES_DIR / "warning_summary.png"
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return fig
