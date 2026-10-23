"""
Streamlit Web Dashboard for Explainable AI Battery Safety & Interpretable Warnings.
(English-only edition with dual Everyday User vs. Battery Engineer modes)

Two scoring modes:
- Evaluation (Diagnosis, Risk Timeline, Model Performance): models are trained
  Leave-One-Battery-Out, so the selected cell is never seen during training.
- Deployment (What-If Simulator, Batch Scoring): models are trained on all cells.
"""

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Project root imports
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
import config
from src.explainability import compute_shap_values, get_local_explanation
from src.feature_engineering import get_all_feature_names, get_raw_feature_names
from src.models import get_model, predict_proba_safe, train_model
from src.multilingual_explainer import translate_warning
from src.warning_module import (
    assess_feature_status,
    classify_risk_level,
    estimate_prediction_uncertainty,
    format_warning_text,
    generate_warning,
)

FEATURES = get_all_feature_names()
RAW_FEATURES = set(get_raw_feature_names())
TARGET = "risk_binary"
MODEL_NAMES = list(config.MODEL_PARAMS)
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
EXPLAINER_NOTES = {
    "logistic_regression": "LinearExplainer on standardized features; values are log-odds shifts.",
    "decision_tree": "TreeExplainer; values are shifts in predicted risk probability.",
    "random_forest": "TreeExplainer; values are shifts in predicted risk probability.",
    "xgboost": "TreeExplainer; values are log-odds shifts.",
}
MAX_BATCH_ROWS = 5000

# Chart palette (validated reference palette; status colors carry icon + label too)
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
RAISES_RISK = "#e34948"
LOWERS_RISK = "#2a78d6"
RULE_GRAY = "#8a8985"
STATUS_STYLE = {
    "LOW": ("#0ca30c", "#ffffff", "✅"),
    "MEDIUM": ("#fab219", "#0b0b0b", "⚠️"),
    "HIGH": ("#ec835a", "#0b0b0b", "🔶"),
    "CRITICAL": ("#d03b3b", "#ffffff", "🛑"),
}

# Page setup
st.set_page_config(
    page_title="EV Battery Safety AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .risk-badge { padding: 4px 14px; border-radius: 20px; font-weight: 600; display: inline-block; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Data, models & explanations (cached)
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset() -> pd.DataFrame:
    data_path = config.DATA_PROCESSED_DIR / "battery_features.csv"
    if not data_path.exists():
        return pd.DataFrame()
    return pd.read_csv(data_path)


@st.cache_resource(show_spinner="Training models…")
def train_models(holdout: str):
    """Train every model on all cells except `holdout` ('' trains on all cells)."""
    data = load_dataset()
    train = data[data["cell_name"] != holdout]
    models = {
        name: train_model(get_model(name), train[FEATURES], train[TARGET])
        for name in MODEL_NAMES
    }
    return models, train[FEATURES].reset_index(drop=True)


def predict_all(models, X: pd.DataFrame) -> pd.DataFrame:
    """Risk probability (class 1) from every model, one column per model."""
    return pd.DataFrame(
        {name: predict_proba_safe(m, X)[:, 1] for name, m in models.items()},
        index=X.index,
    )


@st.cache_data(show_spinner="Scoring unseen battery cycles…")
def lobo_predictions() -> pd.DataFrame:
    """Out-of-fold probabilities for every cycle of every cell (LOBO-CV)."""
    data = load_dataset()
    parts = []
    for cell in sorted(data["cell_name"].unique()):
        models, _ = train_models(cell)
        cell_df = data[data["cell_name"] == cell]
        probs = predict_all(models, cell_df[FEATURES])
        parts.append(pd.concat([cell_df[["cell_name", "cycle_index", TARGET, "risk_score"]], probs], axis=1))
    return pd.concat(parts).sort_values(["cell_name", "cycle_index"]).reset_index(drop=True)


def consensus(probs: dict) -> dict:
    """Inter-model agreement on the binary risk call for a single instance."""
    model_results = {n: {"all_y_pred": [int(p >= config.RISK_THRESHOLD)]} for n, p in probs.items()}
    return estimate_prediction_uncertainty(model_results, 0)


def assess(models, background: pd.DataFrame, model_name: str, x_row: pd.DataFrame, cell="N/A", cycle="N/A"):
    """Predict, explain with SHAP and build the structured warning for one cycle."""
    probs = {n: float(p) for n, p in predict_all(models, x_row).iloc[0].items()}
    shap_exp = compute_shap_values(models[model_name], x_row, model_name, background=background)
    local = get_local_explanation(shap_exp, 0, FEATURES)
    warning = generate_warning(
        risk_score=probs[model_name],
        shap_contributions=local,
        feature_values=x_row.iloc[0].to_dict(),
        uncertainty=consensus(probs),
    )
    warning.update({"cell_name": cell, "cycle_index": cycle, "model": model_name})
    return warning, local, probs


def classification_metrics(y_true, prob) -> dict:
    y_pred = (np.asarray(prob) >= config.RISK_THRESHOLD).astype(int)
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    out["ROC-AUC"] = roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else np.nan
    return out


# ──────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────
def risk_badge(level: str, label: str) -> str:
    bg, fg, icon = STATUS_STYLE.get(level, ("#8a8985", "#ffffff", "•"))
    return f"<span class='risk-badge' style='background:{bg};color:{fg}'>{icon} {label}</span>"


def shap_chart(local: dict, top_k: int = 8) -> alt.Chart:
    rows = local["feature_contributions"][:top_k]
    chart_df = pd.DataFrame({
        "feature": [r["feature"] for r in rows],
        "description": [config.FEATURE_DESCRIPTIONS.get(r["feature"], r["feature"]) for r in rows],
        "value": [r["feature_value"] for r in rows],
        "shap": [r["shap_value"] for r in rows],
    })
    chart_df["effect"] = np.where(chart_df["shap"] >= 0, "Raises risk", "Lowers risk")
    bars = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=4, height=14).encode(
        x=alt.X("shap:Q", title="SHAP contribution to risk"),
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color(
            "effect:N",
            scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RAISES_RISK, LOWERS_RISK]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("description:N", title="Meaning"),
            alt.Tooltip("value:Q", title="Measured value", format=".4g"),
            alt.Tooltip("shap:Q", title="SHAP", format="+.4f"),
        ],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=RULE_GRAY).encode(x="x:Q")
    return (bars + zero).properties(height=36 * len(chart_df) + 40)


def consensus_chart(probs: dict) -> alt.Chart:
    chart_df = pd.DataFrame({
        "model": [MODEL_LABELS[n] for n in probs],
        "probability": list(probs.values()),
    })
    bars = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=4, height=14, color=SERIES_1).encode(
        x=alt.X("probability:Q", title="Predicted risk probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("model:N", sort=None, title=None),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("probability:Q", title="Risk", format=".1%")],
    )
    rule = alt.Chart(pd.DataFrame({"x": [config.RISK_THRESHOLD]})).mark_rule(
        color=RULE_GRAY, strokeDash=[4, 4]
    ).encode(x="x:Q")
    return (bars + rule).properties(height=190)


def render_warning(warning: dict, local: dict, probs: dict, model_name: str, mode: str, key: str):
    """Status row, audience-specific message, SHAP drivers, consensus and downloads."""
    localized = translate_warning(warning, mode=mode)
    level = warning["risk_level"]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown("**Overall Safety Status**")
        st.markdown(f"<h3>{risk_badge(level, localized['risk_level_localized'])}</h3>", unsafe_allow_html=True)
    c2.metric("Predicted risk", f"{warning['risk_score']:.1%}", help=f"Probability from {MODEL_LABELS[model_name]}")
    c3.metric(
        "Model agreement",
        f"{warning['uncertainty_info']['model_agreement']:.0%} · {warning['confidence']}",
        help="Share of the 4 models that make the same risk / no-risk call.",
    )

    msg_text = f"{localized['summary_message']}\n\n**👉 Recommended next steps:** {localized['action']}"
    if level in ["HIGH", "CRITICAL"]:
        st.error(msg_text, icon="🚨")
    elif level == "MEDIUM":
        st.warning(msg_text, icon="⚠️")
    else:
        st.success(msg_text, icon="✅")
    if warning["uncertainty_flag"]:
        st.warning("The models disagree on this cycle — treat this assessment as uncertain.", icon="❓")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("##### Why: primary drivers of this prediction")
        for phrase in localized["explained_factors"]:
            st.markdown(f"- **{phrase}**")
        if mode == "technical":
            st.caption(f"{MODEL_LABELS[model_name]} — {EXPLAINER_NOTES[model_name]}")
            st.markdown("##### Top factors vs. engineering thresholds")
            st.dataframe(
                pd.DataFrame(warning["contributing_factors"])[
                    ["rank", "feature", "actual_value", "unit", "status", "shap_contribution"]
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("💡 Bars to the right pushed the risk up; bars to the left pulled it down.")
    with col_r:
        st.altair_chart(shap_chart(local), width="stretch")
        if mode == "technical":
            st.markdown("##### Model consensus")
            st.altair_chart(consensus_chart(probs), width="stretch")

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download warning report (.txt)",
        format_warning_text(warning),
        file_name=f"warning_{warning['cell_name']}_{warning['cycle_index']}.txt",
        key=f"{key}_txt",
    )
    d2.download_button(
        "⬇️ Download warning (.json)",
        json.dumps(warning, indent=2, default=float),
        file_name=f"warning_{warning['cell_name']}_{warning['cycle_index']}.json",
        mime="application/json",
        key=f"{key}_json",
    )


df = load_dataset()
if df.empty:
    st.error(
        f"Dataset not found at {config.DATA_PROCESSED_DIR / 'battery_features.csv'}. "
        "Run `python experiments/experiment_runner.py` first."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("🔋 BMS Control Center")
st.sidebar.caption("Explainable AI for Battery Risk & Early Degradation")

audience = st.sidebar.radio(
    "Target Audience Mode",
    options=["Everyday EV User (Simple Language)", "Battery Systems Engineer (Technical)"],
    index=0,
)
mode = "simple" if "Everyday" in audience else "technical"

model_name = st.sidebar.selectbox(
    "Prediction model",
    MODEL_NAMES,
    format_func=lambda n: MODEL_LABELS[n] + (" (best)" if n == "logistic_regression" else ""),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Select Battery Pack & Cycle")
available_cells = sorted(df["cell_name"].unique())
selected_cell = st.sidebar.selectbox("Battery Unit ID", available_cells, index=0)

cell_df = df[df["cell_name"] == selected_cell].sort_values("cycle_index").reset_index(drop=True)
min_c = int(cell_df["cycle_index"].min())
max_c = int(cell_df["cycle_index"].max())
cycle_num = st.sidebar.slider(
    "Discharge Cycle Number",
    min_value=min_c,
    max_value=max_c,
    value=min_c + int(0.75 * (max_c - min_c)),
)

other_cells = [c for c in available_cells if c != selected_cell]
st.sidebar.caption(f"Models for {selected_cell} are trained only on {', '.join(other_cells)}.")

# ──────────────────────────────────────────────────────────────────────
# Main Dashboard
# ──────────────────────────────────────────────────────────────────────
st.title("⚡ Electric Vehicle Battery Safety Assistant")

fold_models, fold_background = train_models(selected_cell)
row_idx = int((cell_df["cycle_index"] - cycle_num).abs().idxmin())
row = cell_df.iloc[row_idx]
x_row = cell_df.loc[[row_idx], FEATURES]

tab_diag, tab_timeline, tab_whatif, tab_batch, tab_perf, tab_about = st.tabs([
    "🩺 Diagnosis",
    "📈 Risk Timeline",
    "🧪 What-If Simulator",
    "📂 Batch Scoring",
    "📊 Model Performance",
    "ℹ️ Inputs & Outputs",
])

# ── Diagnosis ─────────────────────────────────────────────────────────
with tab_diag:
    st.caption(f"Cell **{selected_cell}**, cycle **{int(row['cycle_index'])}** — scored as an unseen battery.")

    temp_val = float(row["temp_max"])
    temp_limit = config.SAFETY_THRESHOLDS["temperature"]["normal_max"]
    eol_ah = config.NOMINAL_CAPACITY_AH * config.EOL_CAPACITY_THRESHOLD
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Temperature", f"{temp_val:.1f} °C", delta=f"{temp_val - temp_limit:+.1f} °C vs {temp_limit:.0f} °C limit",
              delta_color="inverse")
    m2.metric("Remaining Capacity", f"{row['capacity']:.2f} Ah", delta=f"{row['capacity'] - eol_ah:+.2f} Ah vs EOL",
              help=f"End-of-life = {config.EOL_CAPACITY_THRESHOLD:.0%} of {config.NOMINAL_CAPACITY_AH} Ah nominal")
    m3.metric("Capacity Fade", f"{row['capacity_fade_pct']:.1f} %", help="Capacity lost since the cell's first cycles")
    m4.metric("Impedance Rise", f"+{row['impedance_rise_pct']:.1f} %", help="Internal resistance growth since first cycles")

    st.markdown("---")
    warning, local, probs = assess(
        fold_models, fold_background, model_name, x_row, selected_cell, int(row["cycle_index"])
    )
    render_warning(warning, local, probs, model_name, mode, key="diag")

    if mode == "technical":
        truth = "elevated risk" if row[TARGET] == 1 else "normal"
        st.caption(f"Ground-truth label for this cycle (multi-factor labeling, score {row['risk_score']:.3f}): **{truth}**.")
        with st.expander("All sensor readings vs. engineering thresholds"):
            st.dataframe(
                pd.DataFrame([
                    {
                        "feature": f,
                        "value": float(row[f]),
                        "unit": config.FEATURE_UNITS.get(f, ""),
                        "status": assess_feature_status(f, float(row[f])),
                        "description": config.FEATURE_DESCRIPTIONS.get(f, f),
                    }
                    for f in FEATURES
                ]),
                hide_index=True,
                width="stretch",
            )

# ── Risk Timeline ─────────────────────────────────────────────────────
with tab_timeline:
    preds = lobo_predictions()
    cell_preds = preds[preds["cell_name"] == selected_cell].copy()
    cell_preds["Model prediction"] = cell_preds[model_name]
    cell_preds["Labeling risk score"] = cell_preds["risk_score"]

    flagged = cell_preds[cell_preds[model_name] >= config.RISK_THRESHOLD]
    labeled = cell_preds[cell_preds[TARGET] == 1]
    eol = cell_df[cell_df["capacity"] < eol_ah]
    k1, k2, k3 = st.columns(3)
    k1.metric("First cycle flagged by model", int(flagged["cycle_index"].min()) if len(flagged) else "Never")
    k2.metric("First cycle labeled at-risk", int(labeled["cycle_index"].min()) if len(labeled) else "Never")
    if len(eol) and len(flagged):
        lead = int(eol["cycle_index"].min() - flagged["cycle_index"].min())
        k3.metric(
            "First alert vs. end-of-life",
            f"{abs(lead)} cycles {'before' if lead >= 0 else 'after'}",
            help=f"End-of-life (capacity < {eol_ah:.1f} Ah) is first reached at cycle {int(eol['cycle_index'].min())}.",
        )
    else:
        k3.metric("Cell reaches EOL (80% capacity)", f"cycle {int(eol['cycle_index'].min())}" if len(eol) else "Not reached")

    long_df = cell_preds.melt(
        id_vars=["cycle_index"], value_vars=["Model prediction", "Labeling risk score"],
        var_name="series", value_name="risk",
    )
    hover = alt.selection_point(fields=["cycle_index"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(long_df).encode(x=alt.X("cycle_index:Q", title="Cycle", scale=alt.Scale(nice=False, zero=False)))
    lines = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("risk:Q", title="Risk (0–1)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=["Model prediction", "Labeling risk score"], range=[SERIES_1, SERIES_2]),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    points = base.mark_point(size=60, filled=True).encode(
        y="risk:Q", color=alt.Color("series:N", legend=None),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    )
    crosshair = base.mark_rule(color=RULE_GRAY).encode(
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[
            alt.Tooltip("cycle_index:Q", title="Cycle"),
            alt.Tooltip("risk:Q", title="Risk", format=".3f"),
            alt.Tooltip("series:N", title="Series"),
        ],
    ).add_params(hover)
    threshold = alt.Chart(pd.DataFrame({"y": [config.RISK_THRESHOLD]})).mark_rule(
        color=RULE_GRAY, strokeDash=[4, 4]
    ).encode(y="y:Q")
    current = alt.Chart(pd.DataFrame({"x": [cycle_num]})).mark_rule(color=RULE_GRAY, strokeWidth=1.5).encode(x="x:Q")
    st.markdown(f"#### Predicted risk over the life of {selected_cell} ({MODEL_LABELS[model_name]})")
    st.altair_chart((lines + points + crosshair + threshold + current).properties(height=300), width="stretch")
    st.caption("Dashed line: 0.5 alert threshold. Solid gray line: selected cycle. "
               "The labeling risk score is the formula used to create the training labels.")

    st.markdown("#### Capacity fade over cycle lifetime")
    cap = alt.Chart(cell_df).mark_line(strokeWidth=2, color=SERIES_1).encode(
        x=alt.X("cycle_index:Q", title="Cycle", scale=alt.Scale(nice=False, zero=False)),
        y=alt.Y("capacity:Q", title="Capacity (Ah)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("cycle_index:Q", title="Cycle"), alt.Tooltip("capacity:Q", title="Capacity (Ah)", format=".3f")],
    )
    eol_rule = alt.Chart(pd.DataFrame({"y": [eol_ah]})).mark_rule(color=RAISES_RISK, strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((cap + eol_rule + current).properties(height=260), width="stretch")
    st.caption(f"Red dashed line: end-of-life threshold ({eol_ah:.1f} Ah = 80% of nominal).")

# ── What-If Simulator ─────────────────────────────────────────────────
with tab_whatif:
    st.markdown("#### Enter your own cycle measurements")
    st.caption(
        f"Pre-filled with {selected_cell} cycle {int(row['cycle_index'])}. Change any value and press **Assess**. "
        "Uses models trained on all four cells."
    )
    full_models, full_background = train_models("")
    with st.form("whatif"):
        inputs = {}
        cols = st.columns(3)
        for i, feat in enumerate(FEATURES):
            unit = config.FEATURE_UNITS.get(feat, "")
            group = "raw" if feat in RAW_FEATURES else "trend"
            inputs[feat] = cols[i % 3].number_input(
                f"{feat} [{unit}]" if unit else feat,
                value=float(row[feat]),
                format="%.6g",
                help=f"{config.FEATURE_DESCRIPTIONS.get(feat, feat)} ({group} feature)",
                key=f"wi_{selected_cell}_{int(row['cycle_index'])}_{feat}",
            )
        submitted = st.form_submit_button("Assess", type="primary")
    if submitted:
        wi_row = pd.DataFrame([inputs], columns=FEATURES)
        wi_warning, wi_local, wi_probs = assess(full_models, full_background, model_name, wi_row, "custom", "custom")
        baseline = float(predict_proba_safe(full_models[model_name], x_row)[0, 1])
        st.markdown("---")
        st.caption(f"Risk for the unmodified cycle with the same model: {baseline:.1%} "
                   f"→ your inputs: {wi_warning['risk_score']:.1%} ({wi_warning['risk_score'] - baseline:+.1%}).")
        render_warning(wi_warning, wi_local, wi_probs, model_name, mode, key="whatif")

# ── Batch Scoring ─────────────────────────────────────────────────────
with tab_batch:
    st.markdown("#### Score many cycles from a CSV file")
    st.caption(
        f"The CSV needs one row per discharge cycle with the {len(FEATURES)} feature columns listed under "
        "*Inputs & Outputs*. Optional `cell_name` / `cycle_index` columns are passed through. "
        "Uses models trained on all four cells."
    )
    template = df[["cell_name", "cycle_index"] + FEATURES].sample(10, random_state=config.RANDOM_SEED)
    st.download_button("⬇️ Download example CSV", template.to_csv(index=False), file_name="battery_cycles_example.csv")
    uploaded = st.file_uploader("Upload cycles CSV", type="csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as err:
            st.error(f"Could not read CSV: {err}")
            batch = None
        if batch is not None:
            missing = [f for f in FEATURES if f not in batch.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            elif len(batch) > MAX_BATCH_ROWS:
                st.error(f"Too many rows ({len(batch)}); the limit is {MAX_BATCH_ROWS}.")
            else:
                X_batch = batch[FEATURES].apply(pd.to_numeric, errors="coerce")
                bad = X_batch.isna().any(axis=1)
                if bad.any():
                    st.warning(f"Skipping {int(bad.sum())} row(s) with missing or non-numeric values.")
                X_batch = X_batch[~bad]
                if X_batch.empty:
                    st.error("No valid rows to score.")
                else:
                    b_probs = predict_all(full_models, X_batch)
                    b_shap = compute_shap_values(full_models[model_name], X_batch, model_name, background=full_background)
                    shap_vals = np.asarray(b_shap.values)
                    top = np.argsort(-np.abs(shap_vals), axis=1)[:, :2]
                    votes = (b_probs >= config.RISK_THRESHOLD).astype(int)
                    agreement = votes.apply(lambda r: max(r.sum(), len(r) - r.sum()) / len(r), axis=1)

                    out = batch.loc[X_batch.index, [c for c in ["cell_name", "cycle_index"] if c in batch.columns]].copy()
                    out["risk_probability"] = b_probs[model_name].round(4)
                    out["risk_level"] = b_probs[model_name].map(classify_risk_level)
                    out["model_agreement"] = agreement.round(2)
                    out["top_driver"] = [FEATURES[i] for i in top[:, 0]]
                    out["second_driver"] = [FEATURES[i] for i in top[:, 1]]
                    out["top_driver_shap"] = shap_vals[np.arange(len(top)), top[:, 0]].round(4)

                    s1, s2, s3 = st.columns(3)
                    s1.metric("Cycles scored", len(out))
                    s2.metric("Flagged at risk", int((out["risk_probability"] >= config.RISK_THRESHOLD).sum()))
                    s3.metric("Low-agreement rows", int((out["model_agreement"] < 0.7).sum()))
                    st.dataframe(out, hide_index=True, width="stretch")
                    st.download_button("⬇️ Download scored results", out.to_csv(index=False), file_name="battery_risk_scores.csv")

# ── Model Performance ─────────────────────────────────────────────────
with tab_perf:
    preds = lobo_predictions()
    st.markdown("#### Leave-One-Battery-Out performance (each cell scored by models that never saw it)")
    overall = pd.DataFrame(
        {MODEL_LABELS[n]: classification_metrics(preds[TARGET], preds[n]) for n in MODEL_NAMES}
    ).T
    st.dataframe(overall.style.format("{:.3f}").highlight_max(axis=0), width="stretch")

    st.markdown(f"#### {MODEL_LABELS[model_name]} — per held-out cell")
    per_cell = pd.DataFrame({
        cell: classification_metrics(g[TARGET], g[model_name]) for cell, g in preds.groupby("cell_name")
    }).T
    per_cell["Cycles"] = preds.groupby("cell_name").size()
    per_cell["At-risk cycles"] = preds.groupby("cell_name")[TARGET].sum()
    st.dataframe(per_cell.style.format("{:.3f}", subset=list(overall.columns)), width="stretch")
    st.caption("F1 / precision / recall use the 0.5 risk threshold. Cells with no at-risk cycles have no ROC-AUC.")

    fig_dir = config.FIGURES_DIR
    figs = [p for p in ["roc_curves.png", "confusion_matrices.png", "trend_vs_raw_importance.png"] if (fig_dir / p).exists()]
    if figs:
        with st.expander("Figures from the full experiment run (E1–E6)"):
            for p in figs:
                st.image(str(fig_dir / p), width="stretch")

# ── Inputs & Outputs ──────────────────────────────────────────────────
with tab_about:
    st.markdown("""
#### Input
One **discharge cycle** of a Li-ion cell, described by 23 numbers:
- **13 raw features** measured during that cycle (temperatures, voltages, current, capacity, duration, impedance).
- **10 trend features** computed from the cell's recent cycles (temperature trend, capacity fade, impedance rise, …).

#### Output
- **Risk probability** (0–1) that the cycle is in an elevated-degradation-risk state, and a **risk level**:
  LOW < 0.30 ≤ MEDIUM < 0.50 ≤ HIGH ≤ 0.75 < CRITICAL.
- **Top contributing factors** (SHAP), each with its measured value, unit and status against engineering thresholds.
- **Recommended action** in plain or technical language.
- **Confidence** from agreement among four models (Logistic Regression, Decision Tree, Random Forest, XGBoost).

*Model-based assessment of degradation risk; not a substitute for physical diagnostics and not a
prediction of thermal runaway.*
""")
    st.dataframe(
        pd.DataFrame([
            {
                "feature": f,
                "type": "raw" if f in RAW_FEATURES else "trend",
                "unit": config.FEATURE_UNITS.get(f, ""),
                "description": config.FEATURE_DESCRIPTIONS.get(f, f),
                "dataset min": df[f].min(),
                "dataset max": df[f].max(),
            }
            for f in FEATURES
        ]),
        hide_index=True,
        width="stretch",
    )

st.markdown("---")
st.caption("Explainable AI for Battery Safety Project • Powered by SHAP and NASA Li-ion Prognostics Data")
