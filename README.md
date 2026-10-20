# Explainable AI for Battery Safety: A Trend-Aware Framework for Reliable Risk Explanations and Interpretable Warnings

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)

A complete research framework that bridges Machine Learning predictions and Explainable AI (XAI) for lithium-ion battery degradation risk. Rather than merely presenting a black-box failure probability, this system translates SHAP/LIME attributions into actionable, natural-language safety warnings tailored for both EV drivers and battery systems engineers.

---

## 🌟 Key Contributions

1. **Trend-Aware Degradation Modeling**: Integrates 10 temporal and dynamic degradation features (e.g., $T_{\text{trend}}$, capacity fade rate, internal impedance rise) alongside instantaneous sensor readings. Trend-only features achieve **$F_1 = 0.866$**, significantly outperforming raw features alone ($F_1 = 0.658$).
2. **Leave-One-Battery-Out Cross-Validation (LOBO-CV)**: Evaluates generalization across completely unseen battery cells (`B0005`, `B0006`, `B0007`, `B0018`) from the NASA Li-ion Prognostics dataset.
3. **Instance-Level Explanation-to-Warning Translation**: Automatically maps Tree-based SHAP attributions into engineering thresholds (`NORMAL`, `WARNING`, `CRITICAL`), accompanied by explicit corrective actions.
4. **Sub-Threshold Early Detection**: Identifies **11.5% of elevated-risk cases** before static univariate BMS limits are tripped.
5. **Interactive Streamlit Dashboard**: A user-friendly, dual-audience (Everyday EV Driver vs. Battery Systems Engineer) web application.

---

## 📊 Experimental Results

Benchmarked under rigorous Leave-One-Battery-Out Cross-Validation (LOBO-CV):

| Model | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.968** | **0.962** | **0.953** | **0.957** | **0.996** |
| Decision Tree | 0.837 | 0.751 | 0.856 | 0.800 | 0.860 |
| Random Forest | 0.753 | 0.681 | 0.661 | 0.671 | 0.922 |
| XGBoost | 0.832 | 0.741 | 0.860 | 0.796 | 0.956 |

### Core Experiments Suite (E1–E6)
- **E1**: Model Performance Comparison (LOBO-CV)
- **E2**: Raw vs. Trend-Aware Feature Importance
- **E3**: Instance Explanations (SHAP & LIME alignment)
- **E4**: Cross-Battery Explanation Stability ($\rho = 0.683$)
- **E5**: Safety Warning Quality Evaluation (100% factor correctness)
- **E6**: Full XAI Framework vs. Prediction-Only Baseline

All generated plots and tables are located in `results/figures/` and `results/tables/`.

---

## 📁 Repository Structure

```
├── config.py                  # Global configurations, thresholds & hyperparams
├── requirements.txt           # Python dependencies
├── app.py                     # Interactive Streamlit dashboard
├── src/
│   ├── data_loader.py         # NASA .mat file parser & extractor
│   ├── preprocessing.py       # Cleaning, validation & outlier handling
│   ├── feature_engineering.py # Raw & trend-aware feature generation
│   ├── risk_labeling.py       # Multi-factor degradation risk labeling
│   ├── models.py              # ML estimators & LOBO-CV pipeline
│   ├── explainability.py      # SHAP & LIME explanation engines
│   ├── warning_module.py      # Translation layer from SHAP to warnings
│   ├── multilingual_explainer.py # Dual-audience explanation logic
│   ├── evaluation.py          # Metrics, calibration & safety evaluation
│   └── visualization.py       # Publication-ready figure generators
├── experiments/
│   ├── experiment_runner.py   # Master pipeline runner (E1-E6)
│   ├── e1_model_comparison.py
│   ├── e2_feature_importance.py
│   ├── e3_local_explanations.py
│   ├── e4_explanation_stability.py
│   ├── e5_warning_evaluation.py
│   └── e6_warning_vs_prediction.py
├── data/
│   ├── raw/                   # NASA Li-ion battery aging files
│   └── processed/             # Cached feature dataset (620 cycles)
└── results/
    ├── figures/               # 14 publication figures (.png)
    ├── tables/                # 11 experiment metrics tables (.csv)
    └── warnings/              # Representative JSON warnings & summaries
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/battery-safety-xai.git
cd battery-safety-xai
pip install -r requirements.txt
```

### 2. Run All Experiments (E1 to E6)
Execute the complete experimental pipeline end-to-end:
```bash
python experiments/experiment_runner.py
```

### 3. Launch the Interactive Dashboard
Run the user-friendly web interface:
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser to explore live predictions, SHAP watermarks, and driver-facing warnings.

---

## 📜 Dataset & Reference
Data sourced from the **NASA Prognostics Center of Excellence (PCoE)** Li-ion Battery Aging Dataset (Cells B0005, B0006, B0007, B0018).
- *Disclaimer*: This project assesses degradation-associated risk under continuous cycling; it does not claim to predict catastrophic thermal runaway events directly.

---

## 📄 License
This project is licensed under the MIT License.
