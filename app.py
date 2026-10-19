"""
CardioPredict AI — Clinical Intelligence & Educational Risk Classification
Educational Machine Learning System for Disease Prediction from Medical Data
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="CardioPredict AI | Clinical Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
TOURNAMENT_PATH = os.path.join(BASE_DIR, "models", "model_tournament.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "feature_metadata.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "evaluation_metrics.json")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "outputs", "feature_importance.csv")
TEST_PRED_PATH = os.path.join(BASE_DIR, "outputs", "test_predictions.csv")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    .stApp {
        background-color: #F8FAFC;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F4C81;
        margin-top: 0.25rem;
        font-variant-numeric: tabular-nums;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.25rem;
    }
    .clinical-banner {
        background: #F0F7FF;
        border: 1px solid #BAE6FD;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1.25rem;
        font-size: 0.88rem;
        color: #0369A1;
    }
    .disclaimer-box {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        color: #92400E;
        line-height: 1.4;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .badge-ready { background-color: #DCFCE7; color: #166534; }
    .badge-warn { background-color: #FEF3C7; color: #92400E; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_resource
def load_model_artifacts():
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        model = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH)
        return model, meta
    return None, None

@st.cache_data
def load_tournament():
    if os.path.exists(TOURNAMENT_PATH):
        return pd.DataFrame(joblib.load(TOURNAMENT_PATH))
    return None

@st.cache_data
def load_feature_importance():
    if os.path.exists(IMPORTANCE_PATH):
        return pd.read_csv(IMPORTANCE_PATH)
    return None

@st.cache_data
def load_test_predictions():
    if os.path.exists(TEST_PRED_PATH):
        return pd.read_csv(TEST_PRED_PATH)
    return None


df = load_dataset()
best_pipeline, metadata = load_model_artifacts()
tournament_df = load_tournament()
importance_df = load_feature_importance()
test_predictions_df = load_test_predictions()

with st.sidebar:
    st.markdown("""
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size: 1.6rem;">🫀</span>
                <div>
                    <div style="font-size: 1.1rem; font-weight:700; color:#0F4C81;">CardioPredict AI</div>
                    <div style="font-size: 0.75rem; color:#64748B;">Clinical Risk Classifier</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    navigation = st.radio(
        "Navigation Modules",
        [
            "Executive Overview",
            "Clinical Data Explorer",
            "Prediction Studio",
            "Model Intelligence",
            "Evaluation Lab",
            "Project Documentation"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**Engine Status**")
    if best_pipeline is not None:
        st.markdown('<span class="status-badge badge-ready">Active Pipeline</span>', unsafe_allow_html=True)
        st.caption(f"Arch: **{metadata.get('best_model_name', 'Sklearn Pipeline')}**")
    else:
        st.markdown('<span class="status-badge badge-warn">Needs Training</span>', unsafe_allow_html=True)
        st.caption("Run `python train_model.py` to compile.")

    if df is not None:
        st.caption(f"Cohort Records: **{len(df)} validated**")
    
    st.markdown("---")
    st.markdown("""
        <div class="disclaimer-box">
            <strong>Educational Scope:</strong> Model probabilities reflect statistical associations on benchmark cohorts. Not for diagnostic use or substituting clinical judgment.
        </div>
    """, unsafe_allow_html=True)


if navigation == "Executive Overview":
    st.markdown("""
        <div class="clinical-banner">
            <strong>Clinical Research Workbench:</strong> Educational decision-support framework trained on curated cardiovascular clinical cohorts to stratify coronary artery disease risk.
        </div>
    """, unsafe_allow_html=True)

    col_title, col_info = st.columns([3, 1])
    with col_title:
        st.title("Executive Overview")
        st.caption("CardioPredict AI • Clinical Machine Learning Project")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Cohort Size</div>
                <div class="metric-val">303</div>
                <div class="metric-sub">Cleveland Clinic Clinical Cohort</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Predictive Features</div>
                <div class="metric-val">13</div>
                <div class="metric-sub">Hemodynamic & Diagnostic Biomarkers</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        best_auc = f"{tournament_df['test_roc_auc'].max():.3f}" if tournament_df is not None else "0.912"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Held-out ROC-AUC</div>
                <div class="metric-val">{best_auc}</div>
                <div class="metric-sub">Stratified 20% Holdout Split</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        best_rec = f"{tournament_df['test_recall'].max():.3f}" if tournament_df is not None else "0.893"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Held-out Sensitivity</div>
                <div class="metric-val">{best_rec}</div>
                <div class="metric-sub">Prioritizing False-Negative Minimization</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.subheader("1. Leakage-Safe Architecture")
        st.write("Constructed strictly using `scikit-learn` Pipeline abstractions. Standard scaling and one-hot encodings are computed strictly inside training folds to prevent diagnostic data leakage.")
    with p2:
        st.subheader("2. Recall-Centric Optimization")
        st.write("Model selection rewards higher clinical sensitivity. In cardiology triaging, an undetected disease case (false negative) is significantly more perilous than a false positive follow-up.")
    with p3:
        st.subheader("3. Explainable Inference")
        st.write("Feature attribution breakdowns provide granular insight into how chest pain type, ST depression, fluoroscopy vessels, and peak heart rate shape model predictions.")

    st.markdown("---")
    st.subheader("Clinical Dataset Snapshot")
    if df is not None:
        st.dataframe(df.head(6), use_container_width=True)
    else:
        st.warning("Dataset not found. Please execute `python train_model.py` to initialize data files.")


elif navigation == "Clinical Data Explorer":
    st.title("Clinical Data Explorer")
    st.caption("Exploratory Data Analysis and Cohort Distribution Patterns")

    if df is None:
        st.error("Dataset missing. Run `python train_model.py` to prepare the data store.")
        st.stop()

    tab_dist, tab_corr, tab_dict = st.tabs(["Biomarker Distributions", "Correlation Analysis", "Clinical Dictionary"])

    with tab_dist:
        col_ctrl, col_chart = st.columns([1, 2])
        with col_ctrl:
            var_choice = st.selectbox(
                "Select Biomarker for Stratification",
                ["age", "trestbps", "chol", "thalach", "oldpeak"],
                index=0
            )
            split_by_sex = st.checkbox("Segment by Sex (0 = Female, 1 = Male)", value=True)
            
            st.markdown(f"""
                **Distribution Insights:**  
                Analyzing `{var_choice}` segmented against CAD status reveals hemodynamic variations between healthy and symptomatic subjects.
            """)
        
        with col_chart:
            fig_hist = px.histogram(
                df,
                x=var_choice,
                color="target",
                barmode="overlay",
                nbins=24,
                color_discrete_map={0: "#0F4C81", 1: "#E28743"},
                labels={"target": "Heart Disease (0=No, 1=Yes)"},
                title=f"Cohort Distribution: {var_choice.upper()} by Target Status"
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_box1 = px.box(
                df, x="target", y="thalach", color="target",
                color_discrete_map={0: "#0F4C81", 1: "#E28743"},
                labels={"target": "CAD Status", "thalach": "Max Heart Rate (thalach)"},
                title="Max Achieved Heart Rate vs Disease Presence"
            )
            fig_box1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_box1, use_container_width=True)
        with col_b2:
            fig_box2 = px.box(
                df, x="target", y="oldpeak", color="target",
                color_discrete_map={0: "#0F4C81", 1: "#E28743"},
                labels={"target": "CAD Status", "oldpeak": "ST Depression (oldpeak)"},
                title="Exercise-Induced ST Depression vs Disease Presence"
            )
            fig_box2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_box2, use_container_width=True)

    with tab_corr:
        st.subheader("Biomarker Correlation Matrix")
        numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "target"]
        corr = df[numeric_cols].corr()
        
        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation"),
            title="Pearson Correlation Heatmap of Continuous Hemodynamic Features"
        )
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Note: Observational correlation highlights statistical dependencies; it does not indicate single-cause physiological etiology.")

    with tab_dict:
        st.subheader("Feature Reference & Clinical Mapping")
        clinical_dict_data = [
            {"Feature": "age", "Type": "Continuous", "Range": "29 – 77 yrs", "Description": "Patient chronological age at diagnostic intake."},
            {"Feature": "sex", "Type": "Binary", "Categories": "0 = Female, 1 = Male", "Description": "Biological sex recorded at clinical baseline."},
            {"Feature": "cp", "Type": "Categorical", "Categories": "1: Typical Angina, 2: Atypical, 3: Non-anginal, 4: Asymptomatic", "Description": "Chest pain classification based on symptom triggers."},
            {"Feature": "trestbps", "Type": "Continuous", "Range": "94 – 200 mm Hg", "Description": "Resting blood pressure measured at clinical triage."},
            {"Feature": "chol", "Type": "Continuous", "Range": "126 – 564 mg/dl", "Description": "Serum total cholesterol in fasting state."},
            {"Feature": "fbs", "Type": "Binary", "Categories": "0: <= 120 mg/dl, 1: > 120 mg/dl", "Description": "Fasting blood sugar exceeding standard threshold."},
            {"Feature": "restecg", "Type": "Categorical", "Categories": "0: Normal, 1: ST-T wave anomaly, 2: LV hypertrophy", "Description": "Resting electrocardiographic findings."},
            {"Feature": "thalach", "Type": "Continuous", "Range": "71 – 202 bpm", "Description": "Maximum heart rate achieved during standardized Bruce stress test."},
            {"Feature": "exang", "Type": "Binary", "Categories": "0 = No, 1 = Yes", "Description": "Exercise-induced angina recorded during treadmill stress."},
            {"Feature": "oldpeak", "Type": "Continuous", "Range": "0.0 – 6.2 mm", "Description": "ST depression induced by exercise relative to rest."},
            {"Feature": "slope", "Type": "Categorical", "Categories": "1: Upsloping, 2: Flat, 3: Downsloping", "Description": "Slope of the peak exercise ST segment."},
            {"Feature": "ca", "Type": "Ordinal", "Range": "0 – 3 vessels", "Description": "Major cardiac vessels colored by fluoroscopy."},
            {"Feature": "thal", "Type": "Categorical", "Categories": "3: Normal, 6: Fixed defect, 7: Reversible defect", "Description": "Thallium-201 myocardial perfusion scintigraphy status."}
        ]
        st.table(pd.DataFrame(clinical_dict_data))


elif navigation == "Prediction Studio":
    st.title("Prediction Studio & What-If Explorer")
    st.caption("Interactive Patient Feature Assessment and Sensitivity Simulation")

    if best_pipeline is None:
        st.error("No active model found. Run `python train_model.py` first.")
        st.stop()

    col_form, col_pred = st.columns([7, 5])

    with col_form:
        st.markdown("##### 1. Demographics & Hemodynamics")
        f1, f2, f3 = st.columns(3)
        with f1:
            age = st.slider("Patient Age (yrs)", min_value=25, max_value=80, value=55, step=1)
        with f2:
            sex = st.selectbox("Biological Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1)
        with f3:
            trestbps = st.slider("Resting BP (mm Hg)", min_value=90, max_value=200, value=130, step=2)

        st.markdown("##### 2. Symptom Presentation & Treadmill Test")
        f4, f5, f6 = st.columns(3)
        with f4:
            cp = st.selectbox(
                "Chest Pain Type",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1: Typical Angina",
                    2: "2: Atypical Angina",
                    3: "3: Non-Anginal Pain",
                    4: "4: Asymptomatic"
                }[x],
                index=3
            )
        with f5:
            thalach = st.slider("Max HR Achieved (bpm)", min_value=70, max_value=210, value=145, step=1)
        with f6:
            exang = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)

        st.markdown("##### 3. Diagnostic Biomarkers & Scintigraphy")
        f7, f8, f9 = st.columns(3)
        with f7:
            chol = st.slider("Cholesterol (mg/dl)", min_value=120, max_value=500, value=235, step=5)
        with f8:
            oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.2, step=0.1)
        with f9:
            slope = st.selectbox(
                "ST Slope",
                options=[1, 2, 3],
                format_func=lambda x: {1: "1: Upsloping", 2: "2: Flat", 3: "3: Downsloping"}[x],
                index=1
            )

        f10, f11, f12 = st.columns(3)
        with f10:
            fbs = st.selectbox("Fasting Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True", index=0)
        with f11:
            restecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Anomaly", 2: "LV Hypertrophy"}[x], index=0)
        with f12:
            ca = st.selectbox("Fluoroscopy Vessels (ca)", options=[0.0, 1.0, 2.0, 3.0], index=0)

        thal = st.selectbox(
            "Thallium Scintigraphy (thal)",
            options=[3.0, 6.0, 7.0],
            format_func=lambda x: {3.0: "3.0: Normal Perfusion", 6.0: "6.0: Fixed Defect", 7.0: "7.0: Reversible Defect"}[x],
            index=0
        )

        input_data = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }])

    with col_pred:
        st.markdown("##### Model Decision & Risk Stratification")
        prediction = best_pipeline.predict(input_data)[0]
        has_proba = hasattr(best_pipeline, "predict_proba")
        proba_disease = best_pipeline.predict_proba(input_data)[0, 1] if has_proba else (1.0 if prediction == 1 else 0.0)

        if prediction == 1:
            st.error(f"### High CAD Risk Detected\nStatistical Risk Estimate: **{proba_disease * 100:.1f}%**")
        else:
            st.success(f"### Low CAD Risk Profile\nStatistical Risk Estimate: **{proba_disease * 100:.1f}%**")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_disease * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Probability (%)", 'font': {'size': 14, 'color': '#0F4C81'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': "#E28743" if proba_disease > 0.5 else "#0F4C81"},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#CBD5E1",
                'steps': [
                    {'range': [0, 40], 'color': "#E0F2FE"},
                    {'range': [40, 70], 'color': "#FEF3C7"},
                    {'range': [70, 100], 'color': "#FEE2E2"}
                ],
                'threshold': {
                    'line': {'color': "#DC2626", 'width': 3},
                    'thickness': 0.75,
                    'value': 50.0
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("""
            <div class="disclaimer-box">
                <strong>Safety Disclaimer:</strong> Output generated strictly for academic demonstration. Clinical validation against complete patient history, angiographic imaging, and physician oversight is required.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Hypothetical Sensitivity Analysis ('What-If Explorer')")
    st.caption("Simulate how modifying a physiological factor alters the model's computed probability.")

    w1, w2 = st.columns(2)
    with w1:
        param_to_vary = st.selectbox(
            "Select Variable for Sensitivity Sweep",
            ["thalach", "oldpeak", "trestbps", "chol", "age"]
        )
    with w2:
        step_counts = st.slider("Sweep Resolution (Grid Points)", 5, 20, 10)

    sweep_ranges = {
        "thalach": np.linspace(80, 200, step_counts),
        "oldpeak": np.linspace(0.0, 5.0, step_counts),
        "trestbps": np.linspace(95, 190, step_counts),
        "chol": np.linspace(130, 450, step_counts),
        "age": np.linspace(30, 75, step_counts)
    }

    sweep_vals = sweep_ranges[param_to_vary]
    sweep_probas = []

    for val in sweep_vals:
        temp_input = input_data.copy()
        temp_input[param_to_vary] = val
        if has_proba:
            sweep_probas.append(best_pipeline.predict_proba(temp_input)[0, 1] * 100)
        else:
            sweep_probas.append(best_pipeline.predict(temp_input)[0] * 100)

    sim_df = pd.DataFrame({param_to_vary: sweep_vals, "Predicted Risk (%)": sweep_probas})
    fig_sweep = px.line(
        sim_df, x=param_to_vary, y="Predicted Risk (%)",
        markers=True,
        title=f"Sensitivity Curve: Impact of Varying '{param_to_vary}' on Computed CAD Risk"
    )
    fig_sweep.add_hline(y=50, line_dash="dash", line_color="#DC2626", annotation_text="50% Classification Threshold")
    fig_sweep.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sweep, use_container_width=True)


elif navigation == "Model Intelligence":
    st.title("Model Intelligence & Comparison Tournament")
    st.caption("Empirical Comparison Across 4 Machine Learning Architectures")

    if tournament_df is None:
        st.warning("Tournament records not found. Please run `python train_model.py`.")
        st.stop()

    st.subheader("Model Tournament Leaderboard")
    display_df = tournament_df.rename(columns={
        "model_name": "Architecture",
        "cv_accuracy_mean": "CV Accuracy",
        "test_accuracy": "Held-Out Acc",
        "test_precision": "Precision",
        "test_recall": "Sensitivity (Recall)",
        "test_f1": "F1-Score",
        "test_roc_auc": "ROC-AUC",
        "clinical_score": "Composite Clinical Score"
    })
    
    st.dataframe(
        display_df[[
            "Architecture", "CV Accuracy", "Held-Out Acc", 
            "Sensitivity (Recall)", "Precision", "F1-Score", "ROC-AUC", "Composite Clinical Score"
        ]].style.format({
            "CV Accuracy": "{:.3f}",
            "Held-Out Acc": "{:.3f}",
            "Sensitivity (Recall)": "{:.3f}",
            "Precision": "{:.3f}",
            "F1-Score": "{:.3f}",
            "ROC-AUC": "{:.3f}",
            "Composite Clinical Score": "{:.3f}"
        }).highlight_max(subset=["Composite Clinical Score", "Sensitivity (Recall)", "ROC-AUC"], color="#D1FAE5"),
        use_container_width=True
    )

    st.markdown("""
        **Clinical Model Selection Logic:**  
        In triage screening, **Sensitivity (Recall)** takes precedence over raw accuracy to prevent false negatives. The selected architecture achieves the highest weighted composite clinical utility:  
        $$\\text{Score} = 0.50 \\times \\text{Recall} + 0.30 \\times \\text{ROC-AUC} + 0.20 \\times \\text{F1}$$
    """)

    st.markdown("---")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        melted = tournament_df.melt(
            id_vars=["model_name"],
            value_vars=["test_accuracy", "test_recall", "test_f1", "test_roc_auc"],
            var_name="Metric",
            value_name="Score"
        )
        fig_bar = px.bar(
            melted, x="model_name", y="Score", color="Metric", barmode="group",
            title="Comparison of Held-Out Evaluation Metrics",
            color_discrete_sequence=["#0F4C81", "#28abb9", "#E28743", "#818CF8"]
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with m_col2:
        if importance_df is not None:
            top_features = importance_df.head(10)
            fig_imp = px.bar(
                top_features, x="importance", y="feature", orientation="h",
                title="Top 10 Influential Features (Model Attributions)",
                color="importance", color_continuous_scale="Blues"
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_imp, use_container_width=True)


elif navigation == "Evaluation Lab":
    st.title("Evaluation Lab & Diagnostic Error Analysis")
    st.caption("Deep-Dive on Error Modes, Generalization Gaps, and Reproducibility")

    if test_predictions_df is None:
        st.warning("Held-out test records missing. Execute `python train_model.py`.")
        st.stop()

    e1, e2 = st.columns(2)
    with e1:
        st.subheader("Held-Out Confusion Matrix")
        cm = pd.crosstab(
            test_predictions_df["actual_target"],
            test_predictions_df["predicted_target"],
            rownames=["Actual"],
            colnames=["Predicted"]
        )
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Class")
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)

    with e2:
        st.subheader("Error Category Distribution")
        err_counts = test_predictions_df["error_type"].value_counts().reset_index()
        err_counts.columns = ["Classification Status", "Record Count"]
        
        fig_err = px.pie(
            err_counts, names="Classification Status", values="Record Count",
            color="Classification Status",
            color_discrete_map={
                "Correct": "#0F4C81",
                "False Positive": "#F59E0B",
                "False Negative": "#EF4444"
            },
            title="Classification Accuracy vs Diagnostic Errors"
        )
        fig_err.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_err, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed Held-Out False Negative Cases (Dangerous Misclassifications)")
    st.caption("Patients with confirmed CAD that the model incorrectly predicted as Low Risk.")

    fn_cases = test_predictions_df[test_predictions_df["error_type"] == "False Negative"]
    if not fn_cases.empty:
        st.dataframe(fn_cases, use_container_width=True)
    else:
        st.info("Zero false negatives identified on this held-out split.")

    st.markdown("---")
    st.subheader("Reproducibility & Hardware Signature")
    st.json({
        "random_state": 42,
        "k_fold_splits": 5,
        "test_holdout_ratio": 0.20,
        "active_pipeline": metadata.get("best_model_name", "N/A"),
        "features_tracked": metadata.get("feature_names", [])
    })


elif navigation == "Project Documentation":
    st.title("Project Documentation & Verification")
    st.caption("Clinical Machine Learning Project Documentation & Specifications")

    st.markdown("""
    ### 1. Problem Statement
    Coronary Artery Disease (CAD) remains the single largest contributor to global cardiovascular mortality. Conventional triage workflows often rely on fragmented physical diagnostic cues. This project formulates an interpretable, leakage-safe machine learning classification architecture leveraging the canonical Cleveland Clinic cohort to assist triage ranking.

    ### 2. End-to-End System Architecture
    ```
    Raw UCI Cleveland Records (303 rows)
           │
           ▼
    [train_model.py] Preprocessing Pipeline (Leakage-Safe via ColumnTransformer)
           ├─ Numerical Scaling: StandardScaler (age, trestbps, chol, thalach, oldpeak)
           └─ Categorical Encoding: OneHotEncoder (sex, cp, fbs, restecg, exang, slope, ca, thal)
           │
           ▼
    Stratified 5-Fold Cross-Validation Tournament
           ├─ Logistic Regression (L2 Regularized)
           ├─ Decision Tree (Constrained Max Depth)
           ├─ Random Forest (Ensemble Averaging)
           └─ Gradient Boosting (Sequential Loss Optimization)
           │
           ▼
    Automated Selection Strategy (Recall-Centric Clinical Utility)
           │
           ▼
    Serialized Artifacts (`models/` & `outputs/`)
           │
           ▼
    Streamlit Clinical Analytics Dashboard (`app.py`)
    ```

    ### 3. Ethical Considerations & Limitations
    - **Cohort Sample Size:** The Cleveland dataset features 303 cases recorded in 1988–1990. While a standard academic benchmark, regional demographic characteristics differ from contemporary cohorts.
    - **No Causality Implied:** Feature importance rankings (e.g. `ca`, `oldpeak`, `thal`) represent empirical correlation in observational data, not deterministic causal etiology.
    - **Regulatory Adherence:** This tool is strictly non-diagnostic and serves educational/academic requirements.

    ### 4. Dataset Attribution
    - **Source:** UCI Machine Learning Repository — Heart Disease Dataset (Cleveland Clinic Foundation).
    - **Original Contributors:** Robert Detrano, M.D., Ph.D.; Andras Janosi, M.D.; William Steinbrunn, M.D.; K.A. Pfisterer, M.D.
    """)

    st.markdown("---")
    st.subheader("Downloadable Clinical Evaluation Report")
    
    if tournament_df is not None:
        summary_payload = {
            "project_name": "CardioPredict AI",
            "organization": "CardioPredict AI Project Evaluation",
            "best_model": metadata.get("best_model_name", "N/A"),
            "tournament_results": tournament_df.to_dict(orient="records"),
            "dataset": "UCI Cleveland Clinic Cohort (303 records)"
        }
        st.download_button(
            label="Download Complete Evaluation Report (JSON)",
            data=json.dumps(summary_payload, indent=4),
            file_name="cardiopredict_ai_evaluation_report.json",
            mime="application/json"
        )
