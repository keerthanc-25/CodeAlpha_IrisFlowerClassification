"""
=====================================================================
  Iris Flower Species Classifier — Premium ML Dashboard
=====================================================================
Frontend-only redesign.

IMPORTANT: The prediction logic, model loading, and ML workflow in
`src/predict.py` (IrisPredictor) are NOT modified. This file only
changes layout, styling, and presentation around the existing
`predictor.predict_species(...)` call.

Tech: Streamlit • Scikit-Learn • Pandas
=====================================================================
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import IrisPredictor  # unchanged backend import


# =====================================================================
# 1. PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Iris Flower Classifier | ML Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# 2. EDITABLE CONFIGURATION
#    Update these constants with the real values from your project.
#    Nothing here touches the model or prediction logic.
# =====================================================================
ASSETS_DIR = Path(__file__).parent / "images"

MODEL_NAME = "SVM Classifier"  # TODO: replace with your actual model name
MODEL_ACCURACY = 0.9667            # TODO: replace with your real test-set accuracy
BEST_ALGORITHM = "SVM"    # TODO: confirm against your model comparison results

DATASET_STATS = {
    "Total Samples": "150",
    "Features": "4",
    "Classes": "3",
    "Samples / Class": "50",
    "Missing Values": "0",
    "Source": "UCI ML Repository (Fisher, 1936)",
}

FEATURE_DESCRIPTIONS = {
    "Sepal Length (cm)": "Length of the outer protective leaf beneath the petals.",
    "Sepal Width (cm)": "Width of that outer protective leaf.",
    "Petal Length (cm)": "Length of the inner, colored flower leaf.",
    "Petal Width (cm)": "Width of the inner, colored flower leaf.",
}

SPECIES_INFO = {
    "Iris-setosa": {
        "emoji": "🌼",
        "color": "#6C63FF",
        "description": "A small, hardy iris with short, rounded petals.",
        "sepal_length": "4.3 - 5.8 cm",
        "sepal_width": "2.3 - 4.4 cm",
        "petal_length": "1.0 - 1.9 cm",
        "petal_width": "0.1 - 0.6 cm",
    },
    "Iris-versicolor": {
        "emoji": "🌿",
        "color": "#00BFA6",
        "description": "A medium-sized iris with moderately proportioned petals.",
        "sepal_length": "4.9 - 7.0 cm",
        "sepal_width": "2.0 - 3.4 cm",
        "petal_length": "3.0 - 5.1 cm",
        "petal_width": "1.0 - 1.8 cm",
    },
    "Iris-virginica": {
        "emoji": "🌺",
        "color": "#FF6B81",
        "description": "The largest of the three species, with long, elegant petals.",
        "sepal_length": "4.9 - 7.9 cm",
        "sepal_width": "2.2 - 3.8 cm",
        "petal_length": "4.5 - 6.9 cm",
        "petal_width": "1.4 - 2.5 cm",
    },
}

# Example evaluation metrics — replace with your actual classification_report
# output once you have it (these are placeholders so the tab isn't empty).
CLASSIFICATION_METRICS = {
    "Iris-setosa":     {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 15},
    "Iris-versicolor": {"precision": 0.94, "recall": 0.94, "f1": 0.94, "support": 16},
    "Iris-virginica":  {"precision": 0.94, "recall": 0.94, "f1": 0.94, "support": 14},
}


# =====================================================================
# 3. CUSTOM CSS — fonts, gradients, glass cards, hover effects
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.main {
    background-color: #F5F7FB;
}

/* ---------- Hero header ---------- */
.hero-banner {
    background: linear-gradient(135deg, #6C63FF 0%, #4CAF50 100%);
    padding: 2.2rem 2rem;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(108, 99, 255, 0.25);
}
.hero-banner h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    color: white;
}
.hero-banner p {
    font-size: 1.05rem;
    opacity: 0.92;
    margin: 0;
}

/* ---------- Generic glass / metric card ---------- */
.metric-card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(108, 99, 255, 0.12);
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 24px rgba(108, 99, 255, 0.18);
}
.metric-card .value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #6C63FF;
}
.metric-card .label {
    font-size: 0.85rem;
    color: #555;
    margin-top: 0.2rem;
}

/* ---------- Prediction result card ---------- */
.prediction-card {
    background: linear-gradient(135deg, #E8F5E9 0%, #E3E0FF 100%);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    border: 1px solid rgba(108, 99, 255, 0.15);
}
.prediction-card .species-emoji {
    font-size: 2.6rem;
}
.prediction-card .species-name {
    font-size: 1.7rem;
    font-weight: 700;
    color: #2E7D32;
    margin-top: 0.3rem;
}
.prediction-card .species-sub {
    font-size: 0.95rem;
    color: #444;
    margin-top: 0.2rem;
}

/* ---------- Species info card ---------- */
.species-card {
    border-radius: 14px;
    padding: 1.2rem;
    background: white;
    border-left: 6px solid var(--accent, #6C63FF);
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.species-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 26px rgba(0,0,0,0.12);
}
.species-card h4 {
    margin-bottom: 0.3rem;
}
.species-card .trait-row {
    font-size: 0.85rem;
    color: #555;
    margin: 0.15rem 0;
}

/* ---------- Probability bar ---------- */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.55rem;
}
.prob-label {
    width: 130px;
    font-size: 0.88rem;
    font-weight: 600;
}
.prob-track {
    flex: 1;
    background: #ECECF4;
    border-radius: 8px;
    height: 14px;
    margin: 0 0.6rem;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 8px;
}
.prob-pct {
    width: 48px;
    font-size: 0.85rem;
    text-align: right;
    font-weight: 600;
}

/* ---------- Footer ---------- */
.app-footer {
    text-align: center;
    color: #888;
    font-size: 0.85rem;
    padding: 1.2rem 0 0.4rem 0;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 4. HELPER FUNCTIONS
# =====================================================================
def get_prediction_and_confidence(predictor: IrisPredictor, features: tuple):
    """
    Calls the existing, unmodified prediction method and — only if the
    predictor happens to expose a probability method — also retrieves
    per-class confidence. If no such method exists, probabilities is
    simply None and the UI degrades gracefully. This never alters or
    assumes anything about src/predict.py's internals beyond what is
    already there.
    """
    species = predictor.predict_species(*features)

    probabilities = None
    for method_name in ("predict_proba", "get_probabilities", "predict_species_proba"):
        if hasattr(predictor, method_name):
            try:
                probabilities = getattr(predictor, method_name)(*features)
            except Exception:
                probabilities = None
            break

    return species, probabilities


def render_metric_card(value: str, label: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>
    """


def render_species_card(name: str, info: dict) -> str:
    return f"""
    <div class="species-card" style="--accent: {info['color']};">
        <h4>{info['emoji']} {name}</h4>
        <p style="color:#555; font-size:0.9rem;">{info['description']}</p>
        <div class="trait-row">🌱 Sepal length: <b>{info['sepal_length']}</b></div>
        <div class="trait-row">🌱 Sepal width: <b>{info['sepal_width']}</b></div>
        <div class="trait-row">🌸 Petal length: <b>{info['petal_length']}</b></div>
        <div class="trait-row">🌸 Petal width: <b>{info['petal_width']}</b></div>
    </div>
    """


def render_probability_bar(label: str, pct: float, color: str) -> str:
    return f"""
    <div class="prob-row">
        <div class="prob-label">{label}</div>
        <div class="prob-track">
            <div class="prob-fill" style="width:{pct}%; background:{color};"></div>
        </div>
        <div class="prob-pct">{pct:.1f}%</div>
    </div>
    """


def image_or_placeholder(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"📎 Add `{path.name}` to the `assets/` folder to display: **{caption}**")


# =====================================================================
# 5. SIDEBAR — project overview, model info, dataset info, tech stack
# =====================================================================
with st.sidebar:
    st.markdown("## 🌸 Iris Classifier")
    st.caption("Machine Learning Dashboard")

    st.markdown("---")
    with st.expander("📖 Project Overview", expanded=True):
        st.write(
            "This app predicts the species of an Iris flower — "
            "*setosa*, *versicolor*, or *virginica* — from four "
            "petal and sepal measurements, using a trained "
            "Scikit-Learn model."
        )

    with st.expander("🤖 Model Information"):
        st.markdown(f"- **Algorithm:** {MODEL_NAME}")
        st.markdown(f"- **Accuracy:** {MODEL_ACCURACY * 100:.2f}%")
        st.markdown("- **Framework:** Scikit-Learn")
        st.markdown("- **Task:** Multi-class classification")

    with st.expander("📊 Dataset Information"):
        for k, v in DATASET_STATS.items():
            st.markdown(f"- **{k}:** {v}")

    with st.expander("🛠️ Technologies Used"):
        st.markdown(
            "- Python\n"
            "- Streamlit\n"
            "- Scikit-Learn\n"
            "- Pandas\n"
        )

    st.markdown("---")
    st.caption("Developed by Keerthan • CodeAlpha Internship Project")


# =====================================================================
# 6. HERO HEADER
# =====================================================================
st.markdown("""
<div class="hero-banner">
    <h1>🌸 Iris Flower Species Prediction</h1>
    <p>Enter flower measurements and get an instant, model-backed species prediction.</p>
</div>
""", unsafe_allow_html=True)

# Top-line metric cards
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(render_metric_card(f"{MODEL_ACCURACY * 100:.1f}%", "Model Accuracy"), unsafe_allow_html=True)
with m2:
    st.markdown(render_metric_card(BEST_ALGORITHM, "Best Algorithm"), unsafe_allow_html=True)
with m3:
    st.markdown(render_metric_card(DATASET_STATS["Total Samples"], "Training Samples"), unsafe_allow_html=True)
with m4:
    st.markdown(render_metric_card(DATASET_STATS["Classes"], "Species Classes"), unsafe_allow_html=True)

st.write("")


# =====================================================================
# 7. INITIALIZE PREDICTOR (unchanged)
# =====================================================================
predictor = IrisPredictor()


# =====================================================================
# 8. TABS
# =====================================================================
tab_predict, tab_dataset, tab_performance, tab_project = st.tabs(
    ["🔮 Prediction", "📊 About Dataset", "📈 Model Performance", "ℹ️ Project Info"]
)


# ---------------------------------------------------------------------
# TAB 1 — PREDICTION
# ---------------------------------------------------------------------
with tab_predict:
    st.subheader("📏 Enter Flower Measurements")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("**🌱 Sepal Measurements**")
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)

    with col2:
        with st.container(border=True):
            st.markdown("**🌸 Petal Measurements**")
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
            petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    st.write("")
    predict_clicked = st.button("🔍 Predict Species", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Analyzing measurements..."):
            species, probabilities = get_prediction_and_confidence(
                predictor,
                (sepal_length, sepal_width, petal_length, petal_width),
            )
            st.session_state["prediction_result"] = species
            st.session_state["prediction_probs"] = probabilities
        st.success("✅ Prediction complete!")

    # Persisted result (survives tab switches / minor reruns)
    if "prediction_result" in st.session_state:
        species = st.session_state["prediction_result"]
        probabilities = st.session_state.get("prediction_probs")
        info = SPECIES_INFO.get(species, {})

        st.write("")
        result_col, info_col = st.columns([1, 1])

        with result_col:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="species-emoji">{info.get('emoji', '🌸')}</div>
                <div class="species-name">{species}</div>
                <div class="species-sub">Predicted Species</div>
            </div>
            """, unsafe_allow_html=True)

            if probabilities is not None:
                try:
                    # Support either a dict {label: prob} or a plain list/array
                    if isinstance(probabilities, dict):
                        prob_items = list(probabilities.items())
                    else:
                        labels = list(SPECIES_INFO.keys())
                        prob_items = list(zip(labels, probabilities))

                    top_label, top_prob = max(prob_items, key=lambda x: x[1])
                    st.write("")
                    st.markdown(f"**Confidence: {top_prob * 100:.1f}%**")
                    st.progress(min(max(top_prob, 0.0), 1.0))
                except Exception:
                    probabilities = None

        with info_col:
            if info:
                st.markdown(render_species_card(species, info), unsafe_allow_html=True)

        # Full probability breakdown across all species
        if probabilities is not None:
            st.write("")
            st.markdown("**📊 Probability by Species**")
            bars_html = ""
            for label, prob in prob_items:
                color = SPECIES_INFO.get(label, {}).get("color", "#6C63FF")
                bars_html += render_probability_bar(label, prob * 100, color)
            st.markdown(bars_html, unsafe_allow_html=True)

            prob_df = pd.DataFrame(prob_items, columns=["Species", "Probability"])
            st.bar_chart(prob_df.set_index("Species"))
        else:
            st.info(
                "ℹ️ Per-species confidence isn't shown because `IrisPredictor` "
                "doesn't currently expose a probability method (e.g. `predict_proba`). "
                "Add one to `src/predict.py` to unlock this section — no other "
                "changes needed here."
            )


# ---------------------------------------------------------------------
# TAB 2 — ABOUT DATASET
# ---------------------------------------------------------------------
with tab_dataset:
    st.subheader("📊 About the Iris Dataset")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(render_metric_card(DATASET_STATS["Total Samples"], "Total Samples"), unsafe_allow_html=True)
    with d2:
        st.markdown(render_metric_card(DATASET_STATS["Features"], "Features"), unsafe_allow_html=True)
    with d3:
        st.markdown(render_metric_card(DATASET_STATS["Classes"], "Classes"), unsafe_allow_html=True)

    st.write("")
    with st.expander("📌 Dataset Details", expanded=True):
        for k, v in DATASET_STATS.items():
            st.markdown(f"- **{k}:** {v}")

    st.write("")
    st.markdown("**🧬 Feature Descriptions**")
    feat_df = pd.DataFrame(
        [{"Feature": k, "Description": v} for k, v in FEATURE_DESCRIPTIONS.items()]
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.write("")
    st.markdown("**🌸 Species Overview**")
    s1, s2, s3 = st.columns(3)
    for col, (name, info) in zip((s1, s2, s3), SPECIES_INFO.items()):
        with col:
            st.markdown(render_species_card(name, info), unsafe_allow_html=True)


# ---------------------------------------------------------------------
# TAB 3 — MODEL PERFORMANCE
# ---------------------------------------------------------------------
with tab_performance:
    st.subheader("📈 Model Performance")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(render_metric_card(f"{MODEL_ACCURACY * 100:.2f}%", "Test Accuracy"), unsafe_allow_html=True)
    with p2:
        st.markdown(render_metric_card(BEST_ALGORITHM, "Best Model"), unsafe_allow_html=True)
    with p3:
        st.markdown(render_metric_card(MODEL_NAME, "Deployed Model"), unsafe_allow_html=True)

    st.write("")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        image_or_placeholder(ASSETS_DIR / "accuracy_comparison.png", "Accuracy Comparison")
    with img_col2:
        image_or_placeholder(ASSETS_DIR / "confusion_matrix.png", "Confusion Matrix")

    st.write("")
    st.markdown("**🧾 Classification Metrics**")
    metrics_df = pd.DataFrame(CLASSIFICATION_METRICS).T
    metrics_df.index.name = "Species"
    metrics_df = metrics_df.rename(
        columns={"precision": "Precision", "recall": "Recall", "f1": "F1-Score", "support": "Support"}
    )
    st.dataframe(
        metrics_df.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"}),
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# TAB 4 — PROJECT INFO
# ---------------------------------------------------------------------
with tab_project:
    st.subheader("ℹ️ Project Information")

    with st.expander("👤 Developer", expanded=True):
        st.markdown("- **Name:** Keerthan")
        st.markdown("- **Project:** CodeAlpha Internship — Iris Flower Classification")

    with st.expander("🛠️ Technologies Used"):
        st.markdown(
            "- Python\n"
            "- Streamlit (UI)\n"
            "- Scikit-Learn (ML model)\n"
            "- Pandas (data handling)\n"
        )

    with st.expander("🎯 Project Goals"):
        st.markdown(
            "- Build an end-to-end ML classification pipeline\n"
            "- Deploy an interactive prediction interface\n"
            "- Present model performance clearly to non-technical viewers\n"
        )

    st.success("Thanks for exploring this project! Switch to the **Prediction** tab to try it out.")


# =====================================================================
# 9. FOOTER
# =====================================================================
st.markdown("""
<div class="app-footer">
    🌸 Developed using Streamlit • Scikit-Learn &nbsp;|&nbsp; CodeAlpha Internship Project &nbsp;|&nbsp; by Keerthan
</div>
""", unsafe_allow_html=True)