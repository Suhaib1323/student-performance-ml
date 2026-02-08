
# AI STUDENT PERFORMANCE INTELLIGENCE SYSTEM 

import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# PAGE CONFIG

st.set_page_config(
    page_title="AI Student Intelligence",
    page_icon="🎓",
    layout="wide"
)


# BACKGROUND UI

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")

model = load_model()


# SIDEBAR INPUTS


st.sidebar.title("🎛 Student Inputs")

age = st.sidebar.slider("Age", 15, 22, 17)
studytime = st.sidebar.slider("Study Time (1–4)", 1, 4, 2)
failures = st.sidebar.slider("Past Failures", 0, 3, 0)
absences = st.sidebar.slider("Absences", 0, 100, 5)
health = st.sidebar.slider("Health (1–5)", 1, 5, 3)
goout = st.sidebar.slider("Going Out (1–5)", 1, 5, 3)


st.title("🚀 AI Student Performance Intelligence Dashboard")
st.markdown("---")


# CREATE INPUT DATAFRAME

input_data = pd.DataFrame([{
    "age": age,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "health": health,
    "goout": goout
}])

prediction = model.predict(input_data)[0]


# FEATURE OVERVIEW + LIVE PREDICTION

col1, col2 = st.columns([2, 1])

# ---------------- FEATURE VISUAL ----------------
with col1:
    st.subheader("📊 Feature Overview")

    feature_dict = {
        "Age": age,
        "Study Time": studytime,
        "Failures": failures,
        "Absences": absences,
        "Health": health,
        "Going Out": goout
    }

    fig = px.bar(
        x=list(feature_dict.keys()),
        y=list(feature_dict.values()),
        color=list(feature_dict.values()),
        color_continuous_scale="Tealgrn"
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- LIVE PREDICTION ----------------
with col2:

    st.subheader("🧠 Live Prediction")

    # Clean metric display
    st.metric("Predicted Final Grade", f"{prediction:.2f}")

    # Stable Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge",
        value=prediction,
        title={
            'text': "Grade Level",
            'font': {'size': 18}
        },
        gauge={
            'axis': {'range': [0, 20]},
            'bar': {'color': "#00FF99"},
            'steps': [
                {'range': [0, 10], 'color': "#ff4b4b"},
                {'range': [10, 15], 'color': "#f9c74f"},
                {'range': [15, 20], 'color': "#43aa8b"}
            ],
        }
    ))

    gauge.update_layout(
        template="plotly_dark",
        height=280,
        width=420,
        margin=dict(l=40, r=40, t=60, b=20)
    )

    st.plotly_chart(gauge)


# WHAT-IF SIMULATION

st.markdown("---")
st.subheader("🔬 What-If Simulation")

sim_study = st.slider("Simulate Increased Study Time", 1, 4, studytime)

sim_data = pd.DataFrame([{
    "age": age,
    "studytime": sim_study,
    "failures": failures,
    "absences": absences,
    "health": health,
    "goout": goout
}])

sim_prediction = model.predict(sim_data)[0]

comparison_df = pd.DataFrame({
    "Scenario": ["Current", "Improved Study"],
    "Predicted Grade": [prediction, sim_prediction]
})

fig_compare = px.bar(
    comparison_df,
    x="Scenario",
    y="Predicted Grade",
    color="Predicted Grade",
    color_continuous_scale="Blues"
)

fig_compare.update_layout(
    template="plotly_dark",
    height=350
)

st.plotly_chart(fig_compare, use_container_width=True)

st.info(f"📈 Improvement Impact: {sim_prediction - prediction:.2f} points change")


# MODEL METRICS

st.markdown("---")
st.subheader("📊 Model Metrics")

m1, m2, m3 = st.columns(3)
m1.metric("R² Score", "0.33")
m2.metric("MAE", "2.96")
m3.metric("RMSE", "3.70")

# FOOTER


st.markdown("---")
st.caption("Developed by Suhaib Shaikh | AI ML Dashboard 2026 🚀")
