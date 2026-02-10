import streamlit as st
import requests
import pandas as pd

# Dashboard Configuration
st.set_page_config(
    page_title="Startup Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.title("🚀 Startup Growth & Success Predictor")
st.markdown(r"""
    **Intelligence Level:** XGBoost Classifier ($88.10\%$)  
    This tool evaluates startup resilience based on chronological milestones and financial health.
""")
st.divider()

# Input Organization: Main Page Dashboard
st.subheader("📝 Input Startup Metrics")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.info("🕒 **Timeline Basics**")
    age_startup = st.number_input("Startup Age (Years)", 0.0, 50.0, 2.0)
    age_first_ms = st.number_input("Age at First Milestone", 0.0, 50.0, 1.0)
    age_last_ms = st.number_input("Age at Last Milestone", 0.0, 50.0, 1.5)

with col_b:
    st.info("💰 **Financial Health**")
    funding_total = st.number_input("Total Funding (USD)", 0, 100000000, 500000)
    rounds = st.slider("Funding Rounds", 1, 20, 2)
    age_first_funding = st.number_input("Age at First Funding", 0.0, 50.0, 0.5)
    age_last_funding = st.number_input("Age at Last Funding", 0.0, 50.0, 1.8)

with col_c:
    st.info("🤝 **Networking & Scale**")
    milestones = st.slider("Total Milestones Reached", 0, 15, 1)
    tier_rel = st.number_input("Tier Relationships (Level)", 1, 10, 3)
    avg_part = st.number_input("Average Participants", 1.0, 100.0, 3.5)

st.divider()

# Action and Results
if st.button("Generate Survival Analysis"):
    payload = {
        "age_startup_year": age_startup,
        "age_last_milestone_year": age_last_ms,
        "age_first_funding_year": age_first_funding,
        "age_first_milestone_year": age_first_ms,
        "funding_total_usd": funding_total,
        "age_last_funding_year": age_last_funding,
        "tier_relationships": tier_rel,
        "avg_participants": avg_part,
        "milestones": milestones,
        "funding_rounds": rounds
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prob = result["survival_probability"]
            status = result["prediction_status"]
            
            st.subheader("📊 Analysis Results")
            res_col1, res_col2, res_col3 = st.columns([1, 1, 2])
            
            with res_col1:
                st.metric("Prediction Status", status)
                if status == "Growth/Success":
                    st.success("High Growth Potential")
                else:
                    st.error("High Risk Detected")

            with res_col2:
                st.metric("Survival Probability", f"{prob * 100:.1f}%")
                
            with res_col3:
                st.write("**Confidence Score Level**")
                st.progress(prob)
                st.caption(r"Based on XGBoost Baseline Accuracy: $88.10\%$")

            with st.expander("🔍 Why this prediction?"):
                st.write("""
                    Our model weighs **Age of Startup Year** and **Total Funding** as primary indicators.
                    Consistent milestones within the first 2 years significantly increase survival probability.
                """)
        else:
            st.error("API Error. Ensure FastAPI is running on port 8000.")

    except Exception as e:
        st.error(f"Backend Offline: {e}")

st.divider()
feat_col1, feat_col2 = st.columns(2)
with feat_col1:
    st.subheader("📈 Top 10 Prediction Drivers")
    st.write("1. Age Startup Year | 2. Last Milestone | 3. First Funding | 4. First Milestone | 5. Total Funding")
    st.write("6. Last Funding | 7. Tier Relationships | 8. Avg Participants | 9. Milestones | 10. Funding Rounds")
    st.caption("Derived from Random Forest & XGBoost Feature Importance.")

with feat_col2:
    st.subheader("🏆 Model Benchmarking")
    st.markdown(r"""
    * **XGBoost Classifier:** $88.10\%$ (Deployed)
    * **Bagging Classifier:** $87.50\%$
    * **Neural Network:** $84.52\%$
    """)