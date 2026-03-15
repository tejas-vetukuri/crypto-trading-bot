import streamlit as st

st.title("Model Evaluation Lab")

st.write("Coin:", st.session_state.coin)
st.write("Frequency:", st.session_state.interval)

st.divider()

# ================= GLOBAL SETTINGS =================
with st.expander("Global Evaluation Settings", expanded=True):
    st.write("Global parameters affecting all evaluations will go here.")

st.divider()

# ================= LSTM =================
with st.expander("LSTM Evaluation"):
    st.write("LSTM model evaluation controls and outputs will go here.")

# ================= XGBOOST =================
with st.expander("XGBoost Evaluation"):
    st.write("XGBoost model evaluation controls and outputs will go here.")

# ================= ENSEMBLE =================
with st.expander("LSTM + XGBoost Ensemble Evaluation"):
    st.write("Ensemble evaluation controls and outputs will go here.")

# ================= MAIN MODEL =================
with st.expander("Ensemble + RL Filtering (Main Model)"):
    st.write("Main model evaluation controls and outputs will go here.")

# ================= ALT LSTM =================
with st.expander("LSTM (Alternative Target)"):
    st.write("Alternative LSTM evaluation controls and outputs will go here.")