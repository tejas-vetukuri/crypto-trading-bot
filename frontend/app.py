# frontend/app.py
import streamlit as st

st.set_page_config(page_title="Trading Dashboard", layout="wide")

# ---- SIDEBAR (SHARED STATE) ----
with st.sidebar:
    coin = st.selectbox(
        "Coin",
        ["BTCUSDT", "ETHUSDT"],
        index=0,   # BTC default
        key="coin"
    )

    interval = st.selectbox(
        "Frequency",
        ["5m", "15m", "1h", "4h"],
        index=2,   # 1h default
        key="interval"
    )

# ---- PAGES ----
live_page = st.Page("pages/live_signal.py", title="Live Signal")
eval_page = st.Page("pages/evaluation.py", title="Evaluation")

pg = st.navigation([live_page, eval_page])
pg.run()