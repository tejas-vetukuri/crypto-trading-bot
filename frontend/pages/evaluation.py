import streamlit as st

st.title("Evaluation Page")

st.write("Coin:", st.session_state.coin)
st.write("Frequency:", st.session_state.interval)