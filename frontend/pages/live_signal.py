import streamlit as st

from services.market_data import fetch_klines
from components.charts import make_candlestick_chart


st.title("Live Signal")

coin = st.session_state.coin
interval = st.session_state.interval

st.subheader(f"{coin} | {interval}")

try:
    df = fetch_klines(symbol=coin, interval=interval, limit=200)

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Close", f"{df['close'].iloc[-1]:,.2f}")
    col2.metric("Last High", f"{df['high'].iloc[-1]:,.2f}")
    col3.metric("Last Low", f"{df['low'].iloc[-1]:,.2f}")

    fig = make_candlestick_chart(df, coin, interval)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show raw candle data"):
        st.dataframe(df.tail(20), use_container_width=True)

except Exception as e:
    st.error(f"Failed to load candle data: {e}")