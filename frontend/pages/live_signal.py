import streamlit as st

from datetime import datetime, timezone

from services.market_data import fetch_klines
from services.predictions import fetch_live_predictions
from services.sentiment import fetch_sentiment_snapshot
from services.trade_history import fetch_recent_rl_replay_trades
from components.charts import make_candlestick_chart


def get_signal_color(signal):
    signal = str(signal).upper().strip()

    if signal in ["UP", "LONG", "TRADE", "WIN", "TP", "BULLISH", "SLIGHTLY BULLISH"]:
        return "#22c55e"
    if signal in ["DOWN", "SHORT", "LOSS", "SL", "BEARISH", "SLIGHTLY BEARISH"]:
        return "#ef4444"
    if signal in ["SIDEWAYS", "DON'T TRADE", "DONT TRADE", "HOLD", "BREAKEVEN", "NEUTRAL", "MIXED"]:
        return "#facc15"
    return "#e5e7eb"


def to_direction_label(signal):
    signal = str(signal).upper().strip()

    if signal in ["LONG", "UP", "BULLISH"]:
        return "UP"
    if signal in ["SHORT", "DOWN", "BEARISH"]:
        return "DOWN"
    if signal in ["SIDEWAYS", "HOLD", "NEUTRAL", "MIXED", "DON'T TRADE", "DONT TRADE"]:
        return "SIDEWAYS"
    return signal


def to_rl_label(signal):
    signal = str(signal).upper().strip()

    if signal in ["LONG", "UP"]:
        return "LONG"
    if signal in ["SHORT", "DOWN"]:
        return "SHORT"
    if signal in ["HOLD", "SIDEWAYS", "NEUTRAL", "MIXED", "DON'T TRADE", "DONT TRADE"]:
        return "HOLD"
    return signal


def prediction_box(title, prediction, prob=None, extra=None):
    normalized_prediction = str(prediction).upper().strip()
    show_dash_for_conf = normalized_prediction in ["SIDEWAYS", "HOLD", "NEUTRAL", "MIXED"]

    if prob is None or show_dash_for_conf:
        prob_text = "—"
    else:
        prob_text = f"{prob:.2%}"

    color = get_signal_color(prediction)

    st.markdown(
        f"""
        <div style="
            border: 1px solid #2e2e2e;
            border-radius: 14px;
            padding: 16px;
            background-color: #111827;
            min-height: 150px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        ">
            <div style="font-size: 0.95rem; color: #9ca3af; margin-bottom: 8px;">
                {title}
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {color}; margin-bottom: 8px;">
                {prediction}
            </div>
            <div style="font-size: 0.95rem; color: #d1d5db; margin-bottom: 6px;">
                <b>Confidence:</b> {prob_text}
            </div>
            <div style="font-size: 0.9rem; color: #cbd5e1;">
                {extra if extra else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_trade_card_html(trade):
    direction = trade.get("direction", "—")
    outcome = trade.get("outcome", "—")
    time_text = trade.get("time", "—")
    r_value = float(trade.get("r", 0.0))
    pnl_value = float(trade.get("pnl", 0.0))

    direction_color = get_signal_color(direction)
    outcome_color = get_signal_color(outcome)
    r_color = "#22c55e" if r_value > 0 else "#ef4444" if r_value < 0 else "#facc15"
    pnl_color = "#22c55e" if pnl_value > 0 else "#ef4444" if pnl_value < 0 else "#facc15"

    return (
        f'<div style="flex:0 0 135px; border:1px solid #2e2e2e; border-radius:12px; '
        f'padding:10px 12px; background-color:#111827; min-height:108px; box-sizing:border-box;">'
        f'<div style="font-size:0.72rem; color:#9ca3af; margin-bottom:6px; line-height:1.2;">{time_text}</div>'
        f'<div style="font-size:1.0rem; font-weight:700; color:{direction_color}; line-height:1.2;">{direction}</div>'
        f'<div style="font-size:0.82rem; color:{outcome_color}; margin-top:4px; line-height:1.2;">{outcome}</div>'
        f'<div style="font-size:0.82rem; margin-top:8px; color:{r_color}; font-weight:700; line-height:1.2;">{r_value:+.2f}R</div>'
        f'<div style="font-size:0.82rem; margin-top:3px; color:{pnl_color}; font-weight:700; line-height:1.2;">{pnl_value:+.2f}</div>'
        f'</div>'
    )


def render_trade_scroll_row(trades):
    cards_html = "".join(build_trade_card_html(trade) for trade in trades)

    st.markdown(
        (
            f'<div style="overflow-x:auto; overflow-y:hidden; padding-bottom:8px;">'
            f'<div style="display:flex; flex-wrap:nowrap; gap:10px; min-width:max-content;">'
            f"{cards_html}"
            f"</div>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def render_trade_summary(trades):
    if not trades:
        return

    total = len(trades)
    wins = sum(1 for t in trades if str(t.get("outcome", "")).upper() == "WIN")
    losses = sum(1 for t in trades if str(t.get("outcome", "")).upper() == "LOSS")
    breakevens = sum(1 for t in trades if str(t.get("outcome", "")).upper() == "BREAKEVEN")

    win_pct = (wins / total * 100.0) if total > 0 else 0.0
    net_r = sum(float(t.get("r", 0.0)) for t in trades)
    avg_r = (net_r / total) if total > 0 else 0.0
    total_pnl = sum(float(t.get("pnl", 0.0)) for t in trades)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Trades", total)
    c2.metric("Win %", f"{win_pct:.1f}%")
    c3.metric("Net R", f"{net_r:+.2f}R")
    c4.metric("Avg R", f"{avg_r:+.2f}R")
    c5.metric("PnL", f"{total_pnl:+.2f}")

    st.caption(f"Wins: {wins} | Losses: {losses} | Breakevens: {breakevens}")


def render_lsr_bar(long_ratio: float, short_ratio: float):
    long_pct = max(0.0, min(100.0, long_ratio * 100.0))
    short_pct = max(0.0, min(100.0, short_ratio * 100.0))

    st.markdown(
        f"""
        <div style="
            width:100%;
            height:18px;
            border-radius:999px;
            overflow:hidden;
            display:flex;
            border:1px solid #2e2e2e;
            background-color:#0f172a;
            margin-top:8px;
            margin-bottom:8px;
        ">
            <div style="
                width:{long_pct}%;
                background-color:#22c55e;
                height:100%;
            "></div>
            <div style="
                width:{short_pct}%;
                background-color:#ef4444;
                height:100%;
            "></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("Live Signal")

now_utc = datetime.now(timezone.utc)
st.caption(f"Last updated: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")

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

    st.markdown("## Sentiment")

    try:
        sentiment = fetch_sentiment_snapshot(
            symbol=coin,
            interval=interval,
            closes_for_price_change=df["close"].tail(2).tolist(),
        )

        oi_value = sentiment["open_interest"]
        oi_change_pct = sentiment["oi_change_pct"]
        long_ratio = sentiment["long_ratio"]
        short_ratio = sentiment["short_ratio"]
        lsr_ratio = sentiment["lsr_ratio"]
        price_change_pct = sentiment["price_change_pct"]
        bias = sentiment["sentiment"]
        bias_reason = sentiment["reason"]

        sent_col1, sent_col2, sent_col3, sent_col4 = st.columns(4)

        with sent_col1:
            st.markdown("**Open Interest**")
            st.metric(
                label="Current OI",
                value=f"{oi_value:,.2f}",
                delta=f"{oi_change_pct:+.2f}%"
            )
            st.caption("Current futures OI with latest interval change.")

        with sent_col2:
            st.markdown("**Long / Short Ratio**")
            st.write(f"Longs: **{long_ratio:.2%}**")
            st.write(f"Shorts: **{short_ratio:.2%}**")
            st.write(f"LSR: **{lsr_ratio:.2f}**")

            render_lsr_bar(long_ratio, short_ratio)
            st.caption("Green = long share, red = short share.")

        with sent_col3:
            bias_color = get_signal_color(bias)

            st.markdown("**Bias from OI + LSR + Price**")
            st.markdown(
                f"""
                <div style="
                    font-size:1.25rem;
                    font-weight:700;
                    color:{bias_color};
                    margin-top:4px;
                    margin-bottom:8px;
                ">
                    {bias}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"Price change: **{price_change_pct:+.2f}%**")
            st.caption(bias_reason)

        with sent_col4:
            st.markdown("**Online Sentiment Score**")

            placeholder_score = 64
            placeholder_label = "BULLISH"
            placeholder_reason = "Placeholder social sentiment metric"

            score_color = get_signal_color(placeholder_label)

            st.markdown(
                f"""
                <div style="
                    font-size:1.25rem;
                    font-weight:700;
                    color:{score_color};
                    margin-top:4px;
                    margin-bottom:8px;
                ">
                    {placeholder_score}/100
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(f"Label: **{placeholder_label}**")
            st.caption(placeholder_reason)

    except Exception as e:
        st.warning(f"Sentiment section unavailable: {e}")

    st.markdown("## Model Predictions")

    predictions = fetch_live_predictions(symbol=coin, interval=interval)

    lstm_pred = to_direction_label(predictions["lstm_label"])
    lstm_prob = predictions["lstm_confidence"]

    xgb_pred = to_direction_label(predictions["xgb_label"])
    xgb_prob = predictions["xgb_confidence"]

    ensemble_pred = to_direction_label(predictions["ensemble_direction"])
    ensemble_prob = predictions["ensemble_confidence"]

    final_action = predictions["final_action"]
    rl_policy_score = predictions.get("policy_score")

    if predictions["rl_reason"] == "ensemble_no_setup":
        rl_display = "HOLD"
        rl_extra = "RL skipped because ensemble gave no setup"
    elif predictions["rl_reason"] == "rl_agent_not_loaded":
        rl_display = to_rl_label(ensemble_pred)
        rl_extra = "RL unavailable, using ensemble signal"
    else:
        rl_display = to_rl_label(final_action)
        rl_extra = "RL policy score"

    top_left, top_mid, top_right = st.columns([1, 0.2, 1])

    with top_left:
        prediction_box(
            "1. LSTM",
            lstm_pred,
            lstm_prob,
            extra="Directional prediction from LSTM"
        )

    with top_mid:
        st.markdown(
            "<div style='text-align:center; font-size:2rem; padding-top:50px; color:#9ca3af;'>+</div>",
            unsafe_allow_html=True
        )

    with top_right:
        prediction_box(
            "2. XGBoost",
            xgb_pred,
            xgb_prob,
            extra="Directional prediction from XGBoost"
        )

    st.markdown(
        "<div style='text-align:center; font-size:1.8rem; margin: 8px 0; color:#9ca3af;'>↓</div>",
        unsafe_allow_html=True
    )

    left_space, center_box, right_space = st.columns([1, 1.2, 1])

    with center_box:
        prediction_box(
            "3. LSTM + XGBoost Ensemble",
            ensemble_pred,
            ensemble_prob,
            extra="Final combined direction"
        )

    st.markdown(
        "<div style='text-align:center; font-size:1.8rem; margin: 8px 0; color:#9ca3af;'>↓</div>",
        unsafe_allow_html=True
    )

    left_space2, center_box2, right_space2 = st.columns([1, 1.2, 1])

    with center_box2:
        prediction_box(
            "4. Ensemble + RL Filter",
            rl_display,
            rl_policy_score,
            extra=rl_extra
        )

    st.markdown("## Last 10 Trades Taken by RL")

    try:
        with st.spinner("Loading RL trade replay..."):
            recent_trades = fetch_recent_rl_replay_trades(
                symbol=coin,
                interval=interval,
                limit=10,
            )

        if recent_trades:
            render_trade_scroll_row(recent_trades)
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            render_trade_summary(recent_trades)
        else:
            st.info("No resolved RL trades found.")

    except Exception as e:
        st.warning(f"Could not load RL trade replay: {e}")

except Exception as e:
    st.error(f"Failed to load live signal data: {e}")