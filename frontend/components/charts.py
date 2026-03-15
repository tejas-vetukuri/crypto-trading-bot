import plotly.graph_objects as go


def make_candlestick_chart(df, symbol: str, interval: str):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["open_time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=f"{symbol} {interval}",
            )
        ]
    )

    fig.update_layout(
        title=f"{symbol} Candlestick Chart ({interval})",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig