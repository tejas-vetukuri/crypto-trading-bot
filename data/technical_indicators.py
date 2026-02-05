import pandas as pd

class TechnicalIndicators:

    @staticmethod
    def calculate_ema(
        df: pd.DataFrame,
        span: int,
        col: str = "close"
    ) -> pd.Series:
        return df[col].ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_rsi(
        df: pd.DataFrame,
        period: int = 14,
        col: str = "close"
    ) -> pd.Series:
        delta = df[col].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_atr(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1
        ).max(axis=1)

        atr = tr.ewm(
            alpha=1 / period,
            adjust=False
        ).mean()

        return atr
