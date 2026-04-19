from models.rl.rl_ensemble import train_rl_policy, RiskConfig

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["5m", "15m"]

if __name__ == "__main__":
    print("\n========== RL BATCH TRAIN ==========\n")

    for symbol in SYMBOLS:
        for resolution in TIMEFRAMES:
            print(f"\n----- Training RL for {symbol} {resolution} -----\n")

            train_rl_policy(
                symbol=symbol,
                resolution=resolution,
                end_date=None,
                train_ratio=0.80,
                lstm_threshold=0.53,   # ← your change kept
                alpha=0.20,
                gamma=0.95,
                eps=0.25,
                episodes=20,
                risk=RiskConfig(
                    capital_usd=5000.0,
                    risk_per_trade=0.02,
                    rr=1.25,
                    leverage=25.0,
                    fee_bps=2.0,
                    trade_penalty_bps=2.0,
                    sl_atr_mult=1.0,
                    min_atr_pct=0.001,
                ),
                ensemble_weight_xgb=0.8,
                ensemble_weight_lstm=0.2,
                ensemble_upper=0.60,
                ensemble_lower=0.40,
                max_horizon=3,
                skip_reward_scale=0.15,
            )

    print("\n========== RL BATCH TRAIN COMPLETE ==========\n")