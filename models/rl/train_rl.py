#train_rl.py

from models.rl.rl_ensemble import train_rl_policy, RiskConfig


if __name__ == "__main__":
    train_rl_policy(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2017-06-01",
        end_date=None,
        train_ratio=0.80,
        xgb_artifacts_path="models/xgboost/xgb_trend_artifacts.joblib",
        lstm_artifacts_path="models/lstm/lstm_artifacts.joblib",
        lstm_threshold=0.52,
        agent_out_path="models/rl/rl_qtable_agent.joblib",
        alpha=0.20,
        gamma=0.95,
        eps=0.25,
        episodes=60,
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
    print("✅ RL filter training complete.")