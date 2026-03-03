# models/rl/train_rl.py
# Runner kept inside models/rl (as you requested)
from models.rl.rl_ensemble import train_rl_policy, RiskConfig

if __name__ == "__main__":
    train_rl_policy(
        symbol="BTCUSD",
        resolution="1h",
        start_date="2019-06-01",
        end_date=None,
        train_ratio=0.80,
        xgb_artifacts_path="models/xgboost/xgb_trend_artifacts.joblib",
        lstm_artifacts_path="models/lstm/lstm_artifacts.joblib",
        lstm_threshold=0.52,
        agent_out_path="models/rl/rl_qtable_agent.joblib",
        alpha=0.10,
        gamma=0.95,
        eps=0.20,
        episodes=3,
        risk=RiskConfig(capital_usd=5000.0, risk_per_trade=0.02, rr=2.0, leverage=25.0),
    )
    print("✅ RL training complete.")