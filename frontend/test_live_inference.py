from __future__ import annotations

from pprint import pprint

from services.predictions import fetch_live_predictions


def main():
    symbol = "BTCUSDT"
    interval = "1h"

    result = fetch_live_predictions(symbol=symbol, interval=interval)

    print("\n================ LIVE INFERENCE TEST ================")
    print(f"Symbol:              {result['symbol']}")
    print(f"Interval:            {result['interval']}")
    print(f"Timestamp:           {result['timestamp']}")
    print(f"Entry / Close:       {result['close_price']}")

    print("\n--- Model Outputs ---")
    print(f"XGB Label:           {result['xgb_label']}")
    print(f"XGB Prob Up:         {result['xgb_prob_up']:.4f}")
    print(f"LSTM Label:          {result['lstm_label']}")
    print(f"LSTM Prob Up:        {result['lstm_prob_up']:.4f}")

    print("\n--- Ensemble ---")
    print(f"Ensemble Direction:  {result['ensemble_direction']}")
    print(f"Ensemble Prob Up:    {result['ensemble_prob_up']:.4f}")
    print(f"Ensemble Confidence: {result['ensemble_confidence']:.4f}")

    print("\n--- RL Filter ---")
    print(f"RL Used:             {result['rl_used']}")
    print(f"RL Decision:         {result['rl_decision']}")
    print(f"RL Reason:           {result['rl_reason']}")
    print(f"RL State:            {result['rl_state']}")
    print(f"Q Skip:              {result['q_skip']}")
    print(f"Q Take:              {result['q_take']}")
    print(f"Q Gap:               {result['q_gap_take_minus_skip']}")
    print(f"Take Visits:         {result['take_visits']}")
    print(f"Skip Visits:         {result['skip_visits']}")

    print("\n--- Final Signal ---")
    print(f"Final Action:        {result['final_action']}")
    print(f"Confidence:          {result['confidence']:.4f}")

    print("\n--- Risk ---")
    print(f"ATR:                 {result['atr']}")
    print(f"ATR %:               {result['atr_pct']}")
    print(f"Stop Loss:           {result['stop_loss']}")
    print(f"Take Profit:         {result['take_profit']}")
    print(f"Position Size USD:   {result['position_size_usd']}")
    print(f"Leverage:            {result['leverage']}")
    print("=====================================================\n")

    print("Raw result:")
    pprint(result)


if __name__ == "__main__":
    main()