from models.xgboost.xgb import train_xgb_model

SYMBOLS = [
    "ETHUSDT"
]

TIMEFRAMES = [
    "4h"
]

def main():

    print("\n========== XGB BATCH GENERATION ==========\n")

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:

            print(f"\n----- Training XGB {symbol} {tf} -----\n")

            train_xgb_model(
                symbol=symbol,
                resolution=tf,

                # keep defaults
                decision_boundary=0.48,
                margin_threshold=0.05,

                # IMPORTANT → tuned params path
                tuned_artifacts_path="models/xgboost/xgb_tuned_artifacts.joblib",
            )

    print("\n========== DONE ALL XGB COMBINATIONS ==========\n")


if __name__ == "__main__":
    main()