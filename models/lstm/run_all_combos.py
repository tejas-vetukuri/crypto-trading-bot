from __future__ import annotations

import traceback

from models.lstm.lstm import train_lstm_model, get_default_start_date


COMBINATIONS = [
    ("BTCUSDT", "5m"),
    ("BTCUSDT", "15m"),
    ("BTCUSDT", "1h"),
    ("BTCUSDT", "4h"),
    ("ETHUSDT", "5m"),
    ("ETHUSDT", "15m"),
    ("ETHUSDT", "1h"),
    ("ETHUSDT", "4h"),
]


def main():
    print("\n================ RUN ALL LSTM COMBINATIONS ================\n")

    successes: list[tuple[str, str]] = []
    failures: list[tuple[str, str, str]] = []

    for idx, (symbol, resolution) in enumerate(COMBINATIONS, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(COMBINATIONS)}] TRAINING {symbol} | {resolution}")
        print("=" * 80)

        try:
            start_date = get_default_start_date(resolution)

            train_lstm_model(
                symbol=symbol,
                resolution=resolution,
                start_date=start_date,
                end_date=None,
                x_window_size=100,
                epochs=10,
                batch_size=64,
                thresholds=(0.50, 0.52, 0.55, 0.60),
            )

            successes.append((symbol, resolution))
            print(f"\n✅ SUCCESS: {symbol} | {resolution}")

        except Exception as e:
            failures.append((symbol, resolution, str(e)))
            print(f"\n❌ FAILED: {symbol} | {resolution}")
            print(f"Reason: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n✅ Successful runs: {len(successes)}")
    for symbol, resolution in successes:
        print(f"  - {symbol} | {resolution}")

    print(f"\n❌ Failed runs: {len(failures)}")
    for symbol, resolution, err in failures:
        print(f"  - {symbol} | {resolution} -> {err}")

    print("\nDone.")


if __name__ == "__main__":
    main()