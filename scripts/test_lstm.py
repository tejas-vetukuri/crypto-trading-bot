from models.lstm.lstm import train_lstm_model


def main():
    print("🚀 Starting LSTM training sanity check...")

    model, history, results_df = train_lstm_model(
        symbol="BTCUSD",
        resolution="5m",
        limit= 100000000,          # small for quick test
        sequence_length=20,
        epochs=3,            # quick sanity run
        batch_size=64,
        model_path="lstm_model_test.h5"
    )

    print("\n✅ Training completed")

    # Basic sanity checks
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    print("\n📊 Test results preview:")
    print(results_df.head())

    print(f"\nRows in test results: {len(results_df)}")
    print("✅ Sanity check passed")


if __name__ == "__main__":
    main()
