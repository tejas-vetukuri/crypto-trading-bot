from data.delta_exchange import DeltaDataClient

client = DeltaDataClient()

df = client.get_candles("BTCUSD", "5m", limit=8000)
print(len(df))
print(df.head(), df.tail())