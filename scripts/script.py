from entsoe import EntsoePandasClient
import pandas as pd

client = EntsoePandasClient(api_key="YOUR_KEY")
prices = client.query_day_ahead_prices(
    "DE_LU",
    start=pd.Timestamp("2023-01-01", tz="Europe/Berlin"),
    end=pd.Timestamp("2024-01-01",   tz="Europe/Berlin"),
)  # returns a pd.Series indexed by UTC timestamp, values in €/MWh
print(prices)