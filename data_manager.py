import pandas as pd
import numpy as np
import yfinance as yf

def get_data(tickers, period='3y'):
    stock_dict = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

        if df.empty:
            print(f"Warning: no data for {ticker}, skipping.")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        stock_dict[ticker] = df[['Close']].dropna()

    return stock_dict
