import pandas as pd
import numpy as np
import yfinance as yf

def get_data(TICKERS, period='3y'):
    stock_dict = {}
    for ticker in TICKERS:
        df = yf.download(ticker, period=period)
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs('Close', axis=1, level=0) 
        
        if ticker in df.columns:
            df = df[[ticker]].rename(columns={ticker: 'Close'})
        else:
            df = df[['Close']]
            
        stock_dict[ticker] = df.dropna()
    return stock_dict
