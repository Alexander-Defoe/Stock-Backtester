import pandas as pd
import numpy as np

def SMA(df, short_window, long_window):
    df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
    return df

def RSI(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def generate_signals(df):
    df['Position'] = np.where((df['SMA_Short'] > df['SMA_Long']) & (df['RSI'] < 70), 1, 0)
    
    df['Signal'] = df['Position'].diff().shift(1).fillna(0)
    return df