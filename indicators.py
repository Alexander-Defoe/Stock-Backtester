import pandas as pd
import numpy as np

def SMA(df, short_window, long_window):
    df = df.copy()
    df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
    return df

def RSI(df, window=14):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(100)
    return df

def MACD(df, fast=12, slow=26, signal=9):
    df = df.copy()
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD_Line']   = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
    return df

def BollingerBands(df, window=20, num_std=2):
    df = df.copy()
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std  = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
    df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
    df['BB_Mid']   = rolling_mean
    return df

def get_ga_features(df):
    df = df.copy()
    df = MACD(df)
    df = BollingerBands(df)

    feat = pd.DataFrame(index=df.index)
    feat['Close'] = df['Close'].values

    # original 3 features
    feat['SMA_Signal']      = (df['SMA_Short'].values > df['SMA_Long'].values).astype(int)
    feat['RSI_Filter']      = (df['RSI'].values < 70).astype(int)
    feat['Price_Above_SMA'] = (df['Close'].values > df['SMA_Short'].values).astype(int)

    # 4 new features
    feat['MACD_Cross']      = (df['MACD_Line'].values > df['MACD_Signal'].values).astype(int)
    feat['BB_Squeeze']      = (df['Close'].values < df['BB_Lower'].values).astype(int)
    feat['RSI_Oversold']    = (df['RSI'].values < 30).astype(int)
    feat['Price_Above_200'] = (df['Close'].values > df['Close'].rolling(200).mean().values).astype(int)

    # target: will price be higher in 5 days?
    feat['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int).values

    return feat.dropna()