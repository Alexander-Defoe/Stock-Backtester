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

def get_ga_features(df):
    feat = pd.DataFrame(index=df.index)
    
    feat['Close'] = df['Close']
    
    feat['SMA_Signal'] = (df['SMA_Short'] > df['SMA_Long']).astype(int)
    
    feat['RSI_Filter'] = (df['RSI'] < 70).astype(int)
    
    feat['Price_Above_SMA'] = (df['Close'] > df['SMA_Short']).astype(int)
    
    feat['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    return feat.dropna()

def calculate_sharpe_ratio(df, risk_free_rate=0):
    returns = df['Daily_Return'].dropna()
    
    avg_return = returns.mean()
    volatility = returns.std()
    
    if volatility == 0:
        return 0
    
    sharpe_daily = (avg_return - (risk_free_rate / 252)) / volatility
    
    sharpe_annualized = sharpe_daily * np.sqrt(252)
    
    return sharpe_annualized