import pandas as pd
import numpy as np

def calculate_drawdown(portfolio_values):
    pv = pd.Series(portfolio_values)
    running_max = pv.cummax()
    drawdown = (pv - running_max) / running_max
    return drawdown.min()

def backtest_strategy(df, start_capital=100000, stop_loss_pct=0.05):
    cash = start_capital
    shares = 0
    portfolio_values = []
    peak_price = 0 
    transaction_cost = 0.001

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]

        if signal == 1 and cash > 0:
            shares = cash // price
            cash -= shares * price * (1 + transaction_cost)
            peak_price = price 
        
        elif shares > 0:
            peak_price = max(peak_price, price)
            
            if price < (peak_price * (1 - stop_loss_pct)) or signal == -1:
                cash += shares * price * (1 - transaction_cost)
                shares = 0
                peak_price = 0
            
        current_val = cash + (shares * price)
        portfolio_values.append(current_val)

    df['Portfolio_Value'] = portfolio_values
    
    df['Daily_Return'] = df['Portfolio_Value'].pct_change()
    returns = df['Daily_Return'].dropna()
    
    if returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0
        
    final_val = portfolio_values[-1]
    total_return = (final_val - start_capital) / start_capital
    mdd = calculate_drawdown(portfolio_values)
    
    return {
        'final_value': final_val,
        'return_pct': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': mdd,
        'df': df
    }