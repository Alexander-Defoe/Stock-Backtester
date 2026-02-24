import pandas as pd

def backtest_strategy(df, start_capital=100000):
    cash = start_capital
    shares = 0
    portfolio_values = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]

        if signal == 1 and cash > 0:
            shares = cash // price
            cash -= shares * price
        
        elif signal == -1 and shares > 0:
            cash += shares * price
            shares = 0
            
        current_val = cash + (shares * price)
        portfolio_values.append(current_val)

    df['Portfolio_Value'] = portfolio_values
    
    final_val = portfolio_values[-1]
    total_return = (final_val - start_capital) / start_capital
    
    return {
        'final_value': final_val,
        'return_pct': total_return,
        'df': df
    }