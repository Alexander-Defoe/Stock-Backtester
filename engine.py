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

    prices         = df['Close'].values
    signals        = df['Signal'].values
    active_signals  = df['Active_Signals'].values

    n_features = active_signals.max() if active_signals.max() > 0 else 1

    for i in range(len(prices)):
        price   = prices[i]
        signal  = signals[i]
        n_active = active_signals[i]

        if signal == 1 and cash > 0:
            conviction = min(n_active / n_features, 1.0)
            position_size = max(0.25, conviction)
            capital_to_deploy = cash * position_size
            shares = capital_to_deploy // price
            cash -= shares * price * (1 + transaction_cost)
            peak_price = price

        elif shares > 0:
            peak_price = max(peak_price, price)

            if price < (peak_price * (1 - stop_loss_pct)) or signal == -1:
                cash += shares * price * (1 - transaction_cost)
                shares = 0
                peak_price = 0

        portfolio_values.append(cash + shares * price)

    df['Portfolio_Value'] = portfolio_values
    df['Daily_Return']    = df['Portfolio_Value'].pct_change()
    returns = df['Daily_Return'].dropna()

    sharpe    = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    final_val = portfolio_values[-1]
    total_ret = (final_val - start_capital) / start_capital
    mdd       = calculate_drawdown(portfolio_values)

    return {
        'final_value':  final_val,
        'return_pct':   total_ret,
        'sharpe_ratio': sharpe,
        'max_drawdown': mdd,
        'df':           df
    }