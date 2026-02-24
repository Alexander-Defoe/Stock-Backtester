from data_manager import get_data
from indicators import SMA, RSI, generate_signals
from engine import backtest_strategy
import matplotlib.pyplot as plt

LIST_OF_STOCKS = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
CAPITAL = 100000

data_dict = get_data(LIST_OF_STOCKS)

summary_results = {}

for ticker, df in data_dict.items():
    df = SMA(df, short_window=20, long_window=50)
    df = RSI(df, window=14)
    
    df = generate_signals(df)

    results = backtest_strategy(df, start_capital=CAPITAL)
    
    summary_results[ticker] = results['return_pct']
    
    print(f"{ticker} Backtest Complete. Return: {results['return_pct']:.2%}")

ranked = sorted(summary_results.items(), key=lambda x: x[1], reverse=True)

print("\n--- FINAL RANKING ---")
for ticker, ret in ranked:
    print(f"{ticker}: {ret:.2%}")