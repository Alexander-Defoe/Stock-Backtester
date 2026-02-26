from data_manager import get_data
from indicators import SMA, RSI, get_ga_features
from engine import backtest_strategy
from ga_engine import GeneticOptimizer

LIST_OF_STOCKS = ['AAPL', 'MSFT', 'NVDA'] 
CAPITAL = 100000

data_dict = get_data(LIST_OF_STOCKS)

for ticker, df in data_dict.items():
    df = SMA(df, 20, 50)
    df = RSI(df, 14)
    
    ga_df = get_ga_features(df)
    feature_cols = [col for col in ga_df.columns if col not in ['Close', 'Target']]
    train_cols = feature_cols + ['Target']
    dataset = ga_df[train_cols].values.astype(int).tolist()
    
    split_idx = int(len(dataset) * 0.7)
    train_data = dataset[:split_idx]
    
    print(f"\nEvolving best strategy for {ticker}...")
    ga = GeneticOptimizer(pop_size=100, generations=150)
    best_rule = ga.evolve(train_data)
    
    genes = best_rule[:-1]
    threshold = best_rule[-1]
    active_signals = (ga_df[feature_cols].values * genes).sum(axis=1)
    ga_df['Final_Position'] = (active_signals >= threshold).astype(int)
    ga_df['Signal'] = ga_df['Final_Position'].diff().fillna(0)
    
    test_df = ga_df.iloc[split_idx:].copy()
    results = backtest_strategy(test_df, start_capital=CAPITAL)
    
    start_price = test_df['Close'].iloc[0]
    end_price = test_df['Close'].iloc[-1]
    benchmark_return = (end_price - start_price) / start_price

    print(f"--- {ticker} GA TEST RESULTS (OUT-OF-SAMPLE) ---")
    print(f"Best Rule Found: {best_rule}")
    print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
    print(f"GA Strategy Return: {results['return_pct']:.2%}")
    print(f"Buy & Hold Return: {benchmark_return:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    
    if results['return_pct'] > benchmark_return:
        print("RESULT: GA strategy BEAT the market!")
    else:
        print("RESULT: Buy & Hold was better.")