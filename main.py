from data_manager import get_data
from indicators import SMA, RSI, MACD, BollingerBands, get_ga_features
from engine import backtest_strategy
from ga_engine import GeneticOptimizer
import config

if __name__ == "__main__":
    data_dict = get_data(config.TICKERS, period=config.PERIOD)

    for ticker, df in data_dict.items():
        df = SMA(df, config.SMA_SHORT, config.SMA_LONG)
        df = RSI(df, config.RSI_WINDOW)
        df = MACD(df)
        df = BollingerBands(df)
        ga_df = get_ga_features(df)

        feature_cols = [col for col in ga_df.columns if col not in ['Close', 'Target']]
        dataset = ga_df[feature_cols + ['Target']].values.tolist()
        n = len(dataset)

        print(f"\n{'='*50}")
        print(f"  Walk-Forward Results: {ticker}")
        print(f"{'='*50}")

        window_results = []

        for w in range(config.N_WINDOWS):
            train_start = int(n * (w * config.STEP_PCT))
            train_end   = int(n * (w * config.STEP_PCT + config.TRAIN_PCT))
            test_end    = int(n * (w * config.STEP_PCT + config.TRAIN_PCT + config.STEP_PCT))

            if test_end > n:
                break

            train_data = dataset[train_start:train_end]
            test_idx   = list(range(train_end, test_end))

            print(f"\n  Window {w+1}: Train [{train_start}:{train_end}] → Test [{train_end}:{test_end}]")
            print(f"  Evolving strategy...")

            ga = GeneticOptimizer(
                pop_size=config.GA_POP_SIZE,
                generations=config.GA_GENERATIONS,
                mut_rate=config.GA_MUT_RATE,
                elite_percent=config.GA_ELITE_PCT,
                seed=config.GA_SEED
            )
            best_rule = ga.evolve(train_data)

            genes     = best_rule[:-1]
            threshold = best_rule[-1]

            active_signals = (ga_df[feature_cols].values * genes).sum(axis=1)
            ga_df['Active_Signals']  = active_signals                              # NEW
            ga_df['Final_Position']  = (active_signals >= threshold).astype(int)
            ga_df['Signal']          = ga_df['Final_Position'].diff().shift(1).fillna(0)
            test_df  = ga_df.iloc[test_idx].copy()
            results  = backtest_strategy(test_df, start_capital=config.CAPITAL, stop_loss_pct=config.STOP_LOSS_PCT)

            start_price      = test_df['Close'].iloc[0]
            end_price        = test_df['Close'].iloc[-1]
            benchmark_return = (end_price - start_price) / start_price

            print(f"  GA Return:      {results['return_pct']:.2%}")
            print(f"  Buy & Hold:     {benchmark_return:.2%}")
            print(f"  Sharpe:         {results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:   {results['max_drawdown']:.2%}")

            window_results.append({
                'return':    results['return_pct'],
                'benchmark': benchmark_return,
                'sharpe':    results['sharpe_ratio'],
                'drawdown':  results['max_drawdown'],
            })

        if window_results:
            avg_ret = sum(r['return']    for r in window_results) / len(window_results)
            avg_bh  = sum(r['benchmark'] for r in window_results) / len(window_results)
            avg_sh  = sum(r['sharpe']    for r in window_results) / len(window_results)
            avg_dd  = sum(r['drawdown']  for r in window_results) / len(window_results)
            wins    = sum(1 for r in window_results if r['return'] > r['benchmark'])

            print(f"\n  {'─'*40}")
            print(f"  AVERAGE ACROSS {len(window_results)} WINDOWS")
            print(f"  {'─'*40}")
            print(f"  Avg GA Return:    {avg_ret:.2%}")
            print(f"  Avg Buy & Hold:   {avg_bh:.2%}")
            print(f"  Avg Sharpe:       {avg_sh:.2f}")
            print(f"  Avg Max Drawdown: {avg_dd:.2%}")
            print(f"  Windows beaten:   {wins}/{len(window_results)}")