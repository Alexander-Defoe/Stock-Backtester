# ── Data ──────────────────────────────────────────
TICKERS = ['AAPL', 'MSFT', 'NVDA']
PERIOD  = '3y'

# ── Capital ───────────────────────────────────────
CAPITAL       = 100000
STOP_LOSS_PCT = 0.05

# ── Indicators ────────────────────────────────────
SMA_SHORT    = 20
SMA_LONG     = 50
RSI_WINDOW   = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
BB_WINDOW    = 20
BB_STD       = 2

# ── Walk-Forward ──────────────────────────────────
N_WINDOWS  = 4
TRAIN_PCT  = 0.6
STEP_PCT   = 0.1

# ── Genetic Algorithm ─────────────────────────────
GA_POP_SIZE    = 100
GA_GENERATIONS = 150
GA_MUT_RATE    = 0.15
GA_ELITE_PCT   = 0.05
GA_SEED        = 42

# ── Position Sizing ───────────────────────────────
POSITION_FLOOR = 0.25