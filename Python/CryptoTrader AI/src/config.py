SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "15m"
DB_PATH = 'db/crypto_data.sqlite'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'
RESULTS_DIR = 'results'
DATA_DIR = "data"
OPTIM_DIR = "optimization"
FEATURES = [
    'EMA_9', 'EMA_21', 'RSI_14', 'MACD_12_26_9', 'ATR_14', 'VWAP', 'OBV',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'BB_upper', 'BB_middle',
    'BB_lower', 'SuperTrend', 'pct_change', 'Stoch_K', 'Stoch_D', 'ADX_14',
    'momentum', 'returns_lag1', 'funding_rate', 'open_interest'
]
TARGET = 'target'
SINCE_DAYS = 1095  # ~3 роки
SEQUENCE_LENGTH = 24
BATCH_SIZE= 64
EPOCHS = 20
PATIENCE = 7
MAX_THREADS = 4