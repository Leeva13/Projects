import ccxt
import pandas as pd
import sqlite3
import os
import time
import pandas_ta as ta
import numpy as np
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from config import SYMBOLS, SINCE_DAYS, MAX_THREADS, FEATURES, TARGET, SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, PATIENCE, TIMEFRAME, DB_PATH, MODELS_DIR, RESULTS_DIR, DATA_DIR
from concurrent.futures import ThreadPoolExecutor

CRASH_PERIODS = {
    "BTC/USDT": [("2020-03-12", "2020-03-13"), ("2022-06-10", "2022-06-18")],
    "ETH/USDT": [("2022-05-12", "2022-05-15")]
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Завантаження API-ключів з .env
load_dotenv()
token_api = os.getenv("BINANCE_API_KEY")
token_secret_key = os.getenv("BINANCE_SECRET_KEY")

# Ініціалізація біржі
exchange = ccxt.binance({
    'apiKey': token_api,
    'secret': token_secret_key,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

def get_historical_fear_greed():
    cache_file = os.path.join(DATA_DIR, "fear_greed_cache.csv")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)['value']
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=0&format=json&date_format=cn")
        data = response.json()['data']
        fg_df = pd.DataFrame(data)
        fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'])
        fg_df['value'] = fg_df['value'].astype(int)
        fg_df.set_index('timestamp')['value'].to_csv(cache_file)
        return fg_df.set_index('timestamp')['value']
    except Exception as e:
        print(f"[ERROR] Помилка завантаження Fear & Greed Index: {e}")
        return pd.Series()

def fetch_ohlcv(symbol: str, timeframe: str, since_days: int):
    print(f"[INFO] Завантаження {symbol} ({timeframe}) за {since_days} днів...")
    since = exchange.milliseconds() - since_days * 24 * 60 * 60 * 1000
    all_ohlcv = []
    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"[ERROR] Помилка завантаження {symbol}: {e}")
            time.sleep(5)
            continue
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def save_to_csv(df: pd.DataFrame, symbol: str, timeframe: str):
    filename = f"{symbol.replace('/', '')}_{timeframe}.csv"
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path)
    print(f"[OK] Збережено до CSV: {path}")

def save_to_sqlite(df, symbol, timeframe, conn=None):
    table_name = f"features_{symbol.replace('/', '_')}_{timeframe}"
    
    # Створюємо нове з’єднання, якщо conn не передано
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close_conn = True
    else:
        close_conn = False
    
    try:
        # Перевіряємо, чи таблиця існує
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = cursor.fetchone()
        
        # Якщо таблиця існує, додаємо нові дані
        if table_exists:
            existing_df = pd.read_sql(f"SELECT timestamp FROM {table_name}", conn, parse_dates=['timestamp'])
            existing_timestamps = set(existing_df['timestamp'])
            new_rows = df[~df['timestamp'].isin(existing_timestamps)]
            if not new_rows.empty:
                new_rows.to_sql(table_name, conn, if_exists="append", index=True)
                print(f"[OK] Додано {len(new_rows)} нових рядків до {table_name}")
            else:
                print(f"[INFO] Немає нових даних для {table_name}")
        else:
            # Створюємо нову таблицю
            df.to_sql(table_name, conn, if_exists="replace", index=True)
            print(f"[OK] Створено таблицю {table_name} з {len(df)} рядками")
        
        conn.commit()
    except Exception as e:
        print(f"[ERROR] Помилка збереження до {table_name}: {e}")
        raise
    finally:
        if close_conn:
            conn.close()

def add_futures_data(df, symbol):
    df = df.copy()
    max_retries = 3
    retry_delay = 5

    # Кеш для funding rate і open interest
    cache_dir = os.path.join(DATA_DIR, "futures_cache")
    os.makedirs(cache_dir, exist_ok=True)
    funding_cache_file = os.path.join(cache_dir, f"{symbol.replace('/', '_')}_funding.csv")
    oi_cache_file = os.path.join(cache_dir, f"{symbol.replace('/', '_')}_oi.csv")

    # Отримання історії ставок фінансування
    if os.path.exists(funding_cache_file):
        print(f"[INFO] Використовуємо кешовані дані funding_rate для {symbol}")
        funding_rates = pd.read_csv(funding_cache_file, index_col='timestamp', parse_dates=True)
        df['funding_rate'] = df.index.map(
            lambda x: funding_rates['fundingRate'].reindex(df.index, method='nearest').get(x, 0.0)
        )
    else:
        for attempt in range(max_retries):
            try:
                futures_data = exchange.fetch_funding_rate_history(symbol)
                if not futures_data:
                    print(f"[WARNING] Немає даних про funding rate для {symbol}")
                    df['funding_rate'] = 0.0
                    break
                print(f"[DEBUG] Funding rate data for {symbol}: {futures_data[:5]}")
                funding_rates = pd.DataFrame(futures_data)
                timestamp_col = 'timestamp' if 'timestamp' in funding_rates.columns else 'fundingTime'
                if timestamp_col not in funding_rates.columns:
                    print(f"[ERROR] Поле 'timestamp' або 'fundingTime' відсутнє у відповіді для {symbol}")
                    df['funding_rate'] = 0.0
                    break
                funding_rates[timestamp_col] = pd.to_datetime(funding_rates[timestamp_col], unit='ms')
                funding_rates.set_index(timestamp_col, inplace=True)
                funding_rates.to_csv(funding_cache_file)
                df['funding_rate'] = df.index.map(
                    lambda x: funding_rates['fundingRate'].reindex(df.index, method='nearest').get(x, 0.0)
                )
                break
            except Exception as e:
                print(f"[ERROR] Спроба {attempt + 1}/{max_retries} не вдалася для funding_rate {symbol}: {e}")
                if attempt + 1 == max_retries:
                    print(f"[ERROR] Не вдалося отримати funding_rate для {symbol} після {max_retries} спроб")
                    df['funding_rate'] = 0.0
                time.sleep(retry_delay)

    # Отримання історії відкритого інтересу
    if os.path.exists(oi_cache_file):
        print(f"[INFO] Використовуємо кешовані дані open_interest для {symbol}")
        open_interest = pd.read_csv(oi_cache_file, index_col='timestamp', parse_dates=True)
        df['open_interest'] = df.index.map(
            lambda x: open_interest['openInterestAmount'].reindex(df.index, method='nearest').get(x, 0.0)
        )
    else:
        for attempt in range(max_retries):
            try:
                oi_data = exchange.fetch_open_interest_history(symbol)
                if not oi_data:
                    print(f"[WARNING] Немає даних про open interest для {symbol}")
                    df['open_interest'] = 0.0
                    break
                print(f"[DEBUG] Open interest data for {symbol}: {oi_data[:5]}")
                open_interest = pd.DataFrame(oi_data)
                timestamp_col = 'timestamp' if 'timestamp' in open_interest.columns else 'fundingTime'
                if timestamp_col not in open_interest.columns:
                    print(f"[ERROR] Поле 'timestamp' або 'fundingTime' відсутнє у відповіді для {symbol}")
                    df['open_interest'] = 0.0
                    break
                open_interest[timestamp_col] = pd.to_datetime(open_interest[timestamp_col], unit='ms')
                open_interest.set_index(timestamp_col, inplace=True)
                open_interest.to_csv(oi_cache_file)
                df['open_interest'] = df.index.map(
                    lambda x: open_interest['openInterestAmount'].reindex(df.index, method='nearest').get(x, 0.0)
                )
                break
            except Exception as e:
                print(f"[ERROR] Спроба {attempt + 1}/{max_retries} не вдалася для open_interest {symbol}: {e}")
                if attempt + 1 == max_retries:
                    print(f"[ERROR] Не вдалося отримати open_interest для {symbol} після {max_retries} спроб")
                    df['open_interest'] = 0.0
                time.sleep(retry_delay)

    return df

def generate_features(df: pd.DataFrame, symbol: str):
    df = df.copy()
    
    df = add_futures_data(df, symbol)
    
    # Виправлення викликів pandas_ta
    df['EMA_9'] = ta.ema(df['close'], length=9).shift(1)
    df['EMA_21'] = ta.ema(df['close'], length=21).shift(1)
    df['RSI_14'] = ta.rsi(df['close'], length=14).shift(1)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD_12_26_9'] = macd['MACD_12_26_9'].shift(1)
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14).shift(1)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['OBV'] = ta.obv(df['close'], df['volume']).shift(1)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_upper'] = bb['BBU_20_2.0'].shift(1)
    df['BB_middle'] = bb['BBM_20_2.0'].shift(1)
    df['BB_lower'] = bb['BBL_20_2.0'].shift(1)
    supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
    df['SuperTrend'] = supertrend['SUPERT_10_3.0'].shift(1)
    df['pct_change'] = df['close'].pct_change().shift(1)
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df['Stoch_K'] = stoch['STOCHk_14_3_3'].shift(1)
    df['Stoch_D'] = stoch['STOCHd_14_3_3'].shift(1)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX_14'] = adx['ADX_14'].shift(1)
    df['fear_greed'] = df.index.map(lambda x: get_historical_fear_greed().get(x.date(), 50))
    df['momentum'] = ta.mom(df['close'], length=10).shift(1)
    df['returns'] = df['close'].pct_change()
    df['returns_lag1'] = df['returns'].shift(1)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).shift(1)

    # Альтернативний таргет: бінарна класифікація (зростання/падіння)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    print(f"[DEBUG] {symbol} - Статистика returns: min={df['returns'].min():.6f}, max={df['returns'].max():.6f}, mean={df['returns'].mean():.6f}")
    print(f"[DEBUG] {symbol} - Статистика ATR_14: min={df['ATR_14'].min():.2f}, max={df['ATR_14'].max():.2f}, mean={df['ATR_14'].mean():.2f}")
    
    df = df.dropna()
    return df[FEATURES + ['target', 'close', 'log_returns']]

def analyze_feature_correlations(df, features, target, symbol):
    correlations = {}
    for feature in features:
        if feature in df.columns:
            corr = df[feature].corr(df[target], method='spearman')
            correlations[feature] = corr
        else:
            correlations[feature] = 0.0
    
    corr_file = os.path.join(RESULTS_DIR, f"feature_correlations_{symbol.replace('/', '_')}.json")
    with open(corr_file, 'w') as f:
        json.dump(correlations, f, indent=4)
    print(f"[OK] Saved feature correlations for {symbol} to {corr_file}")
    
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"[INFO] Feature correlations with target for {symbol}:")
    for feature, corr in sorted_correlations:
        print(f"  {feature}: {corr:.4f}")
    
    return correlations

def save_features_to_sqlite(df: pd.DataFrame, symbol: str, timeframe: str, conn=None):
    table_name = f"features_{symbol.replace('/', '_')}_{timeframe}"
    
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close_conn = True
    else:
        close_conn = False
    
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=True)
        print(f"[OK] Фічі збережено до SQLite: {table_name}")
    except Exception as e:
        print(f"[ERROR] Помилка збереження фіч до {table_name}: {e}")
        raise
    finally:
        if close_conn:
            conn.close()

def process_symbol(symbol, timeframe, days):
    print(f"[INFO] Завантаження {symbol} ({timeframe}) за {days} днів...")
    exchange = ccxt.binance({
        'enableRateLimit': True
    })
    
    try:
        # Завантаження OHLCV даних
        df = fetch_ohlcv(symbol, timeframe, days)
        if df.empty:
            print(f"[WARNING] Немає даних для {symbol}")
            return pd.DataFrame()
        
        # Генерація фіч
        df = generate_features(df, symbol)
        
        # Аналіз кореляцій
        correlations = analyze_feature_correlations(df, FEATURES, TARGET, symbol)
        
        # Збереження в CSV
        save_to_csv(df, symbol, timeframe)
        
        # Збереження в SQLite
        save_features_to_sqlite(df, symbol, timeframe)
        
        return df
    except Exception as e:
        print(f"[ERROR] Помилка обробки {symbol}: {e}")
        return pd.DataFrame()

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(process_symbol, symbol, TIMEFRAME, SINCE_DAYS): symbol for symbol in SYMBOLS}
        for future in tqdm(futures, desc="Завантаження даних"):
            symbol = futures[future]
            try:
                future.result()
                print(f"[OK] Дані для {symbol} оброблено")
            except Exception as e:
                print(f"[ERROR] Помилка для {symbol}: {e}")

if __name__ == '__main__':
    main()