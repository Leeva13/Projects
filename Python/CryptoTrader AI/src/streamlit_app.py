import streamlit as st
import pandas as pd
import sqlite3
import os
import numpy as np
import torch
import torch.nn.functional as F
import json
from train_model import PriceTransformer, FEATURES, preprocess, DEVICE
from web_socket import start_websocket
from config import SYMBOLS, TIMEFRAMES, DB_PATH, MODEL_DIR, TABLE_TEMPLATE, SEQUENCE_LENGTH
from datetime import datetime
import asyncio
import threading

# Глобальна змінна для свічок
latest_candles = {sym: None for sym in SYMBOLS}
# Глобальний буфер
sequence_buffers = {sym: [] for sym in SYMBOLS}

def candle_callback(candle, symbol):
    global sequence_buffers
    sequence_buffers[symbol].append(candle)
    if len(sequence_buffers[symbol]) > SEQUENCE_LENGTH:
        sequence_buffers[symbol].pop(0)

@st.cache_data
def load_features(sym, tf):
    conn = sqlite3.connect(DB_PATH)
    tbl = TABLE_TEMPLATE.format(symbol=sym.replace('/', '_'), timeframe=tf)
    df = pd.read_sql(f"SELECT * FROM {tbl}", conn, parse_dates=['timestamp'], index_col='timestamp')
    conn.close()
    return df.dropna()

@st.cache_resource
def load_model(sym):
    model_path = os.path.join(MODEL_DIR, f"full_model_{sym.replace('/', '_')}.pth")
    if not os.path.exists(model_path):
        st.error(f"Модель для {sym} не знайдено!")
        return None
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = PriceTransformer(input_size=len(checkpoint['features'])).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['scaler'], checkpoint['features']

# Запуск WebSocket для всіх символів
for sym in SYMBOLS:
    start_websocket(sym.replace('/', '').lower(), lambda candle: candle_callback(candle, sym))

st.title("📈 CryptoTrader AI")
st.write("Система прогнозування криптовалют у реальному часі")

# Вибір пари та таймфрейму
sym = st.selectbox("Пара", SYMBOLS)
tf = st.selectbox("Таймфрейм", TIMEFRAMES)

# Кнопка для оновлення прогнозу
if st.button("Оновити прогноз"):
    if len(sequence_buffers[sym]) == SEQUENCE_LENGTH:
        data = pd.DataFrame(sequence_buffers[sym])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        model, scaler, features = load_model(sym)
        if model is None:
            st.error("Не вдалося завантажити модель!")
        else:
            input_data = preprocess(data, scaler, features)
            if input_data is None:
                st.error("Помилка обробки даних!")
            else:
                input_data = input_data.to(DEVICE)
                with torch.no_grad():
                    output = model(input_data)
                    probabilities = F.softmax(output, dim=1)[0].cpu().numpy()
                    prediction = np.argmax(probabilities)
                    signal = ['ПРОДАЖ 🔴', 'УТРИМАННЯ ⚪', 'ПОКУПКА 🟢'][prediction]
                    confidence = probabilities[prediction]
                    st.subheader("🧠 Прогноз моделі:")
                    st.write(f"**Сигнал**: {signal}")
                    st.write(f"**Впевненість**: {confidence:.2%}")
    else:
        st.warning(f"Очікування достатньої кількості свічок: {len(sequence_buffers[sym])}/{SEQUENCE_LENGTH}")

# Звіт про модель
if st.button("Звіт"):
    for f in ['loss', 'acc', 'cm']:
        img_path = os.path.join(MODEL_DIR, f"{sym.replace('/', '_')}_{f}.png")
        if os.path.exists(img_path):
            st.image(img_path, caption=f"Графік {f}")
        else:
            st.warning(f"Графік {f} для {sym} не знайдено!")

# Сайдбар для додаткових налаштувань
st.sidebar.header("Налаштування")
min_confidence = st.sidebar.slider("Мінімальна впевненість сигналу", 0.0, 1.0, 0.7)
min_volume = st.sidebar.slider("Мінімальний обсяг", 0.0, 1.0, 0.0)

# Логування метрик
st.sidebar.subheader("Метрики моделі")
metrics_path = os.path.join(MODEL_DIR, f"best_metrics_{sym.replace('/', '_')}.json")
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    st.sidebar.write(f"Sharpe Ratio: {metrics.get('sharpe', 0):.4f}")
    st.sidebar.write(f"F1 Score: {metrics.get('f1', 0):.4f}")
else:
    st.sidebar.warning("Метрики не знайдено!")