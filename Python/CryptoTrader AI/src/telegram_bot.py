# import ccxt
# import pandas as pd
# import torch
# from telegram.ext import Updater, CommandHandler
# import sqlite3
# from get_data import generate_features

# # Налаштування моделі та біржі
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# exchange = ccxt.binance()
# SYMBOL = 'BTC/USDT'
# model = {'shared': None, 'classifier': None}  # Завантаж моделі тут

# def load_models():
#     model['shared'] = EnhancedLSTM(input_size=30).to(DEVICE)
#     model['classifier'] = PairSpecificClassifier(hidden_size=128).to(DEVICE)
#     checkpoint = torch.load('models/best_BTC_USDT.pth', map_location=DEVICE)
#     model['shared'].load_state_dict(checkpoint['shared_state'])
#     model['classifier'].load_state_dict(checkpoint['classifier_state'])
#     model['shared'].eval()
#     model['classifier'].eval()

# def get_latest_data():
#     ohlcv = exchange.fetch_ohlcv(SYMBOL, '1h', limit=25)
#     df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     return df.set_index('timestamp')

# def predict(update, context):
#     df = get_latest_data()
#     df = generate_features(df)  # Функція з get_data.py
#     window = df.iloc[-24:][['returns_24h', 'EMA_9', 'EMA_21', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
#                             'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'OBV', 'hour', 'day_of_week', 'ADX_14', 'Stoch_K',
#                             'ATR_14', 'Momentum_14', 'CCI_14', 'Stoch_RSI', 'Williams_%R', 'SuperTrend', 'VWAP',
#                             'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'volatility', 'ATR', 'VROC', 'rolling_volatility']]
#     inp = torch.tensor(window.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         features = model['shared'](inp)
#         logits = model['classifier'](features)
#         probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
#         prediction = logits.argmax(dim=1).item()
#     labels = ['Fall', 'Flat', 'Rise']
#     update.message.reply_text(f"Прогноз для {SYMBOL}: {labels[prediction]}\nЙмовірності: {dict(zip(labels, probs))}")

# def start(update, context):
#     update.message.reply_text('Привіт! Я бот для трейдингу криптовалют. Використовуй /predict для прогнозу.')

# if __name__ == '__main__':
#     load_models()
#     updater = Updater('YOUR_TELEGRAM_BOT_TOKEN', use_context=True)
#     dp = updater.dispatcher
#     dp.add_handler(CommandHandler('start', start))
#     dp.add_handler(CommandHandler('predict', predict))
#     updater.start_polling()
#     updater.idle()