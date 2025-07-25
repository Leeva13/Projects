import numpy as np
import pandas as pd
import torch

class Backtester:
    def __init__(self, data, model, initial_capital=10000):
        self.data = data
        self.model = model
        self.capital = initial_capital
        self.positions = []
        self.equity_curve = []

    def run_backtest(self, spread=0.0005, commission=0.001):
        for i in range(24, len(self.data)):  # SEQUENCE_LENGTH = 24
            window = self.data.iloc[i-24:i][['returns_24h', 'EMA_9', 'EMA_21', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
                                             'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'OBV', 'hour', 'day_of_week', 'ADX_14', 'Stoch_K',
                                             'ATR_14', 'Momentum_14', 'CCI_14', 'Stoch_RSI', 'Williams_%R', 'SuperTrend', 'VWAP',
                                             'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'volatility', 'ATR', 'VROC', 'rolling_volatility']]
            current_price = self.data.iloc[i]['close']
            spread_cost = current_price * spread

            # Прогноз моделі
            inp = torch.tensor(window.values, dtype=torch.float32).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            with torch.no_grad():
                features = self.model['shared'](inp)
                logits = self.model['classifier'](features)
                prediction = logits.argmax(dim=1).item()  # 0: Fall, 1: Flat, 2: Rise

            if prediction == 2 and self.capital > 100:  # Rise
                entry_price = current_price + spread_cost
                position_size = min(0.5, self.capital * 0.1 / entry_price)
                self.capital -= position_size * entry_price
                self.positions.append({'type': 'long', 'entry_price': entry_price, 'size': position_size})
            elif prediction == 0 and len(self.positions) > 0:  # Fall
                for pos in self.positions[:]:
                    if pos['type'] == 'long':
                        exit_price = current_price - spread_cost
                        pnl = (exit_price - pos['entry_price']) * pos['size'] - commission * pos['size'] * exit_price
                        self.capital += pos['size'] * exit_price + pnl
                        self.positions.remove(pos)

            equity = self.capital + sum([pos['size'] * current_price for pos in self.positions])
            self.equity_curve.append(equity)

        return self.equity_curve

# Приклад використання:
# df = pd.read_sql("SELECT * FROM features_BTC_USDT_1h", sqlite3.connect('db/market_data.sqlite'), index_col='timestamp')
# model = {'shared': SharedFeatureExtractor(...), 'classifier': PairSpecificClassifier(...)}
# model['shared'].load_state_dict(torch.load('models/best_BTC_USDT.pth')['shared_state'])
# model['classifier'].load_state_dict(torch.load('models/best_BTC_USDT.pth')['classifier_state'])
# backtester = Backtester(df, model)
# equity = backtester.run_backtest()