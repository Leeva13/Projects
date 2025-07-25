import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, initial_capital=10000, max_risk=0.02, max_drawdown=0.05):
        self.capital = initial_capital
        self.max_risk = max_risk
        self.max_drawdown = max_drawdown
        self.peak_equity = initial_capital
        self.equity_curve = [initial_capital]
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trade_history = []
        self.initial_capital = initial_capital    
        self.positions = {}
    
    TRANSACTION_COST = 0.001  # 0.1%
    SLIPPAGE = 0.0005         # 0.05%

    def update_equity(self, new_value):
        self.equity_curve.append(new_value)
        self.peak_equity = max(self.peak_equity, new_value)

    # Критерій Келі для розміру позиції
    def kelly_size(self, win_rate, win_loss_ratio):
        return win_rate - (1 - win_rate) / win_loss_ratio

    def update_kelly_params(self):
        if len(self.trade_history) < 10:  # Чекаємо мінімум 10 угод
            return 0.6, 2.0  # Значення за замовчуванням
        wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        losses = len(self.trade_history) - wins
        win_rate = wins / len(self.trade_history) if self.trade_history else 0.6
        avg_win = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] > 0]) or 1
        avg_loss = np.mean([abs(t['pnl']) for t in self.trade_history if t['pnl'] < 0]) or 1
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0
        return win_rate, win_loss_ratio

    def calculate_pnl(self, entry_price, exit_price, size):
        gross_pnl = size * (exit_price - entry_price)
        costs = size * (entry_price + exit_price) * self.TRANSACTION_COST
        return gross_pnl - costs - self.SLIPPAGE

    # Адаптивний стоп-лос на основі ATR
    def update_stops(self, current_price, atr):
        if self.position > 0:  # Long position
            self.stop_loss = current_price - 1.5 * atr
            self.take_profit = current_price + 3.0 * atr
        elif self.position < 0:  # Short position
            self.stop_loss = current_price + 1.5 * atr
            self.take_profit = current_price - 3.0 * atr

    def open_position(self, action, price, size, volatility):
        """Відкриття позиції"""
        self.positions['current'] = {
            'action': action,
            'price': price,
            'size': size,
            'volatility': volatility
        }
        self.equity_curve.append(self.capital)

    def calculate_position_size(self, confidence, volatility, price):
        """Розрахунок розміру позиції"""
        risk_per_trade = 0.01  # 1% ризику на угоду
        size = (self.capital * risk_per_trade) / (volatility * price)
        return max(0.1, min(size, self.capital / price))  # Обмежуємо розмір

    def update_position(self, price, volatility):
        """Оновлення позиції"""
        if 'current' in self.positions:
            return self.positions['current']['action']
        return None

    def close_position(self, price):
        """Закриття позиції"""
        if 'current' in self.positions:
            pos = self.positions['current']
            if pos['action'] == 'buy':
                pnl = (price - pos['price']) * pos['size']
            else:  # sell
                pnl = (pos['price'] - price) * pos['size']
            self.capital += pnl
            self.equity_curve.append(self.capital)
            del self.positions['current']
            return pnl
        return 0

    def check_drawdown(self):
        """Перевірка просідання"""
        equity = np.array(self.equity_curve)
        max_drawdown = (np.max(equity) - np.min(equity)) / np.max(equity)
        return max_drawdown < 0.3