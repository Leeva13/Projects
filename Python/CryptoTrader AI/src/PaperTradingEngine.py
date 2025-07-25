# Make sure to import or define RiskManager before using it
from risk_manager import RiskManager  # or define the class above if it's local
import numpy as np

class PaperTradingEngine:
    def __init__(self, initial_balance=10000, risk_manager=None):
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.equity_curve = [initial_balance]
        self.risk_manager = risk_manager or RiskManager(initial_balance)
    
    def update_position(self, current_price, volatility):
        if self.position > 0:  # Long
            if current_price > self.entry_price:
                self.trailing_stop = max(self.trailing_stop, current_price - 2 * volatility)
            if current_price <= self.trailing_stop:
                self.execute_trade(self.symbol, "sell", current_price, self.position, volatility)
                print(f"Trailing stop triggered: Closed long at {current_price}")
        elif self.position < 0:  # Short
            if current_price < self.entry_price:
                self.trailing_stop = min(self.trailing_stop, current_price + 2 * volatility)
            if current_price >= self.trailing_stop:
                self.execute_trade(self.symbol, "buy", current_price, -self.position, volatility)
                print(f"Trailing stop triggered: Closed short at {current_price}")

    def execute_trade(self, symbol, action, price, size, volatility):
        if volatility < 0.01:  # Ігнорувати низьковолатильні сигнали
            return False
        
        min_size = 0.001
        if size < min_size:
            size = min_size
        
        required_cash = size * price * (1 + self.risk_manager.TRANSACTION_COST)
        if self.balance < required_cash:
            size = self.balance / (price * (1 + self.risk_manager.TRANSACTION_COST))
        
        executed_price = price * (1 + np.random.uniform(-0.0005, 0.0005))
        confidence = 0.8  # Змінили поріг
        size = self.risk_manager.calculate_position_size(confidence, volatility, price)
        
        if action == "buy":
            cost = size * executed_price * (self.risk_manager.TRANSACTION_COST + self.risk_manager.SLIPPAGE)
            if self.balance >= size * executed_price + cost:
                self.risk_manager.open_position("buy", executed_price, size, volatility)
                self.positions[symbol] = {
                    'size': size,
                    'entry_price': executed_price,
                    'volatility': volatility,
                    'stop_loss': self.risk_manager.stop_loss,
                    'take_profit': self.risk_manager.take_profit
                }
                self.balance -= size * executed_price + cost
                self.trade_history.append(('buy', symbol, executed_price, size, volatility))
                return True
        elif action == "sell":
            if symbol in self.positions:
                position = self.positions[symbol]
                pnl = size * (executed_price - position['entry_price'])
                cost = size * executed_price * (self.risk_manager.TRANSACTION_COST + self.risk_manager.SLIPPAGE)
                self.balance += pnl - cost
                position['size'] -= size
                self.risk_manager.close_position(executed_price)
                if position['size'] <= 1e-8:
                    del self.positions[symbol]
                self.trade_history.append(('sell', symbol, executed_price, size, pnl))
                return True
        return False