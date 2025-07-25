import requests
import numpy as np

class TradingMonitor:
    def __init__(self, trading_engine, telegram_token=None, chat_id=None):
        self.engine = trading_engine
        self.telegram_token = telegram_token
        self.chat_id = chat_id
    
    def send_alert(self, message):
        if self.telegram_token and self.chat_id:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {'chat_id': self.chat_id, 'text': message}
            requests.post(url, json=payload)
        print(f"ALERT: {message}")
    
    def check_risks(self):
        """Моніторинг ключових метрик у реальному часі"""
        current_equity = self.engine.risk_manager.capital
        peak_equity = self.engine.risk_manager.peak_equity
        drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
        
        # Расчет Sharpe Ratio
        returns = np.diff(self.engine.risk_manager.equity_curve)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365*24)
        else:
            sharpe = 0
        
        # Проверка аномалий
        if sharpe < 1.5:
            self.send_alert(f"⚠️ Низький Sharpe Ratio: {sharpe:.2f}")
        if drawdown > 0.05:
            self.send_alert(f"🚨 Небезпечна просадка: {drawdown:.2%}")
        
        # Отчет о состоянии
        report = (f"📊 Звіт:\nEquity: ${current_equity:.2f}\n"
                 f"Sharpe: {sharpe:.2f}\nDrawdown: {drawdown:.2%}\n"
                 f"Активних позицій: {len(self.engine.positions)}")
        self.send_alert(report)

    def daily_report(self):
        report = (f"📈 Денний звіт:\n"
                f"Баланс: ${self.engine.balance:.2f}\n"
                f"Угод: {len(self.engine.trade_history)}\n")
        for trade in self.engine.trade_history:
            report += f"Trade: {trade[0]} {trade[1]} at {trade[2]:.2f}, Size: {trade[3]:.4f}, PnL: {trade[4]:.2f}\n"
        self.send_alert(report)
