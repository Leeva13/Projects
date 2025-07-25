import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import optuna
import time
import json
import ccxt
import ta 
import traceback
import concurrent.futures
import math
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.multiprocessing import Pool, set_start_method
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset
from collections import Counter
from web_socket import start_websocket
from dotenv import load_dotenv
from config import SYMBOLS, FEATURES, TARGET, SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, PATIENCE, TIMEFRAME, DB_PATH, MODELS_DIR, RESULTS_DIR, DATA_DIR, MAX_THREADS, OPTIM_DIR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from numba import njit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from risk_manager import RiskManager  # Імпортуємо оновлений менеджер ризиків
from PaperTradingEngine import PaperTradingEngine
from TradingMonitor import TradingMonitor
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ThreadPoolExecutor

cudnn.benchmark = True
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_sharpes = {sym: -np.inf for sym in SYMBOLS}
latest_candles = {sym: None for sym in SYMBOLS}
sequence_buffers = {sym: [] for sym in SYMBOLS}

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.norm(x + residual)

class EnhancedTransformer(nn.Module):
    def __init__(self, input_size, num_classes, nhead=8, hidden_size=128, num_layers=3, 
                 dropout=0.4, num_res_blocks=2, dim_feedforward=512):  # Додав dim_feedforward
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,  # Використовуємо параметр
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.attention = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_res_blocks)])
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        
        # Додаємо механізм уваги
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # Skip-connection
        
        # Додаємо residual blocks
        for block in self.res_blocks:
            x = block(x)
            
        x = x[:, -1, :]  # Беремо останній крок
        x = self.dropout(x)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)  # Для регресії
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# Гібридна модель
class HybridModel(nn.Module):
    def __init__(self, input_size, num_classes, nhead=8, hidden_size=128, num_layers=3, output_size=3, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Використовуємо hidden_size замість d_model
        self.attention = nn.MultiheadAttention(hidden_size, nhead)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(2)])
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        # Додаємо механізм уваги
        attn_output, _ = self.attention(out, out, out)
        out = out + attn_output  # Skip-connection
        
        # Додаємо residual blocks
        for block in self.res_blocks:
            out = block(out)
            
        return self.fc(out[:, -1, :])

# Нова архітектура Transformer
class PriceTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)  # Для регресії
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# Функція втрат з фінансовим контекстом
class FinancialLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets, potential_gain):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        
        # Ваги на основі потенційного прибутку
        gain_weights = torch.ones_like(targets, dtype=torch.float)
        gain_weights[targets == 2] = potential_gain[targets == 2]  # Більша вага для класу "UP"
        
        loss = (gain_weights * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss

def safe_json_dump(data):
    def replace_inf(obj):
        if isinstance(obj, float) and obj == -np.inf:
            return None  # Заміна -np.inf на null
        return obj
    return json.dumps({k: replace_inf(v) for k, v in data.items()})

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# Оновлений Sharpe з витратами
@njit
def calculate_sharpe(returns, trade_cost=0.001, slippage=0.0005):
    """Покращений розрахунок Sharpe Ratio з урахуванням витрат"""
    if len(returns) == 0:
        return 0
    
    # Учет торгових витрат
    adjusted_returns = returns - trade_cost - slippage
    mean_ret = np.mean(adjusted_returns)
    std_ret = np.std(adjusted_returns)
    
    if std_ret > 1e-6:
        return mean_ret / std_ret * np.sqrt(365 * 24)
    return 0

# Фінансові метрики з використанням RiskManager
def calculate_financial_metrics(y_true, y_pred, prices, volatilities, initial_capital=10000):
    volatilities = np.where(volatilities <= 0, 0.01, volatilities)
    risk_manager = RiskManager(initial_capital=initial_capital)
    position_open = False
    current_position = None
    entry_price = 0
    trade_history = []
    
    for i in range(len(y_pred)):
        current_price = prices[i]
        volatility = volatilities[i]
        
        if not position_open:
            if y_pred[i] == 1:  # Buy signal
                confidence = 0.7
                size = risk_manager.calculate_position_size(confidence, volatility, current_price)
                risk_manager.open_position("buy", current_price, size, volatility)
                position_open = True
                current_position = "long"
                entry_price = current_price
                trade_history.append(("buy", current_price, size, i))
                
            elif y_pred[i] == 0:  # Sell signal
                confidence = 0.7
                size = risk_manager.calculate_position_size(confidence, volatility, current_price)
                risk_manager.open_position("sell", current_price, size, volatility)
                position_open = True
                current_position = "short"
                entry_price = current_price
                trade_history.append(("sell", current_price, size, i))
        else:
            action = risk_manager.update_position(current_price, volatility)
            
            if (current_position == "long" and action == "close_long") or \
               (current_position == "short" and action == "close_short"):
                pnl = risk_manager.close_position(current_price)
                position_open = False
                trade_history.append(("close", current_price, 0, i, pnl))
                
                if not risk_manager.check_drawdown():
                    break
    
    if position_open:
        pnl = risk_manager.close_position(prices[-1])
        trade_history.append(("force_close", prices[-1], 0, len(y_pred)-1, pnl))
    
    equity = risk_manager.equity_curve
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])
    
    sharpe = calculate_sharpe(returns) if len(returns) > 0 else 0
    max_drawdown = (max(equity) - min(equity)) / max(equity) if len(equity) > 0 else 0
    
    return sharpe, max_drawdown, trade_history, equity

# Допоміжні функції
def fetch_df(symbol):
    table = f"features_{symbol.replace('/','_')}_{TIMEFRAME}"
    print(f"[DEBUG] Зчитування даних із таблиці: {table}")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    
    if df.empty:
        raise ValueError(f"No data available for {symbol}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop_duplicates(subset='timestamp', keep='first')
    df = df.set_index('timestamp')
    
    required_columns = FEATURES + [TARGET, 'close', 'log_returns']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing critical columns {missing_cols} for {symbol}. Please check data collection.")
    
    return df[required_columns].dropna()

class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        
    def forward(self, x):
        with torch.no_grad():
            out1 = F.softmax(self.model1(x), dim=1)
            out2 = F.softmax(self.model2(x), dim=1)
            out3 = F.softmax(self.model3(x), dim=1)
        
        # Зважене середнє з урахуванням впевненості моделей
        avg_proba = (out1 + out2 + out3) / 3
        return avg_proba

def evaluate_model(model, X_test, y_test, prices_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(DEVICE))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        accuracy = accuracy_score(y_test_np, preds)
        f1 = f1_score(y_test_np, preds, average='macro')
        return {'accuracy': accuracy, 'f1': f1}

def walk_forward_validation(model, X, y, prices, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    
    for train_index, test_index in tscv.split(X):
        # Розділення даних зі збереженням часового порядку
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        prices_train, prices_test = prices[train_index], prices[test_index]
        
        # Тренування моделі
        model.fit(X_train, y_train, prices_train)
        
        # Оцінка на тестовому наборі
        test_metrics = evaluate_model(model, X_test, y_test, prices_test)
        metrics.append(test_metrics)
    
    # Розрахунок середніх метрик
    avg_metrics = {}
    for key in metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in metrics])
    
    return avg_metrics

# 1. Кастомна функція втрат, що поєднує фінансові метрики
class FinancialSharpeLoss(nn.Module):
    def __init__(self, base_alpha=None, sharpe_weight=0.7):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=base_alpha)
        self.sharpe_weight = sharpe_weight
        
    def forward(self, logits, targets, returns):
        ce_loss = self.ce(logits, targets)
        
        # Розрахунок доходності на основі прогнозів
        preds = torch.argmax(logits, dim=1)
        realized_returns = returns[torch.arange(len(preds)), preds]
        
        # Розрахунок Sharpe Ratio
        mean_return = realized_returns.mean()
        std_return = realized_returns.std()
        sharpe = mean_return / (std_return + 1e-6) * torch.sqrt(torch.tensor(365*24))
        
        # Комбіновані втрати
        combined_loss = ce_loss - self.sharpe_weight * sharpe
        return combined_loss, ce_loss.item(), sharpe.item()

# 2. Функція для розрахунку матриці дохідностей
def calculate_returns_tensor(prices, horizon=1):
    """
    Розраховує матрицю потенційних дохідностей для кожної торгової дії
    Формат повернення: [batch_size, 3] де:
        стовпець 0: дохідність для продажу (short)
        стовпець 1: дохідність для утримання (0)
        стовпець 2: дохідність для покупки (long)
    """

    if len(prices) < horizon + 1:
        return torch.zeros((len(prices), 3), dtype=torch.float32, device=prices.device)
    
    # Переконаємось, що prices є 1D тензором
    if prices.dim() > 1:
        prices = prices.squeeze(-1)  # Видаляємо останній вимір
    
    # Додаємо перевірку на наявність достатньої кількості даних
    if len(prices) < horizon:
        return torch.zeros((len(prices), 3), dtype=torch.float32, device=prices.device)
    
    batch_size = len(prices)
    returns_tensor = torch.zeros((batch_size, 3), dtype=torch.float32, device=prices.device)
    
    # Обчислюємо майбутні ціни
    future_prices = torch.roll(prices, shifts=-horizon, dims=0)
    
    # Обчислюємо дохідність
    valid_mask = prices > 0
    returns = torch.zeros_like(prices, dtype=torch.float32)
    returns[valid_mask] = (future_prices[valid_mask] / prices[valid_mask]) - 1
    
    # Визначаємо дійсний діапазон (без останніх 'horizon' елементів)
    valid_indices = torch.arange(0, batch_size - horizon)
    
    # Заповнюємо матрицю дохідностей
    returns[-horizon:] = 0
    returns_tensor[valid_indices, 0] = -returns[valid_indices]  # Short
    returns_tensor[valid_indices, 1] = 0                        # Hold
    returns_tensor[valid_indices, 2] = returns[valid_indices]    # Long
    
    return returns_tensor

def oversample_minority_classes(X, y, prices):
    class_counts = Counter(y)
    max_count = max(class_counts.values())
    X_balanced, y_balanced, prices_balanced = [], [], []
    
    for cls in class_counts:
        cls_indices = np.where(y == cls)[0]
        oversample_factor = max(1, int(max_count / class_counts[cls]) - 1)
        
        # Додаємо оригінальні дані
        X_balanced.append(X[cls_indices])
        y_balanced.append(y[cls_indices])
        prices_balanced.append(prices[cls_indices])
        
        # Додаємо збалансовані дані
        for _ in range(oversample_factor):
            X_balanced.append(X[cls_indices])
            y_balanced.append(y[cls_indices])
            prices_balanced.append(prices[cls_indices])
    
    # Додаємо перевірку на наявність усіх класів
    class_counts = Counter(y)
    if len(class_counts) < 3:
        print(f"⚠️ Знайдено лише {len(class_counts)} класів. Додаємо фіктивні дані для відсутніх класів")
        for cls in [0, 1, 2]:
            if cls not in class_counts:
                print(f"Додаємо фіктивні дані для класу {cls}")
                # Додаємо невелику кількість фіктивних даних
                dummy_indices = [0]  # використовуємо перший індекс
                X_balanced.append(X[dummy_indices])
                y_balanced.append([cls] * len(dummy_indices))
                prices_balanced.append(prices[dummy_indices])

    # Об'єднуємо всі частини
    X_balanced = np.concatenate(X_balanced)
    y_balanced = np.concatenate(y_balanced)
    prices_balanced = np.concatenate(prices_balanced)
    
    return X_balanced, y_balanced, prices_balanced

def detect_market_regime(df):
    """Визначаємо режим ринку: тренд, флет, волатильність"""
    df['regime'] = 0  # За замовчуванням - невизначено
    
    # Визначення тренду
    sma_50 = df['close'].rolling(50).mean()
    sma_200 = df['close'].rolling(200).mean()
    df.loc[sma_50 > sma_200 * 1.03, 'regime'] = 1  # Верхній тренд
    df.loc[sma_50 < sma_200 * 0.97, 'regime'] = -1  # Нижній тренд
    
    # Визначення волатильності
    volatility = df['close'].pct_change().rolling(24).std()
    df.loc[volatility > volatility.quantile(0.75), 'regime'] = 2  # Висока волатильність
    
    return df

# 3. Оновлена функція підготовки даних
def prepare_sequences(df, sequence_length, features, target_col='target'):
    required_cols = features + [target_col, 'close', 'log_returns']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    df = df[required_cols].dropna()
    X = df[features].values
    y = df[target_col].values
    prices = df['close'].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_prices = MinMaxScaler()
    prices_scaled = scaler_prices.fit_transform(prices.reshape(-1, 1)).flatten()

    X_sequences = []
    y_sequences = []
    price_sequences = []

    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])
        price_sequences.append(prices_scaled[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    price_sequences = np.array(price_sequences)

    X_train, X_test, y_train, y_test, prices_train, prices_test = train_test_split(
        X_sequences, y_sequences, price_sequences, test_size=0.2, shuffle=False
    )

    df_test = df.iloc[-len(X_test):].copy()
    return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32),
            torch.tensor(prices_train, dtype=torch.float32), torch.tensor(prices_test, dtype=torch.float32),
            scaler_X, scaler_prices, df_test)

def calculate_potential_gain(prices, horizon=1):
    """Розрахунок потенційного прибутку для кожного елемента"""
    future_prices = torch.roll(prices, shifts=-horizon, dims=0)
    gain = (future_prices - prices) / prices
    gain[-horizon:] = 0  # Обнуляємо останні елементи
    return gain

def validate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb, _ in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average='macro')
    return f1

# 4. Оновлена функція тренування
def train_model(model, train_loader, val_loader, optimizer, device, symbol, epochs=EPOCHS, patience=PATIENCE):
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    model_file = os.path.join(MODELS_DIR, f"best_model_{symbol.replace('/', '_')}.pth")
    best_val_mse = float('inf')
    no_improve = 0
    
    tscv = TimeSeriesSplit(n_splits=5)
    X_train = torch.cat([xb for xb, _, _ in train_loader], dim=0)
    y_train = torch.cat([yb for _, yb, _ in train_loader], dim=0)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for train_idx, val_idx in tscv.split(X_train):
            X_fold, y_fold = X_train[train_idx], y_train[train_idx]
            fold_dataset = TensorDataset(X_fold, y_fold, torch.zeros_like(y_fold))
            fold_loader = DataLoader(fold_dataset, batch_size=64, shuffle=True)
            
            for xb, yb, _ in fold_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = model(xb).squeeze()
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        train_loss /= (len(fold_loader) * tscv.n_splits)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb).squeeze()
                loss = criterion(outputs, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_mse:
            best_val_mse = val_loss
            torch.save(model.state_dict(), model_file)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

        scheduler.step(val_loss)
    
    return model, best_val_mse

def train_symbol(symbol):
    print(f"Початок тренування для {symbol}")
    df = fetch_df(symbol)
    X_train, X_test, y_train, y_test, prices_train, prices_test, scaler_X, scaler_prices, df_test = prepare_sequences(df, SEQUENCE_LENGTH, FEATURES)
    
    train_dataset = TensorDataset(X_train, y_train, prices_train)
    test_dataset = TensorDataset(X_test, y_test, prices_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    
    model = LSTMModel(input_size=len(FEATURES), hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    model, mse = train_model(model, train_loader, test_loader, optimizer, DEVICE, symbol)
    sharpe, max_drawdown = backtest_model(model, X_test, y_test, prices_test, df_test, symbol)
    
    print(f"{symbol} - MSE: {mse:.6f}, Sharpe: {sharpe:.4f}, Max Drawdown: {max_drawdown:.2%}")
    
    safe_symbol = symbol.replace('/', '_')
    torch.save({
        'model_state': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_prices': scaler_prices,
        'features': FEATURES
    }, f"models/model_{safe_symbol}.pth")
    
    metrics = {'mse': mse, 'sharpe': sharpe, 'max_drawdown': max_drawdown}
    with open(os.path.join(RESULTS_DIR, f"metrics_{safe_symbol}.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

def train_all_models(symbols):
    print(f"Використовується пристрій: {DEVICE}")
    
    def train_symbol_with_device(symbol):
        if 'cuda' in str(DEVICE):
            device_id = symbols.index(symbol) % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            print(f"[INFO] Тренування {symbol} на CUDA:{device_id}")
        train_symbol(symbol)
    
    with ThreadPoolExecutor(max_workers=min(MAX_THREADS, len(symbols))) as executor:
        futures = {executor.submit(train_symbol_with_device, symbol): symbol for symbol in symbols}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="Паралельне тренування"):
            symbol = futures[future]
            try:
                future.result()
                print(f"✅ Тренування для {symbol} завершено")
            except Exception as e:
                print(f"🚨 Помилка тренування для {symbol}: {str(e)}")
    
    print("\nПаралельне тренування завершено!")

# Додаємо цю функцію для автоматичного тренування
def train_missing_models():
    symbols_to_train = []
    for symbol in SYMBOLS:
        model_path = os.path.join(MODELS_DIR, f"full_model_{symbol.replace('/', '_')}.pth")
        if not os.path.exists(model_path):
            print(f"Навччння моделі для {symbol} спершу...")
            train_and_backtest(symbol)
        else:
            print(f"✅ Модель для {symbol} вже існує. Пропускаємо тренування.")
    
    if symbols_to_train:
        print(f"\n⚡ Починаємо тренування для {len(symbols_to_train)} пар...")
        train_all_models(symbols_to_train)
    else:
        print("\n✨ Всі моделі вже натреновані. Немає потреб в тренуванні.")

def load_best_score(symbol):
    """Завантажує найкращий score для конкретного символу"""
    file_path = os.path.join(RESULTS_DIR, f'best_score_{symbol.replace("/", "_")}.json')
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return -np.inf

# Функції для роботи з метриками
def load_best_metrics(symbol):
    """Завантажує метрики для конкретного символу"""
    file_path = os.path.join(RESULTS_DIR, f'best_metrics_{symbol.replace("/", "_")}.json')
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_best_score(symbol, score):
    """Зберігає score для конкретного символу"""
    file_path = os.path.join(RESULTS_DIR, f'best_score_{symbol.replace("/", "_")}.json')
    with open(file_path, 'w') as f:
        json.dump(score, f)

def save_best_metrics(symbol, metrics):
    metrics_file = os.path.join(RESULTS_DIR, f"best_metrics_{symbol.replace('/', '_')}.json")
    try:
        with open(metrics_file, 'r') as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}
    
    # Convert tensors to scalars
    converted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            converted_metrics[key] = value.item()
        else:
            converted_metrics[key] = value
    
    if existing.get('f1', -float('inf')) < converted_metrics['f1']:
        print(f"[INFO] Previous F1-score: {existing.get('f1', -float('inf')):.4f}, New F1-score: {converted_metrics['f1']:.4f}")
        print(f"[INFO] Sharpe Ratio: {converted_metrics['sharpe']:.4f}, Max Drawdown: {converted_metrics['max_drawdown']:.4f}")
        with open(metrics_file, 'w') as f:
            json.dump(converted_metrics, f, indent=4)
        print(f"[OK] Saved best metrics for {symbol} to {metrics_file} with F1-score: {converted_metrics['f1']:.4f}")

def compute_feature_importance(model, X_val, y_val, features, device, symbol):
    model.eval()
    baseline_preds = []
    with torch.no_grad():
        for i in range(len(X_val)):
            input_data = X_val[i].unsqueeze(0).to(device)
            output = model(input_data)
            pred = torch.argmax(output, dim=1).item()
            baseline_preds.append(pred)
    
    baseline_f1 = f1_score(y_val.cpu().numpy(), baseline_preds, average='macro')
    importance_scores = {}
    
    for i, feature in enumerate(features):
        X_val_permuted = X_val.clone()
        # Shuffle the feature across all sequences
        permuted_indices = np.random.permutation(X_val.shape[0])
        X_val_permuted[:, :, i] = X_val_permuted[permuted_indices, :, i]
        
        permuted_preds = []
        with torch.no_grad():
            for j in range(len(X_val_permuted)):
                input_data = X_val_permuted[j].unsqueeze(0).to(device)
                output = model(input_data)
                pred = torch.argmax(output, dim=1).item()
                permuted_preds.append(pred)
        
        permuted_f1 = f1_score(y_val.cpu().numpy(), permuted_preds, average='macro')
        importance_scores[feature] = baseline_f1 - permuted_f1
    
    # Save importance scores
    importance_file = os.path.join(RESULTS_DIR, f"feature_importance_{symbol.replace('/', '_')}.json")
    with open(importance_file, 'w') as f:
        json.dump(importance_scores, f, indent=4)
    print(f"[OK] Saved feature importance for {symbol} to {importance_file}")
    
    # Print sorted importance scores
    sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"[INFO] Feature importance for {symbol}:")
    for feature, score in sorted_importance:
        print(f"  {feature}: {score:.4f}")
    
    return importance_scores

# Оновлення звіту про моделі
def update_model_report(symbol, epoch, train_loss, val_loss, f1, sharpe, drawdown, accuracy, precision, recall, event):
    report_path = os.path.join(RESULTS_DIR, f"model_report_{symbol.replace('/', '_')}.md")
    
    # Якщо файл звіту ще не існує, створюємо заголовок і шапку таблиці
    if not os.path.exists(report_path):
        content = f"# Training Report: {symbol}\n\n"
        content += "| Epoch | Train Loss | Val Loss | Val F1 | Sharpe | Drawdown | Accuracy | Precision | Recall | Event |\n"
        content += "|-------|------------|----------|--------|--------|----------|----------|-----------|--------|-------|\n"
    else:
        with open(report_path, 'r') as f:
            content = f.read()
    
    # Форматуємо значення: якщо метрика не розрахована (None), ставимо "-"
    sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "-"
    drawdown_str = f"{drawdown:.4f}" if drawdown is not None else "-"
    accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "-"
    precision_str = f"{precision:.4f}" if precision is not None else "-"
    recall_str = f"{recall:.4f}" if recall is not None else "-"
    
    # Додаємо новий рядок у звіт
    new_row = f"| {epoch} | {train_loss:.4f} | {val_loss:.4f} | {f1:.4f} | {sharpe_str} | {drawdown_str} | {accuracy_str} | {precision_str} | {recall_str} | {event} |\n"
    content += new_row
    
    # Зберігаємо оновлений звіт
    with open(report_path, 'w') as f:
        f.write(content)

# Функція бектестування
def backtest_model(model, X_test, y_test, prices_test, df_test, symbol):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(len(X_test)):
            input_data = X_test[i].unsqueeze(0).to(DEVICE)
            output = model(input_data).item()
            all_preds.append(output)
    
    atr = df_test['ATR_14'].values
    threshold = 0.25 * atr / df_test['close'].values

    returns = []
    position = 0
    entry_price = 0
    
    for i, (pred, thresh) in enumerate(zip(all_preds, threshold)):
        current_price = df_test['close'].iloc[i]
        if position == 0:
            if pred > thresh:
                position = 1
                entry_price = current_price
            elif pred < -thresh:
                position = -1
                entry_price = current_price
        elif position == 1 and pred < 0:
            returns.append((current_price - entry_price) / entry_price)
            position = 0
        elif position == -1 and pred > 0:
            returns.append((entry_price - current_price) / entry_price)
            position = 0
    
    if position != 0:
        last_price = df_test['close'].iloc[-1]
        returns.append((last_price - entry_price) / entry_price if position == 1 else (entry_price - last_price) / entry_price)
    
    if returns:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365*24) if np.std(returns) > 0 else 0
        max_drawdown = (np.max(np.cumprod([1 + r for r in returns])) - np.min(np.cumprod([1 + r for r in returns]))) / np.max(np.cumprod([1 + r for r in returns]))
    else:
        sharpe, max_drawdown = 0, 0
    
    return sharpe, max_drawdown

def calculate_simple_metrics(prices, signals):
    """Спрощений розрахунок метрик без складного RiskManager"""
    returns = []
    position = 0
    entry_price = 0
    
    for i in range(len(signals)):
        if signals[i] == 2 and position == 0:  # Buy
            position = 1
            entry_price = prices[i]
        elif signals[i] == 0 and position == 1:  # Sell
            returns.append((prices[i] - entry_price) / entry_price)
            position = 0
    
    if position == 1:  # Close open position
        returns.append((prices[-1] - entry_price) / entry_price)
    
    if returns:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365*24)
        cum_returns = np.cumprod([1 + r for r in returns])
        max_drawdown = (np.max(cum_returns) - np.min(cum_returns)) / np.max(cum_returns)
        return sharpe, max_drawdown
    return 0, 0

def optimize_and_train(symbol):
    print(f"\n🚀 Оптимізація гіперпараметрів для {symbol}")
    best_params = optimize_hyperparameters(symbol)
    
    model = LSTMModel(
        input_size=len(FEATURES),
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(DEVICE)
    
    df = fetch_df(symbol)
    X_train, X_test, y_train, y_test, prices_train, prices_test, scaler_X, scaler_prices, df_test = prepare_sequences(df, SEQUENCE_LENGTH, FEATURES)
    
    train_dataset = TensorDataset(X_train, y_train, prices_train)
    test_dataset = TensorDataset(X_test, y_test, prices_test)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
    model, mse = train_model(model, train_loader, test_loader, optimizer, DEVICE, symbol)
    
    sharpe, max_drawdown = backtest_model(model, X_test, y_test, prices_test, df_test, symbol)
    print(f"{symbol} - MSE: {mse:.6f}, Sharpe: {sharpe:.4f}, Max Drawdown: {max_drawdown:.2%}")
    
    safe_symbol = symbol.replace('/', '_')
    torch.save({
        'model_state': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_prices': scaler_prices,
        'features': FEATURES
    }, f"models/model_{safe_symbol}.pth")

def train_and_backtest(symbol):
    df = fetch_df(symbol)
    X_train, X_test, y_train, y_test, prices_train, prices_test, scaler_X, scaler_prices, df_test = prepare_sequences(df, SEQUENCE_LENGTH, FEATURES)
    
    train_dataset = TensorDataset(X_train, y_train, prices_train)
    test_dataset = TensorDataset(X_test, y_test, prices_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = LSTMModel(input_size=len(FEATURES), hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)  # Замінено на LSTMModel
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    model, mse = train_model(model, train_loader, test_loader, optimizer, DEVICE, symbol)
    
    sharpe, max_drawdown = backtest_model(model, X_test, y_test, prices_test, df_test, symbol)
    print(f"{symbol} - MSE: {mse:.6f}, Sharpe: {sharpe:.4f}, Max Drawdown: {max_drawdown:.2%}")
    
    safe_symbol = symbol.replace('/', '_')
    torch.save({
        'model_state': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_prices': scaler_prices,
        'features': FEATURES
    }, f"models/model_{safe_symbol}.pth")

def fetch_realtime_data(symbol, timeframe=TIMEFRAME, limit=SEQUENCE_LENGTH):
    """Отримання реальних даних з Binance (лише для читання)"""
    exchange = ccxt.binance({
        'enableRateLimit': True  # API ключ не потрібний для отримання даних
    })
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Помилка отримання даних для {symbol}: {e}")
        return pd.DataFrame()  # Повертаємо порожній DataFrame у разі помилки

def preprocess(data, scaler, features, sequence_length=SEQUENCE_LENGTH):
    if data.empty:
        return None
        
    data = data.copy()
    
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"⚠️ Відсутні фічі в даних: {missing_features}. Заповнюємо нулями.")
        for f in missing_features:
            data[f] = 0
    
    data = data.fillna(method='ffill').fillna(0)
    
    processed = data[features].copy()
    
    scaled = scaler.transform(processed)
    
    sequence = scaled[-sequence_length:]
    
    return torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

def load_model(model_path, input_size):
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def candle_callback(candle, symbol):
    global sequence_buffers, latest_candles
    latest_candles[symbol] = candle
    sequence_buffers[symbol].append(candle)
    if len(sequence_buffers[symbol]) > SEQUENCE_LENGTH:
        sequence_buffers[symbol].pop(0)

# 5. Оновлена головна функція
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OPTIM_DIR, exist_ok=True)

    with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as executor:
        futures = {executor.submit(optimize_hyperparameters, symbol, n_trials=10): symbol for symbol in SYMBOLS}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(SYMBOLS), desc="Паралельна оптимізація"):
            symbol = futures[future]
            try:
                future.result()
                print(f"✅ Оптимізація для {symbol} завершена")
            except Exception as e:
                print(f"🚨 Помилка оптимізації для {symbol}: {str(e)}")

    train_all_models(SYMBOLS)

# 6. Оновлена функція для оптимізації з Optuna
def objective(trial, symbol, sequence_length, features):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])  # Додано 128
    }

    df = fetch_df(symbol)
    X_train, X_test, y_train, y_test, prices_train, prices_test, scaler_X, scaler_prices, df_test = prepare_sequences(df, sequence_length, features)
    
    model = LSTMModel(
        input_size=len(features),
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)
    train_dataset = TensorDataset(X_train, y_train, prices_train)
    test_dataset = TensorDataset(X_test, y_test, prices_test)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    
    _, mse = train_model(model, train_loader, test_loader, optimizer, DEVICE, symbol)
    return mse

def optimize_hyperparameters(symbol, n_trials=10):
    print(f"\n🚀 Початок оптимізації для {symbol}")
    study = optuna.create_study(direction='minimize')
    for trial in range(n_trials):
        mse = objective(study.trial, symbol, SEQUENCE_LENGTH, FEATURES)
        print(f"Trial {trial+1}/{n_trials} | MSE: {mse:.6f}")
    study.optimize(lambda trial: objective(trial, symbol, SEQUENCE_LENGTH, FEATURES), n_trials=n_trials)
    print(f"✅ Оптимізація завершена для {symbol}")
    print(f"Найкращий MSE: {study.best_value:.4f}")
    with open(os.path.join(OPTIM_DIR, f"best_params_{symbol.replace('/', '_')}.json"), 'w') as f:
        json.dump(study.best_params, f, indent=4)
    return study.best_params

# Функція для паралельної оптимізації всіх пар
def optimize_all_hyperparameters():
    """Паралельна оптимізація для всіх пар"""
    print("\n🚀 Паралельна оптимізація для всіх пар")
    
    # Використовуємо ThreadPoolExecutor для паралелізації
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(optimize_hyperparameters, symbol): symbol for symbol in SYMBOLS}
        
        # Відстежуємо прогрес
        progress_bar = tqdm(concurrent.futures.as_completed(futures), 
                            total=len(SYMBOLS), 
                            desc="Загальний прогрес оптимізації")
        
        for future in progress_bar:
            symbol = futures[future]
            try:
                future.result()
                progress_bar.set_postfix_str(f"✅ {symbol} завершено")
            except Exception as e:
                progress_bar.set_postfix_str(f"🚨 {symbol} помилка")
                print(f"\nПомилка оптимізації для {symbol}: {str(e)}")

def train_with_optimization(symbol):
    """Тренування моделі з використанням оптимізованих параметрів"""
    # Перевірка наявності оптимізованих параметрів
    optimized_params = load_optimization_results(symbol)
    
    if not optimized_params:
        print(f"🔍 Оптимізовані параметри для {symbol} не знайдені. Запускаємо оптимізацію...")
        optimized_params = optimize_hyperparameters(symbol)
    
    # Використання оптимізованих параметрів для тренування
    print(f"🏁 Початок тренування для {symbol} з оптимізованими параметрами")
    train_and_backtest(symbol, optimized_params)

def save_optimization_results(symbol, best_params, best_value):
    os.makedirs(OPTIM_DIR, exist_ok=True)
    file_path = os.path.join(OPTIM_DIR, f"best_params_{symbol.replace('/', '_')}.json")
    
    # Перевіряємо, чи існує файл із попередніми результатами
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                prev_data = json.load(f)
                prev_best_value = prev_data.get('best_value', -float('inf'))
            
            # Якщо новий F1-score не кращий, не перезаписуємо
            if best_value <= prev_best_value:
                print(f"[INFO] Новий F1-score ({best_value:.4f}) не кращий за попередній ({prev_best_value:.4f}). Файл не оновлено.")
                return
            else:
                print(f"[INFO] Попередній F1-score: {prev_best_value:.4f}, Новий F1-score: {best_value:.4f}")
                return
        except Exception as e:
            print(f"[ERROR] Помилка при читанні попереднього файлу: {e}")
    
    # Зберігаємо нові параметри, якщо вони кращі або файл ще не існує
    data = {
        'best_params': best_params,
        'best_value': best_value
    }
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[OK] Збережено найкращі параметри для {symbol} у {file_path} з F1-score: {best_value:.4f}")

def load_optimization_results(symbol):
    """Завантаження результатів оптимізації для символу"""
    file_path = os.path.join(OPTIM_DIR, f"best_params_{symbol.replace('/', '_')}.json")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# Функція для паралельного тренування всіх пар
def train_all_models(symbols):
    print(f"Використовується пристрій: {DEVICE}")
    
    def train_symbol_with_device(symbol):
        if 'cuda' in str(DEVICE):
            device_id = symbols.index(symbol) % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            print(f"[INFO] Тренування {symbol} на CUDA:{device_id}")
        train_symbol(symbol)
    
    with ThreadPoolExecutor(max_workers=min(MAX_THREADS, len(symbols))) as executor:
        futures = {executor.submit(train_symbol_with_device, symbol): symbol for symbol in symbols}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="Паралельне тренування"):
            symbol = futures[future]
            try:
                future.result()
                print(f"✅ Тренування для {symbol} завершено")
            except Exception as e:
                print(f"🚨 Помилка тренування для {symbol}: {str(e)}")
    
    print("\nПаралельне тренування завершено!")

def run_prediction_mode():
    print("\n=== СИСТЕМА ПРОГНОЗУВАННЯ ЗАПУЩЕНА ===")
    print("Режим: Прогнозування в реальному часі")
    print("Періодичність оновлення: кожні 60 секунд")
    print("Для зупинки натисніть Ctrl+C\n")
    
    train_missing_models()
    
    sequence_buffers = {sym: [] for sym in SYMBOLS}
    risk_manager = RiskManager(initial_capital=10000)
    trading_engine = PaperTradingEngine(risk_manager=risk_manager)
    monitor = TradingMonitor(trading_engine, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID) if TELEGRAM_TOKEN else None
    
    models = {}
    scalers = {}
    features_list = {}
    
    for symbol in SYMBOLS:
        model_path = os.path.join(MODELS_DIR, f"model_{symbol.replace('/', '_')}.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE)
                input_size = len(checkpoint['features'])
                model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3).to(DEVICE)
                model.load_state_dict(checkpoint['model_state'])
                model.eval()
                models[symbol] = model
                scalers[symbol] = checkpoint['scaler_X']
                features_list[symbol] = checkpoint['features']
                print(f"✅ Модель для {symbol} успішно завантажена")
            except Exception as e:
                print(f"⚠️ Помилка завантаження моделі для {symbol}: {e}")
        else:
            print(f"🚨 Критична помилка: Модель для {symbol} відсутня")
    
    for symbol in SYMBOLS:
        start_websocket(symbol.replace('/', '').lower(), lambda candle: candle_callback(candle, symbol))
    
    while True:
        try:
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n=== Оновлення прогнозів [{timestamp}] ===")
            
            for symbol in SYMBOLS:
                if symbol not in models or len(sequence_buffers[symbol]) < SEQUENCE_LENGTH:
                    print(f"⏭ Пропущено {symbol} (немає моделі або недостатньо свічок: {len(sequence_buffers[symbol])}/{SEQUENCE_LENGTH})")
                    continue
                
                data = pd.DataFrame(sequence_buffers[symbol])
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)
                
                data['volatility'] = ta.atr(data['high'], data['low'], data['close'], length=14)
                volatility = data['volatility'].iloc[-1] if not pd.isna(data['volatility'].iloc[-1]) else 0.01
                
                input_data = preprocess(data, scalers[symbol], features_list[symbol])
                if input_data is None:
                    print(f"⏭ Пропущено {symbol} (помилка обробки даних)")
                    continue
                
                input_data = input_data.to(DEVICE)
                with torch.no_grad():
                    output = models[symbol](input_data).item()
                
                threshold = 0.25 * volatility / data['close'].iloc[-1]
                signal = "ПОКУПКА 🟢" if output > threshold else \
                         "ПРОДАЖ 🔴" if output < -threshold else \
                         "УТРИМАННЯ ⚪"
                
                confidence = min(abs(output) / threshold, 1.0) if threshold != 0 else 0.0
                print(f"{symbol}: {signal} (впевненість: {confidence:.2%})")
                
                if confidence > 0.7:
                    action = "buy" if output > threshold else "sell" if output < -threshold else None
                    if action:
                        position_size = risk_manager.calculate_position_size(confidence, volatility, data['close'].iloc[-1])
                        trading_engine.execute_trade(
                            symbol=symbol,
                            action=action,
                            price=data['close'].iloc[-1],
                            size=position_size,
                            volatility=volatility
                        )
                
                if monitor and confidence > 0.7:
                    monitor.send_alert(f"{symbol}: {signal}\nВпевненість: {confidence:.2%}\nЦіна: ${data['close'].iloc[-1]:.2f}")
            
            if monitor:
                monitor.check_risks()
            
            time.sleep(60)
            
        except Exception as e:
            error_msg = f"🚨 Критична помилка: {e}"
            print(error_msg)
            if monitor:
                monitor.send_alert(error_msg)
            time.sleep(60)

# Оновлений головний блок
if __name__ == '__main__':
    main()

    '''
    Прогнозування:
    Після тренування запусти run_prediction_mode() для перевірки прогнозів у реальному часі:
    '''
    # run_prediction_mode()

    # Спочатку проводимо оптимізацію для всіх символів
    # with ThreadPoolExecutor(max_workers=min(MAX_THREADS, len(symbols))) as executor:
    #     futures = {executor.submit(optimize_hyperparameters, symbol): symbol for symbol in symbols}
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="Оптимізація"):
    #         symbol = futures[future]
    #         try:
    #             future.result()
    #             print(f"✅ Оптимізація для {symbol} завершена")
    #         except Exception as e:
    #             print(f"🚨 Помилка оптимізації для {symbol}: {str(e)}")

    # # Потім тренуємо моделі
    # train_all_models(symbols)
