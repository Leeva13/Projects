import asyncio
import websockets
import json
import threading

async def binance_websocket(symbol, callback):
    uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_1h"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    kline = data['k']
                    candle = {
                        'timestamp': kline['t'],
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v'])
                    }
                    callback(candle)
        except Exception as e:
            print(f"Помилка WebSocket: {e}. Повторне підключення через 5 секунд...")
            await asyncio.sleep(5)

# Функція для запуску WebSocket в окремому потоці
def start_websocket(symbol, callback):
    async def _run():
        uri = f"wss://stream.binance.com:9443/ws/{symbol}@kline_1h"
        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                candle = data['k']
                callback({
                    'timestamp': candle['t'],
                    'open': float(candle['o']),
                    'high': float(candle['h']),
                    'low': float(candle['l']),
                    'close': float(candle['c']),
                    'volume': float(candle['v'])
                })
    
    def run():
        asyncio.run(_run())
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()