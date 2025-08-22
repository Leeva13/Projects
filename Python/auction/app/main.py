# app/main.py
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, HTTPException
from sqlalchemy.orm import Session
import asyncio

from . import crud, models, schemas
from .database import engine, get_db

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.connections: dict[int, list[WebSocket]] = {}

    async def connect(self, lot_id: int, websocket: WebSocket):
        await websocket.accept()
        if lot_id not in self.connections:
            self.connections[lot_id] = []
        self.connections[lot_id].append(websocket)

    def disconnect(self, lot_id: int, websocket: WebSocket):
        if lot_id in self.connections:
            self.connections[lot_id].remove(websocket)

    async def broadcast(self, lot_id: int, message: dict):
        if lot_id in self.connections:
            for connection in list(self.connections[lot_id]):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.disconnect(lot_id, connection)

manager = ConnectionManager()

@app.post("/lots/", response_model=schemas.Lot)
def create_lot(lot: schemas.LotCreate, db: Session = Depends(get_db)):
    return crud.create_lot(db=db, lot=lot)

@app.get("/lots/", response_model=list[schemas.Lot])
def get_lots(db: Session = Depends(get_db)):
    return crud.get_active_lots(db)

@app.get("/lots/{lot_id}", response_model=schemas.Lot)
def get_lot(lot_id: int, db: Session = Depends(get_db)):
    lot = crud.get_lot(db, lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found")
    return lot

@app.post("/lots/{lot_id}/bids/", response_model=schemas.Bid)
async def create_bid(lot_id: int, bid: schemas.BidCreate, db: Session = Depends(get_db)):
    db_bid, extended, new_end_time = crud.create_bid(db, lot_id, bid)
    bid_message = {
        "type": "bid_placed",
        "lot_id": lot_id,
        "bidder": bid.bidder,
        "amount": bid.amount
    }
    await manager.broadcast(lot_id, bid_message)
    if extended:
        ext_message = {
            "type": "time_extended",
            "lot_id": lot_id,
            "new_end_time": new_end_time.isoformat()
        }
        await manager.broadcast(lot_id, ext_message)
    return db_bid

@app.websocket("/ws/lots/{lot_id}")
async def websocket_endpoint(websocket: WebSocket, lot_id: int, db: Session = Depends(get_db)):
    lot = crud.get_lot(db, lot_id)
    if not lot:
        await websocket.close(code=1003)
        return
    await manager.connect(lot_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # Wait for messages or disconnect
    except WebSocketDisconnect:
        manager.disconnect(lot_id, websocket)
    except Exception:
        manager.disconnect(lot_id, websocket)