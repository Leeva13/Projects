# app/schemas.py (updated)
from pydantic import BaseModel
from typing import List, Optional
import datetime
from .models import LotStatus

# --- Схеми для Ставок (Bid) ---

class BidBase(BaseModel):
    bidder: str
    amount: float

class BidCreate(BidBase):
    pass

class Bid(BidBase):
    id: int
    lot_id: int
    timestamp: datetime.datetime

    class Config:
        from_attributes = True  

# --- Схеми для Лотів (Lot) ---

class LotBase(BaseModel):
    name: str
    description: Optional[str] = None
    start_price: float
    duration_minutes: Optional[int] = 5

class LotCreate(LotBase):
    pass

# Схема для відповіді API (включає ставки)
class Lot(LotBase):
    id: int
    current_price: float
    status: LotStatus
    end_time: datetime.datetime
    bids: List[Bid] = []

    class Config:
        from_attributes = True