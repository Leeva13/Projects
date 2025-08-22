# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional
import datetime
from .models import LotStatus

# --- Betting Schemes (Bid) ---

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

# --- Schemes for Lots (Lot) ---

class LotBase(BaseModel):
    name: str
    description: Optional[str] = None
    start_price: float
    duration_minutes: Optional[int] = 5

class LotCreate(LotBase):
    pass

# Schema for API response (includes bets)
class Lot(LotBase):
    id: int
    current_price: float
    status: LotStatus
    end_time: datetime.datetime
    bids: List[Bid] = []

    class Config:

        from_attributes = True
