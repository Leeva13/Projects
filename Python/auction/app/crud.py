# app/crud.py
from sqlalchemy.orm import Session
from .models import Lot, Bid, LotStatus
from .schemas import LotCreate, BidCreate
from datetime import datetime, timedelta
from fastapi import HTTPException

def create_lot(db: Session, lot: LotCreate):
    end_time = datetime.utcnow() + timedelta(minutes=lot.duration_minutes)
    db_lot = Lot(
        name=lot.name,
        description=lot.description,
        start_price=lot.start_price,
        current_price=lot.start_price,
        status=LotStatus.running,
        end_time=end_time
    )
    db.add(db_lot)
    db.commit()
    db.refresh(db_lot)
    return db_lot

def get_active_lots(db: Session):
    now = datetime.utcnow()
    lots = db.query(Lot).filter(Lot.status == LotStatus.running).all()
    active = []
    for l in lots:
        if l.end_time < now:
            l.status = LotStatus.ended
            db.commit()
        else:
            active.append(l)
    return active

def get_lot(db: Session, lot_id: int):
    lot = db.query(Lot).filter(Lot.id == lot_id).first()
    if lot and lot.status == LotStatus.running:
        now = datetime.utcnow()
        if lot.end_time < now:
            lot.status = LotStatus.ended
            db.commit()
    return lot

def create_bid(db: Session, lot_id: int, bid: BidCreate):
    lot = get_lot(db, lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail="Lot not found")
    if lot.status != LotStatus.running:
        raise HTTPException(status_code=400, detail="Auction has ended")
    if bid.amount <= lot.current_price:
        raise HTTPException(status_code=400, detail="Bid amount must be higher than current price")
    now = datetime.utcnow()
    extended = False
    new_end_time = None
    if lot.end_time - now < timedelta(minutes=1):
        lot.end_time += timedelta(minutes=1)
        extended = True
        new_end_time = lot.end_time
    db_bid = Bid(
        amount=bid.amount,
        bidder=bid.bidder,
        lot_id=lot_id,
        timestamp=now
    )
    lot.bids.append(db_bid)
    lot.current_price = bid.amount
    db.add(db_bid)
    db.commit()
    db.refresh(db_bid)
    return db_bid, extended, new_end_time