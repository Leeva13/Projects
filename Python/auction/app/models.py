# app/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Enum as SQLAlchemyEnum
from sqlalchemy.orm import relationship
from .database import Base
import datetime
import enum

# Create an Enum for the lot status
class LotStatus(str, enum.Enum):
    running = "running"
    ended = "ended"

class Lot(Base):
    __tablename__ = "lots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(String, nullable=True)
    start_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    status = Column(SQLAlchemyEnum(LotStatus), default=LotStatus.running)
    end_time = Column(DateTime, nullable=False)
    
    # One-to-many relationship: one lot can have many bids
    bids = relationship("Bid", back_populates="lot")

class Bid(Base):
    __tablename__ = "bids"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    bidder = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Foreign key referring to the 'lots' table
    lot_id = Column(Integer, ForeignKey("lots.id"))

    # Feedback to the lot

    lot = relationship("Lot", back_populates="bids")
