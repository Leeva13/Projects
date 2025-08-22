# app/models.py (updated)
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Enum as SQLAlchemyEnum
from sqlalchemy.orm import relationship
from .database import Base
import datetime
import enum

# Створюємо Enum для статусу лота
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
    
    # Зв'язок "один до багатьох": один лот може мати багато ставок
    bids = relationship("Bid", back_populates="lot")

class Bid(Base):
    __tablename__ = "bids"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    bidder = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Зовнішній ключ, що посилається на таблицю 'lots'
    lot_id = Column(Integer, ForeignKey("lots.id"))

    # Зворотний зв'язок до лота
    lot = relationship("Lot", back_populates="bids")