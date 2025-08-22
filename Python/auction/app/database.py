from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# We will use environment variables for the settings. 
# Example: postgresql://user:password@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/auction_db")

# Create an engine for connecting to the database
engine = create_engine(DATABASE_URL)

# Create a session factory that will generate new sessions for each request
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# The base class for our ORM models. They will inherit it.
Base = declarative_base()

# A dependency function for retrieving a database session in endpoints
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:

        db.close()
