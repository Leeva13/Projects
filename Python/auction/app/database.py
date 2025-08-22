from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Ми будемо використовувати змінні оточення для налаштувань. 
# Це безпечніше, ніж жорстко кодувати дані в коді.
# Приклад: postgresql://user:password@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/auction_db")

# Створюємо "двигун" для підключення до БД
engine = create_engine(DATABASE_URL)

# Створюємо фабрику сесій, яка буде генерувати нові сесії для кожного запиту
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовий клас для наших моделей ORM. Вони будуть успадковувати його.
Base = declarative_base()

# Функція-залежність (dependency) для отримання сесії БД у ендпоінтах
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()