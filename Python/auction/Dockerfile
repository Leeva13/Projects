# 1. Використовуємо офіційний, легкий образ Python як основу
FROM python:3.10-slim

# 2. Встановлюємо робочу директорію всередині контейнера
WORKDIR /

# 3. Копіюємо файл залежностей та встановлюємо їх
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копіюємо код додатку
COPY app /app

# 5. Вказуємо команду для запуску
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]