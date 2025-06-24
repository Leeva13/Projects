import asyncio
import sqlite3
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton

TOKEN = "7768615088:AAF71qmjbIhDTf7xIuR5y5SH92FogFVhReA"  # Замініть на свій токен

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Підключення до бази даних
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# Створення таблиці, якщо вона ще не існує
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        balance REAL DEFAULT 0.0
    )
""")

# Додавання нових стовпців, якщо вони відсутні
cursor.execute("PRAGMA table_info(users)")
columns = [col[1] for col in cursor.fetchall()]

if "shoe_size" not in columns:
    cursor.execute("ALTER TABLE users ADD COLUMN shoe_size REAL")
if "preferred_style" not in columns:
    cursor.execute("ALTER TABLE users ADD COLUMN preferred_style TEXT")

conn.commit()

@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer(
        "Привіт! Я бот для індивідуального підбору взуття. "
        "Використовуйте /register для реєстрації, /preferences для налаштування вподобань "
        "або /recommend для отримання рекомендацій."
    )

@dp.message(Command("register"))
async def register_user(message: Message):
    user_id = message.from_user.id
    name = message.from_user.full_name
    
    cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    if cursor.fetchone():
        await message.answer("Ви вже зареєстровані!")
    else:
        cursor.execute("INSERT INTO users (id, name) VALUES (?, ?)", (user_id, name))
        conn.commit()
        await message.answer("Ви успішно зареєстровані! Вкажіть свої вподобання за допомогою /preferences.")

@dp.message(Command("balance"))
async def show_balance(message: Message):
    user_id = message.from_user.id
    cursor.execute("SELECT balance FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    if result:
        await message.answer(f"Ваш баланс: {result[0]} грн")
    else:
        await message.answer("Ви ще не зареєстровані! Використовуйте /register.")

@dp.message(Command("preferences"))
async def set_preferences(message: Message):
    user_id = message.from_user.id
    cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    if not cursor.fetchone():
        await message.answer("Ви ще не зареєстровані! Використовуйте /register.")
        return
    
    size_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="38", callback_data="size_38"),
         InlineKeyboardButton(text="39", callback_data="size_39"),
         InlineKeyboardButton(text="40", callback_data="size_40")],
        [InlineKeyboardButton(text="41", callback_data="size_41"),
         InlineKeyboardButton(text="42", callback_data="size_42"),
         InlineKeyboardButton(text="43", callback_data="size_43")]
    ])
    await message.answer("Оберіть ваш розмір взуття:", reply_markup=size_keyboard)

@dp.callback_query(lambda c: c.data.startswith("size_"))
async def process_size(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    size = float(callback.data.split("_")[1])
    
    cursor.execute("UPDATE users SET shoe_size = ? WHERE id = ?", (size, user_id))
    conn.commit()

    style_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Спортивний", callback_data="style_sport"),
         InlineKeyboardButton(text="Класичний", callback_data="style_classic")],
        [InlineKeyboardButton(text="Кежуал", callback_data="style_casual"),
         InlineKeyboardButton(text="Модний", callback_data="style_trendy")]
    ])
    await callback.message.edit_text(f"Ви обрали розмір {size}. Оберіть стиль взуття:", reply_markup=style_keyboard)

@dp.callback_query(lambda c: c.data.startswith("style_"))
async def process_style(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    style = callback.data.split("_")[1]
    
    cursor.execute("UPDATE users SET preferred_style = ? WHERE id = ?", (style, user_id))
    conn.commit()
    
    await callback.message.edit_text(f"Ваші вподобання збережено: стиль - {style}. "
                                     "Отримайте рекомендації за допомогою /recommend.")

@dp.message(Command("recommend"))
async def recommend_shoes(message: Message):
    user_id = message.from_user.id
    cursor.execute("SELECT shoe_size, preferred_style FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    
    if not result:
        await message.answer("Ви ще не зареєстровані або не вказали вподобання! Використовуйте /register та /preferences.")
        return
    
    size, style = result
    recommendation = f"Ось рекомендація для вас:\nРозмір: {size}\nСтиль: {style}\n"
    
    if style == "sport":
        recommendation += "Рекомендуємо спортивні кросівки від Nike або Adidas."
    elif style == "classic":
        recommendation += "Рекомендуємо класичні туфлі від Clarks або Ecco."
    elif style == "casual":
        recommendation += "Рекомендуємо casual-взуття від Timberland."
    elif style == "trendy":
        recommendation += "Рекомендуємо модні кеди від Balenciaga."

    await message.answer(recommendation)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())