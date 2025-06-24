import asyncio
import logging
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command

TOKEN = "7768615088:AAF71qmjbIhDTf7xIuR5y5SH92FogFVhReA"
WEATHER_API_KEY = "cb7d7fd5b044e862d0a5fa92442b806a"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

bot = Bot(token=TOKEN)
dp = Dispatcher()

async def get_weather(city: str):
    params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric", "lang": "ua"}
    async with aiohttp.ClientSession() as session:
        async with session.get(BASE_URL, params=params) as response:
            if response.status == 200:
                data = await response.json()
                temp = data["main"]["temp"]
                desc = data["weather"][0]["description"].capitalize()
                return f"Погода в {city}: {desc}, {temp}°C"
            else:
                return "Не вдалося отримати погоду. Спробуйте інше місто."

@dp.message(Command("start"))
async def start_command(message: types.Message):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Київ", callback_data="weather_Kyiv")],
        [InlineKeyboardButton(text="Львів", callback_data="weather_Lviv")]
    ])
    await message.answer("Привіт! Обери місто, щоб дізнатися погоду:", reply_markup=keyboard)

@dp.callback_query()
async def handle_callback(query: types.CallbackQuery):
    if query.data.startswith("weather_"):
        city = query.data.split("_")[1]
        weather_info = await get_weather(city)
        await query.message.edit_text(weather_info)
    await query.answer()

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())