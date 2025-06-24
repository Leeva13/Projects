import logging
import asyncio
import requests
from aiogram import Bot, Dispatcher, types, Router, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

API_TOKEN = "7982410644:AAH5Wj4rcgJpHt1gQT7GiAAYBw4wl4Rezc0"
WEATHER_API_KEY = "979cf328dcfd6962840c6c2219cece30"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

bot = Bot(token=API_TOKEN)
dp = Dispatcher()
router = Router()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Головне меню
main_kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="🌦 Погода"), KeyboardButton(text="📞 Зв'язатися")]],
    resize_keyboard=True
)

@router.message(F.text == "/start")
async def start(message: types.Message):
    await message.answer("Вітаємо! Натисніть 'Погода', щоб отримати прогноз.", reply_markup=main_kb)

@router.message(F.text == "/help")
async def help_command(message: types.Message):
    await message.answer("Команди: \n/start - Почати \n/help - Допомога \n/weather - Дізнатися погоду")

@router.message(F.text == "🌦 Погода")
async def get_weather(message: types.Message):
    city = "Kyiv"  # За замовчуванням
    params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric", "lang": "uk"}
    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        temp = data['main']['temp']
        description = data['weather'][0]['description']
        await message.answer(f"Погода у {city}: {temp}°C, {description}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Помилка отримання погоди: {e}")
        await message.answer("Не вдалося отримати дані про погоду. Спробуйте пізніше.")

@router.message(F.text == "📞 Зв'язатися")
async def contact_info(message: types.Message):
    await message.answer("Наші контакти: 📧 email@example.com, 📞 +380123456789")

async def main():
    try:
        dp.include_router(router)
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("Бот запущено")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Критична помилка: {e}")
    finally:
        logger.info("Бот завершив роботу")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот зупинено вручну")