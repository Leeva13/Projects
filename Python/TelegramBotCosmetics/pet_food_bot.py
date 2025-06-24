import asyncio
import logging
import requests
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, select, func
from datetime import datetime

# Налаштування токена та бази даних
TOKEN = '765345634564:3456345dgfsty34cx1234zxc56y34'  # Замініть на ваш токен від BotFather
DATABASE_URL = 'sqlite+aiosqlite:///./bot.db'
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Визначення станів для реєстрації
class RegisterStates(StatesGroup):
    name = State()
    phone = State()

# Моделі бази даних
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    tg_id = Column(Integer, unique=True)
    name = Column(String)
    phone = Column(String)
    orders = relationship('Order')

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    price = Column(Integer)  # Ціна в копійках
    category = Column(String)

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    product_id = Column(Integer, ForeignKey('products.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship('User')
    product = relationship('Product')

# Ініціалізація бази даних
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await populate_products()

async def populate_products():
    async with async_session() as session:
        # Виправлено: Використовуємо func.count для підрахунку записів
        count = await session.scalar(select(func.count(Product.id)))
        if count == 0:
            products = [
                Product(name='Преміум корм для собак', description='Високоякісний корм для собак', price=5000, category='Корм для собак'),
                Product(name='Делюкс корм для котів', description='Смачний корм для котів', price=4500, category='Корм для котів'),
                Product(name='Зернова суміш для птахів', description='Натуральний корм для птахів', price=2000, category='Корм для птахів'),
                Product(name='Сухий корм для цуценят', description='Спеціальний корм для цуценят', price=4800, category='Корм для собак'),
            ]
            session.add_all(products)
            await session.commit()

# Функції для роботи з базою даних
async def get_user_by_tg_id(tg_id):
    async with async_session() as session:
        return await session.scalar(select(User).where(User.tg_id == tg_id))

async def add_user(tg_id, name, phone):
    async with async_session() as session:
        user = User(tg_id=tg_id, name=name, phone=phone)
        session.add(user)
        await session.commit()
        return user

async def get_categories():
    async with async_session() as session:
        result = await session.execute(select(Product.category).distinct())
        return [row[0] for row in result]

async def get_products_by_category(category):
    async with async_session() as session:
        result = await session.execute(select(Product).where(Product.category == category))
        return result.scalars().all()

async def get_product(product_id):
    async with async_session() as session:
        return await session.get(Product, product_id)

async def add_order(user_id, product_id):
    async with async_session() as session:
        order = Order(user_id=user_id, product_id=product_id)
        session.add(order)
        await session.commit()

# Клавіатури
main_menu = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text='Каталог'), KeyboardButton(text='Про нас'), KeyboardButton(text='Контакти')]],
    resize_keyboard=True
)

async def categories_keyboard():
    categories = await get_categories()
    buttons = [InlineKeyboardButton(text=cat, callback_data=f'category_{cat}') for cat in categories]
    keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons[i:i+2] for i in range(0, len(buttons), 2)])
    return keyboard

async def products_keyboard(category):
    products = await get_products_by_category(category)
    buttons = [InlineKeyboardButton(text=prod.name, callback_data=f'product_{prod.id}') for prod in products]
    keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons[i:i+2] for i in range(0, len(buttons), 2)])
    return keyboard

# Обробники
router = Router()

@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer('Вітаємо в боті для замовлення корму для тварин!', reply_markup=main_menu)

@router.message(Command('help'))
async def cmd_help(message: Message):
    await message.answer('Команди:\n/start - Почати\n/help - Допомога\n/info - Інформація\n/register - Реєстрація\n/randomdog - Фото собаки')

@router.message(Command('info'))
async def cmd_info(message: Message):
    await message.answer('Бот для замовлення корму для тварин.')

@router.message(F.text == 'Про нас')
async def about(message: Message):
    await message.answer('Магазин кормів для ваших улюбленців.')

@router.message(F.text == 'Контакти')
async def contact(message: Message):
    await message.answer('Зв’яжіться з нами: petfood@example.com')

@router.message(Command('randomdog'))
async def random_dog(message: Message):
    response = requests.get('https://dog.ceo/api/breeds/image/random')
    if response.status_code == 200:
        data = response.json()
        await message.answer_photo(data['message'])
    else:
        await message.answer('Не вдалося отримати фото.')

@router.message(F.photo)
async def handle_photo(message: Message):
    await message.answer(f'ID фото: {message.photo[-1].file_id}\nЩо бажаєте замовити для вашого улюбленця?')

@router.message(Command('register'))
async def start_register(message: Message, state: FSMContext):
    await state.set_state(RegisterStates.name)
    await message.answer('Введіть ваше ім’я:')

@router.message(RegisterStates.name)
async def process_name(message: Message, state: FSMContext):
    await state.update_data(name=message.text)
    await state.set_state(RegisterStates.phone)
    await message.answer('Введіть ваш номер телефону:')

@router.message(RegisterStates.phone)
async def process_phone(message: Message, state: FSMContext):
    data = await state.get_data()
    name = data['name']
    phone = message.text
    tg_id = message.from_user.id
    user = await get_user_by_tg_id(tg_id)
    if not user:
        await add_user(tg_id, name, phone)
    await state.clear()
    await message.answer('Реєстрація успішна!')

@router.message(F.text == 'Каталог')
async def show_catalog(message: Message):
    await message.answer('Виберіть категорію:', reply_markup=await categories_keyboard())

@router.callback_query(F.data.startswith('category_'))
async def show_products(callback: CallbackQuery):
    category = callback.data.split('_')[1]
    await callback.message.edit_text(f'Продукти в категорії {category}:', reply_markup=await products_keyboard(category))
    await callback.answer()

@router.callback_query(F.data.startswith('product_'))
async def show_product(callback: CallbackQuery):
    product_id = int(callback.data.split('_')[1])
    product = await get_product(product_id)
    if product:
        text = f'{product.name}\n{product.description}\nЦіна: {product.price / 100} грн'
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='Замовити', callback_data=f'order_{product.id}')]])
        await callback.message.edit_text(text, reply_markup=keyboard)
    await callback.answer()

@router.callback_query(F.data.startswith('order_'))
async def order_product(callback: CallbackQuery):
    product_id = int(callback.data.split('_')[1])
    user = await get_user_by_tg_id(callback.from_user.id)
    if user is None:
        await callback.message.answer('Зареєструйтесь за допомогою /register')
    else:
        product = await get_product(product_id)
        if product:
            await add_order(user.id, product_id)
            await callback.message.answer(f'Замовлення на {product.name} оформлено!')
    await callback.answer()

@router.message()
async def echo(message: Message):
    await message.answer('Використовуйте /help для списку команд.')

# Головна функція запуску
async def main():
    await init_db()
    bot = Bot(token=TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Бот зупинено')