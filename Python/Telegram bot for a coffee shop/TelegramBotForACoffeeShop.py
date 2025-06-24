import sqlite3
import logging
from telebot import TeleBot, types
from telebot.handler_backends import State, StatesGroup
from telebot.storage import StateMemoryStorage
from dotenv import load_dotenv
import os

# Завантаження змінних із .env
load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')
ADMIN_IDS = list(map(int, os.getenv('ADMIN_IDS').split(',')))
DATABASE_NAME = os.getenv('DATABASE_NAME')

# Ініціалізація логера
logging.basicConfig(filename='bot.log', level=logging.INFO)

# Ініціалізація бота з підтримкою станів
state_storage = StateMemoryStorage()
bot = TeleBot(API_TOKEN, state_storage=state_storage)

# Визначення станів
class OrderStatusStates(StatesGroup):
    select_order = State()
    new_status = State()

# Клас для роботи з базою даних
class Database:
    def __init__(self, db_name):
        self.db_name = db_name
        self._init_tables()

    def _get_connection(self):
        """Створює нове з'єднання з базою даних."""
        return sqlite3.connect(self.db_name, check_same_thread=True)

    def _init_tables(self):
        """Ініціалізує таблиці в базі даних, якщо вони ще не створені."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Видаляємо стару таблицю orders, щоб уникнути конфліктів
            cursor.execute('DROP TABLE IF EXISTS orders')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    price REAL NOT NULL
                )''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    username TEXT,
                    product_id INTEGER NOT NULL,
                    status TEXT DEFAULT 'new'
                )''')
            conn.commit()

    def get_products(self):
        """Отримує список усіх товарів із бази даних."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM products')
            return cursor.fetchall()

    def get_product(self, product_id):
        """Отримує інформацію про товар за його ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
            return cursor.fetchone()

    def add_product(self, name, description, price):
        """Додає новий товар до бази даних."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO products (name, description, price) VALUES (?, ?, ?)',
                           (name, description, price))
            conn.commit()

    def add_order(self, user_id, username, product_id):
        """Додає нове замовлення до бази даних."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO orders (user_id, username, product_id) VALUES (?, ?, ?)',
                           (user_id, username if username else "Невідомий", product_id))
            conn.commit()

    def get_user_orders(self, user_id):
        """Отримує список замовлень користувача за його ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT orders.id, products.name, orders.status 
                FROM orders 
                JOIN products ON orders.product_id = products.id 
                WHERE orders.user_id = ?
            ''', (user_id,))
            return cursor.fetchall()

    def get_all_orders(self):
        """Отримує список усіх замовлень."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT orders.id, orders.user_id, orders.username, products.name, orders.status 
                    FROM orders 
                    JOIN products ON orders.product_id = products.id
                ''')
                return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error in get_all_orders: {e}")
            return []

    def update_order_status(self, order_id, new_status):
        """Оновлює статус замовлення за його ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE orders SET status = ? WHERE id = ?', (new_status, order_id))
            conn.commit()

    def get_order_user_id(self, order_id):
        """Отримує ID користувача, який зробив замовлення, за ID замовлення."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id FROM orders WHERE id = ?', (order_id,))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_order_status(self, order_id):
        """Отримує статус замовлення за його ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT status FROM orders WHERE id = ?', (order_id,))
            result = cursor.fetchone()
            return result[0] if result else None

    def delete_order(self, order_id):
        """Видаляє замовлення за його ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM orders WHERE id = ?', (order_id,))
            conn.commit()

# Клас для обробки логіки бота
class BotHandler:
    def __init__(self, bot, admin_ids, db):
        self.bot = bot
        self.admin_ids = admin_ids
        self.db = db
        self.status_options = ['new', 'preparing', 'done']
        self.deal_of_the_day = "Сьогоднішня акція: Капучино зі знижкою 20%! ☕"

    def get_main_keyboard(self, user_id):
        """Створює головне меню для користувача, додаючи адмін-панель для адміністраторів."""
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
        keyboard.add('📖 Меню', '🛍 Мої замовлення')
        keyboard.add('🎁 Акція дня', '📍 Наші контакти')
        if user_id in self.admin_ids:
            keyboard.add('🔧 Адмін-панель')
        return keyboard

    def log_action(self, user_id, action):
        """Логує дії користувача."""
        logging.info(f"User {user_id} performed action: {action}")

    def start(self, message):
        """Обробляє команду /start, вітаючи користувача."""
        user = message.from_user
        welcome_text = (
            f"Вітаємо, {user.first_name}!\n"
            "Це бот кав'ярні 'Sliwka' ☕\n"
            "Оберіть дію:"
        )
        self.bot.send_message(message.chat.id, welcome_text, reply_markup=self.get_main_keyboard(user.id))
        self.log_action(user.id, 'start')

    def show_menu(self, message):
        """Показує меню з доступними товарами."""
        products = self.db.get_products()
        if not products:
            # Додаємо більше товарів за замовчуванням
            self.db.add_product("Еспресо", "Класична кава", 40.0)
            self.db.add_product("Капучино", "Кава з молочною пінкою", 50.0)
            self.db.add_product("Лате", "Кава з молоком", 55.0)
            self.db.add_product("Американо", "Кава з водою", 45.0)
            self.db.add_product("Чай зелений", "Освіжаючий зелений чай", 35.0)
            self.db.add_product("Гарячий шоколад", "Теплий напій із шоколадом", 60.0)
            self.db.add_product("Круасан з шоколадом", "Свіжий круасан", 35.0)
            self.db.add_product("Круасан із сиром", "Свіжий круасан із сиром", 40.0)
            self.db.add_product("Круасан із мигдалем", "Свіжий круасан із мигдалевою начинкою", 45.0)
            self.db.add_product("Тістечко 'Макарон'", "Смаколик із мигдалевого борошна", 25.0)
            self.db.add_product("Тістечко 'Наполеон'", "Класичний десерт", 60.0)
            self.db.add_product("Чізкейк", "Ніжний чізкейк із ягодами", 70.0)
            self.db.add_product("Смузі манго", "Освіжаючий напій", 70.0)
            self.db.add_product("Лимонад", "Освіжаючий лимонний напій", 50.0)
            products = self.db.get_products()
        
        markup = types.InlineKeyboardMarkup()
        for product in products:
            markup.add(types.InlineKeyboardButton(
                text=f"{product[1]} - {product[3]}₴",
                callback_data=f"product_{product[0]}"
            ))
        self.bot.send_message(message.chat.id, "Наше меню:", reply_markup=markup)

    def product_details(self, call):
        """Показує деталі товару після вибору з меню."""
        product_id = call.data.split('_')[1]
        product = self.db.get_product(product_id)
        if product:
            text = (
                f"☕ {product[1]}\n\n"
                f"{product[2]}\n\n"
                f"Ціна: {product[3]}₴"
            )
            markup = types.InlineKeyboardMarkup()
            markup.add(types.InlineKeyboardButton(
                "Замовити 🛒", callback_data=f"order_{product_id}"
            ))
            self.bot.edit_message_text(
                text, call.message.chat.id, call.message.message_id, reply_markup=markup
            )

    def process_order(self, call):
        """Обробляє замовлення користувача."""
        try:
            product_id = int(call.data.split('_')[1])  # Переконаємося, що product_id є числом
            user_id = call.from_user.id
            username = call.from_user.username if call.from_user.username else call.from_user.first_name
            product = self.db.get_product(product_id)
            if not product:
                raise ValueError(f"Товар з ID {product_id} не знайдено!")
            
            self.db.add_order(user_id, username, product_id)
            admin_text = (
                "Нове замовлення!\n"
                f"Користувач: @{username}\n"
                f"Товар: {product[1]}\n"
                f"Ціна: {product[3]}₴"
            )
            for admin in self.admin_ids:
                self.bot.send_message(admin, admin_text)
            self.bot.answer_callback_query(call.id, "Ваше замовлення прийнято! ✅")
            self.log_action(user_id, f"Ordered product ID {product_id}")
        except (sqlite3.Error, ValueError) as e:
            logging.error(f"Order error: {e}")
            self.bot.answer_callback_query(call.id, f"Помилка при обробці замовлення: {str(e)}")

    def show_user_orders(self, message):
        """Показує список замовлень користувача з можливістю скасування."""
        user_id = message.from_user.id
        orders = self.db.get_user_orders(user_id)
        if not orders:
            self.bot.send_message(message.chat.id, "У вас немає замовлень.")
            return
        
        text = "📋 Ваші замовлення:\n\n"
        markup = types.InlineKeyboardMarkup()
        for order in orders:
            order_id, name, status = order
            text += f"#{order_id} | {name} | Статус: {status}\n"
            if status == 'new':
                markup.add(types.InlineKeyboardButton(
                    f"Скасувати #{order_id}", callback_data=f"cancel_order_{order_id}"
                ))
        self.bot.send_message(message.chat.id, text, reply_markup=markup)

    def cancel_order(self, call):
        """Скасовує замовлення, якщо його статус 'new'."""
        order_id = call.data.split('_')[2]
        user_id = call.from_user.id
        status = self.db.get_order_status(order_id)
        
        if status != 'new':
            self.bot.answer_callback_query(call.id, "Скасувати можна лише замовлення зі статусом 'new'!")
            return
        
        order_user_id = self.db.get_order_user_id(order_id)
        if order_user_id != user_id:
            self.bot.answer_callback_query(call.id, "Ви не можете скасувати чуже замовлення!")
            return
        
        self.db.delete_order(order_id)
        self.bot.answer_callback_query(call.id, f"Замовлення #{order_id} скасовано! 🗑️")
        self.log_action(user_id, f"Canceled order #{order_id}")

    def show_deal_of_the_day(self, message):
        """Показує акцію дня."""
        self.bot.send_message(message.chat.id, self.deal_of_the_day)

    def show_contacts(self, message):
        """Показує контактну інформацію кав'ярні."""
        contacts = (
            "📍 Наші контакти:\n"
            "Адреса: вул. Кавова, 1, м. Київ\n"
            "Графік роботи: Пн-Нд, 8:00-20:00\n"
            "Телефон: +380 123 456 789\n"
            "Слідкуйте за нами в Instagram: @sliwka_coffee"
        )
        self.bot.send_message(message.chat.id, contacts)

    def admin_panel(self, message):
        """Відкриває адмін-панель для адміністраторів."""
        if message.from_user.id not in self.admin_ids:
            self.bot.reply_to(message, "Доступ заборонено ⛔")
            return
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add('Переглянути замовлення 📦', 'Змінити статус замовлення 🔄')
        markup.add('Назад ◀️')
        self.bot.send_message(message.chat.id, "Адмін-панель:", reply_markup=markup)

    def back_to_main_menu(self, message):
        """Повертає користувача до головного меню."""
        self.bot.send_message(message.chat.id, "Повертаємося до головного меню:", reply_markup=self.get_main_keyboard(message.from_user.id))

    def show_all_orders(self, message):
        """Показує всі замовлення для адміністратора."""
        if message.from_user.id not in self.admin_ids:
            self.bot.reply_to(message, "Доступ заборонено ⛔")
            return
        orders = self.db.get_all_orders()
        if not orders:
            self.bot.send_message(message.chat.id, "Немає активних замовлень.")
            return
        text = "📋 Усі замовлення:\n\n"
        for order in orders:
            order_id, user_id, username, product_name, status = order
            display_name = f"@{username}" if username and username != "Невідомий" else f"ID: {user_id}"
            text += f"#{order_id} | Користувач: {display_name} | {product_name} | Статус: {status}\n"
        self.bot.send_message(message.chat.id, text)

    def change_order_status_start(self, message):
        """Починає процес зміни статусу замовлення."""
        if message.from_user.id not in self.admin_ids:
            self.bot.reply_to(message, "Доступ заборонено ⛔")
            return
        orders = self.db.get_all_orders()
        if not orders:
            self.bot.send_message(message.chat.id, "Немає активних замовлень.")
            return
        markup = types.InlineKeyboardMarkup()
        for order in orders:
            markup.add(types.InlineKeyboardButton(
                f"#{order[0]} | {order[3]} | Статус: {order[4]}",
                callback_data=f"change_status_{order[0]}"
            ))
        self.bot.send_message(message.chat.id, "Оберіть замовлення для зміни статусу:", reply_markup=markup)

    def select_order_status(self, call):
        """Дозволяє обрати новий статус для замовлення."""
        order_id = call.data.split('_')[2]
        markup = types.InlineKeyboardMarkup()
        for status in self.status_options:
            markup.add(types.InlineKeyboardButton(status.capitalize(), callback_data=f"set_status_{order_id}_{status}"))
        self.bot.edit_message_text(
            "Оберіть новий статус:", call.message.chat.id, call.message.message_id, reply_markup=markup
        )

    def set_order_status(self, call):
        """Оновлює статус замовлення та сповіщає користувача."""
        order_id = call.data.split('_')[2]
        new_status = call.data.split('_')[3]
        self.db.update_order_status(order_id, new_status)
        user_id = self.db.get_order_user_id(order_id)
        if user_id:
            self.bot.send_message(user_id, f"Статус вашого замовлення #{order_id} змінено на: {new_status.capitalize()}")
        self.bot.answer_callback_query(call.id, "Статус оновлено!")
        self.log_action(call.from_user.id, f"Changed order #{order_id} status to {new_status}")

# Ініціалізація об'єктів
db = Database(DATABASE_NAME)
handler = BotHandler(bot, ADMIN_IDS, db)

# Реєстрація обробників
bot.message_handler(commands=['start'])(handler.start)
bot.message_handler(func=lambda msg: msg.text == '📖 Меню')(handler.show_menu)
bot.message_handler(func=lambda msg: msg.text == '🛍 Мої замовлення')(handler.show_user_orders)
bot.message_handler(func=lambda msg: msg.text == '🎁 Акція дня')(handler.show_deal_of_the_day)
bot.message_handler(func=lambda msg: msg.text == '📍 Наші контакти')(handler.show_contacts)
bot.callback_query_handler(func=lambda call: call.data.startswith('product_'))(handler.product_details)
bot.callback_query_handler(func=lambda call: call.data.startswith('order_'))(handler.process_order)
bot.callback_query_handler(func=lambda call: call.data.startswith('cancel_order_'))(handler.cancel_order)
bot.message_handler(commands=['admin'])(handler.admin_panel)
bot.message_handler(func=lambda msg: msg.text == '🔧 Адмін-панель')(handler.admin_panel)
bot.message_handler(func=lambda msg: msg.text == 'Назад ◀️')(handler.back_to_main_menu)
bot.message_handler(func=lambda msg: msg.text == 'Переглянути замовлення 📦')(handler.show_all_orders)
bot.message_handler(func=lambda msg: msg.text == 'Змінити статус замовлення 🔄')(handler.change_order_status_start)
bot.callback_query_handler(func=lambda call: call.data.startswith('change_status_'))(handler.select_order_status)
bot.callback_query_handler(func=lambda call: call.data.startswith('set_status_'))(handler.set_order_status)

# Запуск бота
if __name__ == '__main__':
    print("Бот запущений...")
    bot.polling(none_stop=True)