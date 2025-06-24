import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
import sqlite3

TOKEN = '7768615088:AAF71qmjbIhDTf7xIuR5y5SH92FogFVhReA'
bot = telebot.TeleBot(TOKEN)

def init_db():
    conn = sqlite3.connect('bot_db.sqlite')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                      id INTEGER PRIMARY KEY,
                      username TEXT,
                      phone TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS orders (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      product TEXT,
                      status TEXT DEFAULT 'Очікує підтвердження')''')
    conn.commit()
    conn.close()

@bot.message_handler(commands=['start'])
def start_message(message):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(KeyboardButton('📋 Каталог косметики'),
               KeyboardButton('🛒 Мої замовлення'),
               KeyboardButton('📞 Зв’язатися з менеджером'))
    bot.send_message(message.chat.id, f'Привіт, {message.from_user.first_name}! Обери опцію:', reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == '📋 Каталог косметики')
def show_catalog(message):
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton('💄 Помада', callback_data='order_lipstick'))
    markup.add(InlineKeyboardButton('👁 Туш для вій', callback_data='order_mascara'))
    markup.add(InlineKeyboardButton('🧴 Крем для обличчя', callback_data='order_cream'))
    bot.send_message(message.chat.id, 'Оберіть категорію:', reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith('order_'))
def order_product(call):
    product = call.data.split('_')[1]
    conn = sqlite3.connect('bot_db.sqlite')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO orders (user_id, product) VALUES (?, ?)', (call.from_user.id, product))
    conn.commit()
    conn.close()
    bot.send_message(call.message.chat.id, f'✅ Ваше замовлення "{product}" прийнято!')

@bot.message_handler(func=lambda message: message.text == '🛒 Мої замовлення')
def my_orders(message):
    conn = sqlite3.connect('bot_db.sqlite')
    cursor = conn.cursor()
    cursor.execute('SELECT product, status FROM orders WHERE user_id = ?', (message.from_user.id,))
    orders = cursor.fetchall()
    conn.close()
    if orders:
        response = '\n'.join([f'🛍 {o[0]} - {o[1]}' for o in orders])
    else:
        response = 'У вас поки що немає замовлень.'
    bot.send_message(message.chat.id, response)

@bot.message_handler(func=lambda message: message.text == '📞 Зв’язатися з менеджером')
def contact_manager(message):
    bot.send_message(message.chat.id, '📱 Телефон менеджера: +380 123 456 789')

if __name__ == '__main__':
    init_db()
    bot.polling(none_stop=True)
