import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
import sqlite3

TOKEN = '7768615088:AAF71qmjbIhDTf7xIuR5y5SH92FogFVhReA'  # Заміни на свій API-токен
bot = telebot.TeleBot(TOKEN)

def init_db():
    conn = sqlite3.connect('repair_bot_db.sqlite')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                      id INTEGER PRIMARY KEY,
                      username TEXT,
                      phone TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS orders (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      service TEXT,
                      status TEXT DEFAULT 'Очікує підтвердження')''')
    conn.commit()
    conn.close()

@bot.message_handler(commands=['start'])
def start_message(message):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(KeyboardButton('🔧 Послуги ремонту'),
               KeyboardButton('📋 Мої замовлення'),
               KeyboardButton('📞 Зв’язатися з майстром'))
    bot.send_message(message.chat.id, f'Привіт, {message.from_user.first_name}! Обери опцію:', reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == '🔧 Послуги ремонту')
def show_services(message):
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton('📱 Заміна екрану', callback_data='order_screen'))
    markup.add(InlineKeyboardButton('🔋 Заміна батареї', callback_data='order_battery'))
    markup.add(InlineKeyboardButton('🔌 Ремонт роз’єму зарядки', callback_data='order_connector'))
    bot.send_message(message.chat.id, 'Оберіть послугу:', reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith('order_'))
def order_service(call):
    service = call.data.split('_')[1]
    service_dict = {
        'screen': 'Заміна екрану',
        'battery': 'Заміна батареї',
        'connector': 'Ремонт роз’єму зарядки'
    }
    service_name = service_dict.get(service, service)
    conn = sqlite3.connect('repair_bot_db.sqlite')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO orders (user_id, service) VALUES (?, ?)', (call.from_user.id, service_name))
    conn.commit()
    conn.close()
    bot.send_message(call.message.chat.id, f'✅ Ваше замовлення "{service_name}" прийнято!')

@bot.message_handler(func=lambda message: message.text == '📋 Мої замовлення')
def my_orders(message):
    conn = sqlite3.connect('repair_bot_db.sqlite')
    cursor = conn.cursor()
    cursor.execute('SELECT service, status FROM orders WHERE user_id = ?', (message.from_user.id,))
    orders = cursor.fetchall()
    conn.close()
    if orders:
        response = '\n'.join([f'🔧 {o[0]} - {o[1]}' for o in orders])
    else:
        response = 'У вас поки що немає замовлень.'
    bot.send_message(message.chat.id, response)

@bot.message_handler(func=lambda message: message.text == '📞 Зв’язатися з майстром')
def contact_master(message):
    bot.send_message(message.chat.id, '📱 Телефон майстра: +380 987 654 321')

if __name__ == '__main__':
    init_db()
    bot.polling(none_stop=True)