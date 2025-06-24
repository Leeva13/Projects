import cv2
import numpy as np
import pyautogui
import time
import keyboard
from threading import Thread, Event
from pynput.mouse import Button, Controller

# Колірний діапазон для сніжинок (у форматі HSV)
lower_color = np.array([35, 100, 100])
upper_color = np.array([40, 255, 255])

# Колірний діапазон для білого блоку (у форматі HSV)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 55, 255])

# Параметри області пошуку
screen_width, screen_height = 1920, 1080
game_width, game_height = 400, 700
game_x = (screen_width - game_width) // 2
game_y = (screen_height - game_height) // 2

# Ініціалізація контролера миші
mouse = Controller()

# Події для управління станами програми
stop_event = Event()
pause_event = Event()
restart_event = Event()  # Додамо подію для перезапуску гри

# Функція для захоплення зображення області екрана
def capture_screen():
    screenshot = pyautogui.screenshot(region=(game_x, game_y, game_width, game_height))
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot

# Функція для визначення положення сніжинок
def detect_snowflakes(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
    return centers

# Функція для визначення положення білого блоку
def detect_white_block(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 100 and h >= 50:
            return (x + w // 2, y + h // 2)
    return None

# Функція для натискання на визначені позиції
def click_positions(positions):
    for (x, y) in positions:
        if stop_event.is_set():
            break
        if pause_event.is_set():
            break
        # Перетворення координат на глобальні координати екрану
        global_x = game_x + x
        global_y = game_y + y
        # Переміщення курсора миші до координат та натискання лівої кнопки миші
        mouse.position = (global_x, global_y)
        mouse.click(Button.left, 1)

# Основна функція для гри
def play_game(duration=40):
    start_time = time.time()
    
    def click_worker():
        while time.time() - start_time < duration:
            if stop_event.is_set():
                break
            if pause_event.is_set():
                return
            screen = capture_screen()
            snowflakes = detect_snowflakes(screen)
            click_positions(snowflakes)
            time.sleep(0.05)

    # Запускаємо окремий потік для натискання на сніжинки
    click_thread = Thread(target=click_worker)
    click_thread.start()
    click_thread.join()

# Функція для перезапуску гри
def restart_game():
    time.sleep(20)  # Очікування 20 секунд перед пошуком білого блоку
    while not stop_event.is_set():
        if pause_event.is_set():
            return
        screen = capture_screen()
        white_block_center = detect_white_block(screen)
        if white_block_center:
            global_x = game_x + white_block_center[0]
            global_y = game_y + white_block_center[1]
            mouse.position = (global_x, global_y)
            mouse.click(Button.left, 1)
            break

# Основна функція для автоматизації гри
def automate_game(total_duration=6000):
    if total_duration < 60:
        print("Тривалість занадто мала. Завершення роботи.")
        return
    
    end_time = time.time() + total_duration
    while time.time() < end_time:
        if stop_event.is_set():
            break
        if pause_event.is_set():
            return
        play_game()
        restart_game()

# Функція для очікування команд від користувача
def wait_for_command():
    def check_keypress():
        while not stop_event.is_set():
            if keyboard.is_pressed('2'):
                print("Пауза.")
                pause_event.set()  # Увімкнути паузу
            elif keyboard.is_pressed('1'):
                print("Продовження/Перезапуск гри.")
                pause_event.clear()  # Зняти паузу
                restart_event.set()  # Перезапустити гру
            elif keyboard.is_pressed('3'):
                print("Завершення програми.")
                stop_event.set()  # Повне завершення
            time.sleep(0.1)

    print("Натисніть '1' для початку/продовження гри, '2' для паузи, '3' для завершення...")

    keypress_thread = Thread(target=check_keypress)
    keypress_thread.start()

    while not stop_event.is_set():
        if restart_event.is_set():  # Якщо натиснуто "1", гра перезапускається
            print("Гра починається!")
            restart_event.clear()
            automate_game()
        time.sleep(0.1)

    keypress_thread.join()

# Запуск очікування команд від користувача
wait_for_command()
