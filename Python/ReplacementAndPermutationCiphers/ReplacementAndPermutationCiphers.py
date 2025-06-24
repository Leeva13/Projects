# Лабораторна робота №5: Шифри заміни і перестановки
# Реалізація на Python для всіх завдань практичної роботи №5

# Імпортуємо потрібні бібліотеки
import string

# Задаємо алфавіт (латинський + пробіл)
alphabet = string.ascii_uppercase + " "

# Функція для видалення символів, яких немає в алфавіті
def clean_text(text, alphabet):
    return ''.join(c for c in text.upper() if c in alphabet)

# Завдання 1: Шифр однобуквеної заміни
def monoalphabetic_cipher(text, substitution):
    text = clean_text(text, alphabet)
    result = ""
    for char in text:
        idx = alphabet.index(char)
        result += substitution[idx]
    return result

# Завдання 2: Гасловий шифр
def keyword_cipher(text, keyword):
    text = clean_text(text, alphabet)
    # Формуємо ключ: спочатку гасло, потім решта алфавіту без повторів
    keyword = clean_text(keyword, alphabet)
    substitution = keyword
    for char in alphabet:
        if char not in keyword:
            substitution += char
    # Шифруємо
    result = ""
    for char in text:
        idx = alphabet.index(char)
        result += substitution[idx]
    return result

# Завдання 3: Шифр Віженера
def vigenere_cipher(text, key):
    text = clean_text(text, alphabet)
    key = clean_text(key, alphabet)
    result = ""
    key_idx = 0
    for char in text:
        p = alphabet.index(char)  # Номер символу в тексті
        k = alphabet.index(key[key_idx % len(key)])  # Номер символу ключа
        c = (p + k) % len(alphabet)  # Зашифрований символ
        result += alphabet[c]
        key_idx += 1
    return result

# Завдання 4: Шифр Плейфера
def playfair_cipher(text, key_matrix):
    text = clean_text(text, alphabet).replace(" ", "")  # Видаляємо пробіли
    # Додаємо 'X' між однаковими буквами та в кінці, якщо довжина непарна
    i = 0
    new_text = ""
    while i < len(text):
        if i + 1 < len(text) and text[i] == text[i + 1]:
            new_text += text[i] + "X"
            i += 1
        else:
            new_text += text[i]
            i += 1
        if i == len(text) - 1:
            new_text += text[i] + "X"
            break
    if len(new_text) % 2 != 0:
        new_text += "X"

    # Шифруємо біграми
    result = ""
    for i in range(0, len(new_text), 2):
        a, b = new_text[i], new_text[i + 1]
        row1, col1 = [(r, c) for r in range(6) for c in range(5) if key_matrix[r][c] == a][0]
        row2, col2 = [(r, c) for r in range(6) for c in range(5) if key_matrix[r][c] == b][0]
        
        if row1 == row2:  # Одна горизонталь
            result += key_matrix[row1][(col1 + 1) % 5] + key_matrix[row2][(col2 + 1) % 5]
        elif col1 == col2:  # Одна вертикаль
            result += key_matrix[(row1 + 1) % 6][col1] + key_matrix[(row2 + 1) % 6][col2]
        else:  # Прямокутник
            result += key_matrix[row1][col2] + key_matrix[row2][col1]
    return result

# Завдання 5: Шифр вертикальної перестановки
def vertical_permutation_cipher(text, key):
    text = clean_text(text, alphabet)
    key = clean_text(key, alphabet)
    cols = len(key)
    rows = (len(text) + cols - 1) // cols  # Округлення вгору
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]
    
    # Заповнюємо таблицю текстом
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < len(text):
                grid[r][c] = text[idx]
                idx += 1
    
    # Сортуємо стовпці за ключем
    key_order = sorted(range(cols), key=lambda k: key[k])
    result = ""
    for col in key_order:
        for row in range(rows):
            result += grid[row][col]
    return result

# Приклад виконання
print("Лабораторна робота №5: Шифри заміни і перестановки")
text = "BRAZHNYK"

# Завдання 1
substitution = "KWRHPBTNUDOZEF.CYVGAIXMQLS"  # Приклад ключа (з прикладу в документі)
print("\nЗавдання 1 (Однобуквена заміна):")
print(f"Відкритий текст: {text}")
print(f"Криптограма: {monoalphabetic_cipher(text, substitution)}")

# Завдання 2
keyword = "SLOGAN"
print("\nЗавдання 2 (Гасловий шифр):")
print(f"Відкритий текст: {text}")
print(f"Криптограма: {keyword_cipher(text, keyword)}")

# Завдання 3
vigenere_key = "BITCOIN"
print("\nЗавдання 3 (Шифр Віженера):")
print(f"Відкритий текст: {text}")
print(f"Криптограма: {vigenere_cipher(text, vigenere_key)}")

# Завдання 4
playfair_key = [
    ['K', 'W', 'R', 'H', ','],
    ['P', 'T', 'B', 'N', 'U'],
    [' ', 'D', 'O', 'Z', 'E'],
    ['J', 'F', '.', 'C', 'Y'],
    ['V', 'G', 'A', 'I', 'X'],
    ['M', '-', 'Q', 'L', 'S']
]
print("\nЗавдання 4 (Шифр Плейфера):")
print(f"Відкритий текст: {text}")
print(f"Криптограма: {playfair_cipher(text, playfair_key)}")

# Завдання 5
long_text = "BRAZHNYK ARTEM MYKHAILOVYCH"  # Довжина > 50 символів можлива з пробілами
vertical_key = "FOREVER"
print("\nЗавдання 5 (Вертикальна перестановка):")
print(f"Відкритий текст: {long_text}")
print(f"Криптограма: {vertical_permutation_cipher(long_text, vertical_key)}")
