import requests
from bs4 import BeautifulSoup
import csv
import time

# Функція для збору цитат з однієї сторінки
def scrape_page(url):
    try:
        # Надсилаємо запит
        response = requests.get(url)
        response.raise_for_status()  # Перевіряємо, чи запит успішний
        soup = BeautifulSoup(response.text, 'html.parser')

        # Знаходимо всі блоки з цитатами
        quotes = soup.find_all('div', class_='quote')
        if not quotes:
            print(f"Цитати не знайдені на {url}")
            return []

        # Список для зберігання даних
        quotes_data = []

        # Обробляємо кожну цитату
        for quote in quotes:
            text = quote.find('span', class_='text').text.strip()  # Текст цитати
            author = quote.find('small', class_='author').text.strip()  # Автор
            tags = [tag.text for tag in quote.find_all('a', class_='tag')]  # Список тегів

            # Додаємо дані до списку
            quotes_data.append({
                'text': text,
                'author': author,
                'tags': ', '.join(tags)  # Об'єднуємо теги в рядок
            })

        return quotes_data

    except requests.RequestException as e:
        print(f"Помилка запиту до {url}: {e}")
        return []

# Функція для отримання посилання на наступну сторінку
def get_next_page(soup):
    next_button = soup.find('li', class_='next')
    if next_button:
        next_link = next_button.find('a')['href']  # Отримуємо значення href
        return 'https://quotes.toscrape.com' + next_link
    return None

# Основна функція
def main():
    base_url = 'https://quotes.toscrape.com/'
    all_quotes = []  # Список для всіх цитат
    current_url = base_url

    # Проходимо по всіх сторінках
    while current_url:
        print(f"Скрапимо сторінку: {current_url}")
        quotes = scrape_page(current_url)
        all_quotes.extend(quotes)  # Додаємо цитати з поточної сторінки

        # Отримуємо HTML для пошуку наступної сторінки
        response = requests.get(current_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        current_url = get_next_page(soup)  # Оновлюємо URL

        # Затримка, щоб не перевантажувати сервер
        time.sleep(2)

    # Зберігаємо дані у CSV
    if all_quotes:
        with open('quotes.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['text', 'author', 'tags'])
            writer.writeheader()  # Записуємо заголовки
            writer.writerows(all_quotes)  # Записуємо дані
        print(f"Збережено {len(all_quotes)} цитат у файл quotes.csv")
    else:
        print("Не вдалося зібрати жодної цитати")

# Запускаємо скрапер
if __name__ == "__main__":
    main()