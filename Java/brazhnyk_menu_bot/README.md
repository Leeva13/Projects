# MyTelegramBot

Простий Telegram-бот на Java з інтерактивним меню та inline-кнопками.

## Опис

Бот дозволяє:
- Відкривати головне меню командою `/start`.
- Перемикатися між головним та другим меню.
- Отримувати текстові відповіді на натискання кнопок.

Використовує бібліотеку `telegrambots` та сучасний синтаксис Java.

## Вимоги

- Java 17+
- Maven
- Залежність: `org.telegram:telegrambots` (6.9.7.1 або новіша)


## Використання

- Знайдіть бота в Telegram за `@YourBotUsername`.
- Надішліть `/start` для головного меню.
- Користуйтеся кнопками:
  - "Кнопка 1" / "Кнопка 2" – текстові відповіді.
  - "Далі" – друге меню.
  - "Назад" – повернення до головного меню.

## Скріншот
![Приклад використання](/images/pryklad-vykorystannya.png)

## Структура

- `src/main/java/org/example/MyBot.java` – логіка бота.
- `src/main/java/org/example/BotConfig.java` – конфігурація.
- `pom.xml` – залежності Maven.

## Встановлення

1. Клонувати репозиторій:
   ```bash
   git clone https://github.com/ваш-username/MyTelegramBot.git  
   
2. Перейти до папки:
   ```
   cd MyTelegramBot

3. Налаштувати BotConfig.java:
    ```
   public class BotConfig {
    private static final String BOT_USERNAME = "YourBotUsername";
    private static final String BOT_TOKEN = "YourBotToken";
    public static String getBotUsername() { return BOT_USERNAME; }
    public static String getBotToken() { return BOT_TOKEN; } 
   }

4. Зібрати проєкт:
    ```
   mvn clean install

5. Запустити `бота`

