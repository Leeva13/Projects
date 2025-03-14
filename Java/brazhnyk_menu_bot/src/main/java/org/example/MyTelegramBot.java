package org.example;

import org.telegram.telegrambots.bots.TelegramLongPollingBot;
import org.telegram.telegrambots.meta.TelegramBotsApi;
import org.telegram.telegrambots.meta.api.methods.send.SendMessage;
import org.telegram.telegrambots.meta.api.objects.Update;
import org.telegram.telegrambots.meta.api.objects.replykeyboard.InlineKeyboardMarkup;
import org.telegram.telegrambots.meta.api.objects.replykeyboard.buttons.InlineKeyboardButton;
import org.telegram.telegrambots.meta.exceptions.TelegramApiException;
import org.telegram.telegrambots.updatesreceivers.DefaultBotSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class MyTelegramBot extends TelegramLongPollingBot {

    private static final Logger logger = LoggerFactory.getLogger(MyTelegramBot.class);

    // Конструктор з токеном і назвою бота
    public MyTelegramBot() {
        super(BotConfig.getBotToken());
    }

    @Override
    public String getBotUsername() {
        return BotConfig.getBotUsername();
    }

    // Обробка вхідних повідомлень і callback
    @Override
    public void onUpdateReceived(Update update) {
        if (update.hasMessage() && update.getMessage().hasText()) {
            String messageText = update.getMessage().getText();
            Long chatId = update.getMessage().getChatId();

            if ("/start".equals(messageText)) {
                sendMainMenu(chatId); // Відправка головного меню
            }
        } else if (update.hasCallbackQuery()) {
            String callbackData = update.getCallbackQuery().getData();
            Long chatId = update.getCallbackQuery().getMessage().getChatId();

            switch (callbackData) {
                case "menu1_button1", "menu2_button1" -> sendTextMessage(chatId, "Кнопка 1");
                case "menu1_button2", "menu2_button2" -> sendTextMessage(chatId, "Кнопка 2");
                case "to_menu2" -> sendMenu2(chatId); // Перехід до другого меню
                case "back_to_menu1" -> sendMainMenu(chatId); // Повернення до головного меню
                default -> logger.warn("Unknown callback: {}", callbackData);
            }
        }
    }

    // Відправка текстового повідомлення
    private void sendTextMessage(Long chatId, String text) {
        SendMessage message = SendMessage.builder()
                .chatId(chatId.toString())
                .text(text)
                .build();
        try {
            execute(message);
        } catch (TelegramApiException e) {
            logger.error("Failed to send text message", e);
        }
    }

    // Відправка меню з inline-клавіатурою
    private void sendMenu(Long chatId, String menuText, List<List<InlineKeyboardButton>> keyboard) {
        SendMessage message = SendMessage.builder()
                .chatId(chatId.toString())
                .text(menuText)
                .replyMarkup(InlineKeyboardMarkup.builder().keyboard(keyboard).build())
                .build();
        try {
            execute(message);
        } catch (TelegramApiException e) {
            logger.error("Failed to send menu", e);
        }
    }

    // Головне меню (Меню 1)
    private void sendMainMenu(Long chatId) {
        List<List<InlineKeyboardButton>> keyboard = new ArrayList<>();
        List<InlineKeyboardButton> row = new ArrayList<>();

        row.add(InlineKeyboardButton.builder().text("Кнопка 1").callbackData("menu1_button1").build());
        row.add(InlineKeyboardButton.builder().text("Кнопка 2").callbackData("menu1_button2").build());
        row.add(InlineKeyboardButton.builder().text("Далі").callbackData("to_menu2").build());
        keyboard.add(row);

        sendMenu(chatId, "Головне меню:", keyboard);
    }

    // Друге меню (Меню 2)
    private void sendMenu2(Long chatId) {
        List<List<InlineKeyboardButton>> keyboard = new ArrayList<>();
        List<InlineKeyboardButton> row = new ArrayList<>();

        row.add(InlineKeyboardButton.builder().text("Кнопка 1").callbackData("menu2_button1").build());
        row.add(InlineKeyboardButton.builder().text("Кнопка 2").callbackData("menu2_button2").build());
        row.add(InlineKeyboardButton.builder().text("Назад").callbackData("back_to_menu1").build());
        keyboard.add(row);

        sendMenu(chatId, "Меню 2:", keyboard);
    }

    // Запуск бота
    public static void main(String[] args) {
        try {
            TelegramBotsApi botsApi = new TelegramBotsApi(DefaultBotSession.class);
            botsApi.registerBot(new MyTelegramBot());
            logger.info("Bot successfully started!");
        } catch (TelegramApiException e) {
            logger.error("Failed to register bot", e);
        }
    }
}