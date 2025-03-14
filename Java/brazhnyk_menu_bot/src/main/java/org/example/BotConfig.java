package org.example;

import io.github.cdimascio.dotenv.Dotenv;

public class BotConfig {
    private static final Dotenv dotenv = Dotenv.load();

    public static String getBotToken() {
        return dotenv.get("BOT_TOKEN");
    }

    public static String getBotUsername() {
        return dotenv.get("BOT_USERNAME");
    }
}

