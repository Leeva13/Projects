package com.brazhnyk.Unveil.component;

import org.springframework.context.ApplicationListener;
import org.springframework.context.event.ContextRefreshedEvent;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Autowired;
import  com.brazhnyk.Unveil.repository.QuestionRepository;

@Component
public class DatabaseCleaning implements ApplicationListener<ContextRefreshedEvent> {

    @Autowired
    private QuestionRepository questionRepository;

    @Override
    public void onApplicationEvent(ContextRefreshedEvent event) {
        // Викликати метод для очищення таблиці question при старті додатка
        questionRepository.deleteAll();
    }
}