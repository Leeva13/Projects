package com.brazhnyk.Unveil.service;

import com.brazhnyk.Unveil.model.Question;
import com.brazhnyk.Unveil.repository.QuestionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;

@Service
public class QuestionService {

    @Autowired
    private QuestionRepository questionRepository;

    public List<Question> getAllQuestions() {
        return questionRepository.findAll();
    }

    public List<Question> getRandomQuestions(int numberOfQuestions) {
        List<Question> allQuestions = questionRepository.findAll();
        Collections.shuffle(allQuestions);
        return allQuestions.subList(0, Math.min(numberOfQuestions, allQuestions.size()));
    }

    public List<Question> getQuestionsByIds(List<Long> questionIds) {
        return questionRepository.findAllById(questionIds);
    }
}
