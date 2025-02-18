package com.brazhnyk.Unveil.controller;

import com.brazhnyk.Unveil.model.Question;
import com.brazhnyk.Unveil.service.QuestionService;
import lombok.Getter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Controller
public class SurveyController {

    @Autowired
    private QuestionService questionService;

    @GetMapping("/")
    public String getNameForm() {
        return "index";
    }

    @PostMapping("/survey")
    public String showSurvey(@RequestParam String name, Model model) {
        List<Question> allQuestions = questionService.getAllQuestions();
        Collections.shuffle(allQuestions);
        List<Question> randomQuestions = allQuestions.stream().distinct().limit(5).collect(Collectors.toList());
        model.addAttribute("name", name);
        model.addAttribute("questions", randomQuestions);
        return "survey";
    }

    @PostMapping("/results")
    public String showResults(@RequestParam Map<String, String> answers, @RequestParam String name, @RequestParam List<Long> questionIds, Model model) {
        List<Question> questions = questionService.getQuestionsByIds(questionIds);
        int correctAnswers = 0;
        List<ResultEntry> results = new ArrayList<>();
        for (Question question : questions) {
            String userAnswer = answers.get("answers[" + question.getId() + "]");
            if (userAnswer != null && userAnswer.equals(question.getCorrectAnswer())) {
                correctAnswers++;
            }
            results.add(new ResultEntry(question.getQuestionText(), userAnswer, question.getCorrectAnswer()));
        }
        model.addAttribute("name", name);
        model.addAttribute("correctAnswers", correctAnswers);
        model.addAttribute("results", results);
        return "results";
    }

    @Getter
    public static class ResultEntry {
        private String questionText;
        private String userAnswer;
        private String correctAnswer;

        public ResultEntry(String questionText, String userAnswer, String correctAnswer) {
            this.questionText = questionText;
            this.userAnswer = userAnswer;
            this.correctAnswer = correctAnswer;
        }
    }
}
