package com.example.hrplatform.controller;

import com.example.hrplatform.model.Candidate;
import com.example.hrplatform.service.CandidateService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/candidates")
public class CandidateController {
    @Autowired
    private CandidateService candidateService;

    @GetMapping
    public String list(Model model) {
        model.addAttribute("candidates", candidateService.findAll());
        return "candidates";
    }

    @PostMapping
    public String add(@ModelAttribute Candidate candidate) {
        candidateService.save(candidate);
        return "redirect:/candidates";
    }

    @DeleteMapping("/{id}")
    public String delete(@PathVariable Long id) {
        candidateService.delete(id);
        return "redirect:/candidates";
    }
}