package com.example.hrplatform.controller;

import com.example.hrplatform.model.Vacation;
import com.example.hrplatform.service.VacationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/vacations")
public class VacationController {
    @Autowired
    private VacationService vacationService;

    @GetMapping
    public String list(Model model) {
        model.addAttribute("vacations", vacationService.findAll());
        return "vacations";
    }

    @PostMapping
    public String add(@ModelAttribute Vacation vacation) {
        vacationService.save(vacation);
        return "redirect:/vacations";
    }

    @DeleteMapping("/{id}")
    public String delete(@PathVariable Long id) {
        vacationService.delete(id);
        return "redirect:/vacations";
    }
}
