package com.example.hrplatform.controller;

import com.example.hrplatform.model.Salary;
import com.example.hrplatform.service.SalaryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/salaries")
public class SalaryController {
    @Autowired
    private SalaryService salaryService;

    @GetMapping
    public String list(Model model) {
        model.addAttribute("salaries", salaryService.findAll());
        return "salaries";
    }

    @PostMapping
    public String add(@ModelAttribute Salary salary) {
        salaryService.save(salary);
        return "redirect:/salaries";
    }

    @DeleteMapping("/{id}")
    public String delete(@PathVariable Long id) {
        salaryService.delete(id);
        return "redirect:/salaries";
    }
}
