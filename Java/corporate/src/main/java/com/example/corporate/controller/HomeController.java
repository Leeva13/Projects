package com.example.corporate.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {

    @GetMapping("/")
    public String home(Model model) {
        model.addAttribute("title", "Welcome to Our Corporate Site");
        return "home";
    }

    @GetMapping("/services")
    public String services(Model model) {
        return "services";
    }

    @GetMapping("/projects")
    public String projects(Model model) {
        return "projects";
    }
}
