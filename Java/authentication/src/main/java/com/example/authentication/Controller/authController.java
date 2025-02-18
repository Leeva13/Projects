package com.example.authentication.Controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class authController {

    @GetMapping("/admin")
    public String writeAdmin(Model model) {
        return "Admin - admin";
    }

    @GetMapping("/user")
    public  String writeUser(Model model) {
        return "User - user";
    }
}
