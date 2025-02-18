package com.example.corporate.controller;

import com.example.corporate.model.ContactMessage;
import com.example.corporate.service.ContactService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
public class ContactController {
    private final ContactService contactService;

    public ContactController(ContactService contactService) {
        this.contactService = contactService;
    }

    @GetMapping("/contact")
    public String contact(Model model) {
        model.addAttribute("title", "Contact Us");
        return "contact";
    }

    @PostMapping("/contact")
    public String submitContact(@RequestParam String name, @RequestParam String email, @RequestParam String message) {
        contactService.save(new ContactMessage(name, email, message));
        return "redirect:/contact";
    }

    @GetMapping("/contacts")
    public String viewContacts(Model model) {
        model.addAttribute("messages", contactService.findAll());
        return "contacts";
    }
}
