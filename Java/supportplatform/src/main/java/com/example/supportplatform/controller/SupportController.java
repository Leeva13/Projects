package com.example.supportplatform.controller;

import com.example.supportplatform.model.Support;
import com.example.supportplatform.service.SupportService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/support")
public class SupportController {

    @Autowired
    private SupportService supportService;

    @GetMapping
    public String getAllSupports(Model model) {
        model.addAttribute("supports", supportService.getAllSupports());
        return "support";
    }

    @GetMapping("/new")
    public String createSupportForm(Model model) {
        model.addAttribute("support", new Support());
        return "support-form";
    }

    @PostMapping
    public String saveSupport(@ModelAttribute("support") Support support) {
        supportService.saveSupport(support);
        return "redirect:/support";
    }

    @GetMapping("/edit/{id}")
    public String editSupportForm(@PathVariable Long id, Model model) {
        model.addAttribute("support", supportService.getSupportById(id));
        return "support-form";
    }

    @PostMapping("/{id}")
    public String updateSupport(@PathVariable Long id, @ModelAttribute("support") Support support) {
        Support existingSupport = supportService.getSupportById(id);
        existingSupport.setQuery(support.getQuery());
        existingSupport.setResponse(support.getResponse());
        supportService.saveSupport(existingSupport);
        return "redirect:/support";
    }

    @GetMapping("/delete/{id}")
    public String deleteSupport(@PathVariable Long id) {
        supportService.deleteSupport(id);
        return "redirect:/support";
    }
}
