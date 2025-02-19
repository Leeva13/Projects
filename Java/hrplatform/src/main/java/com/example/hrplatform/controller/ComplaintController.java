package com.example.hrplatform.controller;

import com.example.hrplatform.model.Complaint;
import com.example.hrplatform.service.ComplaintService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/complaints")
public class ComplaintController {
    @Autowired
    private ComplaintService complaintService;

    @GetMapping
    public String list(Model model) {
        model.addAttribute("complaints", complaintService.findAll());
        return "complaints";
    }

    @PostMapping
    public String add(@ModelAttribute Complaint complaint) {
        complaintService.save(complaint);
        return "redirect:/complaints";
    }

    @DeleteMapping("/{id}")
    public String delete(@PathVariable Long id) {
        complaintService.delete(id);
        return "redirect:/complaints";
    }
}
