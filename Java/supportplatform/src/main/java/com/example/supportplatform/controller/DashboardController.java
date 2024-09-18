package com.example.supportplatform.controller;

import com.example.supportplatform.service.ClientService;
import com.example.supportplatform.service.DocumentService;
import com.example.supportplatform.service.SupportService;
import com.example.supportplatform.service.TaskService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class DashboardController {

    @Autowired
    private ClientService clientService;

    @Autowired
    private TaskService taskService;

    @Autowired
    private DocumentService documentService;

    @Autowired
    private SupportService supportService;

    @GetMapping("/")
    public String showDashboard(Model model) {
        model.addAttribute("clientCount", clientService.getClientCount());
        model.addAttribute("taskCount", taskService.getTaskCount());
        model.addAttribute("documentCount", documentService.getDocumentCount());
        model.addAttribute("supportCount", supportService.getSupportCount());

        model.addAttribute("clients", clientService.getAllClients());
        model.addAttribute("tasks", taskService.getAllTasks());
        model.addAttribute("documents", documentService.getAllDocuments());
        model.addAttribute("supports", supportService.getAllSupports());

        return "index";
    }
}
