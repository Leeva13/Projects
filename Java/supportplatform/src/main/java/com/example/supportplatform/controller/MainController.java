package com.example.supportplatform.controller;

import com.example.supportplatform.service.ClientService;
import com.example.supportplatform.service.DocumentService;
import com.example.supportplatform.service.SupportService;
import com.example.supportplatform.service.TaskService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;

@Controller
public class MainController {

    @Autowired
    private ClientService clientService;

    @Autowired
    private DocumentService documentService;

    @Autowired
    private SupportService supportService;

    @Autowired
    private TaskService taskService;
}
