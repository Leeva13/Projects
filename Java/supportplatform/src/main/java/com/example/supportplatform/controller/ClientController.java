package com.example.supportplatform.controller;

import com.example.supportplatform.model.Client;
import com.example.supportplatform.service.ClientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/clients")
public class ClientController {

    @Autowired
    private ClientService clientService;

    @GetMapping
    public String getAllClients(Model model) {
        model.addAttribute("clients", clientService.getAllClients());
        return "clients";
    }

    @GetMapping("/new")
    public String createClientForm(Model model) {
        model.addAttribute("client", new Client());
        return "client-form";
    }

    @PostMapping
    public String saveClient(@ModelAttribute("client") Client client) {
        clientService.saveClient(client);
        return "redirect:/clients";
    }

    @GetMapping("/edit/{id}")
    public String editClientForm(@PathVariable Long id, Model model) {
        model.addAttribute("client", clientService.getClientById(id));
        return "client-form";
    }

    @PostMapping("/{id}")
    public String updateClient(@PathVariable Long id, @ModelAttribute("client") Client client) {
        Client existingClient = clientService.getClientById(id);
        existingClient.setName(client.getName());
        existingClient.setEmail(client.getEmail());
        clientService.saveClient(existingClient);
        return "redirect:/clients";
    }

    @GetMapping("/delete/{id}")
    public String deleteClient(@PathVariable Long id) {
        clientService.deleteClient(id);
        return "redirect:/clients";
    }
}
