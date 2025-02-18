package com.example.document.controller;

import com.example.document.model.Document;
import com.example.document.service.DocumentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@Controller
public class DocumentController {

    @Autowired
    private DocumentService documentService;

    @GetMapping("/")
    public String index(@RequestParam(value = "search", required = false) String search, Model model) {
        List<Document> documents;
        if (search != null && !search.isEmpty()) {
            documents = documentService.searchDocuments(search);
        } else {
            documents = documentService.getAllDocuments();
        }
        model.addAttribute("documents", documents);
        model.addAttribute("search", search);
        return "index";
    }

    @GetMapping("/document/create")
    public String createDocumentForm(Model model) {
        model.addAttribute("document", new Document());
        return "create_document";
    }

    @PostMapping("/document/create")
    public String createDocument(@ModelAttribute Document document, Model model) {
        if (document.getTitle() == null || document.getTitle().isEmpty()) {
            model.addAttribute("error", "Title cannot be empty");
            return "create_document";
        }
        documentService.addDocument(document);
        return "redirect:/";
    }

    @GetMapping("/document/edit/{id}")
    public String editDocumentForm(@PathVariable Long id, Model model) {
        Optional<Document> document = documentService.getDocumentById(id);
        document.ifPresent(value -> model.addAttribute("document", value));
        return document.isPresent() ? "edit_document" : "redirect:/";
    }

    @PostMapping("/document/edit")
    public String editDocument(@ModelAttribute Document document) {
        documentService.updateDocument(document);
        return "redirect:/";
    }

    @GetMapping("/document/delete/{id}")
    public String deleteDocument(@PathVariable Long id) {
        documentService.deleteDocument(id);
        return "redirect:/";
    }

    @GetMapping("/document/view/{id}")
    public String viewDocument(@PathVariable Long id, Model model) {
        Optional<Document> document = documentService.getDocumentById(id);
        document.ifPresent(value -> model.addAttribute("document", value));
        return document.isPresent() ? "view_document" : "redirect:/";
    }
}
