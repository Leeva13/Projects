package com.example.supportplatform.controller;

import com.example.supportplatform.model.Document;
import com.example.supportplatform.service.DocumentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/documents")
public class DocumentController {

    @Autowired
    private DocumentService documentService;

    @GetMapping
    public String getAllDocuments(Model model) {
        model.addAttribute("documents", documentService.getAllDocuments());
        return "documents";
    }

    @GetMapping("/new")
    public String createDocumentForm(Model model) {
        model.addAttribute("document", new Document());
        return "document-form";
    }

    @PostMapping
    public String saveDocument(@ModelAttribute("document") Document document) {
        documentService.saveDocument(document);
        return "redirect:/documents";
    }

    @GetMapping("/edit/{id}")
    public String editDocumentForm(@PathVariable Long id, Model model) {
        model.addAttribute("document", documentService.getDocumentById(id));
        return "document-form";
    }

    @PostMapping("/{id}")
    public String updateDocument(@PathVariable Long id, @ModelAttribute("document") Document document) {
        Document existingDocument = documentService.getDocumentById(id);
        existingDocument.setName(document.getName());
        existingDocument.setContent(document.getContent());
        documentService.saveDocument(existingDocument);
        return "redirect:/documents";
    }

    @GetMapping("/delete/{id}")
    public String deleteDocument(@PathVariable Long id) {
        documentService.deleteDocument(id);
        return "redirect:/documents";
    }
}
