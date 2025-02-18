package com.example.corporate.controller;

import com.example.corporate.model.BlogArticle;
import com.example.corporate.service.BlogService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
public class BlogController {
    private final BlogService blogService;

    public BlogController(BlogService blogService) {
        this.blogService = blogService;
    }

    @GetMapping("/blog")
    public String blog(Model model) {
        model.addAttribute("articles", blogService.findAll());
        return "blog";
    }

    @GetMapping("/blog/new")
    public String newArticle(Model model) {
        model.addAttribute("title", "New Article");
        return "new-article";
    }

    @PostMapping("/blog")
    public String submitArticle(@RequestParam String title, @RequestParam String content) {
        BlogArticle article = new BlogArticle(null, title, content);
        blogService.save(article);
        return "redirect:/blog";
    }

    @GetMapping("/blog/{id}")
    public String viewArticle(@PathVariable Long id, Model model) {
        blogService.findById(id).ifPresent(article -> model.addAttribute("article", article));
        return "view-article";
    }

    @PostMapping("/blog/delete/{id}")
    public String deleteArticle(@PathVariable Long id) {
        blogService.deleteById(id);
        return "redirect:/blog";
    }
}
