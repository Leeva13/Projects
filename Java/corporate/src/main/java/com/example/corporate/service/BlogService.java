package com.example.corporate.service;

import com.example.corporate.model.BlogArticle;
import com.example.corporate.repository.BlogRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class BlogService {
    private final BlogRepository blogRepository;

    public BlogService(BlogRepository blogRepository) {
        this.blogRepository = blogRepository;
    }

    public List<BlogArticle> findAll() {
        return blogRepository.findAll();
    }

    public Optional<BlogArticle> findById(Long id) {
        return blogRepository.findById(id);
    }

    public BlogArticle save(BlogArticle article) {
        return blogRepository.save(article);
    }

    public void deleteById(Long id) {
        blogRepository.deleteById(id);
    }
}
