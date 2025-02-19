package com.example.corporate.repository;

import com.example.corporate.model.BlogArticle;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

@Repository
public class BlogRepository {
    private List<BlogArticle> articles = new ArrayList<>();
    private AtomicLong counter = new AtomicLong();

    public List<BlogArticle> findAll() {
        return articles;
    }

    public Optional<BlogArticle> findById(Long id) {
        return articles.stream().filter(article -> article.getId().equals(id)).findFirst();
    }

    public BlogArticle save(BlogArticle article) {
        article.setId(counter.incrementAndGet());
        articles.add(article);
        return article;
    }

    public void deleteById(Long id) {
        articles.removeIf(article -> article.getId().equals(id));
    }
}
