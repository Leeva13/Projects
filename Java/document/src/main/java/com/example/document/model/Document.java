package com.example.document.model;

import java.time.LocalDate;

public class Document {


    public Document(String title, String content, LocalDate parse) {
    }

    public Document(){}

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public LocalDate getCreatedDate() {
        return createdDate;
    }

    public void setCreatedDate(LocalDate createdDate) {
        this.createdDate = createdDate;
    }

    private Long id;
    private String title;
    private String content;
    private LocalDate createdDate;
}

