package com.example.document.repository;

import com.example.document.model.Document;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Repository
public class DocumentRepository {
    private List<Document> documents = new ArrayList<>();
    private long nextId = 1;

    // Додати новий документ
    public void addDocument(Document document) {
        document.setId(nextId++);
        documents.add(document);
    }

    // Видалити документ за ідентифікатором
    public void deleteDocument(Long id) {
        documents.removeIf(doc -> doc.getId().equals(id));
    }

    // Знайти документ за ідентифікатором
    public Optional<Document> findById(Long id) {
        return documents.stream()
                .filter(doc -> doc.getId().equals(id))
                .findFirst();
    }

    // Отримати всі документи
    public List<Document> findAll() {
        return new ArrayList<>(documents);
    }

    // Оновити документ
    public void updateDocument(Document updatedDocument) {
        documents.replaceAll(doc -> doc.getId().equals(updatedDocument.getId()) ? updatedDocument : doc);
    }

    // Пошук документів за заголовком
    public List<Document> searchByTitle(String title) {
        return documents.stream()
                .filter(doc -> doc.getTitle().toLowerCase().contains(title.toLowerCase()))
                .toList();
    }
}
