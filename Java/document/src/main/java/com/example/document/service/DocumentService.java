package com.example.document.service;

import com.example.document.model.Document;
import com.example.document.repository.DocumentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class DocumentService {

    @Autowired
    private DocumentRepository documentRepository;

    // Додати документ
    public void addDocument(Document document) {
        documentRepository.addDocument(document);
    }

    // Видалити документ
    public void deleteDocument(Long id) {
        documentRepository.deleteDocument(id);
    }

    // Знайти документ за ідентифікатором
    public Optional<Document> getDocumentById(Long id) {
        return documentRepository.findById(id);
    }

    // Отримати всі документи
    public List<Document> getAllDocuments() {
        return documentRepository.findAll();
    }

    // Оновити документ
    public void updateDocument(Document document) {
        documentRepository.updateDocument(document);
    }

    // Пошук за заголовком
    public List<Document> searchDocuments(String title) {
        return documentRepository.searchByTitle(title);
    }
}
