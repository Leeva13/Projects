package com.example.supportplatform.service;

import com.example.supportplatform.model.Document;
import com.example.supportplatform.repository.DocumentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class DocumentService {
    @Autowired
    private DocumentRepository documentRepository;

    public List<Document> getAllDocuments() {
        return (List<Document>) documentRepository.findAll();
    }

    public long getDocumentCount() {
        return documentRepository.count();
    }

    public Document getDocumentById(Long id) {
        return documentRepository.findById(id).orElse(null);
    }

    public void saveDocument(Document document) {
        documentRepository.save(document);
    }

    public void deleteDocument(Long id) {
        documentRepository.deleteById(id);
    }
}
