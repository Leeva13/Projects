package com.example.supportplatform.repository;

import com.example.supportplatform.model.Document;
import org.springframework.data.repository.CrudRepository;

public interface DocumentRepository extends CrudRepository<Document, Long> {
}
