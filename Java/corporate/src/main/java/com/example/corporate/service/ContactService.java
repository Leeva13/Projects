package com.example.corporate.service;

import com.example.corporate.model.ContactMessage;
import com.example.corporate.repository.ContactRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ContactService {
    private final ContactRepository contactRepository;

    public ContactService(ContactRepository contactRepository) {
        this.contactRepository = contactRepository;
    }

    public List<ContactMessage> findAll() {
        return contactRepository.findAll();
    }

    public void save(ContactMessage message) {
        contactRepository.save(message);
    }
}
