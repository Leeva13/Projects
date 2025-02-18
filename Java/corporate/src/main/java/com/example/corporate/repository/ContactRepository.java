package com.example.corporate.repository;

import com.example.corporate.model.ContactMessage;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;

@Repository
public class ContactRepository {
    private List<ContactMessage> messages = new ArrayList<>();

    public List<ContactMessage> findAll() {
        return messages;
    }

    public void save(ContactMessage message) {
        messages.add(message);
    }
}
