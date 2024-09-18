package com.example.supportplatform.service;

import com.example.supportplatform.model.Support;
import com.example.supportplatform.repository.SupportRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class SupportService {
    @Autowired
    private SupportRepository supportRepository;

    public List<Support> getAllSupports() {
        return (List<Support>) supportRepository.findAll();
    }

    public long getSupportCount() {
        return supportRepository.count();
    }

    public Support getSupportById(Long id) {
        return supportRepository.findById(id).orElse(null);
    }

    public void saveSupport(Support support) {
        supportRepository.save(support);
    }

    public void deleteSupport(Long id) {
        supportRepository.deleteById(id);
    }
}
