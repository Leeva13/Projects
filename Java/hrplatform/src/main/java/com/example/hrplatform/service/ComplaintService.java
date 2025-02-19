package com.example.hrplatform.service;

import com.example.hrplatform.model.Complaint;
import com.example.hrplatform.repository.ComplaintRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ComplaintService {
    @Autowired
    private ComplaintRepository complaintRepository;

    public List<Complaint> findAll() {
        return complaintRepository.findAll();
    }

    public void save(Complaint complaint) {
        complaintRepository.save(complaint);
    }

    public void delete(Long id) {
        complaintRepository.deleteById(id);
    }
}
