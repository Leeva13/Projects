package com.example.hrplatform.service;

import com.example.hrplatform.model.Candidate;
import com.example.hrplatform.repository.CandidateRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CandidateService {
    @Autowired
    private CandidateRepository candidateRepository;

    public List<Candidate> findAll() {
        return candidateRepository.findAll();
    }

    public void save(Candidate candidate) {
        candidateRepository.save(candidate);
    }

    public void delete(Long id) {
        candidateRepository.deleteById(id);
    }
}

