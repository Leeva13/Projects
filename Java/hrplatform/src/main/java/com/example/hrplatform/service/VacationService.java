package com.example.hrplatform.service;

import com.example.hrplatform.model.Vacation;
import com.example.hrplatform.repository.VacationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class VacationService {
    @Autowired
    private VacationRepository vacationRepository;

    public List<Vacation> findAll() {
        return vacationRepository.findAll();
    }

    public void save(Vacation vacation) {
        vacationRepository.save(vacation);
    }

    public void delete(Long id) {
        vacationRepository.deleteById(id);
    }
}
