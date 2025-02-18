package com.example.hrplatform.service;

import com.example.hrplatform.model.Salary;
import com.example.hrplatform.repository.SalaryRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class SalaryService {
    @Autowired
    private SalaryRepository salaryRepository;

    public List<Salary> findAll() {
        return salaryRepository.findAll();
    }

    public void save(Salary salary) {
        salaryRepository.save(salary);
    }

    public void delete(Long id) {
        salaryRepository.deleteById(id);
    }
}
