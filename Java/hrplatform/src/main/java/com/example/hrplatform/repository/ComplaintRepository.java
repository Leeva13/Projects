package com.example.hrplatform.repository;

import com.example.hrplatform.model.Candidate;
import com.example.hrplatform.model.Complaint;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ComplaintRepository extends JpaRepository<Complaint, Long> {
}
