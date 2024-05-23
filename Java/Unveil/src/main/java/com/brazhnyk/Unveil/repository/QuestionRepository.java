package com.brazhnyk.Unveil.repository;

import com.brazhnyk.Unveil.model.Question;
import org.springframework.data.jpa.repository.JpaRepository;

public interface QuestionRepository extends JpaRepository<Question, Long> {
}
