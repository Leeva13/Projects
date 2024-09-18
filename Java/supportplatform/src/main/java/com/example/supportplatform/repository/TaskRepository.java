package com.example.supportplatform.repository;

import com.example.supportplatform.model.Task;
import org.springframework.data.repository.CrudRepository;

public interface TaskRepository extends CrudRepository<Task, Long> {
}
