package com.example.travelplanner.repository;

import com.example.travelplanner.model.Route;
import org.springframework.data.jpa.repository.JpaRepository;

public interface RouteRepository extends JpaRepository<Route, Long> {
}
