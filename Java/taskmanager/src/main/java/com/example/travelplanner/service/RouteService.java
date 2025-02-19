package com.example.travelplanner.service;

import com.example.travelplanner.model.Route;
import com.example.travelplanner.repository.RouteRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class RouteService {

    @Autowired
    private RouteRepository routeRepository;

    public List<Route> findAll() {
        return routeRepository.findAll();
    }

    public Optional<Route> findById(Long id) {
        return routeRepository.findById(id);
    }

    public Route save(Route route) {
        return routeRepository.save(route);
    }

    public void deleteById(Long id) {
        routeRepository.deleteById(id);
    }
}
