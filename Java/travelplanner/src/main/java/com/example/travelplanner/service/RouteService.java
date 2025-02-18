package com.example.travelplanner.service;

import com.example.travelplanner.model.Route;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

@Service
public class RouteService {

    private final List<Route> routes = new ArrayList<>();

    public void addRoute(Route route) {
        routes.add(route);
    }

    public List<Route> getRoutes() {
        return new ArrayList<>(routes);
    }

    public Optional<Route> getRouteById(Long id) {
        return routes.stream().filter(route -> route.getId().equals(id)).findFirst();
    }

    public void updateRoute(Route updatedRoute) {
        getRouteById(updatedRoute.getId()).ifPresent(route -> {
            route.setName(updatedRoute.getName());
            route.setDescription(updatedRoute.getDescription());
            route.setStartDate(updatedRoute.getStartDate());
            route.setEndDate(updatedRoute.getEndDate());
            route.setStartLocation(updatedRoute.getStartLocation());
            route.setEndLocation(updatedRoute.getEndLocation());
        });
    }

    public void deleteRoute(Long id) {
        routes.removeIf(route -> route.getId().equals(id));
    }

    public List<Route> getSortedRoutes(String sortBy, String order) {
        List<Route> sortedRoutes = new ArrayList<>(routes);
        Comparator<Route> comparator;

        switch (sortBy) {
            case "name":
                comparator = Comparator.comparing(Route::getName);
                break;
            case "startDate":
                comparator = Comparator.comparing(Route::getStartDate);
                break;
            case "endDate":
                comparator = Comparator.comparing(Route::getEndDate);
                break;
            case "startLocation":
                comparator = Comparator.comparing(Route::getStartLocation);
                break;
            case "endLocation":
                comparator = Comparator.comparing(Route::getEndLocation);
                break;
            default:
                return sortedRoutes;
        }

        if ("desc".equalsIgnoreCase(order)) {
            comparator = comparator.reversed();
        }

        sortedRoutes.sort(comparator);
        return sortedRoutes;
    }
}
