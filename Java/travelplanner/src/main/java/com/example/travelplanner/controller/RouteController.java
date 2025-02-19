package com.example.travelplanner.controller;

import com.example.travelplanner.model.Route;
import com.example.travelplanner.service.RouteService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Controller
public class RouteController {

    @Autowired
    private RouteService routeService;

    @GetMapping("/")
    public String index(@RequestParam(required = false) String sort,
                        @RequestParam(required = false) String order,
                        Model model) {
        List<Route> routes;
        if (sort != null && !sort.isEmpty()) {
            routes = routeService.getSortedRoutes(sort, order);
        } else {
            routes = routeService.getRoutes();
        }
        model.addAttribute("routes", routes);
        model.addAttribute("sort", sort);
        model.addAttribute("order", order);
        return "index";
    }

    @GetMapping("/create-route")
    public String createRouteForm() {
        return "create_route";
    }

    @PostMapping("/create-route")
    public String createRoute(@RequestParam String name, @RequestParam String description,
                              @RequestParam String startDate, @RequestParam String endDate,
                              @RequestParam String startLocation, @RequestParam String endLocation, Model model) {
        Route route = new Route(name, description, startDate, endDate, startLocation, endLocation);
        routeService.addRoute(route);
        return "redirect:/";
    }

    @GetMapping("/edit-route/{id}")
    public String editRouteForm(@PathVariable Long id, Model model) {
        model.addAttribute("route", routeService.getRouteById(id).orElseThrow(() -> new IllegalArgumentException("Invalid route Id:" + id)));
        return "edit_route";
    }

    @PostMapping("/edit-route")
    public String editRoute(@RequestParam Long id, @RequestParam String name, @RequestParam String description,
                            @RequestParam String startDate, @RequestParam String endDate,
                            @RequestParam String startLocation, @RequestParam String endLocation) {
        Route route = new Route(id, name, description, startDate, endDate, startLocation, endLocation);
        routeService.updateRoute(route);
        return "redirect:/";
    }

    @PostMapping("/delete-route/{id}")
    public String deleteRoute(@PathVariable Long id) {
        routeService.deleteRoute(id);
        return "redirect:/";
    }
}
