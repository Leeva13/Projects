package com.example.travelplanner.controller;

import com.example.travelplanner.model.Route;
import com.example.travelplanner.service.RouteService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.time.format.DateTimeFormatter;
import java.util.List;

@Controller
@RequestMapping("/routes")
public class RouteController {

    @Autowired
    private RouteService routeService;

    @GetMapping
    public String getAllRoutes(@RequestParam("username") String username, Model model) {
        List<Route> routes = routeService.findAll();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");
        model.addAttribute("routes", routes);
        model.addAttribute("username", username);
        return "routes";
    }

    @PostMapping("/create")
    public String createRoute(@RequestParam("username") String username, @ModelAttribute("route") Route route) {
        routeService.save(route);
        return "redirect:/routes?username=" + username;
    }

    @GetMapping("/new")
    public String newRouteForm(@RequestParam("username") String username, Model model) {
        model.addAttribute("route", new Route());
        model.addAttribute("username", username);
        return "route_form";
    }

    @GetMapping("/edit/{id}")
    public String editRouteForm(@PathVariable Long id, @RequestParam("username") String username, Model model) {
        model.addAttribute("route", routeService.findById(id).orElseThrow(() -> new IllegalArgumentException("Invalid route Id:" + id)));
        model.addAttribute("username", username);
        return "route_form";
    }

    @PostMapping("/update/{id}")
    public String updateRoute(@PathVariable Long id, @RequestParam("username") String username, @ModelAttribute("route") Route route) {
        routeService.save(route);
        return "redirect:/routes?username=" + username;
    }

    @GetMapping("/delete/{id}")
    public String deleteRoute(@PathVariable Long id, @RequestParam("username") String username) {
        routeService.deleteById(id);
        return "redirect:/routes?username=" + username;
    }
}
