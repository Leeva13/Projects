package com.example.supportplatform.service;

import com.example.supportplatform.model.Client;
import com.example.supportplatform.repository.ClientRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ClientService {
    @Autowired
    private ClientRepository clientRepository;

    public List<Client> getAllClients() {
        return (List<Client>) clientRepository.findAll();
    }

    public long getClientCount() {
        return clientRepository.count();
    }

    public Client getClientById(Long id) {
        return clientRepository.findById(id).orElse(null);
    }

    public void saveClient(Client client) {
        clientRepository.save(client);
    }

    public void deleteClient(Long id) {
        clientRepository.deleteById(id);
    }
}
