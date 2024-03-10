package com.road.roaddetect.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.road.roaddetect.model.User;
import com.road.roaddetect.repository.UserRepository;

@Service
public class UserServices {
    
    @Autowired
    UserRepository urepo;

    public void save(User u){
        urepo.save(u);
    }
}
