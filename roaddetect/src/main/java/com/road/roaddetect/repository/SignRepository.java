package com.road.roaddetect.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.road.roaddetect.model.Sign;

@Repository
public interface SignRepository extends JpaRepository<Sign,String>{

    
}
