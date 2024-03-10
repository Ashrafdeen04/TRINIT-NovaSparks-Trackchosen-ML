package com.road.roaddetect.repository;


import com.road.roaddetect.model.Image;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ImageRepository extends CrudRepository<Image, Long> {

}