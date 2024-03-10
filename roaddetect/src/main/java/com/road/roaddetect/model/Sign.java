package com.road.roaddetect.model;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import lombok.Getter;
import lombok.Setter;

@Entity
@Getter
@Setter
public class Sign {
    @Id
    String uName;
    String password;
    public Sign(){
        super();
    }
    public Sign(String uName,String password){
        this.uName=uName;
        this.password=password;

    }
}
