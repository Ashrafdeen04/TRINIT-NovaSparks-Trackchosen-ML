package com.road.roaddetect.model;


import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import lombok.Getter;
import lombok.Setter;

@Entity
@Getter
@Setter
public class User {

    @Id
    String uName;
    String eMail;
    String password;
    public User(){
        super();
    }
    public User(String userName,String eMail,String password){
        this.uName=userName;
        this.eMail=eMail;
        this.password=password;
    }
}
