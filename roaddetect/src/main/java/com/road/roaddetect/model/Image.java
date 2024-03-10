package com.road.roaddetect.model;


import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;

import java.sql.Blob;
import java.util.Date;

@Entity
@Table(name = "image_table")
public class Image {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private long id;

    @Lob
    private Blob image;

    private Date date = new Date();

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public Blob getImage() {
        return image;
    }

    public void setImage(Blob image) {
        this.image = image;
    }

    public Date getDate() {
        return date;
    }
}