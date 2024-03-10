package com.road.roaddetect.controller;


import com.road.roaddetect.model.Image;
import com.road.roaddetect.model.Sign;
import com.road.roaddetect.model.User;
import com.road.roaddetect.service.ImageService;
import com.road.roaddetect.service.UserServices;

import jakarta.servlet.http.HttpServletRequest;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.RequestEntity;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;

import javax.sql.rowset.serial.SerialBlob;
import javax.sql.rowset.serial.SerialException;
import java.io.IOException;
import java.net.URI;
import java.sql.Blob;
import java.sql.SQLException;
import java.util.List;
import org.springframework.web.bind.annotation.RequestBody;


@Controller
public class ClientController {

    private final String FLASK_API_URL = "http://127.0.0.1:5000/videocapture";

    @Autowired
    private ImageService imageService;
    @Autowired
    private UserServices service;

    @GetMapping("/ping")
    @ResponseBody
    public String hello_world(){
        return "Hello World!";
    }

    // display image
    @GetMapping("/display")
    public ResponseEntity<byte[]> displayImage(@RequestParam("id") long id) throws IOException, SQLException
    {
        Image image = imageService.viewById(id);
        byte [] imageBytes = null;
        imageBytes = image.getImage().getBytes(1,(int) image.getImage().length());
        return ResponseEntity.ok().contentType(MediaType.IMAGE_JPEG).body(imageBytes);
    }

    // view All images
    @GetMapping("/")
    public String home(){
       /* ModelAndView mv = new ModelAndView("home");
        List<Image> imageList = imageService.viewAll();
        mv.addObject("imageList", imageList);*/
        return "signup";
    }

    // add image - get
    @GetMapping("/add")
    public ModelAndView addImage(){
        return new ModelAndView("addimage");
    }

    // add image - post
    @PostMapping("/add")
    public ModelAndView addImagePost(HttpServletRequest request,@RequestParam("image") MultipartFile file) throws IOException, SerialException, SQLException
    {
        byte[] bytes = file.getBytes();
        Blob blob = new javax.sql.rowset.serial.SerialBlob(bytes);

        Image image = new Image();
        image.setImage(blob);
        ModelAndView mv = new ModelAndView("home");
        List<Image> imageList = imageService.viewAll();
        mv.addObject("imageList", imageList);
        return mv;
    }

    @PostMapping("/signup")
    public ModelAndView signUp(@ModelAttribute User user) {
        service.save(user);
        ModelAndView mv = new ModelAndView("userhome");
        List<Image> imageList = imageService.viewAll();
        mv.addObject("user", user);
        mv.addObject("imageList", imageList);
        return mv;
    }
    @GetMapping("/login")
    public String getMethodName() {
        return "login";
    }
    
    @GetMapping("/aboutus")
    public String aboutUS() {
        return "aboutus";
    }
    
    @PostMapping("/singin")
    public String postMethodName(@ModelAttribute Sign sign) {
        //TODO: process POST request
        return "home";
    }
    @GetMapping("/particular")
    public String location() {
        return "map";
    }
    

    @GetMapping("/capture-video")
    public String captureVideo() throws Exception {
        

        RequestEntity<Void> requestEntity=  RequestEntity
        .post(new URI(FLASK_API_URL))
        .build();
      

        return "index";
    }
    
}    