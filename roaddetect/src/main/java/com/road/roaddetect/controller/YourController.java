package com.road.roaddetect.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.RequestEntity;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.client.RestTemplate;   
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;

import com.road.roaddetect.model.Image;
import com.road.roaddetect.service.ImageService;

import jakarta.servlet.http.HttpServletRequest;

import javax.sql.rowset.serial.SerialException;
import java.io.IOException;
import java.sql.Blob;
import java.sql.SQLException;
import java.util.List;
import java.net.URI;
import java.net.URISyntaxException;

import javax.sql.rowset.serial.SerialBlob;

@Controller
@RequestMapping()
public class YourController {
      
    private final String FLASK_API_URL = "http://127.0.0.1:5000/model";

    @Autowired
    ImageService imageService;

    @PostMapping("/processImage")
    public ModelAndView processImage(HttpServletRequest request,@RequestParam("image") MultipartFile file) throws IOException, SerialException, SQLException, URISyntaxException {
        RestTemplate restTemplate = new RestTemplate();

        // Prepare headers
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        // Prepare the image file to be sent
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", file.getResource());

        // Prepare the request entity
        RequestEntity<MultiValueMap<String, Object>> requestEntity = RequestEntity
                .post(new URI(FLASK_API_URL))
                .headers(headers)
                .body(body);

        // Make the POST request to Flask API
        ResponseEntity<byte[]> responseEntity = restTemplate.exchange(requestEntity, byte[].class);
          byte[] bytes = responseEntity.getBody();
        Blob blob = new SerialBlob(bytes);

        Image image = new Image();
        image.setImage(blob);
        imageService.create(image);

       ModelAndView mv = new ModelAndView("userhome");
        List<Image> imageList = imageService.viewAll();
        mv.addObject("imageList", imageList);
        return mv;
        // Return the response received from Flask API
        /*return ResponseEntity
                .status(responseEntity.getStatusCode())
                .contentType(responseEntity.getHeaders().getContentType())
                .body(responseEntity.getBody());*/
    }
}
