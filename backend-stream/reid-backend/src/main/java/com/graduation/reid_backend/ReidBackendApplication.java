package com.graduation.reid_backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class ReidBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(ReidBackendApplication.class, args);
	}

	@Bean //"이 함수에서 리턴하는 객체를 도구상자(Bean)에 강제로 넣는 메소드"
	public ObjectMapper objectMapper() {
		return new ObjectMapper();
	}
}
