package com.graduation.reid_backend.controller;

import com.graduation.reid_backend.dto.ReidResultDto;
import com.graduation.reid_backend.service.ReidService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class TestController {
    private final ReidService reidService;
    @PostMapping("/api/test/reid")
    public String testReid(@RequestBody ReidResultDto dto) {
        return reidService.findPersonId(dto.getVectorData(), dto.getLocationId());
    }
}
