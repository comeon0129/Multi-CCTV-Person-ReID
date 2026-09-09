package com.graduation.reid_backend.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DetectionDto {
    // 1명의 인물에 대한 정보만 담는 DTO
    private String imageData;
    private float[] vectorData;
    private Map<String, Object> bbox;
    private float confidence;
}