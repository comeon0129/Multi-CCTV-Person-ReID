package com.graduation.reid_backend.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ReidResultDto {

    private Integer schemaVersion;      // null 또는 1 = 레거시, 2 = 신규(그룹)

    // --- 공통 필드 ---
    private String cctvId;        // 예: "cctv_1"
    private String locationId;    // 예: "zone_A" (동일 장소 그룹화용)
    private Integer frameNumber;  // 프레임 번호
    private String timestamp;     // 찍힌 시간 (AI 서버에서 보내는 포맷 그대로 수신)
    private Map<String, Long> timestamps; // 각 구간별 시간 기록용 (E2E Latency 측정용)

    // schemaVersion 1 (레거시) 전용 — 당분간 유지
    private String imageData;
    private float[] vectorData;
    private Map<String, Object> bbox;
    private float confidence;

    // schemaVersion 2 (신규) 전용
    private List<DetectionDto> detections;
}

