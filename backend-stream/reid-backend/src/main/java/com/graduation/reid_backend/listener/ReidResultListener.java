package com.graduation.reid_backend.listener;


import com.fasterxml.jackson.databind.ObjectMapper;
import com.graduation.reid_backend.dto.DetectionDto;
import com.graduation.reid_backend.dto.ReidResultDto;
import com.graduation.reid_backend.entity.ReidResultEntity;
import com.graduation.reid_backend.repository.ReidResultRepository;
import com.graduation.reid_backend.service.ReidService;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.Queue;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@Component
@RequiredArgsConstructor // 의존성 주입(DI): ObjectMapper와 레포지토리를 자동으로 가져오기.
public class ReidResultListener {

    private final ObjectMapper objectMapper; // JSON ➔ 자바객체 변환기
    private final ReidResultRepository reidResultRepository; // DB에 저장할 레포지토리
    private final ReidService reidService;


     // AI 서버가 결과를 던져주는 큐(reid.result.queue)를 실시간으로 감시하는 Listener
     //메시지가 큐에 들어오는 순간 이 메소드가 자동으로 즉시 깨어나서 실행.
     @RabbitListener(queuesToDeclare = @Queue("reid.result.queue"))
     @Transactional //N명의 데이터가 DB에 저장될 때, 하나라도 실패하면 전체 롤백되도록 트랜잭션 보장
     public void receiveReidResult(String message) {
         long listenerStartTime = System.currentTimeMillis();
        log.info("AI 서버로부터 분석 완료 메시지가 도착했습니다.");

        try {
            // 1. 수신된 JSON 텍스트 문자열을 자바에서 쓸 수 있게 DTO 객체로 변환.
            ReidResultDto dto = objectMapper.readValue(message, ReidResultDto.class);
            log.info("[DTO 객체 변환 성공] CCTV ID: {}, 프레임 번호: {}번", dto.getCctvId(), dto.getFrameNumber());

            // 1. 신/구 포맷 통합 로직 (Expand-Contract 패턴 적용)
            List<DetectionDto> detections = dto.getDetections();

            // 만약 AI 서버가 아직 구버전으로 보냈다면 (detections가 null이거나 비어있으면)
            if (detections == null || detections.isEmpty()) {
                DetectionDto singleDetection = DetectionDto.builder()
                        .imageData(dto.getImageData())
                        .vectorData(dto.getVectorData())
                        .bbox(dto.getBbox())
                        .confidence(dto.getConfidence())
                        .build();

                // 불변 리스트로 생성 (메모리 효율성 및 가독성 향상)
                detections = List.of(singleDetection);
            }

            int personCount = 0;

            // 2. 통합된 리스트(1명이든 N명이든)를 순회하며 개별 처리
            for (DetectionDto d : detections) {
                //2. 인물 검색 서비스 호출 및 식별 ID 부여
                String assignedPersonId = reidService.findPersonId(d.getVectorData(),dto.getLocationId());
                log.info("[식별 완료] AI 분석 결과와 기존 DB를 비교하여 ID 부여 완료: {}", assignedPersonId);

                // 3. 앞서 롬복으로 만들어둔 @Builder를 써서 DTO 데이터를 DB에 저장할 수 있게 Entity로 변환(공통 정보는 dto에서, 개인 정보는 d에서 추출)
                ReidResultEntity entity = ReidResultEntity.builder()
                        .cctvId(dto.getCctvId())
                        .locationId(dto.getLocationId())
                        .frameNumber(dto.getFrameNumber())
                        .timestamp(dto.getTimestamp())
                        .imageData(d.getImageData())
                        .vectorData(d.getVectorData())
                        .bbox(d.getBbox())
                        .confidence(d.getConfidence())
                        .personId(assignedPersonId)
                        .build();

                // 4. 데이터베이스에 저장
                reidResultRepository.save(entity);
                personCount++;
            }

            long endTime = System.currentTimeMillis();

            // 리스너 내부에서 소요된 시간 (JSON 파싱 + Vector 검색 + DB 저장)
            long dbAndBackendDuration = endTime - listenerStartTime;

            // [수정] 명확한 파이프라인 소요 시간 계산
            java.util.Map<String, Long> ts = dto.getTimestamps();
            if (ts != null && ts.containsKey("T1_Backend_Arrival")) {
                long t1Arrival = ts.get("T1_Backend_Arrival");
                long aiDuration = ts.getOrDefault("AI_Duration", 0L);

                // 전체 소요 시간 (백엔드 도착 ~ DB 저장 완료까지)
                long totalE2EDuration = endTime - t1Arrival;

                // 순수 네트워크 및 큐 대기 시간 (전체 시간 - AI 연산 시간 - DB/백엔드 연산 시간)
                long networkAndQueueDuration = totalE2EDuration - aiDuration - dbAndBackendDuration;

                log.info("=== 프레임 분석 완료 (CCTV: {}, Frame: {}) ===", dto.getCctvId(), dto.getFrameNumber());
                log.info(" 검출 인원: {}명", personCount);
                log.info(" 시스템 E2E 레이턴시: {}ms", totalE2EDuration);
                log.info("   ├─ AI 서버 추론 시간: {}ms", aiDuration);
                log.info("   ├─ DB 검색 및 저장 시간: {}ms", dbAndBackendDuration);
                log.info("   └─ 네트워크 및 큐 왕복 대기: {}ms", networkAndQueueDuration);
                log.info("=================================================");
            }


        } catch (Exception e) {
            log.error("AI 서버가 보낸 JSON 규격이 맞지 않거나 로직 처리 실패: {}", e.getMessage(), e);
        }
    }
}