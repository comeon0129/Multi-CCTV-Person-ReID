package com.graduation.reid_backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.graduation.reid_backend.dto.ReidResultDto;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.amqp.rabbit.core.RabbitTemplate;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class RabbitMqIntegrationTest {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void AI_서버인척_가짜_메시지_쏘기_테스트() throws Exception {
        Map<String, Long> ts = new HashMap<>();
        ts.put("T1_Backend_Arrival", System.currentTimeMillis() - 100);
        ts.put("T2_Queue_Push", System.currentTimeMillis() - 80);

        // 2048차원의 가짜 벡터 배열 생성
        float[] fakeVector = new float[128];
        for (int i = 0; i < 128; i++) {
            fakeVector[i] = (float) Math.random(); // 0.0 ~ 1.0 사이의 랜덤 값으로 128칸을 채움
        }

        Map<String, Object> fakeBbox = new HashMap<>();
        fakeBbox.put("x", 0.0);
        fakeBbox.put("y", 0.0);
        fakeBbox.put("w", 100.0);
        fakeBbox.put("h", 100.0);

        //1. DTO 구조와 완벽히 일치하도록 데이터 세팅
        ReidResultDto mockDto = ReidResultDto.builder()
                .cctvId("cctv_test_01")
                .locationId("zone_A")
                .frameNumber(1)
                .timestamp("2026-05-27T03:30:00")
                .imageData("base64_fake_image...")
                .vectorData(fakeVector)
                .bbox(fakeBbox)
                .confidence(0.9f)
                .timestamps(ts)
                .build();

        // 2. DTO를 JSON 문자열로 변환
        String jsonPayload = objectMapper.writeValueAsString(mockDto);


        // 첫 번째 테스트: Cache Miss 발생 예상
        rabbitTemplate.convertAndSend("reid.result.queue", jsonPayload);
        System.out.println("첫 번째 가짜 메시지 전송 완료! (Cache Miss 예상)");

        // DB 저장 및 캐시 갱신될 시간을 넉넉히 2초 정도 줍니다.
        // TTL 10초 이내이므로 캐시가 살아있습니다.
        Thread.sleep(2000);

        //두 번째 테스트: Cache Hit 발생 예상
        // frameNumber만 2번으로 바꿔서 똑같은 벡터 전송
        ts.put("T1_Backend_Arrival", System.currentTimeMillis() - 100);
        mockDto.setFrameNumber(2);
        mockDto.setTimestamps(ts); // 갱신된 시간 덮어쓰기
        String jsonPayload2 = objectMapper.writeValueAsString(mockDto);

        rabbitTemplate.convertAndSend("reid.result.queue", jsonPayload2);
        System.out.println("두 번째 가짜 메시지 전송 완료! (Cache Hit 예상)");

        // Listener가 두 번째 메시지를 처리할 때까지 대기
        Thread.sleep(3000);
    }
}