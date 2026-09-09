package com.graduation.reid_backend.handler;

import com.graduation.reid_backend.config.RabbitConfig;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.concurrent.atomic.AtomicInteger;

@Slf4j
@Component
@RequiredArgsConstructor // 롬복이 아래 RabbitTemplate을 자동으로 주입해 줌!
public class VideoFrameHandler extends TextWebSocketHandler { //텍스트로 된 웹소켓메시지(base64)전문적으로 처리

    private final AtomicInteger frameCount = new AtomicInteger(0);

    // 택배 배달을 담당하는 스프링의 기본 우체부 객체
    private final RabbitTemplate rabbitTemplate;

    //json 데이터를 자유자재로 조작하게 해주는 스프링 기본 도구
    private final ObjectMapper objectMapper;


    //프론트엔드에서 이미지 쏘는순간 자동 실행되는 메소드
    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        long startTime = System.currentTimeMillis();
        int currentFrame = frameCount.incrementAndGet();

        // 1. 프론트엔드에서 날아온 JSON 텍스트(cctvId, timestamp, imageData 포함) 받기
        String payload = message.getPayload();

        // 2. 문자열을 다루기 쉬운 JSON 객체로 변환
        ObjectNode jsonNode = (ObjectNode) objectMapper.readTree(payload);

        // 3. 타임스탬프 기록을 위한 노드 생성 및 T1(백엔드 도착 시간) 기록
        ObjectNode timestampNode = objectMapper.createObjectNode();
        timestampNode.put("T1_Backend_Arrival", startTime);

        // 4. 프레임 순서 번호(frameNumber)만 JSON에 쓱 얹어주기
        jsonNode.put("frameNumber", currentFrame);

        // 5. RabbitMQ로 송신 직전 T2(큐 전송 시간) 기록
        timestampNode.put("T2_Queue_Push", System.currentTimeMillis());

        // 6. jsonNode에 timestamps 노드 삽입
        jsonNode.set("timestamps", timestampNode);

        // 7. 다시 문자열로 예쁘게 포장하기
        String finalJsonPayload = objectMapper.writeValueAsString(jsonNode);

        // 8. 포장된 JSON을 reid.exchange 분류 센터로 던지라고 알려주기. 단 방향(ROUTING_KEY)은 ai.image.route로 적어주기
        rabbitTemplate.convertAndSend(RabbitConfig.EXCHANGE_NAME, RabbitConfig.ROUTING_KEY, finalJsonPayload);

        long endTime = System.currentTimeMillis();
        log.info("[발송 완료] {}번째 프레임 RabbitMQ 전송 끝! 소요 시간: {}ms\n-------------------------------------",
                currentFrame, (endTime - startTime));
    }
}