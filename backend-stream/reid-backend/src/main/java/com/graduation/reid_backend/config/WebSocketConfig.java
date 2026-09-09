package com.graduation.reid_backend.config;

import com.graduation.reid_backend.handler.VideoFrameHandler;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;
import org.springframework.web.socket.server.standard.ServletServerContainerFactoryBean; // 이거 추가됨!

@Configuration
@EnableWebSocket //웹소켓 기능 켜줘 하는 스위치
@RequiredArgsConstructor
public class WebSocketConfig implements WebSocketConfigurer {

    private final VideoFrameHandler videoFrameHandler;

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        //프론트엔드가 ws://주소/ws/stream 에 들어오면, 무조건 videoFrameHandler 에게 연결하는 로직
        registry.addHandler(videoFrameHandler, "/ws/stream")
                .setAllowedOrigins("*"); //보안정책인데, 어느 도메인에서 요청이 오던 다 받아준다는 뜻
    }
    // 톰캣은 기본적으로 웹소켓으로 들어오는 텍스트 메시지를 무조건 8kb까지밖에 안받음
    // 웹소켓이 한 번에 받을 수 있는 데이터 크기 제한을 늘려주기!
    @Bean
    public ServletServerContainerFactoryBean createWebSocketContainer() {
        ServletServerContainerFactoryBean container = new ServletServerContainerFactoryBean();
        // Base64 이미지 텍스트를 넉넉하게 받을 수 있도록 최대 1MB(1024 * 1024)로 설정
        container.setMaxTextMessageBufferSize(1024 * 1024);
        container.setMaxBinaryMessageBufferSize(1024 * 1024);
        return container;
    }
}
