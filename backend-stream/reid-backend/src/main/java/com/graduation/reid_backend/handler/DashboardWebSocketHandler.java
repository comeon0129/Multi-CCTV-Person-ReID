package com.graduation.reid_backend.handler;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

@Slf4j
@Component
public class DashboardWebSocketHandler extends TextWebSocketHandler {

    // 현재 대시보드 페이지를 켜놓고 있는 브라우저들의 목록을 안전하게 저장
    private static final List<WebSocketSession> sessions = new CopyOnWriteArrayList<>();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        sessions.add(session);
        log.info("대시보드 클라이언트 접속! 현재 연결 수: {}", sessions.size());
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        sessions.remove(session);
        log.info("대시보드 클라이언트 연결 종료. 남은 연결 수: {}", sessions.size());
    }

    // 백엔드 로직(Listener)에서 이 메소드를 호출하면 모든 대시보드에 데이터가 쫙 뿌려짐
    public void broadcast(String message) {
        for (WebSocketSession session : sessions) {
            if (session.isOpen()) {
                try {
                    session.sendMessage(new TextMessage(message));
                } catch (IOException e) {
                    log.error("대시보드로 메시지 전송 실패", e);
                }
            }
        }
    }
}
