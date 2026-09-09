package com.graduation.reid_backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.GenericJacksonJsonRedisSerializer;
import org.springframework.data.redis.serializer.StringRedisSerializer;

// @Configuration: 스프링 부트에게 "이 클래스는 환경설정 파일이야. 서버 켜질 때 제일 먼저 읽어줘!"라고 알려주는 역할
@Configuration
public class RedisConfig {
    // @Bean: 스프링의 객체 관리소(IoC 컨테이너)에 이 메소드가 반환하는 'RedisTemplate'을 등록해 둠
    // 나중에 다른 서비스(ReidService 등)에서 필요할 때마다 스프링이 알아서 주입해 줌
    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory connectionFactory) {
        //1. Redis와 통신할 수 있는 템플릿 객체 생성
        RedisTemplate<String, Object> template = new RedisTemplate<>();

        //2. 어떤 Redis 서버(호스트,포트)와 연결할지 공장 설정 세팅
        template.setConnectionFactory(connectionFactory);

        // Spring Boot 4.0 이상 최신 방식: Builder 패턴을 사용해 직렬화 설정
        // 직렬화 하는 이유: 기본설정을 쓰면 Sping은 데이터를 사람이 알아볼 수 없는 바이트 코드로 Redis에 저장하는데
        // 이러면 나중에 Redis 터미널에 들어가서 직접 데이터를 눈으로 보고 디버깅 불가. 그래서 규칙 명시
        //Java의 메모리 데이터(객체)를 Redis가 저장할 수 있는 바이트/문자열 형태로 바꾸는 과정
        GenericJacksonJsonRedisSerializer serializer = GenericJacksonJsonRedisSerializer.builder()
                .enableUnsafeDefaultTyping() // 기존처럼 클래스 타입(예: float[])을 JSON에 같이 저장하도록 허용
                .build();

        // Key는 사람이 읽기 쉬운 문자열, Value는 JSON 형태로 저장되도록 직렬화 설정

        //3. 일반 Key-Value 저장 할 때의 규칙
        template.setKeySerializer(new StringRedisSerializer());
        template.setValueSerializer(serializer);

        //4. Hash 구조(HSET,HGET등) 저장시의 규칙
        template.setHashKeySerializer(new StringRedisSerializer());
        template.setHashValueSerializer(serializer);

        return template;
    }
}