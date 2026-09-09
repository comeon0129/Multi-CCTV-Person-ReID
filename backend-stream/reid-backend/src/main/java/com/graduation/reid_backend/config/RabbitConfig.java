package com.graduation.reid_backend.config;

import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.DirectExchange;
import org.springframework.amqp.core.Queue;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitConfig {

    // 우리가 택배를 던져놓을 큐의 이름
    public static final String QUEUE_NAME = "ai.image.queue";
    // 우체국 분류 센터의 이름
    public static final String EXCHANGE_NAME = "reid.exchange";
    // 분류 센터에서 큐로 찾아가는 길 안내판(라우팅 키)
    public static final String ROUTING_KEY = "ai.image.route";

    @Bean
    public Queue imageQueue() {
        return new Queue(QUEUE_NAME, true); // true: 서버가 재시작되어도 큐가 날아가지 않음
    }

    //중앙 분류 센터(exchange)를 만들고 주소가 정확히 일치하는 곳으로 택배 보냄.
    @Bean
    public DirectExchange reidExchange() {

        return new DirectExchange(EXCHANGE_NAME);
    }

    //분류 센터(exchange)와 창고(queue) 연결하는 컨베이어벨트.
    //exchange에 도착한 택배중에 송장에 ROUTING_KEY(ai.image.route)라고 적힌 것들은 무조건 ai.image.queue로 보내게 만듬
    @Bean
    public Binding binding(Queue imageQueue, DirectExchange reidExchange) {
        return BindingBuilder.bind(imageQueue).to(reidExchange).with(ROUTING_KEY);
    }
}