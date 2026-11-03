package com.graduation.reid_backend.config;

import com.graduation.reid_backend.entity.ReidResultEntity;
import com.graduation.reid_backend.repository.ReidResultRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.*;

@Slf4j
@Component
@RequiredArgsConstructor
public class DummyDataLoader implements CommandLineRunner { //이 인터페이스를 구현하면 스프링 부트 애플리케이션이 실행된 직후 run() 메소드가 딱 한번 자동으로 실행됨.

    private final ReidResultRepository repository;

    @Override
    public void run(String... args) throws Exception {
        // 이미 데이터가 존재하면 생성하지 않음
        if (repository.count() > 0) {
            log.info("이미 데이터가 존재하여 더미 데이터 생성을 생략합니다.");
            return;
        }

        int totalDataSize = 10000; // 목표 데이터: 50만 건
        int batchSize = 10000;      // 1만 건씩 쪼개서 저장 (메모리 보호)
        log.warn("병목 성능 측정을 위한 {}건의 더미 데이터 생성을 시작합니다.",totalDataSize);
        Random random = new Random();
        String[] locations = {"zone_A", "zone_B", "zone_C"}; // 테스트용 구역

        for (int i = 0; i < totalDataSize / batchSize; i++) {
            List<ReidResultEntity> dummyList = new ArrayList<>();

            for (int j = 0; j < batchSize; j++) {
                int currentIndex = (i * batchSize) + j;

                // 128차원 랜덤 벡터 생성
                float[] fakeVector = new float[128];
                for (int v = 0; v < 128; v++) {
                    fakeVector[v] = random.nextFloat();
                }

                Map<String, Object> fakeBbox = new HashMap<>();
                fakeBbox.put("x", 0.0);
                fakeBbox.put("y", 0.0);
                fakeBbox.put("w", 100.0);
                fakeBbox.put("h", 100.0);

                ReidResultEntity entity = ReidResultEntity.builder()
                        .cctvId("cctv_" + (currentIndex % 10))
                        .locationId(locations[currentIndex % 3])
                        .personId("Person_Dummy_" + currentIndex)
                        .vectorData(fakeVector)
                        .bbox(fakeBbox)
                        .confidence(0.9f)
                        .build();
                dummyList.add(entity);
            }

            repository.saveAll(dummyList);
            log.info("{} / {} 건 더미 데이터 삽입 완료...", (i + 1) * batchSize, totalDataSize);
        }

        log.warn("총 {}건 더미 데이터 삽입 완료. 검색 성능 테스트 준비가 완료되었습니다.", totalDataSize);
    }
}
