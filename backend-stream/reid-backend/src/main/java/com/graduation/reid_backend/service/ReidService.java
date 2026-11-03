package com.graduation.reid_backend.service;

import com.graduation.reid_backend.dto.NearestPerson;
import com.graduation.reid_backend.repository.ReidResultRepository;
import com.graduation.reid_backend.util.VectorUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import org.springframework.data.redis.core.RedisTemplate;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;

@Slf4j
@Service
@RequiredArgsConstructor
public class ReidService {

    private final ReidResultRepository repository;
    private final RedisTemplate<String, Object> redisTemplate;

    private static final double DISTANCE_THRESHOLD = 0.35 ;

    //Redis에 저장할 Key이름의 접두사 상수(나중에 key 이름 들이 뒤죽박죽 섞이면 관리불가)
    //도메인:목적:세부구분 형태로 하는게 기본 관례
    private static final String CACHE_KEY_PREFIX = "reid:location";


    private final AtomicInteger totalQueries = new AtomicInteger(0);
    private final AtomicInteger cacheHits = new AtomicInteger(0);

    public String findPersonId(float[] newVector, String locationId) {
        totalQueries.incrementAndGet(); // 시작할 때 무조건 1 증가

        // 1. Redis에서 찾을 Key 이름 생성 (예: "reid:location:zone_A")
        String redisKey = CACHE_KEY_PREFIX + locationId;

        // opsForHash().entries(redisKey)는 Redis의 'HGETALL' 명령어와 같음(사람마다 개별적으로 HGET 호출 시 레이턴시 낭비됨)
        // 2. 해당 구역(zone_A)에 있는 모든 사람의 <Person ID, 128차원 벡터> 쌍을 Map 형태로 한 번에 가져오기
        Map<Object, Object> cachedPersons = redisTemplate.opsForHash().entries(redisKey);

        log.info("[Redis 조회] {} 구역에서 캐싱된 인물 {} 명을 메모리로 가져왔습니다.", locationId, cachedPersons.size());

        // 3. 인메모리 코사인 거리 계산(Cache hit 판별)
        String bestMatchPersonId = null;
        float minDistance = Float.MAX_VALUE;

        //Redis에서 가져온 최근 인물들을 하나씩 꺼내서 현재 인물과 벡터 비교
        for (Map.Entry<Object, Object> entry : cachedPersons.entrySet()) {
            String cachedPersonId = (String) entry.getKey();
            //Redis에서 안전한 String 형태로 꺼낸 뒤, float 배열로 수동 복원
            String cachedVectorStr = (String) entry.getValue();
            // "[0.12, 0.34]" 형태의 문자열에서 괄호를 떼고 쉼표로 자름
            String[] stringValues = cachedVectorStr.substring(1, cachedVectorStr.length() - 1).split(",");

            float[] cachedVector = new float[128];
            for (int i = 0; i < 128; i++) {
                // 공백을 제거하고 다시 float 숫자로 변환
                cachedVector[i] = Float.parseFloat(stringValues[i].trim());
            }

            // 2. VectorUtils를 사용해 거리 계산
            float distance = VectorUtils.calculateCosineDistance(newVector, cachedVector);

            // 3. 기존에 찾은 최소 거리보다 더 가깝다면 정답 후보 갱신
            if (distance < minDistance) {
                minDistance = distance;
                bestMatchPersonId = cachedPersonId;
            }
        }

        //4. 찾은 가장 가까운 사람이 임계치(threshold) 이내면 Cache Hit
        if (bestMatchPersonId != null && minDistance <= DISTANCE_THRESHOLD) {
            cacheHits.incrementAndGet(); //캐시 히트 시 1증가
            log.info("[Cache Hit] Redis 메모리에서 인물 매칭 성공! ID: {}, 거리: {}", bestMatchPersonId, minDistance);

            // 방금 또 카메라에 찍혔으므로, 이 구역(locationId) 캐시 데이터의 수명을 다시 10초로 연장
            redisTemplate.expire(redisKey, Duration.ofSeconds(10));

            // DB 조회 없이 메소드 즉시 종료하고 결과 반환 (DB I/O 없음)
            return bestMatchPersonId;
        }

        //5. Cache Miss 처리 및 TTL 등록(DB 조회 및 Redis 업데이트)
        log.info("[Cache Miss] Redis에 조건에 맞는 인물이 없습니다. DB 조회를 시작합니다.");

        /*기존 로직 부분*/
        // 1. 벡터를 DB용 문자열로 변환
        String vectorStr = Arrays.toString(newVector);
        // 2. DB에서 가장 가까운 사람과 거리(distance)만 조회
        NearestPerson nearest = repository.findNearestPersonInLocation(locationId,vectorStr);

        String finalPersonId;

        // 3. 결과 처리
        if (nearest != null && nearest.getDistance() <= DISTANCE_THRESHOLD) {
            log.info("[인물 매칭 성공!] ID: {}, 거리: {}", nearest.getPersonId(), nearest.getDistance());
            finalPersonId = nearest.getPersonId();
        } else {
            // 매칭되는 사람이 없거나 거리가 너무 멂
            finalPersonId = "Person_" + UUID.randomUUID().toString().substring(0, 5);
            log.info("[새로운 인물 발견!] 새 ID 발급: {}", finalPersonId);
        }

        //6. 찾은/새로 발급한 인물 정보를 Redis에 저장하고 TTL 10초 갱신
        redisTemplate.opsForHash().put(redisKey, finalPersonId, vectorStr);
        redisTemplate.expire(redisKey, Duration.ofSeconds(10));

        log.info("[Redis 업데이트 완료] 구역: {}, ID: {} - 캐시 10초 수명으로 저장됨", locationId, finalPersonId);

        return finalPersonId;
    }

    public double getCacheHitRate() {
        int total = totalQueries.get();
        if (total == 0) return 0.0;
        return Math.round(((double) cacheHits.get() / total * 100) * 10.0) / 10.0;
    }
}