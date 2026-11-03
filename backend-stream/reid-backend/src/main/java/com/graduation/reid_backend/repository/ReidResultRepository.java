package com.graduation.reid_backend.repository;

import com.graduation.reid_backend.dto.NearestPerson;
import com.graduation.reid_backend.entity.ReidResultEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface ReidResultRepository extends JpaRepository<ReidResultEntity, Long> {

    // 1. 반환 타입을 List가 아닌 단일 객체(NearestPerson)로 변경
    // 2. SQL에서 personId와 distance를 각각 SELECT (인터페이스 getter 이름과 매칭)
    @Query(value = """
        SELECT person_id AS personId, 
               (vector_data <=> CAST(:vector AS vector)) AS distance 
        FROM reid_result 
        WHERE location_id = :locationId 
        ORDER BY distance 
        LIMIT 1
        """, nativeQuery = true)
    NearestPerson findNearestPersonInLocation(@Param("locationId") String locationId, @Param("vector") String vector);
}
