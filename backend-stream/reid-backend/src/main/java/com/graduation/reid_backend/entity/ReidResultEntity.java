package com.graduation.reid_backend.entity;

import jakarta.persistence.*; //역할: JPA의 모든 기능을 가져다 쓰겠다.
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.util.Map;


@Entity
@Table(name = "reid_result") //실제 DB에 만들어질 테이블 이름 명시
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder // 강력한 디자인 패턴 기능. 나중에 이 Entity에 데이터를 넣을때 어떤 값이 어디에 들어가는지 명시적으로 제시 가능
public class ReidResultEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id; // DB Primary Key (자동 증가)

    private String cctvId;

    private String locationId;

    private Integer frameNumber;

    private String timestamp;

    // 이미지는 Base64 텍스트이므로 길이가 매우 김 따라서 Large Object(LOB)로 지정.
    // [수정 1] 일반 RDBMS -> Vector DB: MySQL 표준인 LONGTEXT를 PostgreSQL 표준인 TEXT로 변경
    @Column(columnDefinition = "TEXT")
    private String imageData;

    // 일반 RDBMS는 벡터 배열을 원시 타입으로 저장하기 어렵다.
    // 따라서 JSON 형태의 긴 String 문자열("[0.12, -0.45, ...]")로 변환해서 통째로 저장
    // [수정 2] 일반 RDBMS -> Vector DB: String(JSON) -> 진짜 pgvector 배열로 변경
    @Column(columnDefinition = "vector(128)")
    private float[] vectorData;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(columnDefinition = "jsonb")
    private Map<String, Object> bbox;

    @Column
    private Float confidence;

    // 나중에 유사도 비교 후 판별된 동일인의 고유 ID를 업데이트할 필드
    private String personId;
}