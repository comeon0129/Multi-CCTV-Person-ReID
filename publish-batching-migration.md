# 발행(Publish) 배칭 — reid.result.queue 메시지 스키마 변경

- **작성일** : 2026-07-17
- **배경** : `rabbitmq-batch-consumer.md` §6-5에서, YOLO+ReID를 프레임 단위로 배치 처리해도 결과 발행(`basic_publish`)이 검출 인원수만큼 순차 반복되어 그 비용이 고스란히 남는 문제가 발견됨. §6-8의 GPU 활용도 실측(최대 부하에서도 연산 사용률 37% 미만)으로 "지연의 원인이 GPU 연산 부족이 아니라 발행 단계"라는 결론이 뒷받침됨.
- **범위** : `ai-server`(발행 방식 변경) + `backend-stream/reid-backend`(수신 스키마 변경).
- **배포 전략** : 스키마 자체는 호환 불가능한 변경(breaking change)이지만, **expand-contract 패턴**으로 배포 시점을 분리해 "동시 배포" 리스크를 없앤다 (§D 참고) — `schemaVersion` 필드 + `config.yaml` 플래그로 전환 시점을 코드 배포가 아닌 설정값 하나로 통제한다.

---

## A. 무엇을 바꾸는가 — `ai-server`

### 현재 (`main.py:166-188`, `_publish_and_ack`)

```python
def _publish_and_ack(channel, rmq_cfg, job, result):
    for detection in result["detections"]:
        message = {
            "cctvId": job["cctv_id"],
            "locationId": job["location_id"],
            "frameNumber": job["frame_number"],
            "timestamp": job["timestamp"],
            "imageData": detection["imageData"],
            "vectorData": detection["vectorData"],
            "bbox": detection["bbox"],
            "confidence": detection["confidence"],
            "timestamps": job["timestamps"],
        }
        channel.basic_publish(
            exchange="",
            routing_key=rmq_cfg["output_queue"],
            body=json.dumps(message),
            properties=pika.BasicProperties(content_type="application/json"),
        )
    channel.basic_ack(delivery_tag=job["delivery_tag"])
```

프레임에 사람이 N명 검출되면 `basic_publish`를 N번 호출 — 검출 0명이면 발행 없이 ack만.

### 변경 후 (제안) — `config.yaml` 플래그로 신/구 포맷을 동시에 지원

```yaml
# config.yaml
rabbitmq:
  ...
  publish_grouped: false   # true로 바꾸면 detections 배열로 프레임당 1회 발행 (기본값 false = 기존 방식 유지)
```

```python
def _publish_and_ack(channel, rmq_cfg, job, result):
    if not result["detections"]:
        channel.basic_ack(delivery_tag=job["delivery_tag"])
        return

    if rmq_cfg.get("publish_grouped", False):
        # 신규: 프레임당 메시지 1건
        message = {
            "schemaVersion": 2,
            "cctvId": job["cctv_id"],
            "locationId": job["location_id"],
            "frameNumber": job["frame_number"],
            "timestamp": job["timestamp"],
            "timestamps": job["timestamps"],
            "detections": [
                {k: d[k] for k in ("imageData", "vectorData", "bbox", "confidence")}
                for d in result["detections"]
            ],
        }
        channel.basic_publish(
            exchange="", routing_key=rmq_cfg["output_queue"],
            body=json.dumps(message),
            properties=pika.BasicProperties(content_type="application/json"),
        )
    else:
        # 레거시: 검출 인원수만큼 반복 발행 (현재 동작과 동일)
        for detection in result["detections"]:
            message = {
                "schemaVersion": 1,
                "cctvId": job["cctv_id"],
                "locationId": job["location_id"],
                "frameNumber": job["frame_number"],
                "timestamp": job["timestamp"],
                "imageData": detection["imageData"],
                "vectorData": detection["vectorData"],
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "timestamps": job["timestamps"],
            }
            channel.basic_publish(
                exchange="", routing_key=rmq_cfg["output_queue"],
                body=json.dumps(message),
                properties=pika.BasicProperties(content_type="application/json"),
            )

    channel.basic_ack(delivery_tag=job["delivery_tag"])
```

- `publish_grouped` 플래그가 꺼져 있으면(`false`, 기본값) **지금과 100% 동일하게 동작** — 이 커밋 자체는 배포해도 아무 위험이 없음.
- 플래그를 켜는 순간부터만 프레임당 1회(`detections` 배열)로 발행 방식이 바뀜 — 즉 **코드 배포와 실제 동작 전환 시점을 분리**할 수 있음 (§D).
- `schemaVersion`을 메시지에 명시해서, 백엔드가 어떤 포맷인지 필드 유무 추론 대신 명확히 분기할 수 있게 함.
- `cctvId`/`locationId`/`frameNumber`/`timestamp`/`timestamps`는 프레임 공통 값이라 신규 포맷에서는 최상위에 한 번만 — 기존에는 검출마다 동일한 값(특히 `timestamps` 딕셔너리 전체)이 중복 직렬화되고 있었음.

`run_reid_batch`, `crop_to_base64` 등 나머지는 변경 없음 — 프레임 내부 처리 로직은 그대로이고 **발행 방식만** 바뀜.

---

## B. 어떤 문제가 해결되는가

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 프레임당 `basic_publish` 호출 | 검출 인원 수만큼 (평균 4.3~4.7회) | 최대 1회 |
| 프레임당 발행 네트워크 왕복 비용 | §4 실측 기준 ~14ms × 4.3~4.7 ≈ **60~70ms** | ~14ms |
| `timestamps` 딕셔너리 직렬화 | 검출 인원 수만큼 중복 | 프레임당 1회 |

- §6-5에서 지목된 "프레임당 p50 지연 240~420ms 바닥"의 상당 부분을 차지하던 순차 발행 비용이 제거됨.
- **주의**: 이 변경만으로 T4-T3 전체가 크게 줄어드는 건 아님. 큐 대기(`_collect_batch`)와 실제 forward 시간은 그대로 남기 때문에, 이번 변경 이후 `rabbitmq-batch-consumer.md` §5의 `max_size`/`max_wait_ms` 실험을 재실행해서 실제 개선폭을 다시 측정해야 함 (§6-7에 이미 "발행 문제 해결 후 §5 재실험" 순서로 명시돼 있음).

---

## C. `backend-stream/reid-backend`에 주는 영향

메시지가 "1메시지 = 1인" → "1메시지 = 1프레임(N명 배열)"로 바뀌므로 수신 쪽 수정이 필요함. 단, §D의 배포 전략에 따라 **신/구 포맷을 동시에 파싱할 수 있게** 만드는 것이 핵심 — 한쪽만 지원하도록 교체하면 안 됨.

### `dto/ReidResultDto.java` — 신/구 필드를 모두 가진 채로 확장

```java
public class ReidResultDto {
    private Integer schemaVersion;      // null 또는 1 = 레거시, 2 = 신규(그룹)
    private String cctvId;
    private String locationId;
    private Integer frameNumber;
    private String timestamp;
    private Map<String, Long> timestamps;

    // schemaVersion 1 (레거시) 전용 — 당분간 유지
    private String imageData;
    private List<Float> vectorData;
    private Object bbox;
    private Float confidence;

    // schemaVersion 2 (신규) 전용
    private List<DetectionDto> detections;
}

// 신설
public class DetectionDto {
    private String imageData;
    private List<Float> vectorData;
    private Object bbox;
    private Float confidence;
}
```

레거시 메시지에는 `detections`가 아예 없으므로 Jackson이 `null`로 남기고, 신규 메시지에는 `imageData` 등 최상위 필드가 없으므로 마찬가지로 `null` — 어느 쪽이 와도 파싱 자체는 실패하지 않음.

### `listener/ReidResultListener.java` (`receiveReidResult`) — 두 포맷을 하나의 경로로 통합 처리

```java
List<DetectionDto> detections = (dto.getDetections() != null)
        ? dto.getDetections()
        : List.of(DetectionDto.builder()
                .imageData(dto.getImageData())
                .vectorData(dto.getVectorData())
                .bbox(dto.getBbox())
                .confidence(dto.getConfidence())
                .build());

for (DetectionDto d : detections) {
    String personId = reidService.findPersonId(d.getVectorData());
    // ... 기존 Entity 빌드/저장 로직을 d 기준으로 수행
}
```

- `reidService.findPersonId`는 벡터 하나를 받는 시그니처 그대로 재사용 — 사람마다 개별 식별이므로 루프 안으로 옮기기만 하면 됨. DB 저장 횟수 자체는 변하지 않음.
- E2E 레이턴시 로그(`T2-T1`~`T5-T4`)는 현재 검출 인원수만큼 중복 출력되는데, 신규 포맷에서는 **프레임당 1회만** 출력됨 — 로그량이 줄어드는 부수 효과. 이 로그를 사람 수 기준으로 집계하는 별도 도구가 있다면 빈도 가정이 깨지므로 확인 필요.

### 그 외 확인 필요 지점

- `ReidService.findPersonId` 자체 시그니처는 벡터 하나를 받는 구조라 변경 불필요 — 호출부(Listener)에서 루프 안으로 옮기기만 하면 됨.
- 메시지 페이로드 크기 증가: 프레임당 최대 12명(관측된 최대 검출 인원, `reid_experiment_log.md` 기준) × (crop 이미지 base64 + 128차원 벡터)만큼 메시지 하나에 들어감. RabbitMQ 기본 최대 메시지 크기(128MB)엔 크게 못 미치지만, CloudAMQP 무료 티어의 flow control(§6-6에서 이미 관찰된 이슈)을 자극할 가능성은 있어 배포 후 모니터링 권장.
- DB INSERT 트랜잭션 경계: 현재는 검출 1건 = 1 트랜잭션(암묵적)으로 저장되는 구조로 보이는데, 프레임 단위로 묶어서 처리할지(N건을 한 트랜잭션으로) 여부는 백엔드 쪽 판단 필요 — 이 문서 범위 밖.

---

## D. 배포 순서 — expand-contract로 "동시 배포" 자체를 없앤다

원래 이 변경은 스키마가 호환 불가능하게 바뀌는 breaking change라 동시 배포가 필요해 보이지만, §A의 `publish_grouped` 플래그와 §C의 이중 파싱 DTO를 쓰면 **세 단계로 쪼개서 각각 독립적으로, 안전하게** 배포할 수 있다. 표준적인 **expand → migrate → contract** 패턴.

### 1단계 (Expand) — backend만 먼저 배포

- §C의 `ReidResultDto`/`ReidResultListener` 변경(신·구 포맷 동시 지원)을 배포.
- 이 시점에 ai-server는 여전히 `publish_grouped: false` 상태 → 레거시 포맷만 보냄 → 새로 배포된 backend는 `detections == null`이므로 자동으로 레거시 분기를 타서 **기존과 완전히 동일하게 동작**.
- 리스크: 사실상 없음. 배포 후 원하는 만큼 오래 지켜봐도 됨 (다음 단계와 시간적으로 묶여있지 않음).

### 2단계 (Migrate) — ai-server 플래그 전환

- ai-server를 재배포하거나(§A 코드 포함) 이미 배포되어 있다면 `config.yaml`의 `publish_grouped`만 `true`로 바꾸고 재시작.
- 이 시점부터 `schemaVersion: 2` 그룹 메시지가 나가기 시작하고, 1단계에서 이미 배포된 backend가 `detections != null` 분기로 정상 처리.
- **롤백이 설정값 하나로 끝남**: 문제가 보이면 `publish_grouped: false`로 되돌리고 재시작만 하면 즉시 레거시 방식으로 복귀 — backend를 다시 건드릴 필요 없음.
- 원하면 테스트 큐 하나에서 먼저 플래그를 켜서 §6-8처럼 격리된 검증을 해본 뒤, 프로덕션 큐에 반영해도 됨.

### 3단계 (Contract) — 정리

- 그룹 포맷이 충분히 안정적으로 운영됨을 확인한 뒤(예: 1~2주 모니터링), 별도 정리 PR로:
  - ai-server: `publish_grouped` 플래그와 레거시(`schemaVersion: 1`) 분기 제거
  - backend: `ReidResultDto`의 레거시 flat 필드와 분기 제거
- 이 단계는 급하지 않고, 두 저장소를 각자 편한 시점에 따로 정리해도 무방 — 이미 신규 포맷만 쓰이고 있어 순서 무관.

### 기존 "동시 배포" 대비 이득

| | 동시 배포 | expand-contract |
|---|---|---|
| 배포 타이밍 제약 | ai-server/backend를 정확히 같은 순간에 맞춰야 함 | 세 단계 모두 독립적 시점에 배포 가능 |
| 실패 시 조치 | 양쪽 다 롤백해야 함 | 설정값(`publish_grouped`) 하나만 되돌리면 됨 |
| 검증 방법 | 스테이징에서 통째로 리허설 필요 | 2단계를 테스트 큐 → 프로덕션 큐 순으로 점진 적용 가능 |

---

## E. 체크리스트

**1단계 (Expand — backend)**
- [x] `ReidResultDto`에 `schemaVersion`/`detections`/`DetectionDto` 추가 (레거시 필드는 유지)
- [x] `ReidResultListener.receiveReidResult`: `detections != null` 여부로 분기해 통합 루프로 처리하도록 수정
- [ ] 배포 후 기존과 동일하게 동작하는지 확인 (레거시 경로만 타는 상태)

**2단계 (Migrate — ai-server)**
- [x] `ai-server/config.yaml`에 `rabbitmq.publish_grouped: false` 추가
- [x] `main.py`의 `_publish_and_ack` 수정 (§A) — 플래그 분기 포함, 기본값 `false`로 배포
- [ ] 테스트 큐에서 `publish_grouped: true`로 격리 검증 (§6-8 패턴)
- [ ] 프로덕션 `config.yaml`에서 `publish_grouped: true`로 전환 + 재시작
- [ ] 문제 발생 시 `publish_grouped: false`로 즉시 롤백 가능한 상태인지 확인

**3단계 (Contract — 정리, 안정화 이후)**
- [ ] ai-server: `publish_grouped` 플래그와 레거시 발행 분기 제거
- [ ] backend: `ReidResultDto` 레거시 필드와 분기 제거

**공통**
- [ ] 배포 후 `rabbitmq-batch-consumer.md` §5 실험 재실행 (실제 개선폭 재측정)
- [ ] CloudAMQP 메시지 크기/flow control 모니터링 (§C 참고)
