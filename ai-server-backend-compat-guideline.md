# ai-server ↔ backend 메시지 호환성 — 조치 가이드라인

- **작성일** : 2026-07-24
- **작성 배경** : `origin/main`의 실제 `backend-stream/reid-backend`를 pull해서 확인한 결과, 로컬(`exp/pca-whitening-dim-reduction`)에 남아있던 `publish-batching-migration.md` 1단계(Expand) 작업은 **커밋/푸시되지 않은 로컬 전용 코드**였고, 동료가 실제로 push한 백엔드에는 해당 작업이 전혀 반영되어 있지 않음이 확인됨.
- **담당 범위** : 이 문서는 `ai-server`만 담당하는 입장에서, 백엔드 코드를 직접 고치지 않고 **ai-server 쪽에서 무엇을 어떻게 바꿔야 지금 당장 통신이 살아나는지**에 초점을 맞춘다.

---

## 1. 지금 당장 깨지는 문제 — `schemaVersion` 필드

### 원인
- ai-server의 `_publish_and_ack`(`main.py`)는 `publish_grouped` 값과 무관하게 모든 발행 메시지에 `"schemaVersion": 1` 또는 `2`를 항상 포함한다.
- 실제 `origin/main`의 `ReidResultDto.java`에는 `schemaVersion` 필드가 없다.
- `ReidBackendApplication.java`의 `ObjectMapper` 빈은 `new ObjectMapper()` 그대로라 `FAIL_ON_UNKNOWN_PROPERTIES`가 Jackson 기본값(`true`)으로 켜져 있다.

### 결과
- 백엔드가 `objectMapper.readValue(message, ReidResultDto.class)`에서 예외를 던지고, `ReidResultListener`의 `catch (Exception e)` 블록에서 로그만 남기고 조용히 버려진다.
- 앱이 죽지는 않지만 **모든 프레임의 ReID 결과가 DB에 저장되지 않는다** — 배포하면 100% 이 상태가 된다.

### 조치 (ai-server 쪽에서 선택)

**옵션 A — `schemaVersion` 필드를 아예 빼고 실제 백엔드 DTO와 100% 동일한 형태로 되돌리기 (권장, 즉시 적용 가능)**

- `main.py`의 `_publish_and_ack` 레거시 분기에서 `"schemaVersion": 1` 키를 제거한다.
- `publish_grouped` 관련 신규 분기(`schemaVersion: 2`, `detections` 배열)는 **백엔드에 대응 코드가 생기기 전까지 발행하지 않는다** — `config.yaml`의 `publish_grouped`를 코드에서 아예 참조하지 않거나, 최소한 `true`로 절대 켜지 않는다.
- 즉, ai-server가 보내는 메시지 필드를 실제 `origin/main`의 `ReidResultDto`와 정확히 맞춘다:

  | 필드 | 백엔드 타입 | 비고 |
  |---|---|---|
  | `cctvId` | `String` | |
  | `locationId` | `String` | `ReidService.findPersonId`가 이제 이 값으로 pgvector 검색 범위를 좁히므로 **반드시 항상 값이 있어야 함** |
  | `frameNumber` | `Integer` | |
  | `timestamp` | `String` | |
  | `imageData` | `String` | |
  | `vectorData` | `float[]` | JSON 배열이면 되므로 Python list와 호환 |
  | `bbox` | `Map<String,Object>` | JSON 객체(`{"x":.., "y":.., "w":.., "h":..}`) 형태만 유지하면 호환 |
  | `confidence` | `float` (primitive) | `null` 전송 금지 — 항상 값이 있어야 함 |
  | `timestamps` | `Map<String,Long>` | |

**옵션 B — 동료가 백엔드 DTO에 `schemaVersion`(및 `detections`)을 먼저 추가할 때까지 배포 보류**

- 동료 쪽에 `ReidResultDto`에 `schemaVersion` 필드 추가를 요청하고, 배포 완료를 확인한 뒤에만 ai-server의 현재 코드를 그대로 배포.
- 이 경우 별도 조치 없이 지금 코드 유지 가능하지만, **동료 배포가 완료되기 전까지는 지금 상태로 배포하면 안 됨**.

> 담당 범위(ai-server)만 고려하면 **옵션 A가 더 빠르고 확실하다** — 백엔드 일정에 의존하지 않고 지금 바로 안전한 상태로 되돌릴 수 있다.

### 체크리스트
- [ ] `main.py` `_publish_and_ack`에서 `schemaVersion` 키 제거 (또는 옵션 B 선택 시 동료 배포 확인 후 보류)
- [ ] `publish_grouped: true`로 전환하는 코드 경로가 실수로도 켜지지 않도록 `config.yaml` 기본값 `false` 유지
- [ ] 위 표의 필드/타입 기준으로 실제 발행 메시지 JSON을 한번 눈으로 확인 (`curl`로 큐 peek 하거나 로그 출력)

---

## 2. `publish_grouped` (그룹 발행) 기능 — 백엔드에 아무 기반이 없음

### 현황
- `publish-batching-migration.md`가 설계한 expand-contract 배포 전략(§D)은 **백엔드가 1단계(Expand: `schemaVersion`/`detections` 파싱 지원)를 먼저 배포해야 안전**하다는 전제였다.
- 그런데 실제 `origin/main`에는 `detections` 필드도, `schemaVersion` 분기 처리도 전혀 없다. 즉 문서가 가정한 "1단계는 이미 완료" 상태가 **사실이 아니었다** — 로컬에만 있던 미푸시 코드였다.
- 지금 이 기능을 ai-server 쪽에서 임의로 진행(플래그를 켜거나, 관련 로직을 더 다듬는 것)해봐야 받아줄 백엔드가 없으므로 의미가 없다.

### 조치
- **ai-server 단독으로 이 기능을 진행하지 않는다.** 아래 내용을 동료와 먼저 맞춘 뒤 재개한다.
  1. `publish-batching-migration.md`(§A~§D) 문서를 동료와 함께 다시 검토 — 특히 §C(백엔드 DTO/Listener 변경)와 §D(배포 순서)에 동의하는지 확인
  2. 동료가 실제로 1단계(Expand: `ReidResultDto`에 `schemaVersion`/`detections` 추가, `ReidResultListener`가 두 포맷 모두 처리)를 배포
  3. 배포 확인 후에만 ai-server의 `publish_grouped` 관련 코드를 다시 검토/활성화
- 그 전까지 ai-server의 `_publish_and_ack`는 **레거시 방식(1번 문제의 옵션 A 형태)만** 사용한다.

### 체크리스트
- [ ] 동료와 `publish-batching-migration.md` 재검토 미팅/논의 진행
- [ ] 동료 쪽 1단계(Expand) 배포 여부 확인 (배포 전엔 ai-server가 `detections`/`schemaVersion` 관련 필드를 절대 보내지 않음)
- [ ] 배포 확인 후 `publish-batching-migration.md`의 체크리스트를 실제 배포 상태 기준으로 다시 갱신 (로컬 전용 완료 표시와 실제 배포 상태를 혼동하지 않도록)

---

## 요약

| 문제 | 지금 상태 | ai-server가 할 일 |
|---|---|---|
| `schemaVersion` 필드 | 배포 시 100% 깨짐 | 필드 제거(옵션 A, 권장) 또는 동료 배포 대기(옵션 B) |
| `publish_grouped`/그룹 발행 | 백엔드 미지원 | 동료와 설계 재논의 전까지 보류, 레거시 방식만 발행 |
