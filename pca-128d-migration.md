# PCA 128차원 전환 — 수정 체크리스트

- **작성일** : 2026-07-03
- **배경** : `ai-server/benchmark/results/eval_dim_reduction_TestG100.md` 실험에서 BoT 2048차원 임베딩을 PCA+Whitening으로 128차원까지 축소해도 mAP가 오히려 +2.30%p 상승(93.50% vs Origin 91.20%)함을 확인. 이 결과를 실험이 아닌 실제 운영 파이프라인에 반영한다.
- **전제** : `origin/main`의 `4ee55ec` 커밋("vector db로 코드 변경")이 아직 로컬 `main`/현재 작업 브랜치에 병합되지 않은 상태. 이 문서는 해당 pgvector 구조가 병합된 이후를 기준으로 작성됨.

---

## A. `ai-server` — 실제로 128차원 벡터를 만드는 곳

- [ ] `ai-server/benchmark/dim_reduction.py`의 `PCAWhiten`에 `save()`/`load()` 추가 — 현재는 실험용이라 fit 결과(`mean_`, `components_`, `explained_variance_`)를 영속화하는 기능이 없음
- [ ] reference set으로 128차원 PCA를 한 번 fit해서 아티팩트 생성 (예: `benchmark/checkpoints/pca_128.npz`)
  - CUHK-SYSU train만 쓸지, 실제 CCTV 도메인 크롭도 섞을지 결정 필요
  - **한 번 fit하면 이후 절대 재fit 금지** (좌표계가 바뀌면 DB에 쌓인 과거 벡터와 비교 불가)
- [ ] `ai-server/main.py`
  - `lifespan()`에서 YOLO/BoT 로딩하듯 PCA 아티팩트도 함께 로드
  - `run_reid()`에서 BoT forward 직후 `pca.transform()` 적용 → 128차원으로 `vectorData` 구성 (현재는 2048 원본 그대로 전송됨)
- [ ] `ai-server/config.yaml`의 `reid:` 섹션에 PCA 아티팩트 경로/차원 설정 추가

## B. `backend-stream/reid-backend` — `2048` 하드코딩/스키마 지점 (4곳)

- [ ] `entity/ReidResultEntity.java`
  ```java
  @Column(columnDefinition = "vector(2048)")
  private float[] vectorData;
  ```
  → `vector(128)`로 변경 (안 맞으면 INSERT 자체가 실패)

- [ ] `config/DummyDataLoader.java`
  ```java
  float[] fakeVector = new float[2048];
  for (int j = 0; j < 2048; j++) { ... }
  ```
  → `128`로 변경

- [ ] `src/test/java/.../RabbitMqIntegrationTest.java`
  ```java
  float[] fakeVector = new float[2048];
  for (int i = 0; i < 2048; i++) { ... }
  ```
  → `128`로 변경

- [ ] **DB 스키마 재생성** — `application.yml`의 `ddl-auto: update`는 기존 `vector(2048)` 컬럼을 `vector(128)`로 자동 변환하지 못함
  - 로컬 `docker-compose down -v` 후 컨테이너 재생성, 또는
  - `ALTER TABLE reid_result ALTER COLUMN vector_data TYPE vector(128);` 직접 실행

## C. `service/ReidService.java` — 임계치 재보정

- [ ] `DISTANCE_THRESHOLD = 0.3` (유사도 0.7의 산술적 반대값일 뿐, 실험적으로 검증된 값 아님) 재실험 필요
  - Whitening을 거치면 코사인 거리 분포 자체가 달라짐
  - 지금까지의 PCA 실험(`eval_dim_reduction_*.md`)은 mAP/CMC 같은 **closed-set 랭킹 지표만** 검증했고, "같은 사람 vs 다른 사람" 거리 분포 기반 open-set 임계치는 검증한 적 없음
  - 128차원 기준으로 same-ID / diff-ID 쌍 거리 분포를 뽑아서 최적 임계치 재산정 필요

## D. (부가) pgvector 인덱스

- [ ] 현재 `ReidResultRepository`의 `<=>` 쿼리는 인덱스 없이 브루트포스 스캔(O(N)이 DB 레벨로 내려갔을 뿐)
- [ ] pgvector의 `ivfflat`/`hnsw` 인덱스는 **2,000차원까지만 지원** — 2048차원에서는 애초에 인덱스 생성 불가, 128차원이면 가능
- [ ] (선택) `CREATE INDEX ... USING hnsw (vector_data vector_cosine_ops)` 추가해 진짜 ANN 검색으로 전환

## 영향 없음 (확인 완료)

- `frontend-client` — `vectorData`를 직접 다루지 않음 (grep 결과 참조 없음)
- WebSocket 핸들러 — 프레임/이미지만 다루고 벡터 미참조
