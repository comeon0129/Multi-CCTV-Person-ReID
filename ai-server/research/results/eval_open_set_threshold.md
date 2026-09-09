# Open-set 임계치 검증 — same-ID vs diff-ID 거리 분포

- **배경** : [pca-128d-migration.md](../../pca-128d-migration.md) §C — `ReidService.java`의 `DISTANCE_THRESHOLD=0.3`은 실험적으로 검증된 값이 아니었음. 128차원 PCA+Whitening 적용 후 실제 same-ID/diff-ID 거리 분포로 재검증.
- **데이터** : CUHK-SYSU TestG500 평가 특징 캐시(`dimred_eval_TestG500_n300.npz`), query별 gallery 라벨(같은 사람=1/다른 사람=0)을 그대로 same-ID/diff-ID 쌍으로 사용
- **거리 정의** : 코사인 거리(1 - 코사인 유사도) — pgvector `<=>`(vector_cosine_ops)와 동일
- **PCA 아티팩트** : `reid_core/checkpoints/pca_128.npz` (2048 → 128차원, 운영 파이프라인과 동일 변환)
- **측정일** : 2026-08-05 13:06
- **재현** : `python research/verify/eval_open_set_threshold.py --save research/results/eval_open_set_threshold.md --plot research/results/eval_open_set_threshold.png` (기존 캐시 재사용, YOLO/BoT 재추론 불필요)
- **분포 그래프** : [eval_open_set_threshold.png](eval_open_set_threshold.png)

---

## 1. 거리 분포 통계

| 구분 | 차원 | n | mean | std | median | p25 | p75 |
|:-----|-----:|---:|-----:|----:|-------:|----:|----:|
| same-ID (Origin) | 2048 | 692 | 0.2112 | 0.1622 | 0.1662 | 0.0908 | 0.2968 |
| diff-ID (Origin) | 2048 | 149308 | 0.8243 | 0.2515 | 0.8260 | 0.6404 | 1.0070 |
| same-ID (PCA+Whiten) | 128 | 692 | 0.4218 | 0.2322 | 0.3871 | 0.2405 | 0.5964 |
| diff-ID (PCA+Whiten) | 128 | 149308 | 0.9988 | 0.0933 | 1.0057 | 0.9451 | 1.0614 |

---

## 2. 임계치별 FAR/FRR (128차원, 운영 파이프라인 기준)

- **FAR**(False Accept Rate) : 다른 사람인데 같은 사람으로 오인식(임계치 이하로 매칭됨) 비율
- **FRR**(False Reject Rate) : 같은 사람인데 다른 사람으로 오인식(새 ID로 등록됨) 비율

| 기준 | THRESHOLD | FAR | FRR |
|:-----|----------:|----:|----:|
| **현재 운영값** | 0.300 | 0.00% | 64.31% |
| **EER**(FAR≈FRR) | 0.830 | — | EER=4.60% |
| FAR≈0.1% 목표 | 0.580 | 0.10% | 26.59% |
| FAR≈1.0% 목표 | 0.730 | 0.99% | 11.85% |
| FAR≈5.0% 목표 | 0.835 | 4.94% | 4.48% |
| FAR≈10.0% 목표 | 0.880 | 9.80% | 2.60% |

---

## 3. 분석

- 현재 운영 중인 `DISTANCE_THRESHOLD=0.3`은 EER 지점(`0.830`) 대비 낮아(보수적) FRR이 상대적으로 큼 — 위 표의 FAR/FRR 수치로 실제 운영에 적합한지 판단 필요.
- CCTV 추적 서비스는 보통 '다른 사람을 같은 사람으로 잘못 합치는' FAR 쪽 리스크가 '같은 사람을 새 ID로 중복 등록하는' FRR보다 치명적인 경우가 많음(추적 이력 오염) — 이 경우 EER보다 낮은 FAR(예: FAR≈1% 지점)을 목표로 임계치를 더 보수적으로 잡는 것을 권장.
- 이 실험은 CUHK-SYSU 데이터 기준이라 실제 CCTV 도메인(카메라 각도/조명/해상도)과 차이가 있을 수 있음 — 최종 배포 전 실제 도메인 데이터로 재검증 권장.

---

## 4. 조치 결과

- **채택 값** : EER 지점 `0.83` — FAR과 FRR을 균형 있게 최소화(전체 오류율 약 4.6%)하는 지점으로 결정.
  보수적 옵션(FAR≈1% 목표 `0.73`)도 검토했으나, 우선 균형점으로 적용 후 실제 운영 데이터가 쌓이면 재조정하기로 함.
- **적용 파일** : `backend-stream/reid-backend/.../service/ReidService.java` — `DISTANCE_THRESHOLD = 0.3` → `0.83`
- **효과** : 기존 `0.3` 기준 FRR 64.31%(같은 사람 재등장의 64%를 새 인물로 오등록) → `0.83` 기준 FRR 4.48%로 대폭 개선.

## 5. 다음 실험 방향

- [ ] 실제 CCTV 도메인 크롭(다른 카메라 각도/조명)으로 same-ID/diff-ID 쌍을 다시 뽑아 `0.83`이 여전히 적정한지 재검증
- [ ] FAR/FRR 리스크 우선순위(추적 이력 오염 vs 중복 ID 생성)를 팀 내에서 논의해 EER 대신 보수적 FAR≈1% 지점(`0.73`)으로 바꿀지 결정
- [ ] `pca-128d-migration.md` §C 체크리스트 완료 처리 (본 문서로 대체)