# ReID 실험 전체 기록

- **작성일** : 2026-05-28
- **실험 환경** : RTX 3060 Laptop (6 GB VRAM), CUDA, CUHK-SYSU 데이터셋

---

## 목차

1. [BoT Baseline 성능 평가](#1-bot-baseline-성능-평가)
2. [PoseReID (Method 3) 학습 시도](#2-posreid-method-3-학습-시도)
3. [Part-based ReID 학습 시도](#3-part-based-reid-학습-시도)
4. [공통 붕괴 원인 분석](#4-공통-붕괴-원인-분석)
5. [다음 실험 방향](#5-다음-실험-방향)

---

## 1. BoT Baseline 성능 평가

### 1-1. 모델 구성

| 항목 | 내용 |
|:-----|:-----|
| 모델 | BoT ResNet50 (`bot_best.pth`) |
| 학습 데이터 | Market-1501 (파인튜닝) |
| 평가 데이터 | CUHK-SYSU TestG500 (300 쿼리) |
| 검출 모델 | YOLOv8n (distractor crop용) |
| 디바이스 | CUDA |
| 소요 시간 | 3,058초 |

### 1-2. 평가 결과

| 지표 | CUHK-SYSU (zero-shot) | Market-1501 (학습 기준) |
|:-----|:---------------------:|:-----------------------:|
| mAP | **81.62%** | 78.13% |
| Rank-1 | **87.00%** | 90.86% |
| Rank-5 | **94.33%** | — |
| Rank-10 | **96.67%** | — |

> Market-1501만으로 학습했음에도 CUHK-SYSU에서 zero-shot으로 mAP 81.62% 달성.
> 이 수치가 이후 모든 실험의 **비교 기준**이 됨.

### 1-3. 파이프라인 성능 (200장 기준)

| 구간 | 평균 시간 | 비율 |
|:-----|:--------:|:----:|
| 이미지 로드 | 5.33ms | 12.3% |
| YOLO 검출 | 18.67ms | 43.1% |
| Crop 추출 | 0.48ms | 1.1% |
| ReID 추출 | 18.86ms | 43.5% |
| **전체** | **43.34ms** | **23.1 FPS** |

**주요 발견**: Crop 단계(0.48ms)는 무시 가능한 수준. 실제 병목은 YOLO와 ReID 추론 각 43%.
ReID는 1~6명까지 GPU 병렬 처리로 약 9~11ms로 거의 동일 (배치 처리 효과).

---

## 2. PoseReID (Method 3) 학습 시도

### 2-1. 방법론

BoT backbone에서 추출한 appearance feature(2048-dim)와
YOLO-pose 키포인트 벡터(51-dim)를 concat하여 fused feature(512-dim) 생성.

```
이미지 → BoT (frozen) → 2048dim ─┐
                                    ├─ concat → FusionHead → 512dim
keypoints (51dim) → PoseEncoder ───┘
```

| 항목 | 값 |
|:-----|:---|
| 학습 파라미터 | 5,636,480개 (PoseEncoder + FusionHead만) |
| Backbone | Frozen |
| 데이터 | CUHK-SYSU train (15,088샘플, 5,532명) |
| Optimizer | Adam (lr=3e-4), StepLR (step=20, γ=0.1) |
| Loss | Triplet (margin=0.3) + ID CrossEntropy |
| 배치 | P=16, K=4 (배치 64) |

### 2-2. 학습 경과

| Epoch | total | triplet | id | 비고 |
|------:|------:|--------:|---:|:-----|
| 1 | 21.99 | 13.24 | 8.76 | |
| 5 | 14.42 | 5.89 | 8.54 | |
| 10 | 10.08 | 1.69 | 8.38 | |
| 12 | 9.00 | 0.42 | 8.58 | |
| **13** | **8.92** | **0.300** | **8.618** | **붕괴 시작** |
| 34 | 8.92 | 0.300 | 8.618 | 중단 |

### 2-3. 붕괴 원인

**핵심: 키포인트 벡터 자체에 신원 구분력이 없음**

| 비교 | 코사인 유사도 |
|:-----|:------------:|
| 같은 사람 간 | 0.8901 |
| 다른 사람 간 | 0.8545 |
| **차이** | **0.0356** |

- CUHK-SYSU 이미지 특성상 사람들이 유사한 자세 (서있거나 걷는)
- raw keypoint 좌표로는 신원 구분이 불가능
- Frozen backbone에서 FusionHead만 학습 → 의미없는 pose feature에 적응하지 못하고 붕괴

---

## 3. Part-based ReID 학습 시도

### 3-1. 방법론

키포인트로 신체 부위(머리/상체/하체)를 **crop**하여 각각의 appearance feature를 추출.
좌표 자체가 아닌 **부위별 이미지 텍스처/색상**을 특징으로 사용.

```
[head crop ]
[upper crop] × N명  →  (N×3, C, H, W)  →  backbone 1회 forward
[lower crop]
                    →  (N×3, 2048dim)  →  reshape  →  (N, 6144)
                    →  PartFusionHead  →  (N, 512dim)
```

**실시간 오버헤드 측정 결과**

| 사람 수 | 현재 BoT | Part-based (3 parts 배치) | 오버헤드 |
|:-------:|:--------:|:-------------------------:|:--------:|
| 1명 | 9.97ms | 8.66ms | 0.87x |
| 4명 | 11.72ms | 19.94ms | 1.70x |
| 6명 | 11.28ms | 27.15ms | 2.41x |

### 3-2. 실험 1 — Frozen Backbone

| 항목 | 값 |
|:-----|:---|
| Backbone | Frozen |
| AMP | 미적용 |
| 학습 파라미터 | 9,652,736개 |
| 배치 | P=16, K=4 (GPU 배치 192장) |

**학습 경과**

| Epoch | total | triplet | id |
|------:|------:|--------:|---:|
| 1 | 23.09 | 14.34 | 8.76 |
| 5 | 15.06 | 6.51 | 8.56 |
| 12 | 8.98 | 0.378 | 8.60 |
| **13** | **8.92** | **0.300** | **8.618** |
| 18 | 8.92 | 0.300 | 8.618 |

→ PoseReID와 동일한 epoch에서 동일한 값으로 붕괴. **중단.**

### 3-3. 실험 2 — Unfrozen Backbone + AMP

VRAM 측정 결과에 따라 AMP(fp16) 적용.

| 설정 | fp32 VRAM | AMP VRAM |
|:-----|:---------:|:--------:|
| P=4, K=4 (GPU 48장) | 3.11 GB | 1.74 GB |
| P=8, K=4 (GPU 96장) | 6.05 GB ⚠️ | 3.25 GB ✓ |
| P=16, K=4 (GPU 192장) | OOM | ~6.3 GB ⚠️ |

| 항목 | 값 |
|:-----|:---|
| Backbone | **Unfrozen** |
| AMP | **활성화 (fp16)** |
| 학습 파라미터 | **33,162,816개** (backbone 포함) |
| 배치 | P=8, K=4 (GPU 배치 96장) |
| 저장 파일 | `part_reid_unfreeze_best.pth` (기존 파일과 분리) |

**학습 경과**

| Epoch | total | triplet | id |
|------:|------:|--------:|---:|
| 1 | 23.24 | 14.37 | 8.87 |
| 5 | 11.56 | 3.00 | 8.56 |
| 7 | 8.94 | 0.322 | 8.62 |
| **8** | **8.92** | **0.301** | **8.618** |
| 15 | 8.92 | 0.300 | 8.618 |

→ Frozen 대비 epoch 13 → **epoch 8**로 오히려 더 빨리 붕괴. **중단.**

---

## 4. 공통 붕괴 원인 분석

### 4-1. 수렴 지점이 항상 동일

모든 실험이 예외 없이 동일한 값으로 수렴:

```
triplet loss = 0.300  ← margin의 수학적 하한
id loss      = 8.618  ← log(5532), 무작위 추측의 정보 엔트로피
─────────────────────
total loss   = 8.918  ← 로컬 미니멈
```

이는 모델 구조 문제가 아닌 **학습 다이나믹스의 문제**임.

### 4-2. 원인별 정리

| 원인 | 설명 |
|:-----|:-----|
| 고정 LR + StepLR | 초반 과도한 LR로 로컬 미니멈에 빠진 뒤 탈출 불가 |
| ID Loss 난이도 | 5,532 클래스 CrossEntropy → 초반 gradient가 매우 작음 |
| Triplet margin | margin=0.3 도달 후 대부분 triplet의 gradient=0 → 학습 신호 소실 |
| Backbone unfreeze 역효과 | 학습 파라미터 33M으로 늘어나 같은 로컬 미니멈에 더 빨리 수렴 |

---

## 5. 다음 실험 방향

로컬 미니멈 탈출을 위한 세 가지 변경:

### 5-1. Warm-up + Cosine Annealing LR

```
현재: flat 3e-4 → step decay
개선: 0 → 3e-4 (5 epoch warm-up) → cosine decay
```

초반에 LR을 서서히 올려 안정적인 gradient flow 확보 후,
cosine schedule로 로컬 미니멈 주기적 탈출.

### 5-2. Label Smoothing (ε=0.1)

```
현재: CrossEntropy (hard label)
개선: LabelSmoothing (ε=0.1) → 5,532 클래스 gradient를 고르게 분산
```

### 5-3. Triplet Margin 축소 (0.3 → 0.1)

```
현재: margin=0.3 → 빨리 하한 도달, gradient 소실
개선: margin=0.1 → 더 오래 triplet이 학습 신호 공급
```

### 예상 효과

세 변경을 동시 적용 시 로컬 미니멈 탈출 가능성 높음.
id loss가 8.618 이하로 감소하는지 여부가 핵심 지표.
