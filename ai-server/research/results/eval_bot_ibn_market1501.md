# BoT vs BoT-IBN 백본 비교 실험

- **데이터셋** : Market-1501 (train 12,936장 / query 3,368장 / gallery 19,732장)
- **환경** : RTX 3060 Laptop (6GB VRAM), CUDA
- **측정일** : 2026-07-29
- **관련 문서** : [reid_experiment_log.md](reid_experiment_log.md) §1 (BoT baseline), [pca-128d-migration.md](../../../pca-128d-migration.md)

---

## 1. 변경 사항

기존 BoT(ResNet50)는 그대로 두고, **백본만 IBN-Net(ResNet50-IBN-a)으로 교체한 대조군**을 추가했다.
Bag-of-Tricks 구성(last-stride 2→1, BN-Neck)과 학습 코드는 BoT와 완전히 동일 — 차이는 backbone 한 곳뿐.

### 신규 파일

| 파일 | 내용 |
|:-----|:-----|
| `ai-server/research/benchmark/models/bot_ibn.py` | `BoTIBNReID` — 배포용 추론 모델. 기존 `models/bot.py`의 `BoTReID`와 동일 구조에서 `resnet50(weights=None)` → `resnet50_ibn_a(num_classes=1, pretrained=True)`로 백본만 교체 |
| `ai-server/research/benchmark/train_market1501.py` | Market-1501 파인튜닝 스크립트(신규). `TrainableOSNet`/`TrainableBoT`/`TrainableBoTIBN`/`TrainableLightMBN` 4개 모델 지원. `TrainableBoTIBN`은 `TrainableBoT`와 동일 코드에서 backbone 생성자만 IBN으로 교체 |
| `ai-server/research/benchmark/eval_market1501.py` | mAP/Rank-1/5/10 + Params/GFLOPs/추론 레이턴시 통합 벤치마크(신규) |
| `ai-server/research/benchmark/utils.py` | 벤치마크용 공용 유틸(`BenchmarkResult`, FLOPs/레이턴시 측정 함수)(신규) |

### 모델 코드 diff (`models/bot.py` → `models/bot_ibn.py`)

```diff
- from torchvision.models import resnet50
+ from torchreid.reid.models.resnet_ibn_a import resnet50_ibn_a

- class BoTReID(nn.Module):
+ class BoTIBNReID(nn.Module):
      def __init__(self) -> None:
          super().__init__()
-         base = resnet50(weights=None)
+         base = resnet50_ibn_a(num_classes=1, pretrained=True)

          # 이하 last-stride 변경 + BN-Neck 구성은 100% 동일
          base.layer4[0].downsample[0].stride = (1, 1)
          base.layer4[0].conv2.stride = (1, 1)
          ...
```

- IBN-Net(Pan et al., ECCV 2018)은 backbone 앞단 일부 BatchNorm을 Instance Norm과 절반씩 섞어(IN+BN) **외형(색·조명 등 도메인 변화)에 덜 민감하게** 만드는 것이 목적.
- 학습용 클래스(`TrainableBoT`/`TrainableBoTIBN`, classifier 포함) 기준 파라미터는 두 모델 모두 **25.05M로 동일**. 배포용 추론 모델(`bot.py`/`bot_ibn.py`, classifier 제외) 기준으로는 **23.51M로 동일** — IN/BN 혼합은 채널을 반으로 나눠 쓰는 방식이라 추가 파라미터 비용은 없음.
- **다만 GFLOPs는 동일(4.087)해도 실측 추론 레이턴시는 IBN 쪽이 더 느림** — 아래 2-3절 참조.

### 학습 설정 (BoT와 완전히 동일 — 백본 외 모든 조건 고정)

| 항목 | 값 |
|:-----|:---|
| Epochs | 60 (10 epoch마다 평가) |
| 배치 | P=16 identity × K=4 = 64 |
| Optimizer | Adam, lr=3.5e-4, weight_decay=5e-4 |
| Scheduler | Linear warmup 10 epoch → MultiStepLR([30, 50], γ=0.1) |
| Loss | CrossEntropy(label smoothing 0.1) + Triplet(batch-hard, margin=0.3) |
| Augmentation | Resize(288,144)→RandomCrop(256,128), HFlip, RandomErasing(p=0.5) |

---

## 2. 결과

### 2-1. 학습 곡선 (BoT-IBN)

| Epoch | CE Loss | Triplet Loss | LR | mAP | Rank-1 |
|---:|---:|---:|---:|---:|---:|
| 10 | 3.0417 | 1.2815 | 3.50e-04 | 50.66% | 73.13% |
| 20 | 1.6559 | 1.2125 | 3.50e-04 | 67.53% | 85.12% |
| 30 | 1.5852 | 1.1618 | 3.50e-04 | 70.90% | 87.17% |
| 40 | 1.4944 | 1.1166 | 3.50e-05 | 73.29% | 88.18% |
| 50 | 1.2518 | 0.9710 | 3.50e-05 | 78.34% | 90.91% |
| 60 | 1.2360 | 0.9371 | 3.50e-06 | **78.86%** | **90.94%** |

에폭당 약 15~19초 소요, 총 학습 시간 약 16분.

### 2-2. 최종 비교 (동일 평가 코드로 재측정)

| 모델 | 백본 | mAP | Rank-1 | Rank-5 | Rank-10 | 체크포인트 |
|:-----|:-----|---:|---:|---:|---:|:-----|
| BoT | ResNet50 | 78.13% | 90.86% | 96.62% | 97.74% | `checkpoints/bot_best.pth` (epoch 50, 2026-05-22) |
| BoT-IBN | ResNet50-IBN-a | **78.86%** | **90.94%** | 96.29% | 97.54% | `checkpoints/bot_ibn_best.pth` (epoch 60, 2026-07-29) |
| **차이** | | **+0.73%p** | **+0.08%p** | -0.33%p | -0.20%p | |

> BoT 수치는 학습 당시 저장된 값(`reid_experiment_log.md` §1)과 이번에 `evaluate()`로 재측정한 값이 완전히 일치함(78.13%/90.86%) — 비교 코드/데이터 split이 동일함을 확인.

### 2-3. 효율성 비교 (`eval_market1501.py`, batch=1, warmup=10, runs=100, 입력 3×256×128)

전체 4개 모델(OSNet/BoT/LightMBN/BoT-IBN) 통합 결과는 [market1501_benchmark_20260729_235552.md](market1501_benchmark_20260729_235552.md) 참조. BoT 계열만 발췌:

| 모델 | Params (M) | GFLOPs | Out-Dim | Mean (ms) | Min (ms) | Max (ms) | Std (ms) |
|:-----|---:|---:|---:|---:|---:|---:|---:|
| BoT | 23.51 | 4.087 | 2048 | 10.69 | 8.77 | 17.10 | 1.65 |
| BoT-IBN | 23.51 | 4.087 | 2048 | 14.24 | 12.19 | 21.16 | 1.55 |
| **차이** | +0.00 | +0.000 | — | **+3.55ms (+33%)** | +3.42ms | +4.06ms | -0.10 |

> **GFLOPs는 완전히 동일한데 실측 레이턴시는 BoT-IBN이 평균 33% 더 느림.** FLOPs 카운터는 곱셈-누산 연산만 세기 때문에, Instance Norm이 배치 정규화보다 실제 GPU에서 더 비싼 연산(샘플별 평균/분산 계산, 채널 split-concat)이라는 점이 잡히지 않음 — "FLOPs 동일 = 속도 동일"이 아님을 보여주는 사례.

---

## 3. 분석

- 백본을 IBN-Net으로 교체하는 것만으로 **파라미터·GFLOPs 증가 없이** mAP +0.73%p 개선 — 방향은 맞지만 폭은 크지 않음.
- 반면 **추론 레이턴시는 평균 10.69ms → 14.24ms로 33% 증가**. 실시간 CCTV 파이프라인(`reid_experiment_log.md` §1-3, ReID 추론이 전체 파이프라인의 43%를 차지)에 그대로 적용하면 mAP 0.73%p 이득 대비 레이턴시 비용이 작지 않음 — 프레임당 처리 시간이 늘어나면 FPS가 그만큼 줄어듦.
- Market-1501은 카메라 6대 모두 비슷한 실내 조명/각도 조건이라 **도메인(카메라 간 외형 차이)이 크지 않음** — IBN이 노리는 "도메인 불변성"이 발휘될 여지가 애초에 적은 데이터셋. 원 논문에서도 IBN의 이득은 **cross-domain(예: Market-1501 학습 → CUHK-SYSU/타 데이터셋 평가)** 상황에서 더 크게 나타남.
- 따라서 이번 결과만으로 "IBN 효과가 미미하다"고 결론 내리기보다는, **CUHK-SYSU 등 다른 도메인에서 zero-shot 평가**(BoT baseline이 이미 mAP 81.62%를 기록한 것과 동일한 조건)로 재검증할 필요가 있음. 다만 **레이턴시 33% 증가는 도메인과 무관하게 고정 비용**이므로, mAP 이득이 cross-domain에서 더 크게 나오지 않는 한 실전 도입은 신중해야 함.

## 4. 다음 실험 방향

- [ ] `bot_ibn_best.pth`로 CUHK-SYSU TestG500 zero-shot 평가 → BoT의 81.62%(mAP) 기준과 비교
- [ ] 128차원 PCA+Whitening을 BoT-IBN 임베딩에도 적용해 identifiability 유지되는지 확인 (`pca-128d-migration.md` 절차와 동일)
- [ ] 레이턴시 +33% 원인이 Instance Norm 자체인지 `resnet_ibn_a` 구현 세부사항(예: split-concat 오버헤드)인지 프로파일링(`torch.profiler`)으로 확인
