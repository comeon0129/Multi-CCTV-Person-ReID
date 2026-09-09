# BoT vs ViT-Small/16 백본 비교 실험

- **데이터셋** : Market-1501 (train 12,936장 / query 3,368장 / gallery 19,732장)
- **환경** : RTX 3060 Laptop (6GB VRAM), CUDA
- **측정일** : 2026-07-30
- **관련 문서** : [eval_bot_ibn_market1501.md](eval_bot_ibn_market1501.md) (BoT vs BoT-IBN, 같은 실험 설계), [reid_experiment_log.md](reid_experiment_log.md) §1 (BoT baseline)

---

## 1. 변경 사항

BoT-IBN 실험과 같은 방식으로, **백본만 트랜스포머(ViT)로 교체한 대조군**을 추가했다.
BN-Neck + classifier + 학습 루프(Loss/Optimizer/Scheduler)는 CNN 계열(BoT/BoT-IBN)과 완전히 동일 — 차이는 backbone 계열(CNN → Transformer) 하나뿐.

### 코드 변경

| 파일 | 내용 |
|:-----|:-----|
| `ai-server/research/benchmark/train_market1501.py` | `TrainableViT` 클래스 추가. `timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0, img_size=(256,128))`로 backbone 구성 후 동일한 BN-Neck+classifier 부착. `--model` 선택지에 `vit` 추가(기존 `bot_ibn`처럼 기본 `--model all`에는 포함하지 않음) |
| `ai-server/research/benchmark/eval_market1501.py` | `build_entries()`에 `vit_best.pth` 존재 시 `ViT-Small/16` 항목을 추가로 로드하도록 수정 |
| `ai-server/venv` | 신규 의존성 `timm==1.0.28` 설치 |

```python
class TrainableViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224", pretrained=True, num_classes=0, img_size=(256, 128),
        )
        feat_dim = self.backbone.num_features  # 384
        self.neck = nn.BatchNorm1d(feat_dim)
        self.neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, x):
        f = self.backbone(x)
        feat = self.neck(f)
        if not self.training:
            return feat
        return feat, self.classifier(feat)
```

- 224×224 ImageNet 사전학습 위치 임베딩을 256×128 입력(16×8=128패치+CLS)에 맞춰 timm이 자동 보간.
- 모델 크기는 **ViT-Small**(embed_dim 384, 21.57M params)로 선택 — BoT/BoT-IBN(23.51M)과 파라미터 규모를 맞춰 "가장 가벼운 ViT(Tiny, 5.7M)"가 아닌 **비슷한 체급끼리 백본 계열만 바꾼 비교**가 되도록 함(사용자 확인 하에 결정).

### 학습 설정 (BoT/BoT-IBN과 완전히 동일 — 백본 계열 외 모든 조건 고정)

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

### 2-1. 학습 곡선 (ViT-Small/16)

| Epoch | CE Loss | Triplet Loss | LR | mAP | Rank-1 |
|---:|---:|---:|---:|---:|---:|
| 10 | 4.8154 | 0.9377 | 3.50e-04 | 43.08% | 62.77% |
| 20 | 2.5629 | 1.0299 | 3.50e-04 | 58.40% | 75.68% |
| 30 | 1.7853 | 1.0329 | 3.50e-04 | 64.88% | 82.16% |
| 40 | 1.5708 | 1.0240 | 3.50e-05 | 66.58% | 83.19% |
| 50 | 1.2796 | 0.9139 | 3.50e-05 | 71.64% | 85.93% |
| 60 | 1.2518 | 0.8757 | 3.50e-06 | **71.72%** | **86.16%** |

에폭당 약 15~17초(BoT-IBN과 비슷), 총 학습 시간 약 16분.

참고로 같은 시점의 BoT-IBN 대비 전 구간에서 뒤처짐(10ep: 50.66/73.13 → 20ep: 67.53/85.12 → 30ep: 70.90/87.17 → 40ep: 73.29/88.18 → 50ep: 78.34/90.91) — ViT가 CNN보다 수렴이 느리다는 통설과 일치.

### 2-2. 최종 비교 (동일 평가 코드, 같은 실행에서 측정)

| 모델 | 백본 | 파라미터 | GFLOPs | Out-Dim | mAP | Rank-1 | Rank-5 | Rank-10 | Mean 레이턴시 |
|:-----|:-----|---:|---:|---:|---:|---:|---:|---:|---:|
| BoT | ResNet50 | 23.51M | 4.087 | 2048 | 78.13% | 90.86% | 96.62% | 97.74% | 12.85ms |
| BoT-IBN | ResNet50-IBN-a | 23.51M | 4.087 | 2048 | 78.86% | 90.94% | 96.29% | 97.54% | 14.74ms |
| **ViT-Small/16** | ViT | 21.57M | **2.777** | **384** | 71.72% | 86.16% | 93.97% | 95.93% | **9.10ms** |
| ViT − BoT | | -1.94M | **-32%** | **-1664차원** | **-6.41%p** | **-4.70%p** | -2.65%p | -1.81%p | **-29%(-3.75ms)** |

전체 5개 모델(OSNet 포함) 원본은 [market1501_benchmark_20260730_002457.md](market1501_benchmark_20260730_002457.md) 참조.

---

## 3. 분석

- **정확도는 ViT가 확실히 낮음** — mAP -6.41%p, Rank-1 -4.70%p. CNN용으로 튜닝된 하이퍼파라미터(60 epoch, lr=3.5e-4, Adam)를 그대로 ViT에 적용한 결과라, ViT에 불리한 비교일 가능성이 큼. 학습 곡선을 보면 60 epoch 시점에도 상승세가 완전히 꺾이지 않아(50→60ep: mAP +0.08%p로 사실상 수렴) 이 설정 안에서는 이 이상 나오기 어려움 — 더 나은 결과를 보려면 **ViT 전용 레시피**(더 긴 warmup, layer-wise LR decay, 더 많은 epoch, 또는 TransReID처럼 overlapping patch/SIE 등 ReID 특화 구조)가 필요.
- **효율성은 ViT가 뚜렷한 우위** — GFLOPs -32%, 레이턴시 -29%, 게다가 임베딩 차원이 **384로 BoT(2048)의 1/5** 수준. `pca-128d-migration.md`에서 BoT 2048차원을 PCA로 128차원까지 억지로 줄였던 것과 비교하면, ViT는 PCA 없이도 이미 384차원 — DB 저장/유사도 검색 비용 관점에서 흥미로운 지점.
- 종합하면 이번 결과는 "**같은 학습 조건에서 정확도는 CNN이 낫고, 속도·임베딩 크기는 ViT가 낫다**"는 트레이드오프를 보여줌. ViT 쪽 정확도가 튜닝 부족 때문인지 구조적 한계인지는 이번 실험만으로 판단 불가.

## 4. 다음 실험 방향

- [ ] ViT 전용 학습 레시피(layer-wise LR decay, warmup 연장, epoch 증가)로 재학습 후 동일 비교 반복 — 지금 결과가 "구조적 한계"인지 "설정 미스매치"인지 분리
- [ ] `vit_best.pth`로 CUHK-SYSU TestG500 zero-shot 평가 → BoT 81.62%(mAP) 기준과 비교 (BoT-IBN과 동일한 다음 단계)
- [ ] ViT 임베딩(384차원)에 PCA 128차원 적용 시 identifiability 손실 여부 확인 — 이미 저차원이라 추가 축소 이득이 작을 수 있음
