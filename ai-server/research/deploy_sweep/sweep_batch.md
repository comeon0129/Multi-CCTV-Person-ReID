# `sweep_batch.py` 설명

- **작성일** : 2026-07-17
- **목적** : 보고서용 측정 — `max_wait_ms`(배치 최대 대기시간)와 `dim_reduction.enabled`(PCA 차원 축소 적용 여부)를 각각 바꿔가며 지연시간에 미치는 영향을 측정.

---

## 왜 새 스크립트로 만들었나

- `pipeline_test.py`는 이미지 1장씩 순차 처리라서 `max_wait_ms`(여러 프레임을 모아 배치로 묶는 대기시간) 효과를 측정할 수 없음 — 배치 개념 자체가 없음.
- `main.py`의 실제 배치 로직(`_collect_batch`, `run_reid_batch`)은 RabbitMQ 컨슈머와 강하게 얽혀 있어서, 그대로 가져다 쓰려면 실제 브로커 연결이 필요함. 프로덕션 큐를 이런 반복 실험에 쓰는 건 피하는 게 맞아서, 이번 스크립트는 **RabbitMQ 없이 로컬에서 같은 배치 수집 알고리즘만 재현**하도록 만듦.
- FLOPs 측정, 구간별(load/crop 등) 세부 브레이크다운처럼 이번 측정 목적과 무관한 부분은 넣지 않음 — 이 스크립트는 "배치 대기시간 × PCA 여부"에만 집중.

## 무엇을 측정하는가

1. **`max_wait_ms` 스윕**: 기본 `[10, 30, 50, 100]` (config.yaml 기본값 30을 포함, `rabbitmq-batch-consumer.md` §6-2 실험과 같은 후보값)
2. **`dim_reduction` on/off**: PCA+Whitening(128차원) 적용 여부 — BoT 원본 2048차원 그대로 쓸 때와 비교
3. 두 축을 전부 조합해서(기본 4×2=8가지) 각각 실행

## 어떻게 측정하는가

- 가상 프로듀서 스레드가 지정한 fps(기본 15 — `rabbitmq-batch-consumer.md` §6-3에서 확인된 이 하드웨어의 안정적 처리량 기준)로 로컬 CUHK-SYSU 이미지를 큐에 넣음 (매번 랜덤 샘플링, 이미지 50장을 미리 메모리에 올려 디스크 I/O 노이즈 최소화).
- 컨슈머 루프는 `main.py`의 `_collect_batch`와 **동일한 알고리즘**(첫 프레임은 블로킹으로 기다렸다가, `max_size`에 닿거나 `max_wait_ms`가 지날 때까지 더 모음)으로 배치를 구성.
- 배치가 모이면 YOLO → crop → ReID → (옵션) PCA transform까지 실제로 수행하고, `torch.cuda.synchronize()`로 GPU 비동기 실행이 섞이지 않도록 정확히 시간을 측정.
- 각 프레임의 지연시간 = 그 프레임이 큐에 들어간 시각부터, 그 프레임이 속한 배치의 forward가 끝난 시각까지.
- 앞쪽 `warmup_s`(기본 3초) 동안 들어온 프레임은 통계에서 제외 (CUDA/모델 워밍업 영향 배제).
- 매 조합 실행 전 동일한 시드로 랜덤을 리셋해서, 조합마다 같은 순서의 이미지가 발행되도록 함 — `max_wait_ms`/`dim_reduction` 차이 외의 변수를 최대한 통제하기 위함.

## 실행 방법

```bash
cd /home/lookie14/Intelligent-CCTV-ReID/ai-server
python research/deploy_sweep/sweep_batch.py
# 커스텀 예시
python research/deploy_sweep/sweep_batch.py --max_wait_ms 10 30 50 100 --dim_reduction both --fps 15 --measure_s 15
```

결과는 기본적으로 `research/results/batch_wait_dimred_sweep.md`에 표로 저장됨 (`--save`로 경로 변경 가능).

## 알려진 한계

- 실제 RabbitMQ 큐 통신(TLS 왕복, CloudAMQP flow control 등)은 재현하지 않음 — 순수하게 "배치 수집 + 모델 forward" 구간만 측정. 실제 운영 E2E 지연에는 이 값 외에 큐/발행 단계 비용이 추가로 붙는다는 점을 감안해서 해석해야 함 (`rabbitmq-batch-consumer.md` §6-5 참고).
- 이미지 풀이 CUHK-SYSU 50장 고정이라, 실제 CCTV 화면(해상도·인원 수 분포)과는 차이가 있을 수 있음.
