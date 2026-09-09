# Intelligent CCTV ReID

사람 탐지(YOLO) + 특징 벡터 추출(ReID)을 수행하는 FastAPI 기반 추론 서버입니다.

## 아키텍처

```
이미지 입력
    │
    ▼
YOLOv8 (사람 탐지)
    │  → bounding box + confidence
    ▼
OSNet (Re-ID 특징 추출)
    │  → 512차원 임베딩 벡터
    ▼
JSON 응답
```

## 요구사항

- Python 3.9+
- CUDA (선택사항, CPU로도 동작)

## 설치

```bash
# 1. 저장소 클론
git clone git@github.com:lookie14/Intelligent-CCTV-ReID.git
cd Intelligent-CCTV-ReID

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# torchreid는 별도 설치 필요
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

## 설정

`config.yaml` 에서 모델과 파라미터를 변경할 수 있습니다.

```yaml
yolo:
  model: yolov8n.pt     # n / s / m / l / x
  confidence: 0.25      # 탐지 신뢰도 임계값
  iou: 0.45             # NMS IoU 임계값

reid:
  model: osnet_x0_25    # osnet_x0_25 / x0_5 / x0_75 / x1_0 / osnet_ain_x0_25
  device: auto          # auto / cpu / cuda
```

## 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

### `GET /health`

서버 상태 및 로드된 모델 확인

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "yolo": "yolov8n.pt",
  "reid": "osnet_x0_25"
}
```

---

### `POST /embed`

이미지에서 사람을 탐지하고 ReID 임베딩 벡터를 반환합니다.

```bash
curl -X POST http://localhost:8000/embed \
  -F "file=@image.jpg"
```

**응답 예시**

```json
{
  "count": 2,
  "detections": [
    {
      "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 300},
      "confidence": 0.92,
      "vector": [0.021, -0.013, ...]
    }
  ]
}
```

| 필드 | 설명 |
|------|------|
| `count` | 탐지된 사람 수 |
| `bbox` | 바운딩 박스 좌표 (픽셀) |
| `confidence` | YOLO 탐지 신뢰도 |
| `vector` | 512차원 ReID 임베딩 벡터 |

## 모델 옵션

### YOLO

| 모델 | 속도 | 정확도 |
|------|------|--------|
| yolov8n.pt | 빠름 | 낮음 |
| yolov8s.pt | 보통 | 보통 |
| yolov8m.pt | 느림 | 높음 |

### ReID (OSNet)

| 모델 | 크기 | 정확도 |
|------|------|--------|
| osnet_x0_25 | 작음 | 낮음 |
| osnet_x0_5 | 보통 | 보통 |
| osnet_x1_0 | 큼 | 높음 |
