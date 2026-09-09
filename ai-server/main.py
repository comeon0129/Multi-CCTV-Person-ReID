from contextlib import asynccontextmanager
import base64
import functools
import io
import json
import queue
import threading
import time

import numpy as np
import pika
import torch
import torchvision.transforms as T
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from reid_core.dim_reduction import PCAWhiten
from reid_core.models.bot import BoTReID


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_models: dict = {}
_cfg: dict = {}

# on_message(IO 스레드)가 프레임을 던져두면 batch_worker(별도 스레드)가 모아서
# GPU에 한 번에 넣는다. 크기는 RabbitMQ prefetch_count로 자연히 제한된다
# (ack 전까진 브로커가 그 이상 배달하지 않으므로).
_frame_queue: "queue.Queue[dict]" = queue.Queue()

# start_consumer()가 (재)연결할 때마다 최신 connection/channel로 갱신.
# batch_worker는 항상 이 딕셔너리를 통해 최신 채널을 참조한다.
_rmq_ctx: dict = {}

_REID_TF = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def crop_to_base64(img_np: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_reid_batch(img_list: list[np.ndarray]) -> list[dict]:
    """여러 프레임을 한 번의 YOLO forward + 한 번의 ReID forward로 처리한다.

    각 crop이 어느 프레임 소속인지 frame_idx로 추적해서, 결과를 다시
    프레임 단위로 regroup해 반환한다 (반환 리스트는 img_list와 1:1 순서).
    """
    yolo_cfg = _cfg["yolo"]

    results = _models["yolo"](
        img_list,
        classes=[0],
        conf=yolo_cfg["confidence"],
        iou=yolo_cfg["iou"],
        verbose=False,
    )

    all_crops, all_meta, frame_idx_of_crop = [], [], []
    for frame_idx, (img_np, res) in enumerate(zip(img_list, results)):
        h, w = img_np.shape[:2]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            all_crops.append(crop)
            all_meta.append({
                "imageData": crop_to_base64(crop),
                "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                "confidence": float(box.conf[0]),
            })
            frame_idx_of_crop.append(frame_idx)

    frame_results = [{"count": 0, "detections": []} for _ in img_list]
    if not all_crops:
        return frame_results

    device = _cfg["reid"]["device"]
    tensors = torch.stack([_REID_TF(Image.fromarray(c)) for c in all_crops]).to(device)
    with torch.no_grad():
        features: torch.Tensor = _models["reid"](tensors)

    feats_np = features.cpu().numpy()
    pca: PCAWhiten | None = _models.get("pca")
    if pca is not None:
        feats_np = pca.transform(feats_np)

    for m, feat, frame_idx in zip(all_meta, feats_np, frame_idx_of_crop):
        frame_results[frame_idx]["detections"].append({**m, "vectorData": feat.tolist()})

    for r in frame_results:
        r["count"] = len(r["detections"])

    return frame_results


def run_reid(img_np: np.ndarray) -> dict:
    return run_reid_batch([img_np])[0]


def base64_to_image(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def on_message(ch, method, properties, body):
    """IO 스레드에서 실행됨 — 무거운 추론은 하지 않고 큐에 넣기만 하고 바로 리턴,
    다음 메시지를 계속 당겨올 수 있게 한다. 실제 추론은 batch_worker가 담당."""
    try:
        t3 = int(time.time() * 1000)
        data = json.loads(body)
        timestamps = data.get("timestamps", {})
        timestamps["T3_AI_Consumption"] = t3

        job = {
            "delivery_tag": method.delivery_tag,
            "cctv_id": data.get("cctvId", "unknown"),
            "location_id": data.get("locationId", "unknown"),
            "frame_number": data.get("frameNumber", -1),
            "timestamp": data.get("timestamp", ""),
            "timestamps": timestamps,
            "img_np": base64_to_image(data["imageData"]),
        }
        print(f"[Consumer] 프레임 {job['frame_number']} 수신 (cctv: {job['cctv_id']})")
        _frame_queue.put(job)

    except Exception as e:
        print(f"[Consumer] 프레임 파싱 오류: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def _collect_batch(max_size: int, max_wait_s: float) -> list[dict]:
    """최소 1개는 기다렸다가 받고, 그 뒤론 max_size에 닿거나 max_wait_s가
    지날 때까지(둘 중 먼저 오는 조건) 큐에 이미 도착한 프레임들을 마저 모은다."""
    batch = [_frame_queue.get()]
    deadline = time.monotonic() + max_wait_s
    while len(batch) < max_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            batch.append(_frame_queue.get(timeout=remaining))
        except queue.Empty:
            break
    return batch


def _publish_and_ack(channel, rmq_cfg, job, result):
    """connection.add_callback_threadsafe로 IO 스레드에서 실행됨 —
    pika BlockingConnection은 다른 스레드에서 채널을 직접 건드리면 안 되기 때문."""
    if not result["detections"]:
        channel.basic_ack(delivery_tag=job["delivery_tag"])
        print(f"[Consumer] 프레임 {job['frame_number']} 처리 완료 — 검출: 0명")
        return

    if rmq_cfg.get("publish_grouped", False):
        # 신규: 프레임당 메시지 1건 (detections 배열)
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
        # 레거시: 검출 인원수만큼 반복 발행 (기존 동작과 동일)
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
    print(f"[Consumer] 프레임 {job['frame_number']} 처리 완료 — 검출: {result['count']}명")


def batch_worker():
    batch_cfg = _cfg["reid"].get("batch") or {}
    max_size = batch_cfg.get("max_size", 16)
    max_wait_s = batch_cfg.get("max_wait_ms", 30) / 1000

    while True:
        batch = _collect_batch(max_size, max_wait_s)
        connection = _rmq_ctx.get("connection")
        channel = _rmq_ctx.get("channel")

        try:
            frame_results = run_reid_batch([job["img_np"] for job in batch])
        except Exception as e:
            print(f"[BatchWorker] 배치 추론 오류: {e}")
            for job in batch:
                connection.add_callback_threadsafe(
                    functools.partial(channel.basic_nack, delivery_tag=job["delivery_tag"], requeue=False)
                )
            continue

        rmq_cfg = _cfg["rabbitmq"]
        for job, result in zip(batch, frame_results):
            # [수정] AI 서버의 종료 시간을 구한 뒤, 절대 시간이 아닌 '소요 시간(Duration)'을 계산해서 넣습니다.
            t4_end_time = int(time.time() * 1000)
            t3_start_time = job["timestamps"].get("T3_AI_Consumption", t4_end_time)

            # AI 서버 내부에서 걸린 순수 연산 시간 (ms)
            job["timestamps"]["AI_Duration"] = t4_end_time - t3_start_time

            connection.add_callback_threadsafe(
                functools.partial(_publish_and_ack, channel, rmq_cfg, job, result)
            )

        print(f"[BatchWorker] 배치 처리 — {len(batch)}프레임 / 검출 {sum(r['count'] for r in frame_results)}명")


def start_consumer():
    import ssl as _ssl
    rmq_cfg = _cfg["rabbitmq"]

    while True:
        try:
            credentials = pika.PlainCredentials(rmq_cfg["username"], rmq_cfg["password"])
            ssl_options = None
            if rmq_cfg.get("ssl", False):
                ssl_context = _ssl.create_default_context()
                ssl_options = pika.SSLOptions(ssl_context)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=rmq_cfg["host"],
                    port=rmq_cfg["port"],
                    virtual_host=rmq_cfg.get("vhost", "/"),
                    credentials=credentials,
                    ssl_options=ssl_options,
                )
            )
            channel = connection.channel()

            channel.queue_declare(queue=rmq_cfg["input_queue"], durable=True)
            channel.queue_declare(queue=rmq_cfg["output_queue"], durable=True)

            channel.basic_qos(prefetch_count=rmq_cfg.get("prefetch_count", 1))
            channel.basic_consume(queue=rmq_cfg["input_queue"], on_message_callback=on_message)

            _rmq_ctx["connection"] = connection
            _rmq_ctx["channel"] = channel

            if "worker" not in _rmq_ctx:
                worker_thread = threading.Thread(target=batch_worker, daemon=True)
                worker_thread.start()
                _rmq_ctx["worker"] = worker_thread
                print("[BatchWorker] 배치 추론 쓰레드 시작")

            print(f"[Consumer] '{rmq_cfg['input_queue']}' 구독 시작")
            channel.start_consuming()

        except Exception as e:
            print(f"[Consumer] RabbitMQ 연결 실패, 5초 후 재시도: {e}")
            time.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _cfg.update(cfg)

    reid_cfg = cfg["reid"]
    device = reid_cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _cfg["reid"]["device"] = device

    from ultralytics import YOLO
    _models["yolo"] = YOLO(cfg["yolo"]["model"])

    reid_model = BoTReID().to(device)
    ckpt = torch.load(reid_cfg["weights"], map_location=device, weights_only=False)
    reid_model.load_state_dict(ckpt["state_dict"], strict=False)
    reid_model.eval()
    _models["reid"] = reid_model

    dim_cfg = reid_cfg.get("dim_reduction") or {}
    if dim_cfg.get("enabled", False):
        pca = PCAWhiten.load(dim_cfg["path"])
        _models["pca"] = pca
        print(f"[config] 차원 축소 적용: {dim_cfg['path']} → {pca.n_components}차원 "
              f"(설명된 분산 {pca.explained_variance_ratio_ * 100:.1f}%)")
    else:
        _models["pca"] = None
        print("[config] 차원 축소 미적용 — BoT 원본 2048차원 그대로 사용")

    print(f"[config] YOLO={cfg['yolo']['model']}  ReID=BoT(bot_best.pth)  "
          f"Rank-1={ckpt['rank1']:.2f}%  mAP={ckpt['mAP']:.2f}%  device={device}")

    consumer_thread = threading.Thread(target=start_consumer, daemon=True)
    consumer_thread.start()
    print("[Consumer] 백그라운드 쓰레드 시작")

    yield

    _models.clear()
    _cfg.clear()


app = FastAPI(title="Person ReID API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "yolo": _cfg.get("yolo", {}).get("model"),
        "reid": "BoT (bot_best.pth)",
        "reid_rank1": f"{_cfg.get('reid', {}).get('rank1', 'N/A')}",
    }


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    return run_reid(np.array(img))
