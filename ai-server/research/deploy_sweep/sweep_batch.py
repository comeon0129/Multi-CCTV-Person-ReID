import argparse
import queue
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from reid_core.dim_reduction import PCAWhiten
from reid_core.models.bot import BoTReID

DATASET_DIR = Path("/home/lookie14/capstone/dataset/cuhk_sysu/Image/SSM")
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

_REID_TF = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_pool(n: int) -> list[np.ndarray]:
    paths = sorted(DATASET_DIR.glob("*.jpg"))[:max(n, 200)]
    sample = random.sample(paths, n)
    return [np.array(Image.open(p).convert("RGB")) for p in sample]


def load_models(cfg: dict):
    device = cfg["reid"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(cfg["yolo"]["model"])

    reid = BoTReID().to(device)
    ckpt = torch.load(cfg["reid"]["weights"], map_location=device, weights_only=False)
    reid.load_state_dict(ckpt["state_dict"], strict=False)
    reid.eval()

    pca = PCAWhiten.load(cfg["reid"]["dim_reduction"]["path"])

    return yolo, reid, pca, device


def run_batch(img_list, yolo, reid, pca, device, cfg, apply_pca: bool) -> int:
    yolo_cfg = cfg["yolo"]
    results = yolo(img_list, classes=[0], conf=yolo_cfg["confidence"], iou=yolo_cfg["iou"], verbose=False)

    crops = []
    for img_np, res in zip(img_list, results):
        h, w = img_np.shape[:2]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img_np[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

    if not crops:
        return 0

    tensors = torch.stack([_REID_TF(Image.fromarray(c)) for c in crops]).to(device)
    with torch.no_grad():
        features = reid(tensors)
    if device == "cuda":
        torch.cuda.synchronize()

    feats_np = features.cpu().numpy()
    if apply_pca:
        feats_np = pca.transform(feats_np)

    return len(crops)


def collect_batch(q: "queue.Queue", max_size: int, max_wait_s: float, stop_event: threading.Event):
    while True:
        try:
            first = q.get(timeout=0.5)
            break
        except queue.Empty:
            if stop_event.is_set() and q.empty():
                return None

    batch = [first]
    deadline = time.monotonic() + max_wait_s
    while len(batch) < max_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            batch.append(q.get(timeout=remaining))
        except queue.Empty:
            break
    return batch


def producer(q, pool, fps, duration_s, stop_event):
    interval = 1.0 / fps
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        t0 = time.monotonic()
        img = random.choice(pool)
        q.put((img, time.perf_counter()))
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))
    stop_event.set()


def run_combo(pool, yolo, reid, pca, device, cfg, apply_pca, max_wait_ms, max_size, fps, warmup_s, measure_s, seed):
    random.seed(seed)
    q: "queue.Queue" = queue.Queue()
    stop_event = threading.Event()

    t_start = time.perf_counter()
    measure_from = t_start + warmup_s

    prod_thread = threading.Thread(
        target=producer, args=(q, pool, fps, warmup_s + measure_s, stop_event), daemon=True,
    )
    prod_thread.start()

    latencies_ms, batch_sizes = [], []

    while True:
        batch = collect_batch(q, max_size, max_wait_ms / 1000, stop_event)
        if batch is None:
            break
        imgs = [b[0] for b in batch]
        arrivals = [b[1] for b in batch]

        t0 = time.perf_counter()
        run_batch(imgs, yolo, reid, pca, device, cfg, apply_pca)
        t1 = time.perf_counter()

        if arrivals[0] >= measure_from:
            batch_sizes.append(len(batch))
            for a in arrivals:
                latencies_ms.append((t1 - a) * 1000)

    prod_thread.join()

    arr = np.array(latencies_ms)
    return {
        "dim_reduction": apply_pca,
        "max_wait_ms": max_wait_ms,
        "n_frames": len(latencies_ms),
        "n_batches": len(batch_sizes),
        "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
        "mean_batch_size": mean(batch_sizes) if batch_sizes else 0.0,
        "max_batch_size": max(batch_sizes) if batch_sizes else 0,
    }


def save_markdown(path, cfg, device, fps, max_size, warmup_s, measure_s, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# max_wait_ms x dim_reduction 스윕 결과",
        "",
        f"- **측정일** : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- **디바이스** : {device.upper()}",
        f"- **데이터셋** : CUHK-SYSU (`Image/SSM/`, 로컬)",
        f"- **모델** : YOLO=`{cfg['yolo']['model']}`, ReID=BoT(`bot_best.pth`)",
        f"- **고정값** : max_size={max_size}, 시뮬레이션 fps={fps}, warmup={warmup_s}s, 측정={measure_s}s",
        "- **방법** : RabbitMQ를 거치지 않고 `main.py`의 `_collect_batch`와 동일한 배치 수집 로직을 로컬로 재현. "
        "가상 프로듀서 스레드가 지정 fps로 로컬 이미지를 큐에 발행하고, 컨슈머가 배치를 모아 YOLO+ReID(+선택적 PCA) forward.",
        "",
        "| dim_reduction | max_wait_ms | p50(ms) | p95(ms) | p99(ms) | 평균 배치크기 | 최대 배치 | 배치 수 | 샘플 프레임 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {'on' if r['dim_reduction'] else 'off'} | {r['max_wait_ms']} | "
            f"{r['p50']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} | "
            f"{r['mean_batch_size']:.2f} | {r['max_batch_size']} | {r['n_batches']} | {r['n_frames']} |"
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[저장] {out.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--max_wait_ms", type=int, nargs="+", default=[10, 30, 50, 100])
    parser.add_argument("--dim_reduction", choices=["on", "off", "both"], default="both")
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--warmup_s", type=float, default=3.0)
    parser.add_argument("--measure_s", type=float, default=15.0)
    parser.add_argument("--pool_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="research/results/batch_wait_dimred_sweep.md")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    max_size = args.max_size or cfg["reid"]["batch"]["max_size"]
    dim_reduction_options = {"on": [True], "off": [False], "both": [True, False]}[args.dim_reduction]

    print(f"[설정] max_wait_ms={args.max_wait_ms}  dim_reduction={dim_reduction_options}  "
          f"max_size={max_size}  fps={args.fps}")

    random.seed(args.seed)
    pool = load_pool(args.pool_size)
    yolo, reid, pca, device = load_models(cfg)
    print(f"[모델] YOLO={cfg['yolo']['model']}  ReID=BoT  device={device}  풀 이미지={len(pool)}장")

    print("[워밍업] JIT/CUDA 초기화 중...")
    run_batch(pool[:4], yolo, reid, pca, device, cfg, apply_pca=True)

    rows = []
    total = len(args.max_wait_ms) * len(dim_reduction_options)
    i = 0
    for apply_pca in dim_reduction_options:
        for wait_ms in args.max_wait_ms:
            i += 1
            print(f"\n[{i}/{total}] dim_reduction={'on' if apply_pca else 'off'}  max_wait_ms={wait_ms}  실행 중...")
            result = run_combo(
                pool, yolo, reid, pca, device, cfg, apply_pca,
                wait_ms, max_size, args.fps, args.warmup_s, args.measure_s, args.seed,
            )
            rows.append(result)
            print(f"  p50={result['p50']:.1f}ms  p95={result['p95']:.1f}ms  "
                  f"평균배치={result['mean_batch_size']:.2f}  배치수={result['n_batches']}")

    save_markdown(args.save, cfg, device, args.fps, max_size, args.warmup_s, args.measure_s, rows)


if __name__ == "__main__":
    main()
