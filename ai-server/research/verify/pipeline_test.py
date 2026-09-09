"""
CUHK-SYSU 데이터셋을 이용한 파이프라인 테스트
측정 항목: 각 구간별 FLOPs / 처리 시간 (ms)

파이프라인:
  [1] 이미지 로드
  [2] YOLO 사람 검출
  [3] Crop 추출
  [4] BoT ReID 특징 벡터 추출

실행:
  cd /home/lookie14/Intelligent-CCTV-ReID/ai-server
  python research/verify/pipeline_test.py                          # 기본 100장
  python research/verify/pipeline_test.py --n 0                    # 전체 이미지 사용
  python research/verify/pipeline_test.py --n 1000 --save research/results/pipeline_1000.md
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from thop import profile
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from reid_core.models.bot import BoTReID

# ── 경로 설정 ───────────────────────────────────────────────────────
DATASET_DIR = Path("/home/lookie14/capstone/dataset/cuhk_sysu/Image/SSM")
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

# ── ReID 전처리 ─────────────────────────────────────────────────────
_REID_TF = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── FLOPs 측정 ──────────────────────────────────────────────────────
def measure_yolo_flops(model) -> float:
    """YOLO FLOPs: ultralytics 내장 info() 사용 — 640×640 기준 GFLOPs"""
    # info()는 (layers, params, grads, gflops) 반환
    _, _, _, gflops = model.info(verbose=True, detailed=False)
    return gflops


def measure_reid_flops(device: str) -> float:
    """BoT ReID FLOPs: 추론 모델과 분리된 fresh 인스턴스로 측정 (hook 오염 방지)"""
    fresh = BoTReID().to(device).eval()
    dummy = torch.zeros(1, 3, 256, 128).to(device)
    with torch.no_grad():
        flops, _ = profile(fresh, inputs=(dummy,), verbose=False)
    del fresh
    return flops / 1e9  # GFLOPs


# ── 모델 로드 ───────────────────────────────────────────────────────
def load_models(cfg: dict):
    device = cfg["reid"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[설정] device={device}")

    yolo = YOLO(cfg["yolo"]["model"])

    reid = BoTReID().to(device)
    ckpt = torch.load(cfg["reid"]["weights"], map_location=device, weights_only=False)
    reid.load_state_dict(ckpt["state_dict"], strict=False)
    reid.eval()

    print(f"[모델] YOLO={cfg['yolo']['model']}  "
          f"ReID=BoT(Rank-1={ckpt['rank1']:.2f}%  mAP={ckpt['mAP']:.2f}%)")
    return yolo, reid, device


# ── 단일 이미지 파이프라인 실행 ─────────────────────────────────────
def run_pipeline(img_path: Path, yolo, reid, device: str, cfg: dict):
    times = {}

    # [1] 이미지 로드
    t0 = time.perf_counter()
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    times["load_ms"] = (time.perf_counter() - t0) * 1000

    # [2] YOLO 검출
    t0 = time.perf_counter()
    results = yolo(
        img_np,
        classes=[0],
        conf=cfg["yolo"]["confidence"],
        iou=cfg["yolo"]["iou"],
        verbose=False,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    times["yolo_ms"] = (time.perf_counter() - t0) * 1000

    boxes = results[0].boxes
    n_det = len(boxes) if boxes is not None else 0

    # [3] Crop 추출
    t0 = time.perf_counter()
    crops = []
    h, w = img_np.shape[:2]
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img_np[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
    times["crop_ms"] = (time.perf_counter() - t0) * 1000

    # [4] ReID 특징 추출
    if crops:
        t0 = time.perf_counter()
        tensors = torch.stack([_REID_TF(Image.fromarray(c)) for c in crops]).to(device)
        with torch.no_grad():
            features = reid(tensors)
        if device == "cuda":
            torch.cuda.synchronize()
        times["reid_ms"] = (time.perf_counter() - t0) * 1000
    else:
        times["reid_ms"] = 0.0
        features = None

    times["total_ms"] = times["load_ms"] + times["yolo_ms"] + times["crop_ms"] + times["reid_ms"]

    return times, n_det, len(crops), features


# ── 결과 출력 ───────────────────────────────────────────────────────
def print_stats(label: str, values: list[float], unit: str = "ms"):
    if not values:
        return
    arr = np.array(values)
    print(f"  {label:<18} mean={mean(values):7.2f}{unit}  "
          f"min={min(values):7.2f}{unit}  max={max(values):7.2f}{unit}  "
          f"std={stdev(values) if len(values) > 1 else 0:.2f}{unit}  "
          f"p50={np.percentile(arr,50):7.2f}{unit}  "
          f"p95={np.percentile(arr,95):7.2f}{unit}  "
          f"p99={np.percentile(arr,99):7.2f}{unit}")


# ── 메인 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=100,  help="테스트 이미지 수 (기본 100)")
    parser.add_argument("--warmup",  type=int, default=5,    help="워밍업 이미지 수 (기본 5)")
    parser.add_argument("--config",  type=str, default=str(CONFIG_PATH))
    parser.add_argument("--save",    type=str, default=None, help="결과 저장 경로 (예: results/report.md)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    yolo, reid, device = load_models(cfg)

    # ── FLOPs 측정 (1회) ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  FLOPs 측정")
    print("="*60)

    sample_img = np.array(Image.open(next(DATASET_DIR.iterdir())).convert("RGB"))

    yolo_flops = measure_yolo_flops(yolo)
    reid_flops = measure_reid_flops(device)

    print(f"  YOLO ({cfg['yolo']['model']})        : {yolo_flops:.3f} GFLOPs  (입력 {sample_img.shape[1]}×{sample_img.shape[0]})")
    print(f"  BoT ReID (bot_best.pth)   : {reid_flops:.3f} GFLOPs  (입력 256×128, crop 1장 기준)")
    print(f"  합계 (1 crop 기준)         : {yolo_flops + reid_flops:.3f} GFLOPs")

    # ── 이미지 목록 준비 ──────────────────────────────────────────
    all_images = sorted(DATASET_DIR.glob("*.jpg"))
    if args.n == 0:
        args.n = max(0, len(all_images) - args.warmup)
        print(f"[설정] --n 0 → 전체 이미지 사용: {args.n}장 (워밍업 {args.warmup}장 제외)")
    elif len(all_images) < args.n + args.warmup:
        print(f"[경고] 요청 이미지 수({args.n}+{args.warmup})가 데이터셋({len(all_images)})보다 많음. 전체 사용.")
        args.n = max(0, len(all_images) - args.warmup)
    test_images = all_images[:args.warmup + args.n]

    # ── 워밍업 ────────────────────────────────────────────────────
    print(f"\n[워밍업] {args.warmup}장 처리 중...")
    for img_path in test_images[:args.warmup]:
        run_pipeline(img_path, yolo, reid, device, cfg)

    # ── 본 측정 ───────────────────────────────────────────────────
    print(f"[측정] {args.n}장 처리 중...\n")

    stats = {k: [] for k in ["load_ms", "yolo_ms", "crop_ms", "reid_ms", "total_ms"]}
    det_counts, crop_counts = [], []

    for i, img_path in enumerate(test_images[args.warmup:], 1):
        times, n_det, n_crop, _ = run_pipeline(img_path, yolo, reid, device, cfg)
        for k, v in times.items():
            stats[k].append(v)
        det_counts.append(n_det)
        crop_counts.append(n_crop)

        if i % 20 == 0:
            print(f"  [{i:>4}/{args.n}] total={times['total_ms']:.1f}ms  "
                  f"yolo={times['yolo_ms']:.1f}ms  reid={times['reid_ms']:.1f}ms  "
                  f"검출={n_det}명")

    # ── 최종 결과 ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  처리 시간 통계 ({args.n}장, device={device})")
    print("="*60)
    print_stats("[1] 이미지 로드",    stats["load_ms"])
    print_stats("[2] YOLO 검출",      stats["yolo_ms"])
    print_stats("[3] Crop 추출",      stats["crop_ms"])
    print_stats("[4] ReID 추출",      stats["reid_ms"])
    print("-"*60)
    print_stats("전체 파이프라인",    stats["total_ms"])

    print(f"\n  검출 인원   mean={mean(det_counts):.2f}명  "
          f"min={min(det_counts)}명  max={max(det_counts)}명")
    print(f"  유효 crop   mean={mean(crop_counts):.2f}개  "
          f"min={min(crop_counts)}개  max={max(crop_counts)}개")

    no_det = sum(1 for x in det_counts if x == 0)
    print(f"  미검출 이미지: {no_det}/{args.n}장 ({no_det/args.n*100:.1f}%)")

    fps = 1000 / mean(stats["total_ms"])
    print(f"\n  추정 처리량: {fps:.1f} FPS")
    print("="*60)

    if args.save:
        save_markdown(
            path=args.save,
            cfg=cfg,
            device=device,
            n=args.n,
            yolo_flops=yolo_flops,
            reid_flops=reid_flops,
            stats=stats,
            det_counts=det_counts,
            crop_counts=crop_counts,
            fps=fps,
        )


def save_markdown(path, cfg, device, n, yolo_flops, reid_flops, stats, det_counts, crop_counts, fps):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    avg_persons = mean(det_counts)
    total_flops = yolo_flops + reid_flops * avg_persons
    no_det = sum(1 for x in det_counts if x == 0)

    lines = [
        f"# 파이프라인 테스트 결과",
        f"",
        f"- **데이터셋** : CUHK-SYSU (`Image/SSM/`)",
        f"- **테스트 이미지** : {n}장",
        f"- **디바이스** : {device.upper()}",
        f"- **측정일** : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"---",
        f"",
        f"## 1. 모델 구성",
        f"",
        f"| 구성 요소 | 모델 | 비고 |",
        f"|:---------|:-----|:-----|",
        f"| 사람 검출 | YOLOv8n (`{cfg['yolo']['model']}`) | conf={cfg['yolo']['confidence']}, iou={cfg['yolo']['iou']} |",
        f"| 특징 추출 | BoT ResNet50 (`bot_best.pth`) | Market-1501 파인튜닝, Rank-1 90.86%, mAP 78.13% |",
        f"| 벡터 차원 | 2048-dim | BN-Neck 출력 |",
        f"",
        f"---",
        f"",
        f"## 2. FLOPs (이론적 연산량)",
        f"",
        f"| 구간 | GFLOPs | 기준 입력 |",
        f"|:-----|-------:|:---------|",
        f"| [2] YOLO 검출 | {yolo_flops:.3f} | 640×640 |",
        f"| [4] BoT ReID | {reid_flops:.3f} | 256×128, crop 1장 |",
        f"| 합계 (crop 1장 기준) | {yolo_flops + reid_flops:.3f} | — |",
        f"| **합계 (평균 {avg_persons:.1f}명 기준)** | **{total_flops:.3f}** | **실측 평균 인원 적용** |",
        f"",
        f"> ReID FLOPs는 검출된 사람 수에 비례하여 증가합니다.",
        f"",
        f"---",
        f"",
        f"## 3. 구간별 처리 시간",
        f"",
        f"| 구간 | 평균 (ms) | 최소 (ms) | 최대 (ms) | 표준편차 (ms) | p50 (ms) | p95 (ms) | p99 (ms) | 비율 (%) |",
        f"|:-----|----------:|----------:|----------:|-------------:|---------:|---------:|---------:|---------:|",
    ]

    total_mean = mean(stats["total_ms"])
    stage_labels = [
        ("load_ms",  "[1] 이미지 로드"),
        ("yolo_ms",  "[2] YOLO 검출"),
        ("crop_ms",  "[3] Crop 추출"),
        ("reid_ms",  "[4] ReID 추출"),
        ("total_ms", "**전체 파이프라인**"),
    ]
    for key, label in stage_labels:
        v = stats[key]
        arr = np.array(v)
        m = mean(v)
        sd = stdev(v) if len(v) > 1 else 0
        p50 = np.percentile(arr, 50)
        p95 = np.percentile(arr, 95)
        p99 = np.percentile(arr, 99)
        ratio = m / total_mean * 100 if key != "total_ms" else 100.0
        sep = "**" if key == "total_ms" else ""
        lines.append(
            f"| {label} | {sep}{m:.2f}{sep} | {sep}{min(v):.2f}{sep} | "
            f"{sep}{max(v):.2f}{sep} | {sep}{sd:.2f}{sep} | "
            f"{sep}{p50:.2f}{sep} | {sep}{p95:.2f}{sep} | {sep}{p99:.2f}{sep} | "
            f"{sep}{ratio:.1f}{sep} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## 4. 검출 통계",
        f"",
        f"| 항목 | 값 |",
        f"|:-----|:---|",
        f"| 평균 검출 인원 | {mean(det_counts):.2f}명 |",
        f"| 최소 검출 인원 | {min(det_counts)}명 |",
        f"| 최대 검출 인원 | {max(det_counts)}명 |",
        f"| 평균 유효 crop | {mean(crop_counts):.2f}개 |",
        f"| 미검출 이미지 | {no_det}장 / {n}장 ({no_det/n*100:.1f}%) |",
        f"",
        f"---",
        f"",
        f"## 5. 종합 성능",
        f"",
        f"| 항목 | 값 |",
        f"|:-----|:---|",
        f"| 추정 처리량 | **{fps:.1f} FPS** |",
        f"| 평균 전체 지연 | **{total_mean:.2f} ms** |",
        f"| 최저 지연 | {min(stats['total_ms']):.2f} ms |",
        f"| 최고 지연 | {max(stats['total_ms']):.2f} ms |",
        f"",
        f"### 병목 구간",
        f"",
        f"ReID 추출이 전체 처리 시간의 **{mean(stats['reid_ms'])/total_mean*100:.1f}%** 를 차지합니다.",
        f"검출 인원이 많을수록 ReID 시간이 선형 증가하므로, 다중 카메라 환경에서는 배치 처리 최적화가 필요합니다.",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[저장] {out.resolve()}")


if __name__ == "__main__":
    main()
