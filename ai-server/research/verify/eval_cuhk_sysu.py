"""
CUHK-SYSU 데이터셋 정확도 평가

평가 지표:
  - Rank-1 / Rank-5 / Rank-10 정확도
  - mAP (mean Average Precision)

평가 방식:
  - Query/Gallery GT bbox로 crop → BoT ReID 벡터 추출
  - Distractor 이미지는 YOLO로 주요 인물 crop (없으면 더미 벡터 사용)
  - 코사인 유사도 기반 랭킹 후 표준 CMC/mAP 계산

실행:
  cd /home/lookie14/Intelligent-CCTV-ReID/ai-server
  python research/verify/eval_cuhk_sysu.py
  python research/verify/eval_cuhk_sysu.py --n 500 --gallery TestG50
  python research/verify/eval_cuhk_sysu.py --n 200 --gallery TestG1000 --save research/results/eval_cuhk_sysu.md
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from reid_core.models.bot import BoTReID
from research.benchmark.models.bot_ibn import BoTIBNReID

MODEL_MAP = {
    "bot":     (BoTReID,     "BoT ResNet50",           "reid_core/checkpoints/bot_best.pth"),
    "bot_ibn": (BoTIBNReID,  "BoT ResNet50-IBN-a",     "research/benchmark/checkpoints/bot_ibn_best.pth"),
}

# ── 경로 ───────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/lookie14/capstone/dataset/cuhk_sysu")
IMAGE_DIR    = DATASET_ROOT / "Image" / "SSM"
ANNOT_DIR    = DATASET_ROOT / "annotation"

# ── 전처리 ─────────────────────────────────────────────────────────
_TF = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Crop 유틸 ──────────────────────────────────────────────────────
def xywh_to_xyxy(box):
    """[x, y, w, h] → [x1, y1, x2, y2]"""
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    return x, y, x + w, y + h


def crop_from_gt(imname: str, idlocate) -> Image.Image | None:
    """GT bbox로 인물 crop"""
    img_path = IMAGE_DIR / imname
    if not img_path.exists():
        return None
    try:
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        x1, y1, x2, y2 = xywh_to_xyxy(idlocate)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return img.crop((x1, y1, x2, y2))
    except Exception:
        return None


def crop_top_detection(imname: str, yolo, conf=0.25) -> Image.Image | None:
    """Distractor 이미지에서 YOLO로 가장 신뢰도 높은 인물 crop"""
    img_path = IMAGE_DIR / imname
    if not img_path.exists():
        return None
    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        results = yolo(img_np, classes=[0], conf=conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        best = boxes[int(boxes.conf.argmax())]
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        iw, ih = img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        return img.crop((x1, y1, x2, y2))
    except Exception:
        return None


# ── 특징 추출 ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_feature(crop: Image.Image, reid, device: str) -> np.ndarray:
    t = _TF(crop).unsqueeze(0).to(device)
    feat = reid(t)
    feat = F.normalize(feat, dim=1)
    return feat.cpu().numpy()[0]


def extract_batch(crops: list[Image.Image], reid, device: str, batch_size: int = 32) -> np.ndarray:
    feats = []
    for i in range(0, len(crops), batch_size):
        batch = torch.stack([_TF(c) for c in crops[i:i + batch_size]]).to(device)
        with torch.no_grad():
            f = reid(batch)
            f = F.normalize(f, dim=1)
        feats.append(f.cpu().numpy())
    return np.vstack(feats)


# ── CMC / mAP 계산 ─────────────────────────────────────────────────
def compute_cmc_map(query_feats, gallery_feats, gallery_labels):
    """
    query_feats  : (Q, D)
    gallery_feats: (G, D)
    gallery_labels: (G,) — 1=positive, 0=distractor
    Returns cmc (length 10), mAP
    """
    sims = query_feats @ gallery_feats.T          # (Q, G) 코사인 유사도
    ranks = np.argsort(-sims, axis=1)             # 높은 유사도 순 정렬

    Q = query_feats.shape[0]
    cmc_acc = np.zeros(10)
    all_ap  = []

    for qi in range(Q):
        ordered_labels = gallery_labels[qi][ranks[qi]]  # 이 쿼리의 gallery label 순서
        pos_mask = ordered_labels == 1

        if pos_mask.sum() == 0:
            continue

        # CMC
        found = np.minimum(pos_mask.cumsum(), 1)
        cmc_acc += found[:10]

        # AP
        tp       = pos_mask.cumsum()
        prec     = tp / (np.arange(len(pos_mask)) + 1)
        all_ap.append((prec * pos_mask).sum() / pos_mask.sum())

    cmc_acc /= Q
    mAP = np.mean(all_ap) * 100
    return cmc_acc * 100, mAP


# ── 메인 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int,   default=300,       help="평가 쿼리 수 (기본 300)")
    parser.add_argument("--gallery", type=str,   default="TestG50", help="TestG50 / TestG100 / TestG500 / TestG1000")
    parser.add_argument("--model",   type=str,   default="bot",     choices=list(MODEL_MAP), help="bot / bot_ibn")
    parser.add_argument("--weights", type=str,   default=None,      help="기본값: --model에 대응하는 체크포인트")
    parser.add_argument("--yolo",    type=str,   default="yolov8n.pt")
    parser.add_argument("--device",  type=str,   default="auto")
    parser.add_argument("--save",    type=str,   default=None,      help="결과 저장 경로 (.md)")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    reid_cls, model_label, default_weights = MODEL_MAP[args.model]
    weights_path = args.weights or default_weights
    print(f"[설정] device={device}  gallery={args.gallery}  n={args.n}  model={model_label}")

    # 모델 로드
    yolo = YOLO(args.yolo)

    reid = reid_cls().to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    reid.load_state_dict(ckpt["state_dict"], strict=False)
    reid.eval()
    print(f"[모델] {model_label} Rank-1={ckpt['rank1']:.2f}%  mAP={ckpt['mAP']:.2f}%  (Market-1501 기준)")

    # 어노테이션 로드
    mat_path = ANNOT_DIR / "test" / "train_test" / f"{args.gallery}.mat"
    mat = sio.loadmat(str(mat_path), simplify_cells=True)
    test_data = mat[args.gallery]
    total_queries = len(test_data)
    n = min(args.n, total_queries)
    print(f"[데이터] 전체 쿼리: {total_queries}  평가: {n}\n")

    query_feats  = []
    gallery_feat_list  = []
    gallery_label_list = []
    skip_count   = 0
    t_start      = time.perf_counter()

    for i, entry in enumerate(test_data[:n]):
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1:>4}/{n}] {elapsed:.0f}s 경과...")

        # Query crop
        q = entry["Query"]
        q_crop = crop_from_gt(q["imname"], q["idlocate"])
        if q_crop is None:
            skip_count += 1
            continue

        q_feat = extract_feature(q_crop, reid, device)

        # Gallery
        gallery = entry["Gallery"]
        g_crops  = []
        g_labels = []

        for g in gallery:
            has_bbox = isinstance(g["idlocate"], np.ndarray) and g["idlocate"].size > 0
            if has_bbox:
                crop = crop_from_gt(g["imname"], g["idlocate"])
                label = 1
            else:
                crop = crop_top_detection(g["imname"], yolo)
                label = 0

            if crop is None:
                crop = Image.new("RGB", (64, 128), (128, 128, 128))  # 더미

            g_crops.append(crop)
            g_labels.append(label)

        g_feats  = extract_batch(g_crops, reid, device)
        g_labels = np.array(g_labels)

        query_feats.append(q_feat)
        gallery_feat_list.append(g_feats)
        gallery_label_list.append(g_labels)

    elapsed = time.perf_counter() - t_start
    evaluated = len(query_feats)
    print(f"\n[완료] {evaluated}개 쿼리 평가  ({elapsed:.1f}s, 스킵: {skip_count})\n")

    if evaluated == 0:
        print("평가 가능한 쿼리가 없습니다.")
        return

    # 유사도 계산 — 각 쿼리마다 gallery 크기가 같으므로 numpy stacking
    Q_feats = np.vstack(query_feats)                 # (Q, D)
    G_feats = np.stack(gallery_feat_list)            # (Q, G, D)
    G_labels = np.stack(gallery_label_list)          # (Q, G)

    # 쿼리별 유사도: (Q, G)
    sims = np.einsum("qd,qgd->qg", Q_feats, G_feats)
    ranks = np.argsort(-sims, axis=1)

    cmc_acc = np.zeros(10)
    all_ap  = []

    for qi in range(evaluated):
        ordered = G_labels[qi][ranks[qi]]
        pos_mask = ordered == 1
        if pos_mask.sum() == 0:
            continue
        found = np.minimum(pos_mask.cumsum(), 1)
        cmc_acc += found[:10]
        tp   = pos_mask.cumsum()
        prec = tp / (np.arange(len(pos_mask)) + 1)
        all_ap.append((prec * pos_mask).sum() / pos_mask.sum())

    cmc_acc = cmc_acc / evaluated * 100
    mAP     = np.mean(all_ap) * 100

    # 결과 출력
    print("=" * 55)
    print(f"  CUHK-SYSU 정확도 평가  ({args.gallery}, {evaluated}쿼리)")
    print("=" * 55)
    print(f"  mAP      : {mAP:.2f}%")
    print(f"  Rank-1   : {cmc_acc[0]:.2f}%")
    print(f"  Rank-5   : {cmc_acc[4]:.2f}%")
    print(f"  Rank-10  : {cmc_acc[9]:.2f}%")
    print("=" * 55)

    if args.save:
        save_markdown(args, device, evaluated, skip_count, ckpt, mAP, cmc_acc, elapsed, model_label, weights_path)


def save_markdown(args, device, evaluated, skip_count, ckpt, mAP, cmc_acc, elapsed, model_label, weights_path):
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# CUHK-SYSU 파이프라인 정확도 평가",
        "",
        f"- **데이터셋** : CUHK-SYSU (`{args.gallery}`)",
        f"- **평가 쿼리** : {evaluated}개",
        f"- **디바이스** : {device.upper()}",
        f"- **측정일** : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 1. 모델 구성",
        "",
        "| 구성 요소 | 내용 |",
        "|:---------|:-----|",
        f"| ReID 모델 | {model_label} (`{Path(weights_path).name}`) |",
        f"| 학습 데이터 | Market-1501 (파인튜닝) |",
        f"| Market-1501 Rank-1 | {ckpt['rank1']:.2f}% |",
        f"| Market-1501 mAP | {ckpt['mAP']:.2f}% |",
        f"| 검출 모델 | YOLOv8n (distractor 처리용) |",
        "",
        "---",
        "",
        "## 2. 평가 방법",
        "",
        "- **Query**: GT bbox로 인물 crop → BoT 벡터 추출",
        "- **Gallery (positive)**: GT bbox로 인물 crop → BoT 벡터 추출",
        "- **Gallery (distractor)**: YOLO 최상위 검출로 crop → BoT 벡터 추출",
        "- **랭킹**: 코사인 유사도 기준 내림차순 정렬",
        "- **지표**: CMC (Rank-1/5/10), mAP",
        "",
        "---",
        "",
        "## 3. 정확도 결과",
        "",
        "| 지표 | CUHK-SYSU | Market-1501 (참고) |",
        "|:-----|----------:|------------------:|",
        f"| mAP | **{mAP:.2f}%** | {ckpt['mAP']:.2f}% |",
        f"| Rank-1 | **{cmc_acc[0]:.2f}%** | {ckpt['rank1']:.2f}% |",
        f"| Rank-5 | **{cmc_acc[4]:.2f}%** | — |",
        f"| Rank-10 | **{cmc_acc[9]:.2f}%** | — |",
        "",
        "---",
        "",
        "## 4. 분석",
        "",
        "### 도메인 차이 (Market-1501 vs CUHK-SYSU)",
        "",
        "| 항목 | Market-1501 | CUHK-SYSU |",
        "|:-----|:-----------|:---------|",
        "| 이미지 유형 | 고정 카메라 crop | 실제 씬 이미지 |",
        "| 평균 인물 크기 | 비교적 균일 | 크기 다양 |",
        "| 배경 복잡도 | 낮음 | 높음 |",
        "| 학습 여부 | 파인튜닝 완료 | 미학습 (zero-shot) |",
        "",
        "모델이 Market-1501로만 학습되어 CUHK-SYSU는 **zero-shot 일반화** 성능을 보여줍니다.",
        f"평가 소요 시간: {elapsed:.1f}초 ({evaluated}쿼리)",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {out.resolve()}")


if __name__ == "__main__":
    main()
