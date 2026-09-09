"""
Market-1501 ReID 통합 벤치마크
  - 정확도: mAP, Rank-1 / Rank-5 / Rank-10
  - 효율성: Params(M), GFLOPs, 추론 레이턴시

사용:
    cd /home/lookie14/Intelligent-CCTV-ReID/ai-server
    python -m research.benchmark.eval_market1501
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchreid.reid.utils.feature_extractor import FeatureExtractor

from research.benchmark.utils import measure_flops, time_inference

# ── 설정 ──────────────────────────────────────────────────────────
MARKET_ROOT = Path("/home/lookie14/Intelligent-CCTV-ReID/ai-server/data/Market-1501-v15.09.15")
OUTPUT_DIR  = Path("/home/lookie14/Intelligent-CCTV-ReID/ai-server/research/results")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 256
NUM_WORKERS = 4
LATENCY_RUNS   = 100
LATENCY_WARMUP = 10

_transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Market-1501 데이터 로더 ────────────────────────────────────────
def _parse_id(fname: str):
    m = re.match(r"(-?\d+)_c(\d+)", fname)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def load_set(folder: Path):
    paths, pids, cids = [], [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".jpg"):
            continue
        pid, cid = _parse_id(fname)
        paths.append(str(folder / fname))
        pids.append(pid)
        cids.append(cid)
    return paths, np.array(pids), np.array(cids)


# ── 특징 추출 ──────────────────────────────────────────────────────
class NNWrapper:
    """nn.Module을 GPU 배치 추론 래퍼로 감쌈."""

    def __init__(self, model: nn.Module, device: str):
        self.model  = model.eval().to(device)
        self.device = device

    def __call__(self, crops: list) -> torch.Tensor:
        tensors = torch.stack([_transform(Image.fromarray(c)) for c in crops]).to(self.device)
        with torch.no_grad():
            return self.model(tensors)

    def forward_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(batch.to(self.device))


class _ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return _transform(img)


def extract_features(model: nn.Module, img_paths: list) -> np.ndarray:
    """DataLoader + 멀티워커로 CPU 로딩과 GPU 추론을 파이프라인화."""
    dataset = _ImageDataset(img_paths)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        prefetch_factor=2,
        persistent_workers=True,
    )
    feats, n, done = [], len(img_paths), 0
    with torch.no_grad():
        for batch in loader:
            f = model(batch.to(DEVICE, non_blocking=True))
            feats.append(f.cpu().numpy())
            done += len(batch)
            print(f"\r    {done:>5}/{n}", end="", flush=True)
    print()
    return np.vstack(feats)


# ── ReID 평가 ─────────────────────────────────────────────────────
def compute_distmat(qf: np.ndarray, gf: np.ndarray) -> np.ndarray:
    """코사인 거리 행렬 (q × g)."""
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    return 1.0 - qf @ gf.T


def eval_reid(distmat, q_pids, g_pids, q_cids, g_cids, max_rank=10):
    """Market-1501 표준 프로토콜로 mAP 및 CMC 계산."""
    num_q   = distmat.shape[0]
    all_AP  = []
    cmc_acc = np.zeros(max_rank)

    for qi in range(num_q):
        order = np.argsort(distmat[qi])
        gp, gc = g_pids[order], g_cids[order]

        # 같은 카메라 동일 ID(cross-camera 제외) 및 junk(-1) 제거
        keep    = ~((gp == -1) | ((gp == q_pids[qi]) & (gc == q_cids[qi])))
        matches = (gp[keep] == q_pids[qi]).astype(np.float32)
        if matches.sum() == 0:
            continue

        cmc_acc += np.minimum(matches.cumsum(), 1)[:max_rank]

        tp   = matches.cumsum()
        prec = tp / (np.arange(len(matches)) + 1)
        all_AP.append((prec * matches).sum() / matches.sum())

    mAP = np.mean(all_AP) * 100.0
    CMC = cmc_acc / num_q * 100.0
    return mAP, CMC


# ── 모델 빌드 (파인튜닝 체크포인트 로드) ───────────────────────────
def build_entries():
    from research.benchmark.train_market1501 import (
        TrainableOSNet, TrainableBoT, TrainableBoTIBN, TrainableViT, TrainableLightMBN,
        NUM_CLASSES, CKPT_DIR,
    )

    def load(cls, fname, weights_info):
        model = cls(NUM_CLASSES)
        ckpt  = torch.load(CKPT_DIR / fname, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(DEVICE)
        r1  = ckpt.get("rank1", 0.0)
        mAP = ckpt.get("mAP",   0.0)
        print(f"  [{fname}] 로드 완료  Rank-1={r1:.2f}%  mAP={mAP:.2f}%")
        return model, f"{weights_info} (R1={r1:.1f}%)"

    osnet, osnet_info = load(TrainableOSNet,    "osnet_best.pth", "fine-tuned Market-1501")
    bot,   bot_info   = load(TrainableBoT,      "bot_best.pth",   "fine-tuned Market-1501")
    lmbn,  lmbn_info  = load(TrainableLightMBN, "lmbn_best.pth",  "fine-tuned Market-1501")

    entries = [
        ("OSNet-AIN x1.0", osnet, osnet_info),
        ("BoT (ResNet50)",  bot,   bot_info),
        ("LightMBN",        lmbn,  lmbn_info),
    ]

    # bot_ibn_best.pth / vit_best.pth는 학습 후에만 생기므로, 없으면 건너뛰고 나머지만 평가
    bot_ibn_ckpt = CKPT_DIR / "bot_ibn_best.pth"
    if bot_ibn_ckpt.exists():
        bot_ibn, bot_ibn_info = load(TrainableBoTIBN, "bot_ibn_best.pth", "fine-tuned Market-1501")
        entries.append(("BoT-IBN (ResNet50-IBN-a)", bot_ibn, bot_ibn_info))

    vit_ckpt = CKPT_DIR / "vit_best.pth"
    if vit_ckpt.exists():
        vit, vit_info = load(TrainableViT, "vit_best.pth", "fine-tuned Market-1501")
        entries.append(("ViT-Small/16", vit, vit_info))

    return entries


# ── 결과 저장 ─────────────────────────────────────────────────────
def save_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Market-1501 ReID 통합 벤치마크 결과\n",
        f"- **측정일** : {ts}",
        f"- **디바이스** : {DEVICE}",
        f"- **데이터셋** : Market-1501-v15.09.15",
        f"- **입력 크기** : 3 × 256 × 128",
        f"- **갤러리** : {rows[0]['gallery_size']:,}장 / **쿼리** : {rows[0]['query_size']:,}장",
        f"- **레이턴시** : batch=1, warmup={LATENCY_WARMUP}, runs={LATENCY_RUNS}\n",
        "## 1. ReID 정확도 (Market-1501 표준 프로토콜)\n",
        "| 모델 | 가중치 | mAP (%) | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) |",
        "|:-----|:-------|--------:|-----------:|-----------:|------------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['weights']} "
            f"| {r['mAP']:.2f} | {r['rank1']:.2f} | {r['rank5']:.2f} | {r['rank10']:.2f} |"
        )

    lines += [
        "\n## 2. 모델 효율성\n",
        "| 모델 | Params (M) | GFLOPs | Out-Dim | Mean (ms) | Min (ms) | Max (ms) | Std (ms) |",
        "|:-----|----------:|-------:|--------:|----------:|---------:|---------:|---------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['params_m']:.2f} | {r['gflops']:.3f} | {r['out_dim']} "
            f"| {r['mean_ms']:.2f} | {r['min_ms']:.2f} | {r['max_ms']:.2f} | {r['std_ms']:.2f} |"
        )

    lines += [
        "\n---",
        "> 3개 모델 모두 Market-1501 train split(751 ID, 12,936장)으로 파인튜닝된 결과입니다.",
        "> 학습 설정: CE + Triplet Loss / Adam lr=3.5e-4 / 60 epochs / MultiStepLR(30,50)",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[저장] {path}")


# ── 메인 ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Market-1501 ReID 통합 벤치마크")
    print(f"  device : {DEVICE}")
    print("=" * 60)

    q_paths, q_pids, q_cids = load_set(MARKET_ROOT / "query")
    g_paths, g_pids, g_cids = load_set(MARKET_ROOT / "bounding_box_test")
    print(f"\n  쿼리: {len(q_paths):,}장  갤러리: {len(g_paths):,}장\n")

    dummy_t       = torch.zeros(1, 3, 256, 128)
    dummy_t_gpu   = dummy_t.to(DEVICE)

    rows = []
    for name, model, weights_info in build_entries():
        print(f"\n{'─'*60}")
        print(f"  {name}  [{weights_info}]")
        print(f"{'─'*60}")

        # 효율성 측정
        gflops, params_m = measure_flops(model, dummy_t)
        timing = time_inference(
            fn=lambda m=model: m(dummy_t_gpu),
            warmup=LATENCY_WARMUP,
            runs=LATENCY_RUNS,
            device=DEVICE,
        )

        # 출력 차원
        with torch.no_grad():
            out_dim = int(model(dummy_t_gpu).shape[1])

        print(f"  Params: {params_m:.2f}M  GFLOPs: {gflops:.3f}  OutDim: {out_dim}")
        print(f"  Mean: {timing['mean_ms']:.2f}ms  Min: {timing['min_ms']:.2f}ms")

        # 특징 추출
        print("  쿼리 특징 추출 중...")
        qf = extract_features(model, q_paths)
        print("  갤러리 특징 추출 중...")
        gf = extract_features(model, g_paths)

        # 거리 행렬 & 평가
        print("  평가 중...")
        distmat      = compute_distmat(qf, gf)
        mAP, cmc     = eval_reid(distmat, q_pids, g_pids, q_cids, g_cids)

        print(f"  mAP={mAP:.2f}%  R1={cmc[0]:.2f}%  R5={cmc[4]:.2f}%  R10={cmc[9]:.2f}%")

        rows.append({
            "model":        name,
            "weights":      weights_info,
            "mAP":          mAP,
            "rank1":        cmc[0],
            "rank5":        cmc[4],
            "rank10":       cmc[9],
            "params_m":     params_m,
            "gflops":       gflops,
            "out_dim":      out_dim,
            "mean_ms":      timing["mean_ms"],
            "min_ms":       timing["min_ms"],
            "max_ms":       timing["max_ms"],
            "std_ms":       timing["std_ms"],
            "gallery_size": len(g_paths),
            "query_size":   len(q_paths),
        })

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # 저장
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"market1501_benchmark_{ts}.md"
    save_markdown(rows, path)


if __name__ == "__main__":
    main()
