"""
PCA+Whitening 차원 축소 — Identifiability(mAP/CMC) 비교 테스트

원본 BoT 임베딩(2048-dim)과, PCA+Whitening으로 축소한 임베딩의
CUHK-SYSU 식별 성능(mAP, Rank-1/5/10)을 동일한 query/gallery로 비교한다.

절차:
  1. 참조(reference) 특징 추출 — CUHK-SYSU **Train** split에서 crop 샘플링.
     query/gallery(TestG*)와 인물 ID가 겹치지 않는 split이므로
     PCA가 평가 데이터를 보지 않는다 (data leakage 방지).
  2. 평가용 query/gallery 특징 추출 — eval_cuhk_sysu.py와 완전히 동일한
     crop/전처리 protocol (같은 함수 재사용).
  3. 두 특징 집합 모두 .npz로 캐싱 → 이후 --dims 값만 바꿔 재실행해도
     YOLO/BoT 재추론 없이 즉시 비교 가능.
  4. 각 목표 차원 k 에 대해 참조 특징에서 PCA+Whitening을 fit, query/gallery에
     적용 → Origin(원본) 대비 mAP/CMC/유사도 계산 시간 비교.

기본 설정(--gallery TestG500 --n 300)은 research/results/legacy/eval_g500.md와
동일하다 — Origin 행의 결과가 그 문서의 mAP 81.62%와 일치하면 정상 동작 확인.

실행:
  cd /home/lookie14/Intelligent-CCTV-ReID/ai-server
  python research/verify/eval_dim_reduction.py --n 30 --gallery TestG50 --ref_n 300   # 빠른 검증용
  python research/verify/eval_dim_reduction.py                                        # 기본(TestG500, 300쿼리)
  python research/verify/eval_dim_reduction.py --dims 512,256,128,64,32
  python research/verify/eval_dim_reduction.py --refresh                              # 캐시 무시, 특징 재추출
  python research/verify/eval_dim_reduction.py --save research/results/eval_dim_reduction.md
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from reid_core.dim_reduction import PCAWhiten
from reid_core.models.bot import BoTReID
from eval_cuhk_sysu import (
    ANNOT_DIR,
    IMAGE_DIR,
    crop_from_gt,
    crop_top_detection,
    extract_batch,
    extract_feature,
)

TRAIN_MAT = ANNOT_DIR / "test" / "train_test" / "Train.mat"
CACHE_DIR = Path("research/cache")


# ── 참조(reference) 특징 — PCA fit 전용, query/gallery와 ID 겹치지 않음 ──────
def load_train_crops() -> list[tuple[str, tuple[int, int, int, int]]]:
    mat = sio.loadmat(str(TRAIN_MAT), simplify_cells=True)
    train = mat["Train"]
    samples = []
    for entry in train:
        for scene in entry["scene"]:
            img_path = IMAGE_DIR / scene["imname"]
            if not img_path.exists():
                continue
            x, y, w, h = map(int, scene["idlocate"])
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = x1 + w, y1 + h
            if x2 <= x1 or y2 <= y1:
                continue
            samples.append((scene["imname"], (x1, y1, x2, y2)))
    return samples


def extract_reference_features(reid, device: str, n_samples: int, seed: int) -> np.ndarray:
    samples = load_train_crops()
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(samples))
    idx = rng.choice(len(samples), size=n, replace=False)

    crops = []
    for i in idx:
        imname, (x1, y1, x2, y2) = samples[i]
        img = Image.open(IMAGE_DIR / imname).convert("RGB")
        crops.append(img.crop((x1, y1, x2, y2)))

    print(f"[참조 특징] Train split {len(samples)}개 중 {n}개 crop 추출 중...")
    return extract_batch(crops, reid, device)


# ── 평가용 query/gallery 특징 — eval_cuhk_sysu.py와 동일 protocol ───────────
def extract_eval_features(yolo, reid, device: str, gallery_name: str, n: int):
    mat_path = ANNOT_DIR / "test" / "train_test" / f"{gallery_name}.mat"
    mat = sio.loadmat(str(mat_path), simplify_cells=True)
    test_data = mat[gallery_name]
    n = min(n, len(test_data))

    query_feats, gallery_feat_list, gallery_label_list = [], [], []
    skip_count = 0
    t0 = time.perf_counter()

    for i, entry in enumerate(test_data[:n]):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1:>4}/{n}] {time.perf_counter()-t0:.0f}s 경과...")

        q = entry["Query"]
        q_crop = crop_from_gt(q["imname"], q["idlocate"])
        if q_crop is None:
            skip_count += 1
            continue
        q_feat = extract_feature(q_crop, reid, device)

        g_crops, g_labels = [], []
        for g in entry["Gallery"]:
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

        g_feats = extract_batch(g_crops, reid, device)
        query_feats.append(q_feat)
        gallery_feat_list.append(g_feats)
        gallery_label_list.append(np.array(g_labels))

    elapsed = time.perf_counter() - t0
    print(f"[평가 특징] {len(query_feats)}개 쿼리 추출 완료 ({elapsed:.1f}s, 스킵 {skip_count})")

    Q = np.vstack(query_feats)                  # (Q, D)
    G = np.stack(gallery_feat_list)              # (Q, Gsize, D)
    L = np.stack(gallery_label_list)             # (Q, Gsize)
    return Q, G, L


# ── CMC / mAP — query별 gallery가 다른 CUHK-SYSU protocol (eval_cuhk_sysu.py main()과 동일 로직) ──
def compute_cmc_map(Q_feats: np.ndarray, G_feats: np.ndarray, G_labels: np.ndarray):
    sims = np.einsum("qd,qgd->qg", Q_feats, G_feats)
    ranks = np.argsort(-sims, axis=1)

    n_query = Q_feats.shape[0]
    cmc_acc = np.zeros(10)
    all_ap = []
    for qi in range(n_query):
        ordered = G_labels[qi][ranks[qi]]
        pos_mask = ordered == 1
        if pos_mask.sum() == 0:
            continue
        found = np.minimum(pos_mask.cumsum(), 1)
        cmc_acc += found[:10]
        tp = pos_mask.cumsum()
        prec = tp / (np.arange(len(pos_mask)) + 1)
        all_ap.append((prec * pos_mask).sum() / pos_mask.sum())

    cmc_acc = cmc_acc / n_query * 100
    mAP = np.mean(all_ap) * 100
    return cmc_acc, mAP


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=300,       help="평가 쿼리 수 (기본 300, eval_g500.md와 동일)")
    parser.add_argument("--gallery", type=str, default="TestG500", help="TestG50/.../TestG1000")
    parser.add_argument("--ref_n",   type=int, default=3000,      help="PCA 참조(reference) 샘플 수")
    parser.add_argument("--dims",    type=str, default="512,256,128,64", help="비교할 PCA 목표 차원, 콤마구분")
    parser.add_argument("--weights", type=str, default="reid_core/checkpoints/bot_best.pth")
    parser.add_argument("--yolo",    type=str, default="yolov8n.pt")
    parser.add_argument("--device",  type=str, default="auto")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--refresh", action="store_true", help="캐시 무시하고 특징 재추출")
    parser.add_argument("--save",    type=str, default=None,      help="결과 저장 경로 (.md)")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[설정] device={device}  gallery={args.gallery}  n={args.n}  ref_n={args.ref_n}  dims={args.dims}")

    yolo = YOLO(args.yolo)
    reid = BoTReID().to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    reid.load_state_dict(ckpt["state_dict"], strict=False)
    reid.eval()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 참조 특징 (캐시) ──
    ref_cache = CACHE_DIR / f"dimred_ref_n{args.ref_n}_seed{args.seed}.npz"
    if ref_cache.exists() and not args.refresh:
        ref_feats = np.load(ref_cache)["feats"]
        print(f"[캐시] 참조 특징 로드: {ref_cache}  {ref_feats.shape}")
    else:
        ref_feats = extract_reference_features(reid, device, args.ref_n, args.seed)
        np.savez_compressed(ref_cache, feats=ref_feats)
        print(f"[캐시] 참조 특징 저장: {ref_cache}  {ref_feats.shape}")

    # ── 평가 특징 (캐시) ──
    eval_cache = CACHE_DIR / f"dimred_eval_{args.gallery}_n{args.n}.npz"
    if eval_cache.exists() and not args.refresh:
        d = np.load(eval_cache)
        Q, G, L = d["Q"], d["G"], d["L"]
        print(f"[캐시] 평가 특징 로드: {eval_cache}  Q={Q.shape} G={G.shape}")
    else:
        Q, G, L = extract_eval_features(yolo, reid, device, args.gallery, args.n)
        np.savez_compressed(eval_cache, Q=Q, G=G, L=L)
        print(f"[캐시] 평가 특징 저장: {eval_cache}")

    D = Q.shape[1]
    dims = sorted({int(d) for d in args.dims.split(",")}, reverse=True)
    print(f"\n[원본 차원] {D}\n")

    results = []

    # Origin — 축소 없음
    t0 = time.perf_counter()
    cmc, mAP = compute_cmc_map(Q, G, L)
    sim_ms = (time.perf_counter() - t0) * 1000
    results.append(dict(label="Origin", dim=D, mAP=mAP, rank1=cmc[0], rank5=cmc[4],
                         rank10=cmc[9], evr=1.0, sim_ms=sim_ms))
    print(f"  [Origin      dim={D:>4}] mAP={mAP:6.2f}%  Rank-1={cmc[0]:6.2f}%  "
          f"Rank-5={cmc[4]:6.2f}%  Rank-10={cmc[9]:6.2f}%  sim={sim_ms:6.1f}ms")

    # PCA + Whitening 스윕
    for k in dims:
        if k >= D:
            print(f"  [건너뜀] dim={k} >= 원본 차원({D})")
            continue

        pca = PCAWhiten(n_components=k).fit(ref_feats)
        Qk = pca.transform(Q)
        Gk = pca.transform(G.reshape(-1, D)).reshape(G.shape[0], G.shape[1], k)

        t0 = time.perf_counter()
        cmc, mAP = compute_cmc_map(Qk, Gk, L)
        sim_ms = (time.perf_counter() - t0) * 1000

        results.append(dict(label="PCA+Whiten", dim=k, mAP=mAP, rank1=cmc[0], rank5=cmc[4],
                             rank10=cmc[9], evr=pca.explained_variance_ratio_, sim_ms=sim_ms))
        drop = results[0]["mAP"] - mAP
        print(f"  [PCA+Whiten  dim={k:>4}] mAP={mAP:6.2f}%  Rank-1={cmc[0]:6.2f}%  "
              f"Rank-5={cmc[4]:6.2f}%  Rank-10={cmc[9]:6.2f}%  "
              f"설명분산={pca.explained_variance_ratio_*100:5.1f}%  sim={sim_ms:6.1f}ms  "
              f"(mAP {-drop:+.2f}%p)")

    print()
    if args.save:
        save_markdown(args, device, D, results)


def save_markdown(args, device: str, origin_dim: int, results: list[dict]):
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)

    origin = results[0]
    lines = [
        "# PCA+Whitening 차원 축소 — Identifiability 비교",
        "",
        f"- **데이터셋** : CUHK-SYSU (`{args.gallery}`, {args.n}쿼리)",
        f"- **원본 차원** : {origin_dim} (BoT ResNet50 BN-Neck 출력)",
        f"- **PCA 참조 샘플** : Train split {args.ref_n}개 (query/gallery와 ID 미겹침)",
        f"- **디바이스** : {device.upper()}",
        f"- **측정일** : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 1. 방법",
        "",
        "- **Origin**: BoT 원본 임베딩, 축소 없음",
        "- **PCA+Whiten**: CUHK-SYSU Train split 특징으로 PCA(whiten=True) fit → "
        "query/gallery에 적용 후 L2 재정규화",
        "- query/gallery는 두 조건 모두 **동일** (같은 crop, 같은 원본 특징에서 사영) — "
        "차이는 오직 차원 축소 유무",
        "- 랭킹: 코사인 유사도 내림차순, 지표: CMC(Rank-1/5/10), mAP",
        "",
        "---",
        "",
        "## 2. 결과",
        "",
        "| 방법 | 차원 | mAP | Rank-1 | Rank-5 | Rank-10 | 설명된 분산 | 유사도 계산(ms) | mAP 변화 |",
        "|:-----|-----:|----:|-------:|-------:|--------:|-----------:|----------------:|---------:|",
    ]
    for r in results:
        drop = origin["mAP"] - r["mAP"]
        drop_str = "—" if r["label"] == "Origin" else f"{-drop:+.2f}%p"
        lines.append(
            f"| {r['label']} | {r['dim']} | {r['mAP']:.2f}% | {r['rank1']:.2f}% | "
            f"{r['rank5']:.2f}% | {r['rank10']:.2f}% | {r['evr']*100:.1f}% | "
            f"{r['sim_ms']:.1f} | {drop_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. 분석",
        "",
        "- **mAP 변화**가 음수(-)면 원본 대비 식별 성능 손실, 무시할 수준(예: 1~2%p 이내)이면 "
        "해당 차원까지 축소해도 identifiability를 유지한다고 볼 수 있음.",
        "- **설명된 분산**이 낮은데 mAP 손실이 작다면, 나머지 분산은 신원 구분과 무관한 "
        "잡음(조명/포즈/배경)일 가능성이 높음 — 오히려 PCA가 노이즈를 제거해 mAP가 "
        "Origin보다 **상승**할 수도 있음.",
        "- **유사도 계산(ms)**은 이번 쿼리 수 기준 참고값. 실제 이득은 갤러리 규모가 커질수록 "
        "(예: 전체 DB 대상 검색) 차원에 비례해 커짐.",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {out.resolve()}")


if __name__ == "__main__":
    main()
