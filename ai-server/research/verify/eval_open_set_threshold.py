"""
Open-set 임계치 검증 — same-ID vs diff-ID 거리 분포 기반 threshold 재산정

배경 (pca-128d-migration.md §C):
  ReidService.java의 DISTANCE_THRESHOLD=0.3은 "유사도 0.7의 산술적 반대값"일
  뿐, 실제 same-ID/diff-ID 거리 분포로 검증된 값이 아니다. PCA+Whitening을
  거치면 코사인 거리 분포 자체가 달라지므로(§eval_dim_reduction_*.md는
  closed-set 랭킹 지표만 검증) 128차원 기준 open-set 임계치 재검증이 필요.

방법:
  eval_dim_reduction.py가 이미 만들어 둔 CUHK-SYSU TestG500 평가 특징 캐시
  (Q, G, L)를 재사용한다 — YOLO/BoT 재추론 불필요. query별 gallery 라벨(L)이
  이미 "같은 사람(1)/다른 사람(0)"으로 표시돼 있으므로 그대로 same-ID/diff-ID
  쌍으로 쓴다. CUHK-SYSU search 프로토콜 자체가 "한 프레임에서 검출된 사람
  벡터로 갤러리(=DB)를 검색"하는 실제 운영 시나리오와 형태가 같아 open-set
  임계치 검증에 적합하다.

  거리 정의는 pgvector의 `<=>` 연산자(vector_cosine_ops)와 동일하게
  코사인 거리(= 1 - 코사인 유사도)를 사용한다.

실행:
  cd /home/lookie14/Intelligent-CCTV-ReID/ai-server
  python research/verify/eval_open_set_threshold.py
  python research/verify/eval_open_set_threshold.py --save research/results/eval_open_set_threshold.md \
                                            --plot research/results/eval_open_set_threshold.png
"""

import argparse
from pathlib import Path
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from reid_core.dim_reduction import PCAWhiten

CACHE_DIR = Path("research/cache")
CURRENT_THRESHOLD = 0.3  # ReidService.java DISTANCE_THRESHOLD (검증 대상)


# ── 거리 계산 ──────────────────────────────────────────────────────────────
def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def cosine_distance_pairs(Q: np.ndarray, G: np.ndarray, L: np.ndarray):
    """Q:(n,D) G:(n,Gsize,D) L:(n,Gsize) -> (same_dists, diff_dists)"""
    sims = np.einsum("nd,ngd->ng", Q, G)
    dists = 1.0 - sims
    return dists[L == 1], dists[L == 0]


# ── FAR/FRR ───────────────────────────────────────────────────────────────
def far_frr(same: np.ndarray, diff: np.ndarray, threshold: float):
    far = float((diff <= threshold).mean())  # 다른 사람인데 매칭 성공(오인식)
    frr = float((same > threshold).mean())   # 같은 사람인데 매칭 실패(신규 등록됨)
    return far, frr


def sweep_thresholds(same: np.ndarray, diff: np.ndarray, lo=0.0, hi=1.6, step=0.005):
    ths = np.arange(lo, hi + step, step)
    fars = np.array([float((diff <= t).mean()) for t in ths])
    frrs = np.array([float((same > t).mean()) for t in ths])
    eer_idx = int(np.argmin(np.abs(fars - frrs)))
    return ths, fars, frrs, float(ths[eer_idx]), float((fars[eer_idx] + frrs[eer_idx]) / 2)


def threshold_at_far(ths, fars, frrs, target_far):
    idx = int(np.argmin(np.abs(fars - target_far)))
    return float(ths[idx]), float(fars[idx]), float(frrs[idx])


def describe(name: str, arr: np.ndarray) -> str:
    return (f"{name}: n={len(arr):>6}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
            f"min={arr.min():.4f}  p25={np.percentile(arr, 25):.4f}  "
            f"median={np.median(arr):.4f}  p75={np.percentile(arr, 75):.4f}  max={arr.max():.4f}")


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_cache", type=str, default="dimred_eval_TestG500_n300.npz")
    parser.add_argument("--pca", type=str, default="reid_core/checkpoints/pca_128.npz")
    parser.add_argument("--save", type=str, default=None, help="결과 저장 경로 (.md)")
    parser.add_argument("--plot", type=str, default=None, help="분포 히스토그램 저장 경로 (.png)")
    args = parser.parse_args()

    cache_path = CACHE_DIR / args.eval_cache
    d = np.load(cache_path)
    Q, G, L = d["Q"], d["G"], d["L"]
    print(f"[캐시 로드] {cache_path}  Q={Q.shape} G={G.shape} L={L.shape}")

    D = Q.shape[1]

    # Origin(원본 차원) — 이미 L2 정규화되어 캐시됨(eval_cuhk_sysu.extract_feature)
    Qo, Go = l2_normalize(Q), l2_normalize(G)
    same_o, diff_o = cosine_distance_pairs(Qo, Go, L)

    # PCA+Whiten(운영과 동일한 128차원 변환)
    pca = PCAWhiten.load(args.pca)
    Qp = pca.transform(Q)
    Gp = pca.transform(G.reshape(-1, D)).reshape(G.shape[0], G.shape[1], pca.n_components)
    same_p, diff_p = cosine_distance_pairs(Qp, Gp, L)

    print(f"\n[Origin {D}차원]")
    print(" ", describe("same-ID", same_o))
    print(" ", describe("diff-ID", diff_o))

    print(f"\n[PCA+Whiten {pca.n_components}차원] (운영 파이프라인과 동일 변환)")
    print(" ", describe("same-ID", same_p))
    print(" ", describe("diff-ID", diff_p))

    ths, fars, frrs, eer_th, eer = sweep_thresholds(same_p, diff_p)
    cur_far, cur_frr = far_frr(same_p, diff_p, CURRENT_THRESHOLD)

    print(f"\n[128차원 기준 open-set 성능 — pgvector `<=>` 코사인 거리]")
    print(f"  현재 THRESHOLD={CURRENT_THRESHOLD:.3f}  ->  FAR={cur_far*100:6.2f}%  FRR={cur_frr*100:6.2f}%")
    print(f"  EER   THRESHOLD={eer_th:.3f}  ->  EER={eer*100:6.2f}% (FAR≈FRR 지점)")

    far_targets = (0.001, 0.01, 0.05, 0.10)
    target_rows = []
    for target in far_targets:
        t, f, r = threshold_at_far(ths, fars, frrs, target)
        target_rows.append((target, t, f, r))
        print(f"  FAR≈{target*100:>5.1f}% 지점 THRESHOLD={t:.3f}  ->  FAR={f*100:6.2f}%  FRR={r*100:6.2f}%")

    if args.plot:
        plot_distributions(same_o, diff_o, same_p, diff_p, CURRENT_THRESHOLD, eer_th, args.plot)

    if args.save:
        save_markdown(args, D, pca.n_components, same_o, diff_o, same_p, diff_p,
                       eer_th, eer, cur_far, cur_frr, target_rows)


# ── 시각화 ───────────────────────────────────────────────────────────────
def plot_distributions(same_o, diff_o, same_p, diff_p, cur_th, eer_th, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 1.6, 80)
    for ax, same, diff, title in [
        (axes[0], same_o, diff_o, "Origin (raw dim)"),
        (axes[1], same_p, diff_p, "PCA+Whiten (128d, production pipeline)"),
    ]:
        ax.hist(diff, bins=bins, alpha=0.55, density=True, label=f"diff-ID (n={len(diff)})", color="#d9534f")
        ax.hist(same, bins=bins, alpha=0.55, density=True, label=f"same-ID (n={len(same)})", color="#428bca")
        ax.set_title(title)
        ax.set_xlabel("cosine distance (1 - cosine similarity)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    axes[1].axvline(cur_th, color="black", linestyle="--", linewidth=1.2)
    axes[1].axvline(eer_th, color="green", linestyle="--", linewidth=1.2)
    axes[1].text(cur_th, axes[1].get_ylim()[1] * 0.9, f" current={cur_th}", fontsize=8)
    axes[1].text(eer_th, axes[1].get_ylim()[1] * 0.8, f" EER={eer_th:.3f}", fontsize=8, color="green")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"[저장] {out_path}")


# ── 리포트 저장 ──────────────────────────────────────────────────────────
def save_markdown(args, D, k, same_o, diff_o, same_p, diff_p,
                   eer_th, eer, cur_far, cur_frr, target_rows):
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Open-set 임계치 검증 — same-ID vs diff-ID 거리 분포",
        "",
        f"- **배경** : [pca-128d-migration.md](../../pca-128d-migration.md) §C — "
        "`ReidService.java`의 `DISTANCE_THRESHOLD=0.3`은 실험적으로 검증된 값이 "
        "아니었음. 128차원 PCA+Whitening 적용 후 실제 same-ID/diff-ID 거리 분포로 재검증.",
        f"- **데이터** : CUHK-SYSU TestG500 평가 특징 캐시(`{args.eval_cache}`), "
        "query별 gallery 라벨(같은 사람=1/다른 사람=0)을 그대로 same-ID/diff-ID 쌍으로 사용",
        f"- **거리 정의** : 코사인 거리(1 - 코사인 유사도) — pgvector `<=>`(vector_cosine_ops)와 동일",
        f"- **PCA 아티팩트** : `{args.pca}` ({D} → {k}차원, 운영 파이프라인과 동일 변환)",
        f"- **측정일** : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 1. 거리 분포 통계",
        "",
        "| 구분 | 차원 | n | mean | std | median | p25 | p75 |",
        "|:-----|-----:|---:|-----:|----:|-------:|----:|----:|",
        f"| same-ID (Origin) | {D} | {len(same_o)} | {same_o.mean():.4f} | {same_o.std():.4f} | "
        f"{np.median(same_o):.4f} | {np.percentile(same_o,25):.4f} | {np.percentile(same_o,75):.4f} |",
        f"| diff-ID (Origin) | {D} | {len(diff_o)} | {diff_o.mean():.4f} | {diff_o.std():.4f} | "
        f"{np.median(diff_o):.4f} | {np.percentile(diff_o,25):.4f} | {np.percentile(diff_o,75):.4f} |",
        f"| same-ID (PCA+Whiten) | {k} | {len(same_p)} | {same_p.mean():.4f} | {same_p.std():.4f} | "
        f"{np.median(same_p):.4f} | {np.percentile(same_p,25):.4f} | {np.percentile(same_p,75):.4f} |",
        f"| diff-ID (PCA+Whiten) | {k} | {len(diff_p)} | {diff_p.mean():.4f} | {diff_p.std():.4f} | "
        f"{np.median(diff_p):.4f} | {np.percentile(diff_p,25):.4f} | {np.percentile(diff_p,75):.4f} |",
        "",
        "---",
        "",
        "## 2. 임계치별 FAR/FRR (128차원, 운영 파이프라인 기준)",
        "",
        "- **FAR**(False Accept Rate) : 다른 사람인데 같은 사람으로 오인식(임계치 이하로 매칭됨) 비율",
        "- **FRR**(False Reject Rate) : 같은 사람인데 다른 사람으로 오인식(새 ID로 등록됨) 비율",
        "",
        "| 기준 | THRESHOLD | FAR | FRR |",
        "|:-----|----------:|----:|----:|",
        f"| **현재 운영값** | {CURRENT_THRESHOLD:.3f} | {cur_far*100:.2f}% | {cur_frr*100:.2f}% |",
        f"| **EER**(FAR≈FRR) | {eer_th:.3f} | — | EER={eer*100:.2f}% |",
    ]
    for target, t, f, r in target_rows:
        lines.append(f"| FAR≈{target*100:.1f}% 목표 | {t:.3f} | {f*100:.2f}% | {r*100:.2f}% |")

    lines += [
        "",
        "---",
        "",
        "## 3. 분석",
        "",
        f"- 현재 운영 중인 `DISTANCE_THRESHOLD={CURRENT_THRESHOLD}`은 EER 지점(`{eer_th:.3f}`) 대비 "
        f"{'낮아(보수적) FRR이 상대적으로 큼' if CURRENT_THRESHOLD < eer_th else '높아 FAR이 상대적으로 큼'} — "
        "위 표의 FAR/FRR 수치로 실제 운영에 적합한지 판단 필요.",
        "- CCTV 추적 서비스는 보통 '다른 사람을 같은 사람으로 잘못 합치는' FAR 쪽 리스크가 "
        "'같은 사람을 새 ID로 중복 등록하는' FRR보다 치명적인 경우가 많음(추적 이력 오염) — "
        "이 경우 EER보다 낮은 FAR(예: FAR≈1% 지점)을 목표로 임계치를 더 보수적으로 잡는 것을 권장.",
        "- 이 실험은 CUHK-SYSU 데이터 기준이라 실제 CCTV 도메인(카메라 각도/조명/해상도)과 "
        "차이가 있을 수 있음 — 최종 배포 전 실제 도메인 데이터로 재검증 권장.",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {out.resolve()}")


if __name__ == "__main__":
    main()
