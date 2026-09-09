"""
운영 파이프라인용 PCA+Whitening 아티팩트 생성.

eval_dim_reduction.py 실험에서 사용한 것과 동일한 reference 캐시
(research/cache/dimred_ref_n3000_seed42.npz, CUHK-SYSU Train split 3000개,
query/gallery와 ID 미겹침)로 fit하여 research/results/eval_dim_reduction_TestG100.md에
기록된 128차원 결과(mAP 93.50%)와 동일한 변환을 재현한다.

한 번 생성한 아티팩트는 운영 중 재fit하지 않는다 — 좌표계가 바뀌면 DB에
이미 쌓인 벡터와 비교할 수 없게 된다.

실행:
  cd /home/lookie14/Intelligent-CCTV-ReID/ai-server
  python research/verify/fit_pca.py --dim 128
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from reid_core.dim_reduction import PCAWhiten

CACHE_DIR = Path("research/cache")
CHECKPOINT_DIR = Path("reid_core/checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--ref_cache", type=str, default="dimred_ref_n3000_seed42.npz")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    ref_path = CACHE_DIR / args.ref_cache
    ref_feats = np.load(ref_path)["feats"]
    print(f"[참조 특징] {ref_path} 로드: {ref_feats.shape}")

    pca = PCAWhiten(n_components=args.dim).fit(ref_feats)
    print(
        f"[PCA fit 완료] {ref_feats.shape[1]} -> {args.dim}차원, "
        f"설명된 분산 {pca.explained_variance_ratio_ * 100:.1f}%"
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else CHECKPOINT_DIR / f"pca_{args.dim}.npz"
    pca.save(str(out_path))
    print(f"[저장 완료] {out_path}")


if __name__ == "__main__":
    main()
