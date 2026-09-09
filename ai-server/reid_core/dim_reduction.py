"""
PCA + Whitening 차원 축소 유틸리티

원본 ReID 임베딩을 낮은 차원으로 투영하면서 각 주성분의 분산을 1로
정규화(whitening)한다. sklearn 의존성 없이 numpy SVD로 구현.

주의: fit()은 반드시 평가용 query/gallery와 겹치지 않는 참조(reference)
특징 집합에서만 수행해야 한다. 평가 데이터로 fit하면 PCA가 테스트 분포에
맞춰져 mAP가 실제보다 부풀려진다 (data leakage).
"""

from __future__ import annotations

import numpy as np


class PCAWhiten:
    """PCA + whitening 변환기.

    변환 후 벡터는 다시 L2 정규화한다 — whitening은 주성분별 분산을
    독립적으로 맞추므로 변환 후 벡터 norm이 1이 아니게 되고, 그 상태로
    코사인 유사도를 계산하면 원본과 동일한 랭킹 지표를 비교할 수 없다.
    """

    def __init__(self, n_components: int, eps: float = 1e-6):
        self.n_components = n_components
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None            # (k, D)
        self.explained_variance_: np.ndarray | None = None    # (k,)
        self.explained_variance_ratio_: float | None = None   # scalar

    def fit(self, X: np.ndarray) -> "PCAWhiten":
        """X: (N, D) 참조 특징. N >= n_components 여야 한다."""
        if X.shape[0] < self.n_components:
            raise ValueError(
                f"참조 샘플 수({X.shape[0]})가 목표 차원({self.n_components})보다 "
                f"적습니다. --ref_n을 늘리세요."
            )
        self.mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean_
        # SVD 기반 PCA: Xc = U S Vt → 공분산 고유값 = S^2 / (N-1)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = self.n_components
        self.components_ = Vt[:k]
        var_all = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ = var_all[:k]
        total_var = var_all.sum()
        self.explained_variance_ratio_ = (
            float(var_all[:k].sum() / total_var) if total_var > 0 else 0.0
        )
        return self

    def transform(self, X: np.ndarray, l2_normalize: bool = True) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")
        Xc = X - self.mean_
        proj = Xc @ self.components_.T                          # (N, k)
        whitened = proj / np.sqrt(self.explained_variance_ + self.eps)
        if l2_normalize:
            norm = np.linalg.norm(whitened, axis=1, keepdims=True)
            whitened = whitened / np.clip(norm, 1e-12, None)
        return whitened.astype(np.float32)

    def save(self, path: str) -> None:
        """fit() 결과를 .npz로 저장 — 운영 파이프라인은 이 아티팩트를 로드해서
        재사용해야 하며, 로드 후 절대 재fit하면 안 된다(좌표계가 바뀌면 DB에
        쌓인 기존 벡터와 더 이상 비교할 수 없다)."""
        if self.components_ is None:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")
        np.savez(
            path,
            n_components=self.n_components,
            eps=self.eps,
            mean_=self.mean_,
            components_=self.components_,
            explained_variance_=self.explained_variance_,
            explained_variance_ratio_=self.explained_variance_ratio_,
        )

    @classmethod
    def load(cls, path: str) -> "PCAWhiten":
        data = np.load(path)
        obj = cls(n_components=int(data["n_components"]), eps=float(data["eps"]))
        obj.mean_ = data["mean_"]
        obj.components_ = data["components_"]
        obj.explained_variance_ = data["explained_variance_"]
        obj.explained_variance_ratio_ = float(data["explained_variance_ratio_"])
        return obj
