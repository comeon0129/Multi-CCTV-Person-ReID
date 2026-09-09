"""
Market-1501 ReID 파인튜닝 (3개 모델)

손실   : CrossEntropy (ID 분류) + Triplet Loss (batch-hard)
배치   : P=16 identities × K=4 images = 64
스케줄 : linear warmup(10 epoch) → MultiStepLR([30,50], γ=0.1)
저장   : research/benchmark/checkpoints/<model>_best.pth  (Rank-1 기준)

사용:
    cd /home/lookie14/Intelligent-CCTV-ReID/ai-server
    python -m research.benchmark.train_market1501
    python -m research.benchmark.train_market1501 --model osnet --epochs 30
"""

import argparse
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchreid
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import ResNet50_Weights, resnet50
from torchreid.reid.models.resnet_ibn_a import resnet50_ibn_a

# ── 경로 설정 ──────────────────────────────────────────────────────
MARKET_ROOT  = Path("/home/lookie14/Intelligent-CCTV-ReID/ai-server/data/Market-1501-v15.09.15")
CKPT_DIR     = Path("/home/lookie14/Intelligent-CCTV-ReID/ai-server/research/benchmark/checkpoints")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── 하이퍼파라미터 ─────────────────────────────────────────────────
P            = 16       # 배치당 identity 수
K            = 4        # identity당 이미지 수
BATCH_SIZE   = P * K    # 64
NUM_WORKERS  = 4
LR           = 3.5e-4
WEIGHT_DECAY = 5e-4
TRIPLET_MARGIN = 0.3
WARMUP_EPOCHS  = 10
NUM_CLASSES    = 751    # Market-1501 train identity 수


# ── 데이터셋 ───────────────────────────────────────────────────────
_train_tf = T.Compose([
    T.Resize((288, 144)),
    T.RandomCrop((256, 128)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.5, scale=(0.02, 0.4), value=0),
])

_test_tf = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _parse_id(fname):
    m = re.match(r"(-?\d+)_c(\d+)", fname)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


class Market1501Train(Dataset):
    def __init__(self):
        folder = MARKET_ROOT / "bounding_box_train"
        self.samples = []
        pid_set = sorted({
            _parse_id(f.name)[0]
            for f in folder.iterdir()
            if f.suffix == ".jpg" and _parse_id(f.name)[0] != -1
        })
        self.pid2label = {pid: i for i, pid in enumerate(pid_set)}
        for f in sorted(folder.iterdir()):
            if f.suffix != ".jpg":
                continue
            pid, _ = _parse_id(f.name)
            if pid == -1:
                continue
            self.samples.append((str(f), self.pid2label[pid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return _train_tf(img), label


class Market1501Test(Dataset):
    def __init__(self, folder):
        self.samples = []
        self.pids, self.cids = [], []
        for f in sorted(Path(folder).iterdir()):
            if f.suffix != ".jpg":
                continue
            pid, cid = _parse_id(f.name)
            self.samples.append(str(f))
            self.pids.append(pid)
            self.cids.append(cid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return _test_tf(Image.open(self.samples[idx]).convert("RGB"))


class RandomIdentitySampler(Sampler):
    """배치마다 P개 identity에서 각 K장을 샘플링 (PK sampler)."""

    def __init__(self, dataset: Market1501Train, p: int, k: int):
        self.p, self.k = p, k
        id2idx = defaultdict(list)
        for i, (_, label) in enumerate(dataset.samples):
            id2idx[label].append(i)
        self.id2idx = {lb: idxs for lb, idxs in id2idx.items() if len(idxs) >= k}
        self.labels = list(self.id2idx.keys())
        self.n_batches = len(self.labels) // p

    def __len__(self):
        return self.n_batches * self.p * self.k

    def __iter__(self):
        np.random.shuffle(self.labels)
        for i in range(0, self.n_batches * self.p, self.p):
            batch_labels = self.labels[i:i + self.p]
            indices = []
            for lb in batch_labels:
                pool = self.id2idx[lb]
                chosen = np.random.choice(pool, self.k, replace=len(pool) < self.k)
                indices.extend(chosen.tolist())
            yield from indices


# ── 손실 함수 ──────────────────────────────────────────────────────
class TripletLoss(nn.Module):
    """Batch-hard triplet loss."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n = feats.size(0)
        feats = F.normalize(feats, dim=1)
        dist = torch.cdist(feats, feats, p=2)

        mask_pos = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        mask_neg = ~mask_pos
        mask_pos.fill_diagonal_(False)

        hardest_pos = (dist * mask_pos.float()).max(dim=1).values
        hardest_neg = (dist + 1e6 * mask_pos.float()).min(dim=1).values

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()


# ── 학습 가능 모델 ──────────────────────────────────────────────────
class TrainableOSNet(nn.Module):
    """OSNet-AIN x1.0: fc hook으로 512-dim 피처를 캡처 → (feat, logits) 반환."""

    def __init__(self, num_classes):
        super().__init__()
        base = torchreid.models.build_model(
            name="osnet_ain_x1_0",
            num_classes=num_classes,
            pretrained=True,      # loss 기본값 'softmax': train→logits, eval→feat
        )
        self._base = base
        self._feat  = None
        # fc 레이어 출력(512-dim)을 캡처해 triplet loss에 활용
        base.fc.register_forward_hook(self._capture_feat)

    def _capture_feat(self, module, inp, out):
        self._feat = out

    def forward(self, x):
        if self.training:
            logits = self._base(x)       # softmax 모드 → 클래스 logits
            return self._feat, logits    # (512-dim feat, num_classes logits)
        return self._base(x)             # eval 모드 → 512-dim feat


class TrainableBoT(nn.Module):
    """ResNet50(ImageNet) + Bag-of-Tricks + BN-Neck + classifier."""

    def __init__(self, num_classes):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        base.layer4[0].downsample[0].stride = (1, 1)
        base.layer4[0].conv2.stride = (1, 1)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.BatchNorm1d(2048)
        self.neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        f = self.gap(self.backbone(x)).flatten(1)
        feat = self.neck(f)
        if not self.training:
            return feat
        return feat, self.classifier(feat)


class TrainableBoTIBN(nn.Module):
    """ResNet50-IBN-a(ImageNet) + Bag-of-Tricks + BN-Neck + classifier.

    TrainableBoT와 동일한 구조/트릭이며 backbone만 IBN-Net으로 교체 —
    백본 교체 하나만의 효과를 비교하기 위한 대조군.
    """

    def __init__(self, num_classes):
        super().__init__()
        base = resnet50_ibn_a(num_classes=1, pretrained=True)
        base.layer4[0].downsample[0].stride = (1, 1)
        base.layer4[0].conv2.stride = (1, 1)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.BatchNorm1d(2048)
        self.neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        f = self.gap(self.backbone(x)).flatten(1)
        feat = self.neck(f)
        if not self.training:
            return feat
        return feat, self.classifier(feat)


class TrainableViT(nn.Module):
    """ViT-Small/16(ImageNet) + BN-Neck + classifier.

    TrainableBoT/TrainableBoTIBN과 동일한 head 구조(BN-Neck, classifier, forward 시그니처)를
    유지하고 backbone만 CNN(ResNet50)에서 트랜스포머(ViT)로 교체 — 백본 계열 교체 하나만의
    효과를 비교하기 위한 대조군. 224×224용 위치 임베딩을 256×128 입력에 맞게 보간(timm 내부 처리).
    """

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224", pretrained=True, num_classes=0, img_size=(256, 128),
        )
        feat_dim = self.backbone.num_features  # 384
        self.neck = nn.BatchNorm1d(feat_dim)
        self.neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, x):
        f = self.backbone(x)
        feat = self.neck(f)
        if not self.training:
            return feat
        return feat, self.classifier(feat)


class TrainableLightMBN(nn.Module):
    """ResNet50(ImageNet) + global/local 멀티브랜치 + classifier."""

    NUM_STRIPES = 6
    GLOBAL_DIM  = 2048
    LOCAL_DIM   = 256
    FEAT_DIM    = GLOBAL_DIM + NUM_STRIPES * LOCAL_DIM  # 3584

    def __init__(self, num_classes):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        base.layer4[0].downsample[0].stride = (1, 1)
        base.layer4[0].conv2.stride = (1, 1)
        self.backbone   = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.global_gap = nn.AdaptiveAvgPool2d(1)
        self.global_bn  = nn.BatchNorm1d(self.GLOBAL_DIM)
        self.global_bn.bias.requires_grad_(False)
        self.local_fcs  = nn.ModuleList(
            [nn.Linear(self.GLOBAL_DIM, self.LOCAL_DIM, bias=False) for _ in range(self.NUM_STRIPES)]
        )
        self.local_bns  = nn.ModuleList(
            [nn.BatchNorm1d(self.LOCAL_DIM) for _ in range(self.NUM_STRIPES)]
        )
        for bn in self.local_bns:
            bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.FEAT_DIM, num_classes, bias=False)

    def forward(self, x):
        feat_map = self.backbone(x)
        sh = max(1, feat_map.size(2) // self.NUM_STRIPES)
        g  = self.global_bn(self.global_gap(feat_map).flatten(1))
        ls = [
            bn(fc(feat_map[:, :, i * sh:(i + 1) * sh, :].mean(dim=[2, 3])))
            for i, (fc, bn) in enumerate(zip(self.local_fcs, self.local_bns))
        ]
        feat = torch.cat([g, *ls], dim=1)
        if not self.training:
            return feat
        return feat, self.classifier(feat)


# ── 평가 ──────────────────────────────────────────────────────────
def extract_test_features(model, folder) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = Market1501Test(folder)
    loader  = DataLoader(dataset, batch_size=128, num_workers=4, pin_memory=True)
    model.eval()
    feats = []
    with torch.no_grad():
        for batch in loader:
            feats.append(model(batch.to(DEVICE)).cpu().numpy())
    model.train()
    return np.vstack(feats), np.array(dataset.pids), np.array(dataset.cids)


def evaluate(model) -> tuple[float, float]:
    qf, q_pids, q_cids = extract_test_features(model, MARKET_ROOT / "query")
    gf, g_pids, g_cids = extract_test_features(model, MARKET_ROOT / "bounding_box_test")

    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
    distmat = 1.0 - qf @ gf.T

    all_AP, cmc_acc = [], np.zeros(10)
    for qi in range(distmat.shape[0]):
        order    = np.argsort(distmat[qi])
        gp, gc   = g_pids[order], g_cids[order]
        keep     = ~((gp == -1) | ((gp == q_pids[qi]) & (gc == q_cids[qi])))
        matches  = (gp[keep] == q_pids[qi]).astype(np.float32)
        if matches.sum() == 0:
            continue
        cmc_acc += np.minimum(matches.cumsum(), 1)[:10]
        tp   = matches.cumsum()
        prec = tp / (np.arange(len(matches)) + 1)
        all_AP.append((prec * matches).sum() / matches.sum())

    mAP   = np.mean(all_AP) * 100.0
    rank1 = cmc_acc[0] / distmat.shape[0] * 100.0
    return mAP, rank1


# ── 학습 루프 ──────────────────────────────────────────────────────
def train(model_name: str, epochs: int, eval_interval: int):
    print(f"\n{'='*60}")
    print(f"  [{model_name}] 파인튜닝 시작  ({epochs} epochs, device={DEVICE})")
    print(f"{'='*60}")

    # 모델
    model_map = {
        "osnet":   TrainableOSNet,
        "bot":     TrainableBoT,
        "bot_ibn": TrainableBoTIBN,
        "vit":     TrainableViT,
        "lmbn":    TrainableLightMBN,
    }
    model = model_map[model_name](NUM_CLASSES).to(DEVICE)

    # 데이터
    train_ds = Market1501Train()
    sampler  = RandomIdentitySampler(train_ds, P, K)
    loader   = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
    )

    # 최적화
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    ce_loss   = nn.CrossEntropyLoss(label_smoothing=0.1)
    tri_loss  = TripletLoss(TRIPLET_MARGIN)

    best_rank1 = 0.0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path  = CKPT_DIR / f"{model_name}_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()

        # Linear warmup
        if epoch <= WARMUP_EPOCHS:
            lr_scale = epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale

        total_ce = total_tri = total_n = 0
        t0 = time.time()

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            feat, logits = model(imgs)

            loss = ce_loss(logits, labels) + tri_loss(feat, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_ce  += ce_loss(logits, labels).item() * len(imgs)
            total_tri += tri_loss(feat, labels).item()  * len(imgs)
            total_n   += len(imgs)

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:>3}/{epochs}  "
            f"CE={total_ce/total_n:.4f}  Tri={total_tri/total_n:.4f}  "
            f"lr={cur_lr:.2e}  [{elapsed:.0f}s]",
            end=""
        )

        # 평가
        if epoch % eval_interval == 0 or epoch == epochs:
            mAP, rank1 = evaluate(model)
            mark = " ★" if rank1 > best_rank1 else ""
            print(f"  →  mAP={mAP:.2f}%  R1={rank1:.2f}%{mark}")
            if rank1 > best_rank1:
                best_rank1 = rank1
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "rank1": rank1, "mAP": mAP}, ckpt_path)
        else:
            print()

    print(f"\n  [{model_name}] 완료  Best Rank-1={best_rank1:.2f}%  →  {ckpt_path}")
    return best_rank1


# ── 메인 ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="all", choices=["all", "osnet", "bot", "bot_ibn", "vit", "lmbn"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-interval", type=int, default=10)
    args = parser.parse_args()

    targets = ["osnet", "bot", "lmbn"] if args.model == "all" else [args.model]

    results = {}
    for name in targets:
        results[name] = train(name, args.epochs, args.eval_interval)

    print("\n" + "="*60)
    print("  파인튜닝 최종 결과 (Best Rank-1)")
    for name, r1 in results.items():
        print(f"    {name:<8}: {r1:.2f}%  ({CKPT_DIR / f'{name}_best.pth'})")
    print("="*60)


if __name__ == "__main__":
    main()
