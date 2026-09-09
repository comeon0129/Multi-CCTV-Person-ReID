import torch.nn as nn
from torchvision.models import resnet50


class BoTReID(nn.Module):
    """
    ResNet50 + Bag of Tricks for ReID.
    핵심 변경: last_stride 2→1 (특징 맵 해상도 유지) + BN-Neck.
    ref: "Bag of Tricks and a Strong Baseline for Deep Person Re-Identification" (Luo et al., 2019)
    """

    def __init__(self) -> None:
        super().__init__()
        base = resnet50(weights=None)

        # Bag of Tricks: layer4의 첫 블록 stride를 2→1로 변경
        base.layer4[0].downsample[0].stride = (1, 1)
        base.layer4[0].conv2.stride = (1, 1)

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # BN-Neck: classifier 없이 특징 벡터 정규화만 수행
        self.neck = nn.BatchNorm1d(2048)
        self.neck.bias.requires_grad_(False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x).flatten(1)   # (B, 2048)
        return self.neck(x)           # (B, 2048)
