import torch.nn as nn
from torchreid.reid.models.resnet_ibn_a import resnet50_ibn_a


class BoTIBNReID(nn.Module):
    """
    ResNet50-IBN-a + Bag of Tricks for ReID.
    bot.py의 BoTReID와 동일한 트릭(last_stride 2→1 + BN-Neck)을 유지하고
    backbone만 IBN-Net(Instance-Batch Normalization)으로 교체 — 백본 하나만의 효과를 비교하기 위함.
    ref: "Bag of Tricks..." (Luo et al., 2019) + "Two at Once: IBN-Net" (Pan et al., ECCV 2018)
    """

    def __init__(self) -> None:
        super().__init__()
        base = resnet50_ibn_a(num_classes=1, pretrained=True)

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
