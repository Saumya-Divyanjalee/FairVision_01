"""
FairVision CNN Model Architecture
==================================
Custom CNN designed from scratch using PyTorch.
No pretrained models or transfer learning used.

Architecture:
- 5 ConvBlocks: Conv2d → BatchNorm2d → ReLU → MaxPool2d
- Filters: 3 → 32 → 64 → 128 → 256 → 256
- Spatial: 224 → 112 → 56 → 28 → 14 → 7
- Classifier: Flatten → Dropout(0.5) → FC(1024) → ReLU → Dropout(0.3) → FC(512) → ReLU → FC(9)

Justification:
- Increasing filter depth captures low→mid→high-level face features
- BatchNorm stabilizes training across demographic groups
- Dropout prevents overfitting on majority demographic classes
- Moderate depth balances capacity vs training time on FairFace scale
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Single convolution block: Conv → BatchNorm → ReLU → (optional MaxPool)"""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FairVisionCNN(nn.Module):
    """
    Custom CNN for Age Group Classification on FairFace dataset.
    Input : (B, 3, 224, 224)
    Output: (B, num_classes)  — default num_classes=9
    """

    def __init__(self, num_classes: int = 9, dropout1: float = 0.5, dropout2: float = 0.3):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # → (B, 32,  112, 112)
            ConvBlock(32,  64,  pool=True),   # → (B, 64,  56,  56)
            ConvBlock(64,  128, pool=True),   # → (B, 128, 28,  28)
            ConvBlock(128, 256, pool=True),   # → (B, 256, 14,  14)
            ConvBlock(256, 256, pool=True),   # → (B, 256, 7,   7)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout1),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FairVisionCNN(num_classes=9)
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"Output shape   : {out.shape}")          # (2, 9)
    print(f"Total params   : {count_parameters(model):,}")
    print(model)
