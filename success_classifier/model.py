"""ResNet18 binary success/fail classifier (aligned with RLinf ResNetRewardModel)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetSuccessClassifier(nn.Module):
    """ImageNet-pretrained ResNet18 with a single logit for BCEWithLogitsLoss."""

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if arch != "resnet18":
            raise ValueError(f"only resnet18 supported in this trainer, got {arch}")

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.image_size = image_size
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer(
            "_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Accept uint8 [0,255] or float [0,1], NCHW or NHWC → NCHW normalized."""
        if images.dim() != 4:
            raise ValueError(f"expected 4D images, got shape {tuple(images.shape)}")
        if images.shape[-1] in (1, 3, 4) and images.shape[1] not in (1, 3, 4):
            images = images.permute(0, 3, 1, 2).contiguous()
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif not images.is_floating_point():
            raise TypeError(f"unsupported image dtype: {images.dtype}")
        else:
            # Assume already [0, 1] if float.
            images = images.float()
        images = TF.resize(
            images,
            [self.image_size, self.image_size],
            antialias=True,
        )
        return (images - self._mean) / self._std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B,)."""
        x = self.preprocess(images)
        feats = self.backbone(x)
        logits = self.head(feats).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(images))


def build_model(
    arch: str = "resnet18",
    pretrained: bool = True,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    image_size: int = 224,
    checkpoint: Optional[str] = None,
    map_location: str | torch.device = "cpu",
) -> ResNetSuccessClassifier:
    model = ResNetSuccessClassifier(
        arch=arch,
        pretrained=pretrained if checkpoint is None else False,
        hidden_dim=hidden_dim,
        dropout=dropout,
        image_size=image_size,
    )
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=map_location, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
    return model
