"""
src/model_def.py  —  Single source of truth for SilicosisNet.

Architecture confirmed from best_silicosis_model.pth tensor inspection:
  DenseNet121 backbone
  ILO head : Dropout(0.3) -> Linear(1024,256) -> ReLU -> Dropout(0.15) -> Linear(256,12)
  PMF head : Dropout(0.3) -> Linear(1024,128) -> ReLU -> Dropout(0.15) -> Linear(128, 4)
  NIH head : Dropout(0.3) -> Linear(1024,2)   (Phase-1 only, kept for weight compatibility)

Grad-CAM target layer: model.features.denseblock4.denselayer16.conv2
  (as defined in your notebook Cell 10 and Cell 17)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import yaml

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)


class SilicosisNet(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        do    = _cfg["model_settings"]["dropout"]        # 0.3
        n_ilo = _cfg["model_settings"]["num_ilo_classes"] # 12
        n_pmf = _cfg["model_settings"]["num_pmf_classes"] #  4

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base    = models.densenet121(weights=weights)
        self.features    = base.features
        self.in_features = base.classifier.in_features   # 1024
        self.pool        = nn.AdaptiveAvgPool2d(1)

        # NIH head — Linear(1024, 1) to match saved weights.
        # Notebook used BCEWithLogitsLoss so output dim is 1, not 2.
        self.nih_head = nn.Sequential(
            nn.Dropout(do),
            nn.Linear(self.in_features, 1),
        )

        # ILO head  Linear(1024 -> 256 -> 12)
        self.ilo_head = nn.Sequential(
            nn.Dropout(do),
            nn.Linear(self.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(do / 2),
            nn.Linear(256, n_ilo),
        )

        # PMF head  Linear(1024 -> 128 -> 4)
        self.pmf_head = nn.Sequential(
            nn.Dropout(do),
            nn.Linear(self.in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(do / 2),
            nn.Linear(128, n_pmf),
        )

    def _featurize(self, x):
        f = F.relu(self.features(x), inplace=True)
        return self.pool(f).flatten(1)          # [B, 1024]

    def forward(self, x):
        """Returns (ilo_logits [B,12], pmf_logits [B,4])"""
        f = self._featurize(x)
        return self.ilo_head(f), self.pmf_head(f)

    def forward_nih(self, x):
        return self.nih_head(self._featurize(x))

    @torch.no_grad()
    def predict(self, x):
        """Returns (ilo_probs, pmf_probs) — no grad, softmax applied."""
        ilo, pmf = self.forward(x)
        return F.softmax(ilo, dim=1), F.softmax(pmf, dim=1)


def load_model(weights_path: str, device: str = "cpu") -> SilicosisNet:
    """
    Load SilicosisNet from best_silicosis_model.pth.
    Handles both raw state_dict and {'model_state_dict': ...} checkpoint formats.
    """
    model = SilicosisNet(pretrained=False)
    raw   = torch.load(weights_path, map_location=device, weights_only=False)
    # Unwrap if torch.save was called with a dict wrapper
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_model] Missing  : {len(missing)} keys (first: {missing[0]})")
    if unexpected:
        print(f"[load_model] Unexpected: {len(unexpected)} keys")
    model.to(device).eval()
    return model
