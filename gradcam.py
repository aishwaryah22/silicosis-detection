"""
src/gradcam.py
==============
Grad-CAM matching your notebook Cell 17 exactly.

Target layer: model.features.denseblock4.denselayer16.conv2
  (same as notebook: layer = model.features.denseblock4.denselayer16.conv2)

ONNX Runtime has no .backward() — so Grad-CAM runs on the PyTorch model
loaded from best_silicosis_model.pth on CPU.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
import yaml
from src.model_def import SilicosisNet, load_model

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

ILO_CLASSES = _cfg["ilo_classes"]
PMF_CLASSES  = _cfg["pmf_classes"]

# ── Singleton ─────────────────────────────────────────────────────────
_model_instance = None


def _get_model() -> SilicosisNet:
    global _model_instance
    if _model_instance is None:
        path = _cfg["model_settings"]["gradcam_weights_path"]
        _model_instance = load_model(path, device="cpu")
        print(f"[gradcam] PyTorch model loaded: {path}")
    return _model_instance


# ── GradCAM class (matches your notebook Cell 17) ─────────────────────

class GradCAM:
    """
    Grad-CAM hooked onto denseblock4.denselayer16.conv2
    — exact layer from your notebook Cell 10 + Cell 17.
    """

    def __init__(self, model: SilicosisNet):
        self.model        = model
        self._activations = None
        self._gradients   = None

        # Hook onto the exact layer from your notebook
        layer = model.features.denseblock4.denselayer16.conv2
        layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_activations", o.detach())
        )
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_gradients", go[0].detach())
        )

    def __call__(self,
                 img_tensor: torch.Tensor,
                 task: str = "ilo",
                 class_idx: int = None,
                 orig_img: np.ndarray = None):
        """
        Parameters
        ----------
        img_tensor : [1, 3, H, W]  float32 on CPU
        task       : 'ilo' or 'pmf'
        class_idx  : target class (None = argmax)
        orig_img   : HxWx3 uint8 numpy for overlay

        Returns
        -------
        cam_norm  : np.ndarray (H, W) float32 in [0,1]
        overlay   : np.ndarray (H, W, 3) uint8 or None
        class_idx : int
        """
        self.model.eval()
        self.model.zero_grad()

        ilo_l, pmf_l = self.model(img_tensor)
        logits = ilo_l if task == "ilo" else pmf_l

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        logits[0, class_idx].backward()

        grads   = self._gradients[0]              # (C, H, W)
        acts    = self._activations[0]            # (C, H, W)
        weights = grads.mean(dim=(1, 2))          # (C,)
        cam     = F.relu((weights[:, None, None] * acts).sum(0))

        # Normalise
        cam_n = cam - cam.min()
        if (cam.max() - cam.min()) > 1e-8:
            cam_n = cam_n / (cam.max() - cam.min())
        cam_np = cam_n.cpu().numpy().astype(np.float32)

        overlay = None
        if orig_img is not None:
            h, w    = orig_img.shape[:2]
            cam_big = cv2.resize(cam_np, (w, h))
            heatmap = cv2.applyColorMap(
                (cam_big * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay  = cv2.addWeighted(
                orig_img.astype(np.uint8), 0.55,
                heatmap.astype(np.uint8), 0.45, 0
            )

        return cam_np, overlay, class_idx

    @staticmethod
    def to_base64(image_array: np.ndarray) -> str:
        """RGB numpy array -> Base64 PNG string (no disk write)."""
        bgr     = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("imencode failed in GradCAM.to_base64")
        return base64.b64encode(buf).decode("utf-8")


# ── Module-level convenience getter ───────────────────────────────────

_gcam_instance = None

def get_gradcam() -> GradCAM:
    """Return cached GradCAM instance (loads model once at startup)."""
    global _gcam_instance
    if _gcam_instance is None:
        _gcam_instance = GradCAM(_get_model())
        print("[gradcam] GradCAM ready (hooked onto denseblock4.denselayer16.conv2)")
    return _gcam_instance
