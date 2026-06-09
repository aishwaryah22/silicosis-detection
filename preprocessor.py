"""
src/preprocessor.py
====================
Single-image preprocessing for inference — matches your notebook exactly.

Notebook (Cell 7):
  val/test: Resize(512) -> CenterCrop(448) -> Normalize -> ToTensorV2
"""

import yaml
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

_img = _cfg["image_settings"]

# Inference transform — identical to notebook get_transforms("val")
_TRANSFORM = T.Compose([
    T.Resize((_img["size"], _img["size"]), interpolation=InterpolationMode.BILINEAR),
    T.CenterCrop(_img["crop"]),
    T.ToTensor(),
    T.Normalize(mean=_img["mean"], std=_img["std"]),
])


def preprocess(pil_image: Image.Image) -> np.ndarray:
    """
    PIL image -> float32 numpy array [1, 3, 448, 448] for ONNX Runtime.
    """
    img_rgb = pil_image.convert("RGB")
    tensor  = _TRANSFORM(img_rgb)
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def preprocess_to_tensor(pil_image: Image.Image):
    """
    PIL image -> torch.Tensor [1, 3, 448, 448].
    Used by Grad-CAM (needs autograd).
    """
    img_rgb = pil_image.convert("RGB")
    return _TRANSFORM(img_rgb).unsqueeze(0)
