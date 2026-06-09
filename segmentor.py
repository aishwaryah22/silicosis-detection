"""
src/segmentor.py
================
Lung ROI extraction using classical CV — no extra model needed.
CLAHE + Otsu thresholding + morphological closing + top-2 contours.
"""

import cv2
import numpy as np
import base64
from PIL import Image


def segment_lungs(pil_image: Image.Image):
    """
    Returns:
        masked_image : np.ndarray (H, W, 3) uint8 — background zeroed
        seg_mask     : np.ndarray (H, W)    uint8 — binary mask 0/255
    """
    gray = np.array(pil_image.convert("L"))
    h, w = gray.shape

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        rgb = np.array(pil_image.convert("RGB"))
        return rgb, np.full((h, w), 255, dtype=np.uint8)

    top2 = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, top2, -1, 255, thickness=cv2.FILLED)

    k2   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=4)

    rgb    = np.array(pil_image.convert("RGB"))
    mask3  = np.stack([mask] * 3, axis=-1)
    masked = np.where(mask3 > 0, rgb, 0).astype(np.uint8)
    return masked, mask


def mask_to_base64(mask: np.ndarray) -> str:
    """Binary mask -> Base64 PNG string (no file written to disk)."""
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("imencode failed for seg mask")
    return base64.b64encode(buf).decode("utf-8")
