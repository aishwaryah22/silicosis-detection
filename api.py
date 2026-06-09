"""
src/api.py
==========
FastAPI server — POST /predict

Sponsor-required JSON keys (exact):
  prediction            str
  confidence_score      float
  inference_time_ms     float
  visualizations        dict
    gradcam_overlay_base64    str  (Base64 PNG)
    segmentation_mask_base64  str  (Base64 PNG)

No image files are written to disk. All visuals are Base64 in-memory.
All settings from config.yaml.

Run locally:
  uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import io
import yaml
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.preprocessor import preprocess, preprocess_to_tensor
from src.inference     import run_inference, ILO_CLASSES, PMF_CLASSES
from src.segmentor     import segment_lungs, mask_to_base64
from src.gradcam       import get_gradcam, GradCAM
from src.report        import generate_report

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="rises.AI — Silicosis Detection Service",
    version="1.0.0",
    description="DenseNet-121 · ILO Profusion Scoring · PMF Detection · Grad-CAM",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ────────────────────────────────────────
@app.on_event("startup")
async def startup():
    # Warm-up ONNX session
    from src.inference import _get_session
    _get_session()
    # Load PyTorch model + hook Grad-CAM
    get_gradcam()
    print("[api] All models ready.")


# ── Health ─────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status":  "ok",
        "service": "silicosis-ai",
        "model":   _cfg["model_settings"]["architecture"],
    }


# ── Predict ────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    POST /predict
    Input : multipart/form-data  field='file'  (JPEG / PNG chest X-ray)
    Output: JSON — prediction, confidence_score, inference_time_ms,
            visualizations (Base64), additional_inference_metadata
    """

    # 1. Read image
    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    # 2. Lung segmentation
    try:
        masked_arr, seg_mask = segment_lungs(pil_img)
        masked_pil           = Image.fromarray(masked_arr)
        seg_b64              = mask_to_base64(seg_mask)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

    # 3. Preprocess segmented image for ONNX
    input_array = preprocess(masked_pil)          # [1, 3, 448, 448] float32

    # 4. ONNX inference
    try:
        result = run_inference(input_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # 5. Grad-CAM on original (unmasked) image
    try:
        gcam      = get_gradcam()
        tensor    = preprocess_to_tensor(masked_pil)   # [1, 3, 448, 448]
        orig_arr  = np.array(pil_img)                  # original for overlay
        cam_np, overlay, ilo_idx = gcam(
            tensor,
            task="ilo",
            class_idx=None,
            orig_img=orig_arr,
        )
        gcam_b64 = GradCAM.to_base64(overlay)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {e}")

    # 6. Clinical report
    report_text = generate_report(
        ilo_class  = result["ilo_class"],
        pmf_class  = result["pmf_class"],
        confidence = result["confidence_score"],
        ilo_probs  = result["ilo_probs"],
        pmf_probs  = result["pmf_probs"],
    )

    # 7. Return sponsor-compliant JSON
    return {
        # ── Mandatory top-level keys ────────────────────────────────
        "prediction":         result["prediction"],
        "confidence_score":   result["confidence_score"],
        "inference_time_ms":  result["inference_time_ms"],
        "visualizations": {
            "gradcam_overlay_base64":   gcam_b64,
            "segmentation_mask_base64": seg_b64,
        },

        # ── Extended metadata ────────────────────────────────────────
        "additional_inference_metadata": {
            "ilo_profusion_score":      result["ilo_class"],
            "pmf_grade":                result["pmf_class"],
            "ilo_severity":             result["ilo_severity"],
            "ilo_class_probabilities":  dict(zip(ILO_CLASSES, result["ilo_probs"])),
            "pmf_class_probabilities":  dict(zip(PMF_CLASSES,  result["pmf_probs"])),
            "lung_segmentation_applied": True,
            "model_architecture":       _cfg["model_settings"]["architecture"],
            "clinical_report":          report_text,
        },
    }
