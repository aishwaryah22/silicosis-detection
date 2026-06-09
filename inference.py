"""
src/inference.py
================
ONNX Runtime inference — singleton session, softmax, severity mapping.
All config from config.yaml — zero hardcoded values.
"""

import time
import yaml
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

ILO_CLASSES = _cfg["ilo_classes"]   # 12 labels
PMF_CLASSES  = _cfg["pmf_classes"]  #  4 labels
THRESHOLD    = _cfg["model_settings"]["confidence_threshold"]

# Severity description per ILO index
_SEVERITY = {
    0:  "Normal (sub-threshold)",
    1:  "Normal (no opacity)",
    2:  "Borderline / Suspected",
    3:  "Borderline / Suspected",
    4:  "Mild Silicosis",
    5:  "Mild–Moderate Silicosis",
    6:  "Moderate Silicosis",
    7:  "Moderate Silicosis",
    8:  "Moderate–Severe Silicosis",
    9:  "Severe Silicosis",
    10: "Severe Silicosis",
    11: "Very Severe Silicosis",
}

# ── Singleton ONNX session ────────────────────────────────────────────
_session = None

def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        path     = _cfg["model_settings"]["weights_path"]
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        print(f"[inference] ONNX session loaded: {path}")
    return _session


def run_inference(input_array: np.ndarray) -> dict:
    """
    Parameters
    ----------
    input_array : float32  [1, 3, 448, 448]

    Returns
    -------
    dict with keys: prediction, confidence_score, inference_time_ms,
                    ilo_class, pmf_class, ilo_probs, pmf_probs, ilo_severity
    """
    sess       = _get_session()
    input_name = sess.get_inputs()[0].name

    t0      = time.perf_counter()
    outputs = sess.run(None, {input_name: input_array})
    t1      = time.perf_counter()

    ilo_logits = outputs[0][0]   # [12]
    pmf_logits = outputs[1][0]   # [4]

    ilo_probs = softmax(ilo_logits).tolist()
    pmf_probs  = softmax(pmf_logits).tolist()

    ilo_idx = int(np.argmax(ilo_probs))
    pmf_idx  = int(np.argmax(pmf_probs))

    return {
        "prediction":        _SEVERITY[ilo_idx],
        "confidence_score":  round(float(max(ilo_probs)), 4),
        "inference_time_ms": round((t1 - t0) * 1000, 2),
        "ilo_class":         ILO_CLASSES[ilo_idx],
        "pmf_class":         PMF_CLASSES[pmf_idx],
        "ilo_probs":         [round(p, 4) for p in ilo_probs],
        "pmf_probs":         [round(p, 4) for p in pmf_probs],
        "ilo_severity":      _SEVERITY[ilo_idx],
    }
