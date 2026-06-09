"""
pipeline/convert_to_onnx.py
============================
Converts best_silicosis_model.pth -> models/silicosis_model.onnx

Run from project root:
  python pipeline/convert_to_onnx.py

Steps:
  1. Load SilicosisNet from src/model_def.py
  2. Load best_silicosis_model.pth weights
  3. Export as ONNX (opset 12, two output nodes)
  4. Verify with onnxruntime
"""

import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.model_def import load_model

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)


def export():
    pth_path  = cfg["model_settings"]["gradcam_weights_path"]   # best_silicosis_model.pth
    onnx_path = cfg["model_settings"]["weights_path"]            # silicosis_model.onnx

    if not os.path.exists(pth_path):
        raise FileNotFoundError(
            f"Weights not found: '{pth_path}'\n"
            "Copy best_silicosis_model.pth into models/ folder."
        )

    print(f"[convert] Loading weights from : {pth_path}")
    model = load_model(pth_path, device="cpu")
    model.eval()

    # Dummy input — matches inference crop size (448x448)
    dummy = torch.randn(
        1,
        cfg["image_settings"]["channels"],
        cfg["image_settings"]["crop"],
        cfg["image_settings"]["crop"],
    )

    os.makedirs("models", exist_ok=True)
    print(f"[convert] Exporting ONNX -> {onnx_path} ...")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["ilo_output", "pmf_output"],
        dynamic_axes={
            "input":      {0: "batch_size"},
            "ilo_output": {0: "batch_size"},
            "pmf_output": {0: "batch_size"},
        },
    )

    # Size check
    size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    print(f"[convert] File size : {size_mb:.1f} MB", end="  ")
    print("✓ OK" if size_mb > 5 else "⚠ WARNING: very small")

    # Sanity inference
    print("[convert] Running ONNX sanity check ...")
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp  = dummy.numpy().astype(np.float32)
    ilo_out, pmf_out = sess.run(None, {"input": inp})

    assert ilo_out.shape == (1, cfg["model_settings"]["num_ilo_classes"]), \
        f"ILO output shape wrong: {ilo_out.shape}"
    assert pmf_out.shape == (1, cfg["model_settings"]["num_pmf_classes"]), \
        f"PMF output shape wrong: {pmf_out.shape}"

    print(f"  ILO output : {ilo_out.shape}  ✓")
    print(f"  PMF output : {pmf_out.shape}  ✓")
    print(f"\n[convert] Done. ONNX model ready at: {onnx_path}")
    print("  Next: docker build -t silicosis-service .")


if __name__ == "__main__":
    export()
