# =====================================================================
# Dockerfile — silicosis-detector
# Base   : python:3.10-slim
# Port   : 8000
# No Streamlit / Gradio / UI frameworks
# =====================================================================

FROM python:3.10-slim

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy runtime files only
COPY config.yaml            .
COPY models/silicosis_model.onnx      ./models/silicosis_model.onnx
COPY models/best_silicosis_model.pth  ./models/best_silicosis_model.pth
COPY src/                   ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
