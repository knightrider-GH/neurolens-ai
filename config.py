"""
NeuroLens AI — Configuration Module
Centralized configuration for models, paths, and runtime settings.
"""

import os
import platform
from pathlib import Path

# ─── Application Info ──────────────────────────────────────────────────────────
APP_NAME = "NeuroLens AI"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "On-Device Visual Intelligence & Accessibility Platform for Snapdragon"
APP_AUTHOR = "NeuroLens Team"

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "model_weights"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
for d in [MODELS_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Model Configuration ──────────────────────────────────────────────────────
MODELS = {
    "yolov8n": {
        "name": "YOLOv8 Nano",
        "filename": "yolov8n.onnx",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
        "input_size": (640, 640),
        "description": "Real-time object detection — optimized for Snapdragon NPU",
        "task": "detection",
    },
    "real_esrgan_x4": {
        "name": "Real-ESRGAN x4",
        "filename": "realesrgan_x4.onnx",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "input_size": None,
        "description": "4× AI image super-resolution",
        "task": "enhancement",
    },
    "u2net": {
        "name": "U²-Net",
        "filename": "u2net.onnx",
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        "input_size": (320, 320),
        "description": "Salient object segmentation for background removal",
        "task": "segmentation",
    },
}

# ─── ONNX Runtime / NPU Settings ──────────────────────────────────────────────
# Execution providers in order of preference
# QNNExecutionProvider = Qualcomm Neural Network (Snapdragon NPU)
# DmlExecutionProvider = DirectML (GPU fallback on Windows)
# CPUExecutionProvider = CPU fallback
PREFERRED_PROVIDERS = [
    "QNNExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]

# QNN-specific options for Snapdragon NPU
QNN_OPTIONS = {
    "backend_path": "QnnHtp.dll",        # HTP (Hexagon Tensor Processor) backend
    "enable_htp_fp16_precision": "1",     # FP16 for speed on NPU
    "htp_graph_finalization_optimization_mode": "3",
}

# ─── Server Settings ──────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8765
DEBUG = True
CORS_ORIGINS = ["*"]

# ─── Detection Settings ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.50
MAX_DETECTIONS = 100

# COCO class names for YOLOv8
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ─── Enhancement Settings ─────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 20
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]

# ─── Benchmark Settings ───────────────────────────────────────────────────────
BENCHMARK_ITERATIONS = 50
WARMUP_ITERATIONS = 5
