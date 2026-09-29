"""
NeuroLens AI — Main Application Server
FastAPI-based backend serving the NeuroLens AI visual intelligence platform.
Provides REST API endpoints for all AI features and serves the web UI.
"""

import asyncio
import base64
import io
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import config
from utils.npu_manager import npu_manager
from utils.model_downloader import model_downloader

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-24s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neurolens.app")

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

# ─── Global State ──────────────────────────────────────────────────────────────
models_loaded = {
    "detector": False,
    "enhancer_sr": False,
    "enhancer_bg": False,
    "ocr": False,
}


def load_models_lazy():
    """Load models on first use (lazy loading)."""
    global models_loaded

    if not models_loaded["detector"]:
        try:
            from models.detector import detector
            if detector.load():
                models_loaded["detector"] = True
        except Exception as e:
            logger.warning(f"Detector load skipped: {e}")

    if not models_loaded["ocr"]:
        try:
            from models.ocr_engine import ocr_engine
            if ocr_engine.load():
                models_loaded["ocr"] = True
        except Exception as e:
            logger.warning(f"OCR load skipped: {e}")

    if not models_loaded["enhancer_bg"]:
        try:
            from models.enhancer import enhancer
            if enhancer.load_background_removal():
                models_loaded["enhancer_bg"] = True
        except Exception as e:
            logger.warning(f"BG removal load skipped: {e}")


# ─── Helper Functions ──────────────────────────────────────────────────────────


def decode_base64_image(data: str) -> np.ndarray:
    """Decode base64 image string to numpy array."""
    # Strip data URL prefix if present
    if "," in data:
        data = data.split(",", 1)[1]
    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def encode_image_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode numpy image to base64 string."""
    _, buffer = cv2.imencode(fmt, image)
    return base64.b64encode(buffer).decode("utf-8")


async def read_upload_image(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into a numpy image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return image


# ─── Routes: Pages ─────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main application UI."""
    return templates.TemplateResponse("index.html", {"request": request})


# ─── Routes: System ────────────────────────────────────────────────────────────


@app.get("/api/status")
async def system_status():
    """Get system status, hardware info, and model availability."""
    hw_info = npu_manager.get_hardware_info()
    model_status = model_downloader.get_status()

    return {
        "app": {
            "name": config.APP_NAME,
            "version": config.APP_VERSION,
        },
        "hardware": hw_info,
        "models": model_status,
        "models_loaded": models_loaded,
    }


@app.post("/api/models/download/{model_key}")
async def download_model(model_key: str):
    """Download a specific model."""
    if model_key not in config.MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_key}")

    path = model_downloader.download_model(model_key)
    if path:
        return {"status": "success", "path": str(path)}
    raise HTTPException(status_code=500, detail="Download failed")


@app.post("/api/models/download-all")
async def download_all_models():
    """Download all models."""
    results = model_downloader.download_all()
    return {
        "status": "complete",
        "results": {k: str(v) if v else None for k, v in results.items()},
    }


@app.post("/api/models/load")
async def load_models():
    """Load all available models into memory."""
    load_models_lazy()
    return {"status": "loaded", "models": models_loaded}


# ─── Routes: Object Detection ─────────────────────────────────────────────────


@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Run object detection on an uploaded image."""
    from models.detector import detector
    from models.scene_narrator import narrator

    if not detector.loaded:
        detector.load()

    if not detector.loaded:
        raise HTTPException(
            status_code=503, detail="Detector model not available. Please download the model first."
        )

    image = await read_upload_image(file)
    h, w = image.shape[:2]

    detections, inference_ms = detector.detect(image)

    # Generate scene narration
    narration = narrator.narrate(detections, w, h, verbose=True)

    # Draw annotated image
    annotated = detector.draw_detections(image, detections)
    annotated_b64 = encode_image_base64(annotated)

    return {
        "detections": detections,
        "count": len(detections),
        "inference_ms": inference_ms,
        "narration": narration,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "provider": npu_manager.active_provider,
        "image_size": {"width": w, "height": h},
    }


@app.post("/api/detect/base64")
async def detect_objects_base64(payload: dict):
    """Run object detection on a base64-encoded image (for webcam frames)."""
    from models.detector import detector
    from models.scene_narrator import narrator

    if not detector.loaded:
        detector.load()

    if not detector.loaded:
        raise HTTPException(status_code=503, detail="Detector not available")

    image = decode_base64_image(payload["image"])
    h, w = image.shape[:2]

    detections, inference_ms = detector.detect(image)
    narration = narrator.narrate(detections, w, h)

    annotated = detector.draw_detections(image, detections)
    annotated_b64 = encode_image_base64(annotated)

    return {
        "detections": detections,
        "count": len(detections),
        "inference_ms": inference_ms,
        "narration": narration,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "provider": npu_manager.active_provider,
    }


# ─── Routes: Image Enhancement ────────────────────────────────────────────────


@app.post("/api/enhance/upscale")
async def upscale_image(file: UploadFile = File(...)):
    """Upscale an image using AI super-resolution."""
    from models.enhancer import enhancer

    if not enhancer.sr_loaded:
        enhancer.load_super_resolution()

    image = await read_upload_image(file)
    result, inference_ms = enhancer.upscale(image)
    result_b64 = encode_image_base64(result, ".png")

    return {
        "enhanced_image": f"data:image/png;base64,{result_b64}",
        "original_size": {"width": image.shape[1], "height": image.shape[0]},
        "enhanced_size": {"width": result.shape[1], "height": result.shape[0]},
        "inference_ms": inference_ms,
        "provider": npu_manager.active_provider,
    }


@app.post("/api/enhance/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    """Remove background from an image."""
    from models.enhancer import enhancer

    if not enhancer.bg_loaded:
        enhancer.load_background_removal()

    if not enhancer.bg_loaded:
        raise HTTPException(status_code=503, detail="Background removal model not available")

    image = await read_upload_image(file)
    result, inference_ms = enhancer.remove_background(image)

    # Encode as PNG to preserve alpha channel
    result_b64 = encode_image_base64(result, ".png")

    return {
        "result_image": f"data:image/png;base64,{result_b64}",
        "inference_ms": inference_ms,
        "provider": npu_manager.active_provider,
    }


@app.post("/api/enhance/replace-bg")
async def replace_bg(file: UploadFile = File(...), color: str = "255,255,255"):
    """Replace image background with a solid color."""
    from models.enhancer import enhancer

    if not enhancer.bg_loaded:
        enhancer.load_background_removal()

    if not enhancer.bg_loaded:
        raise HTTPException(status_code=503, detail="Background removal model not available")

    image = await read_upload_image(file)
    bg_color = tuple(int(c) for c in color.split(","))
    result, inference_ms = enhancer.replace_background(image, bg_color)

    result_b64 = encode_image_base64(result, ".png")

    return {
        "result_image": f"data:image/png;base64,{result_b64}",
        "inference_ms": inference_ms,
        "provider": npu_manager.active_provider,
    }


# ─── Routes: OCR ──────────────────────────────────────────────────────────────


@app.post("/api/ocr")
async def extract_text(file: UploadFile = File(...)):
    """Extract text from an image using OCR."""
    from models.ocr_engine import ocr_engine

    if not ocr_engine.loaded:
        ocr_engine.load()

    if not ocr_engine.loaded:
        raise HTTPException(status_code=503, detail="OCR engine not available")

    image = await read_upload_image(file)
    result, inference_ms = ocr_engine.extract_text(image, detail=True)

    # Draw annotated image
    annotated = ocr_engine.draw_ocr_results(image, result["blocks"])
    annotated_b64 = encode_image_base64(annotated)

    result["annotated_image"] = f"data:image/jpeg;base64,{annotated_b64}"
    result["inference_ms"] = inference_ms
    result["provider"] = npu_manager.active_provider

    return result


# ─── Routes: Benchmark ────────────────────────────────────────────────────────


@app.post("/api/benchmark/{model_key}")
async def run_benchmark(model_key: str):
    """Run inference benchmark for a specific model."""
    from models.detector import detector

    if model_key == "yolov8n":
        if not detector.loaded:
            detector.load()
        if not detector.loaded:
            raise HTTPException(status_code=503, detail="Model not available")

        # Generate random test input
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        input_data = {detector.input_name: dummy_input}

        # Run benchmark
        results = npu_manager.benchmark_inference(
            detector.session,
            input_data,
            iterations=config.BENCHMARK_ITERATIONS,
            warmup=config.WARMUP_ITERATIONS,
        )
        results["model"] = "YOLOv8 Nano"
        return results

    raise HTTPException(status_code=404, detail=f"Benchmark not available for: {model_key}")


@app.get("/api/benchmark/hardware")
async def get_hardware_benchmark():
    """Get hardware capability summary."""
    return npu_manager.get_hardware_info()


# ─── WebSocket: Real-time Detection ───────────────────────────────────────────


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint for real-time webcam object detection."""
    from models.detector import detector
    from models.scene_narrator import narrator

    await websocket.accept()
    logger.info("WebSocket client connected for real-time detection.")

    if not detector.loaded:
        detector.load()

    if not detector.loaded:
        await websocket.send_json({"error": "Detector not loaded"})
        await websocket.close()
        return

    previous_detections = []

    try:
        while True:
            # Receive base64 frame from client
            data = await websocket.receive_text()
            image = decode_base64_image(data)
            h, w = image.shape[:2]

            # Detect
            detections, inference_ms = detector.detect(image)
            narration = narrator.narrate(detections, w, h)

            # Check for scene changes
            changes = narrator.get_scene_changes(detections, previous_detections)
            previous_detections = detections

            # Draw and encode
            annotated = detector.draw_detections(image, detections)
            annotated_b64 = encode_image_base64(annotated)

            # Send results
            await websocket.send_json({
                "detections": detections,
                "count": len(detections),
                "inference_ms": inference_ms,
                "fps": round(1000 / inference_ms, 1) if inference_ms > 0 else 0,
                "narration": narration,
                "changes": changes,
                "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# ─── Startup ───────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    """Application startup handler."""
    logger.info("=" * 60)
    logger.info(f"  {config.APP_NAME} v{config.APP_VERSION}")
    logger.info(f"  {config.APP_DESCRIPTION}")
    logger.info("=" * 60)
    logger.info(f"  Provider: {npu_manager.active_provider}")
    logger.info(f"  NPU Available: {npu_manager.npu_available}")
    logger.info(f"  Models Dir: {config.MODELS_DIR}")
    logger.info("=" * 60)
    logger.info(f"  → Open http://{config.HOST}:{config.PORT} in your browser")
    logger.info("=" * 60)


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info",
    )
