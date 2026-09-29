# NeuroLens AI — Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    NeuroLens AI Platform                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Web Frontend (Browser)                    │  │
│  │  ┌─────────┐ ┌──────────┐ ┌─────┐ ┌──────────────────┐  │  │
│  │  │Detection │ │Enhance   │ │ OCR │ │ Benchmark        │  │  │
│  │  │  Panel   │ │  Panel   │ │Panel│ │   Dashboard      │  │  │
│  │  └────┬─────┘ └────┬─────┘ └──┬──┘ └────────┬─────────┘  │  │
│  │       │             │          │              │            │  │
│  │  ┌────┴─────────────┴──────────┴──────────────┴────────┐  │  │
│  │  │           JavaScript Application (app.js)            │  │  │
│  │  │   • REST API Client  • WebSocket Client              │  │  │
│  │  │   • Webcam Manager   • File Upload Handler           │  │  │
│  │  │   • TTS Engine       • Toast Notifications           │  │  │
│  │  └────────────────────────┬────────────────────────────┘  │  │
│  └───────────────────────────┼───────────────────────────────┘  │
│                              │ HTTP / WebSocket                  │
│  ┌───────────────────────────┼───────────────────────────────┐  │
│  │              FastAPI Backend (app.py)                       │  │
│  │  ┌────────────────────────┴────────────────────────────┐  │  │
│  │  │              API Router Layer                        │  │  │
│  │  │   /api/detect    /api/enhance/*   /api/ocr          │  │  │
│  │  │   /api/benchmark /api/models/*    /ws/detect        │  │  │
│  │  └────────────────────────┬────────────────────────────┘  │  │
│  │                           │                                │  │
│  │  ┌────────────────────────┴────────────────────────────┐  │  │
│  │  │              AI Model Layer                          │  │  │
│  │  │  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────┐  │  │  │
│  │  │  │ Detector │ │ Enhancer │ │   OCR   │ │Narrator│  │  │  │
│  │  │  │ (YOLOv8) │ │(ESRGAN/  │ │(EasyOCR)│ │(NLP)   │  │  │  │
│  │  │  │          │ │ U²-Net)  │ │         │ │        │  │  │  │
│  │  │  └────┬─────┘ └────┬─────┘ └────┬────┘ └───┬────┘  │  │  │
│  │  └───────┼────────────┼────────────┼───────────┼───────┘  │  │
│  │          │            │            │           │           │  │
│  │  ┌───────┴────────────┴────────────┴───────────┴───────┐  │  │
│  │  │           NPU Manager (npu_manager.py)               │  │  │
│  │  │   • Provider Detection  • Session Factory             │  │  │
│  │  │   • Benchmarking        • Hardware Info               │  │  │
│  │  └────────────────────────┬────────────────────────────┘  │  │
│  └───────────────────────────┼───────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┼───────────────────────────────┐  │
│  │              ONNX Runtime                                  │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐  │  │
│  │  │     QNN EP   │ │  DirectML EP │ │     CPU EP       │  │  │
│  │  │ (Snapdragon  │ │  (GPU        │ │  (Fallback)      │  │  │
│  │  │   NPU)       │ │   Fallback)  │ │                  │  │  │
│  │  └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘  │  │
│  └─────────┼────────────────┼──────────────────┼─────────────┘  │
│            │                │                  │                 │
│  ┌─────────┴────────────────┴──────────────────┴─────────────┐  │
│  │                  Hardware Layer                             │  │
│  │   Qualcomm Hexagon NPU │ Adreno GPU │ ARM CPU (Oryon)     │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Object Detection Pipeline
```
Input Image → Letterbox Resize (640×640) → Normalize → CHW Transpose
    → ONNX Runtime (NPU/CPU) → YOLOv8 Output (1,84,8400)
    → Transpose → Confidence Filter → NMS → Bounding Boxes
    → Scene Narrator → Natural Language Description → TTS
```

### Image Enhancement Pipeline
```
Input Image → Tile Splitting → Normalize → CHW Transpose
    → ONNX Runtime (NPU/CPU) → Real-ESRGAN Output
    → Tile Stitching → Denormalize → 4× Upscaled Image
```

### Background Removal Pipeline
```
Input Image → Resize (320×320) → ImageNet Normalize → CHW Transpose
    → ONNX Runtime (NPU/CPU) → U²-Net Mask → Resize to Original
    → Apply Alpha Channel → Transparent Background Output
```

### OCR Pipeline
```
Input Image → EasyOCR Detection → Text Blocks with Bounding Boxes
    → Entity Extraction (regex) → Document Type Classification
    → Structured Analysis Output
```

## Key Design Decisions

1. **Lazy Model Loading**: Models load on first API call, not at startup, to reduce initial boot time.
2. **Provider Fallback Chain**: QNN (NPU) → DirectML (GPU) → CPU — ensures the app works everywhere.
3. **WebSocket for Real-time**: Live webcam detection uses WebSocket for low-latency bidirectional communication.
4. **Tiled Processing**: Large images are processed in tiles to prevent out-of-memory errors on the NPU.
5. **Scene Narration**: Detection results are converted to natural language for accessibility, with built-in TTS.
6. **Base64 Transport**: Images are encoded as base64 for easy JSON transport between frontend and backend.

## NPU Optimization Strategy

| Optimization           | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| FP16 Precision         | Models run in half-precision on NPU for 2× speed            |
| Graph Optimization     | ONNX Runtime applies operator fusion and constant folding    |
| QNN HTP Backend        | Uses Hexagon Tensor Processor for dedicated AI acceleration  |
| Memory Arena           | Reuses memory allocations across inference runs              |
| Batch Size 1           | Optimal for real-time single-image inference on NPU          |
