<p align="center">
  <img src="https://img.shields.io/badge/Platform-Snapdragon%20X%20Elite%20%7C%20Plus-blue?style=for-the-badge&logo=qualcomm&logoColor=white" alt="Platform" />
  <img src="https://img.shields.io/badge/Runtime-ONNX%20Runtime%20QNN-green?style=for-the-badge" alt="Runtime" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge" alt="License" />
</p>

<h1 align="center">🧠 NeuroLens AI</h1>

<p align="center">
  <strong>On-Device Visual Intelligence & Accessibility Platform for Snapdragon</strong>
</p>

<p align="center">
  Real-time object detection • AI image enhancement • Smart OCR • Scene narration<br/>
  <em>All running locally on your Snapdragon-powered HP PC — no cloud, full privacy, blazing speed.</em>
</p>

---

## 🎯 What is NeuroLens AI?

**NeuroLens AI** is a comprehensive on-device visual intelligence platform built from the ground up for **Snapdragon-powered HP PCs**. It leverages the **Qualcomm Hexagon NPU** (Neural Processing Unit) through ONNX Runtime's QNN Execution Provider to deliver real-time AI inference without relying on any cloud service.

The platform combines four powerful AI capabilities into a single, beautifully designed web application:

| Feature | Model | Description |
|---------|-------|-------------|
| 🔍 **Real-Time Detection** | YOLOv8 Nano | Detect 80+ object categories from webcam or uploaded images at real-time speeds |
| 🖼️ **Image Enhancement** | Real-ESRGAN / U²-Net | 4× AI super-resolution and intelligent background removal |
| 📄 **Smart Document OCR** | EasyOCR | Extract text from documents with automatic entity recognition (emails, dates, amounts) |
| 🗣️ **Scene Narration** | Custom NLP | Natural language scene descriptions with text-to-speech for visually impaired users |

### Why NeuroLens AI?

- **🔒 Privacy-First**: All AI processing happens locally on-device. Your images and documents never leave your machine.
- **⚡ NPU-Accelerated**: Leverages the dedicated Snapdragon NPU (up to 45 TOPS) for inference speeds that rival or exceed cloud APIs.
- **♿ Accessibility**: Built-in scene narration with TTS enables visually impaired users to understand their visual environment.
- **📊 Benchmarkable**: Includes a comprehensive CPU vs NPU benchmark dashboard to demonstrate Snapdragon's AI advantage.
- **🌐 Zero Dependencies on Cloud**: No API keys, no internet required (after initial model download). Works completely offline.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           NeuroLens AI Web Interface              │
│   (HTML5 + CSS3 + Vanilla JS — Dark Glassmorphism) │
├─────────────────────────────────────────────────┤
│           FastAPI Backend (Python)                 │
│   REST API + WebSocket (real-time detection)       │
├──────────┬──────────┬──────────┬─────────────────┤
│ Detector │ Enhancer │   OCR    │   Narrator       │
│ (YOLOv8) │(ESRGAN/  │(EasyOCR) │  (Custom NLP)   │
│          │  U²-Net) │          │                   │
├──────────┴──────────┴──────────┴─────────────────┤
│         NPU Manager (Provider Abstraction)         │
│   QNN EP (NPU) → DirectML (GPU) → CPU (Fallback)  │
├─────────────────────────────────────────────────┤
│         ONNX Runtime + Qualcomm QNN SDK            │
├─────────────────────────────────────────────────┤
│     Snapdragon X Elite/Plus Hardware (Hexagon NPU) │
└─────────────────────────────────────────────────┘
```

### Key Technical Highlights

- **Automatic Provider Detection**: The NPU Manager automatically detects the best available execution provider (QNN → DirectML → CPU) and configures ONNX Runtime sessions accordingly.
- **FP16 NPU Inference**: All models run in half-precision (FP16) on the Hexagon NPU for maximum throughput.
- **WebSocket Real-Time Pipeline**: Live webcam detection uses WebSocket for low-latency bidirectional frame streaming.
- **Tiled Super-Resolution**: Large images are processed in overlapping tiles to prevent NPU memory exhaustion.
- **Lazy Model Loading**: Models initialize on first use, not at startup, keeping boot time under 2 seconds.

---

## 🚀 Quick Start

### Prerequisites

- **Hardware**: Snapdragon X Elite or X Plus powered HP PC (e.g., HP OmniBook Ultra, HP OmniBook 3)
- **OS**: Windows 11 on ARM
- **Python**: 3.10 or higher
- **Webcam** (optional): For real-time detection features

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/knightrider-GH/neurolens-ai.git
cd neurolens-ai

# 2. Run the automated setup (Windows)
setup.bat

# OR manual setup:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. For Snapdragon NPU support (recommended):
pip install onnxruntime-qnn
```

### Running

```bash
# Activate virtual environment
venv\Scripts\activate

# Start the server
python app.py

# Or use the quick-run script:
run.bat
```

Open **http://127.0.0.1:8765** in your browser.

### First-Time Setup

1. Navigate to the **Setup** tab in the web interface
2. Click **Download** next to each model to download AI weights
3. Click **Load All Models** to initialize them
4. Start using Detection, Enhancement, or OCR!

---

## 📸 Features in Detail

### 🔍 Real-Time Object Detection

- **YOLOv8 Nano** model optimized for Snapdragon NPU
- Detects **80+ object categories** (COCO dataset)
- **Live webcam detection** via WebSocket with FPS counter
- **Image upload** detection with annotated results
- Configurable confidence threshold
- Color-coded bounding boxes with class labels

### 🗣️ Scene Narration (Accessibility)

- Converts detection results into **natural language descriptions**
- Spatial reasoning: describes object positions (left, center, right)
- **Scene change detection**: alerts when new objects appear or leave
- Built-in **Web Speech API** text-to-speech
- Designed specifically for **visually impaired users**

### 🖼️ AI Image Enhancement

- **4× Super-Resolution** with Real-ESRGAN
  - Tiled processing for large images
  - NPU-accelerated inference
- **Background Removal** with U²-Net
  - Salient object segmentation
  - Transparent PNG output
  - Background replacement with custom colors
- Before/after comparison view

### 📄 Smart Document OCR

- **EasyOCR** text extraction with confidence scores
- **Intelligent entity recognition**:
  - 📧 Email addresses
  - 📅 Dates
  - 💰 Currency amounts
  - 📞 Phone numbers
  - 🔗 URLs
- **Automatic document type classification** (invoice, resume, letter, legal)
- Annotated image output with color-coded confidence
- Copy-to-clipboard functionality

### 📊 Performance Benchmark

- **CPU vs NPU comparison** with visual bar chart
- Metrics: Average latency, P95, Min/Max, FPS
- Configurable iterations (default: 50 with 5 warmup)
- Hardware information display

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | HTML5, CSS3, Vanilla JS | Premium glassmorphism UI with animations |
| **Backend** | FastAPI + Uvicorn | High-performance async API server |
| **AI Runtime** | ONNX Runtime + QNN EP | Cross-provider inference (NPU/GPU/CPU) |
| **Detection** | YOLOv8 Nano (ONNX) | Real-time object detection |
| **Enhancement** | Real-ESRGAN / U²-Net | Super-resolution & segmentation |
| **OCR** | EasyOCR | Optical character recognition |
| **Narration** | Custom NLP + Web Speech API | Scene description & TTS |
| **Communication** | WebSocket + REST | Real-time & standard API patterns |

### Qualcomm AI Hub Integration

NeuroLens AI is designed to leverage models from **[Qualcomm AI Hub](https://aihub.qualcomm.com/)**, which provides pre-optimized models for Snapdragon platforms. The NPU Manager automatically configures the QNN Execution Provider with HTP backend settings optimized for Snapdragon X Elite/Plus chipsets.

**Key Qualcomm technologies used:**
- **QNN (Qualcomm Neural Network) SDK** — via ONNX Runtime QNN EP
- **Hexagon Tensor Processor (HTP)** — dedicated AI accelerator
- **FP16 Precision** — half-precision inference for NPU optimization

---

## 📁 Project Structure

```
neurolens-ai/
├── app.py                  # FastAPI main application server
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── setup.bat               # Automated Windows setup
├── run.bat                 # Quick-start script
├── LICENSE                 # MIT License
├── .gitignore
│
├── models/                 # AI model integration
│   ├── __init__.py
│   ├── detector.py         # YOLOv8 object detection
│   ├── enhancer.py         # Super-resolution & BG removal
│   ├── ocr_engine.py       # OCR with document analysis
│   └── scene_narrator.py   # Natural language scene narration
│
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── npu_manager.py      # Snapdragon NPU detection & management
│   └── model_downloader.py # Model download with progress tracking
│
├── static/                 # Frontend assets
│   ├── css/
│   │   └── style.css       # Complete design system
│   └── js/
│       └── app.js          # Frontend application logic
│
├── templates/
│   └── index.html          # Main application template
│
├── docs/
│   └── ARCHITECTURE.md     # Architecture documentation
│
├── model_weights/          # Downloaded model files (gitignored)
├── uploads/                # Uploaded files (gitignored)
└── outputs/                # Processed outputs (gitignored)
```

---

## 🔧 Configuration

All configuration is centralized in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `8765` | Server port |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum detection confidence |
| `NMS_THRESHOLD` | `0.50` | Non-maximum suppression threshold |
| `MAX_UPLOAD_SIZE_MB` | `20` | Maximum upload file size |
| `BENCHMARK_ITERATIONS` | `50` | Number of benchmark runs |

### Execution Provider Priority

```python
PREFERRED_PROVIDERS = [
    "QNNExecutionProvider",     # Snapdragon NPU (best)
    "DmlExecutionProvider",     # DirectML GPU (good)
    "CPUExecutionProvider",     # CPU fallback (works everywhere)
]
```

---

## 📊 Performance

*Expected performance on Snapdragon X Elite (45 TOPS NPU):*

| Model | NPU (QNN) | CPU | Speedup |
|-------|-----------|-----|---------|
| YOLOv8n Detection | ~8ms | ~45ms | **5.6×** |
| Real-ESRGAN (256px) | ~15ms | ~120ms | **8×** |
| U²-Net BG Removal | ~12ms | ~80ms | **6.7×** |

*Actual performance depends on hardware configuration and system load.*

---

## 🌐 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve web UI |
| `GET` | `/api/status` | System status & hardware info |
| `POST` | `/api/detect` | Object detection (file upload) |
| `POST` | `/api/detect/base64` | Object detection (base64 image) |
| `POST` | `/api/enhance/upscale` | AI super-resolution |
| `POST` | `/api/enhance/remove-bg` | Background removal |
| `POST` | `/api/ocr` | Text extraction |
| `POST` | `/api/benchmark/{model}` | Run performance benchmark |
| `POST` | `/api/models/download/{key}` | Download a model |
| `POST` | `/api/models/load` | Load all models |

### WebSocket

| Endpoint | Protocol | Description |
|----------|----------|-------------|
| `/ws/detect` | WebSocket | Real-time webcam detection stream |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Qualcomm](https://www.qualcomm.com/)** — Snapdragon platform & AI Hub
- **[ONNX Runtime](https://onnxruntime.ai/)** — Cross-platform AI inference engine
- **[Ultralytics](https://ultralytics.com/)** — YOLOv8 object detection
- **[xinntao](https://github.com/xinntao/Real-ESRGAN)** — Real-ESRGAN super-resolution
- **[danielgatis](https://github.com/danielgatis/rembg)** — U²-Net segmentation
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** — Optical character recognition

---

<p align="center">
  <strong>Built for the Snapdragon® AI Lab Build & Present Challenge</strong><br/>
  <em>Powered by Snapdragon NPU × ONNX Runtime × Qualcomm AI Hub</em>
</p>
