/**
 * NeuroLens AI — Frontend Application
 * Handles all UI interactions, webcam streaming, API calls,
 * and real-time detection via WebSocket.
 */

// ─── State ─────────────────────────────────────────────────────────────────────
const state = {
  activeTab: 'detect',
  webcamActive: false,
  webcamStream: null,
  wsConnection: null,
  isProcessing: false,
  systemStatus: null,
  detectionResults: [],
  lastNarration: '',
  fpsHistory: [],
};

// ─── DOM Helpers ───────────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ─── API Client ────────────────────────────────────────────────────────────────
const API = {
  base: '',

  async get(path) {
    const res = await fetch(`${this.base}${path}`);
    if (!res.ok) throw new Error(`API Error: ${res.statusText}`);
    return res.json();
  },

  async postFile(path, file) {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${this.base}${path}`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  async postJson(path, data) {
    const res = await fetch(`${this.base}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`API Error: ${res.statusText}`);
    return res.json();
  },
};

// ─── Toast Notifications ───────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
  const container = $('#toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-message">${message}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(40px)';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ─── Tab Navigation ────────────────────────────────────────────────────────────
function initTabs() {
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      switchTab(tab);
    });
  });
}

function switchTab(tabId) {
  state.activeTab = tabId;

  // Update buttons
  $$('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });

  // Update panels
  $$('.tab-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === `panel-${tabId}`);
  });

  // Stop webcam if leaving detect tab
  if (tabId !== 'detect' && state.webcamActive) {
    stopWebcam();
  }
}

// ─── System Status ─────────────────────────────────────────────────────────────
async function fetchSystemStatus() {
  try {
    const status = await API.get('/api/status');
    state.systemStatus = status;
    updateStatusUI(status);
    return status;
  } catch (err) {
    console.error('Failed to fetch status:', err);
    updateStatusOffline();
  }
}

function updateStatusUI(status) {
  const dot = $('#status-dot');
  const text = $('#status-text');
  const providerEl = $('#provider-text');

  if (status.hardware.npu_available) {
    dot.className = 'status-dot';
    text.textContent = 'NPU Active';
    providerEl.textContent = 'Snapdragon NPU';
  } else if (status.hardware.gpu_available) {
    dot.className = 'status-dot';
    text.textContent = 'GPU Active';
    providerEl.textContent = 'DirectML GPU';
  } else {
    dot.className = 'status-dot loading';
    text.textContent = 'CPU Mode';
    providerEl.textContent = 'CPU Fallback';
  }

  // Update model status
  updateModelStatusUI(status.models);
}

function updateModelStatusUI(models) {
  const container = $('#model-status-list');
  if (!container) return;

  container.innerHTML = '';
  for (const [key, info] of Object.entries(models)) {
    const item = document.createElement('div');
    item.className = 'model-download-item';
    item.innerHTML = `
      <div class="model-info">
        <span class="model-name">${info.name}</span>
        <span class="model-desc">${info.description}</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        ${info.downloaded
          ? `<span class="model-status-badge downloaded">✓ Ready ${info.size_mb ? `(${info.size_mb} MB)` : ''}</span>`
          : `<span class="model-status-badge missing">Not Downloaded</span>
             <button class="btn btn-sm btn-primary" onclick="downloadModel('${key}')">Download</button>`
        }
      </div>
    `;
    container.appendChild(item);
  }
}

function updateStatusOffline() {
  const dot = $('#status-dot');
  const text = $('#status-text');
  if (dot) dot.className = 'status-dot offline';
  if (text) text.textContent = 'Offline';
}

async function downloadModel(key) {
  showToast(`Downloading model...`, 'info');
  try {
    await API.postJson(`/api/models/download/${key}`, {});
    showToast('Model downloaded successfully!', 'success');
    fetchSystemStatus();
  } catch (err) {
    showToast(`Download failed: ${err.message}`, 'error');
  }
}

async function loadModels() {
  showToast('Loading models into memory...', 'info');
  try {
    const result = await API.postJson('/api/models/load', {});
    showToast('Models loaded!', 'success');
    fetchSystemStatus();
  } catch (err) {
    showToast(`Failed to load models: ${err.message}`, 'error');
  }
}

// ─── Object Detection ──────────────────────────────────────────────────────────

// File upload detection
function initDetectionUpload() {
  const zone = $('#detect-upload-zone');
  const input = $('#detect-file-input');

  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      handleDetectionFile(e.dataTransfer.files[0]);
    }
  });
  input.addEventListener('change', () => {
    if (input.files.length) handleDetectionFile(input.files[0]);
  });
}

async function handleDetectionFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Please upload an image file.', 'warning');
    return;
  }

  setProcessing(true, 'detect');
  try {
    const result = await API.postFile('/api/detect', file);
    displayDetectionResults(result);
    showToast(`Detected ${result.count} objects in ${result.inference_ms}ms`, 'success');
  } catch (err) {
    showToast(`Detection failed: ${err.message}`, 'error');
  } finally {
    setProcessing(false, 'detect');
  }
}

function displayDetectionResults(result) {
  // Show annotated image
  const imgContainer = $('#detect-result-image');
  if (imgContainer && result.annotated_image) {
    imgContainer.innerHTML = `<img src="${result.annotated_image}" alt="Detection results" />`;
  }

  // Update stats
  updateStat('detect-count', result.count);
  updateStat('detect-time', `${result.inference_ms}ms`);
  updateStat('detect-provider', result.provider?.replace('ExecutionProvider', '') || 'N/A');
  updateStat('detect-fps', result.inference_ms > 0 ? `${(1000 / result.inference_ms).toFixed(1)}` : 'N/A');

  // Show detection list
  const listContainer = $('#detection-list');
  if (listContainer) {
    const colors = [
      '#ff6b6b', '#4ecdc4', '#45b7d1', '#9670d6',
      '#ff9f43', '#2ed573', '#74b9ff', '#dfe4ea',
    ];

    if (result.detections.length === 0) {
      listContainer.innerHTML = `
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
          <h4>No Objects Detected</h4>
          <p>Try a different image with more visible objects.</p>
        </div>`;
    } else {
      listContainer.innerHTML = result.detections.map((det, i) => `
        <div class="detection-item">
          <div class="detection-color" style="background:${colors[det.class_id % colors.length]}"></div>
          <div class="detection-info">
            <div class="detection-label">${det.label}</div>
            <div class="detection-meta">[${det.bbox.join(', ')}]</div>
          </div>
          <div class="detection-confidence">${(det.confidence * 100).toFixed(1)}%</div>
        </div>
      `).join('');
    }
  }

  // Show narration
  const narrationEl = $('#narration-text');
  if (narrationEl && result.narration) {
    narrationEl.textContent = result.narration;
    state.lastNarration = result.narration;
  }
}

// ─── Webcam Detection ──────────────────────────────────────────────────────────
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'environment' }
    });

    state.webcamStream = stream;
    state.webcamActive = true;

    const video = $('#webcam-video');
    video.srcObject = stream;
    video.play();

    $('#webcam-placeholder').classList.add('hidden');
    video.classList.remove('hidden');
    $('#webcam-start-btn').classList.add('hidden');
    $('#webcam-stop-btn').classList.remove('hidden');

    showToast('Webcam started. Connecting to detection server...', 'info');

    // Start WebSocket connection for real-time detection
    connectWebSocket();

  } catch (err) {
    showToast(`Camera access denied: ${err.message}`, 'error');
  }
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(t => t.stop());
    state.webcamStream = null;
  }
  state.webcamActive = false;

  if (state.wsConnection) {
    state.wsConnection.close();
    state.wsConnection = null;
  }

  const video = $('#webcam-video');
  if (video) {
    video.srcObject = null;
    video.classList.add('hidden');
  }

  const placeholder = $('#webcam-placeholder');
  if (placeholder) placeholder.classList.remove('hidden');

  const startBtn = $('#webcam-start-btn');
  const stopBtn = $('#webcam-stop-btn');
  if (startBtn) startBtn.classList.remove('hidden');
  if (stopBtn) stopBtn.classList.add('hidden');
}

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/detect`);

  ws.onopen = () => {
    showToast('Real-time detection connected!', 'success');
    state.wsConnection = ws;
    sendFrame();
  };

  ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    if (result.error) {
      showToast(result.error, 'error');
      stopWebcam();
      return;
    }

    // Display annotated frame
    const resultImg = $('#webcam-result');
    if (resultImg && result.annotated_image) {
      resultImg.src = result.annotated_image;
      resultImg.classList.remove('hidden');
    }

    // Update FPS
    const fpsEl = $('#webcam-fps');
    if (fpsEl) fpsEl.textContent = `${result.fps} FPS`;

    // Update detection count
    updateStat('detect-count', result.count);
    updateStat('detect-time', `${result.inference_ms}ms`);

    // Update narration
    const narrationEl = $('#narration-text');
    if (narrationEl && result.narration) {
      narrationEl.textContent = result.narration;
    }

    // Update detection list
    displayDetectionResults(result);

    // Send next frame
    if (state.webcamActive) {
      requestAnimationFrame(() => setTimeout(sendFrame, 50)); // ~20 FPS cap
    }
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
    showToast('Detection connection error', 'error');
  };

  ws.onclose = () => {
    state.wsConnection = null;
  };
}

function sendFrame() {
  if (!state.wsConnection || state.wsConnection.readyState !== WebSocket.OPEN) return;
  if (!state.webcamActive) return;

  const video = $('#webcam-video');
  if (!video || video.readyState < 2) {
    setTimeout(sendFrame, 100);
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
  state.wsConnection.send(dataUrl);
}

// ─── Scene Narration (TTS) ─────────────────────────────────────────────────────
function speakNarration() {
  const text = state.lastNarration || $('#narration-text')?.textContent;
  if (!text) {
    showToast('No narration available to speak.', 'warning');
    return;
  }

  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;
    // Try to use a natural-sounding voice
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v => v.name.includes('Google') || v.name.includes('Natural'));
    if (preferred) utterance.voice = preferred;
    window.speechSynthesis.speak(utterance);
    showToast('🔊 Speaking narration...', 'info', 2000);
  } else {
    showToast('Speech synthesis not supported in this browser.', 'warning');
  }
}

// ─── Image Enhancement ─────────────────────────────────────────────────────────
function initEnhancementUpload() {
  const zone = $('#enhance-upload-zone');
  const input = $('#enhance-file-input');

  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      state._enhanceFile = e.dataTransfer.files[0];
      previewEnhanceFile(e.dataTransfer.files[0]);
    }
  });
  input.addEventListener('change', () => {
    if (input.files.length) {
      state._enhanceFile = input.files[0];
      previewEnhanceFile(input.files[0]);
    }
  });
}

function previewEnhanceFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const container = $('#enhance-original-image');
    if (container) {
      container.innerHTML = `<img src="${e.target.result}" alt="Original" />
        <span class="comparison-label original">Original</span>`;
    }
  };
  reader.readAsDataURL(file);
}

async function runUpscale() {
  const file = state._enhanceFile || $('#enhance-file-input')?.files[0];
  if (!file) {
    showToast('Please upload an image first.', 'warning');
    return;
  }

  setProcessing(true, 'enhance');
  try {
    const result = await API.postFile('/api/enhance/upscale', file);
    const container = $('#enhance-result-image');
    if (container) {
      container.innerHTML = `<img src="${result.enhanced_image}" alt="Enhanced" />
        <span class="comparison-label enhanced">Enhanced 4×</span>`;
    }
    updateStat('enhance-time', `${result.inference_ms}ms`);
    updateStat('enhance-original-size', `${result.original_size.width}×${result.original_size.height}`);
    updateStat('enhance-new-size', `${result.enhanced_size.width}×${result.enhanced_size.height}`);
    showToast(`Image upscaled to ${result.enhanced_size.width}×${result.enhanced_size.height} in ${result.inference_ms}ms`, 'success');
  } catch (err) {
    showToast(`Upscale failed: ${err.message}`, 'error');
  } finally {
    setProcessing(false, 'enhance');
  }
}

async function runRemoveBg() {
  const file = state._enhanceFile || $('#enhance-file-input')?.files[0];
  if (!file) {
    showToast('Please upload an image first.', 'warning');
    return;
  }

  setProcessing(true, 'enhance');
  try {
    const result = await API.postFile('/api/enhance/remove-bg', file);
    const container = $('#enhance-result-image');
    if (container) {
      // Checkerboard background for transparency
      container.innerHTML = `
        <div style="background:repeating-conic-gradient(#1a1a2e 0% 25%, #252540 0% 50%) 50%/20px 20px; border-radius:12px;">
          <img src="${result.result_image}" alt="Background Removed" style="display:block;" />
        </div>
        <span class="comparison-label enhanced">BG Removed</span>`;
    }
    updateStat('enhance-time', `${result.inference_ms}ms`);
    showToast(`Background removed in ${result.inference_ms}ms`, 'success');
  } catch (err) {
    showToast(`Background removal failed: ${err.message}`, 'error');
  } finally {
    setProcessing(false, 'enhance');
  }
}

// ─── OCR ───────────────────────────────────────────────────────────────────────
function initOCRUpload() {
  const zone = $('#ocr-upload-zone');
  const input = $('#ocr-file-input');

  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      handleOCRFile(e.dataTransfer.files[0]);
    }
  });
  input.addEventListener('change', () => {
    if (input.files.length) handleOCRFile(input.files[0]);
  });
}

async function handleOCRFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Please upload an image file.', 'warning');
    return;
  }

  setProcessing(true, 'ocr');
  try {
    const result = await API.postFile('/api/ocr', file);
    displayOCRResults(result);
    showToast(`Extracted ${result.block_count} text blocks in ${result.inference_ms}ms`, 'success');
  } catch (err) {
    showToast(`OCR failed: ${err.message}`, 'error');
  } finally {
    setProcessing(false, 'ocr');
  }
}

function displayOCRResults(result) {
  // Annotated image
  const imgContainer = $('#ocr-result-image');
  if (imgContainer && result.annotated_image) {
    imgContainer.innerHTML = `<img src="${result.annotated_image}" alt="OCR Results" />`;
  }

  // Extracted text
  const textOutput = $('#ocr-text-output');
  if (textOutput) {
    textOutput.textContent = result.text || 'No text found.';
  }

  // Stats
  updateStat('ocr-blocks', result.block_count);
  updateStat('ocr-confidence', `${(result.avg_confidence * 100).toFixed(1)}%`);
  updateStat('ocr-time', `${result.inference_ms}ms`);
  updateStat('ocr-words', result.analysis?.word_count || 0);

  // Analysis
  const analysisContainer = $('#ocr-analysis');
  if (analysisContainer && result.analysis) {
    const a = result.analysis;
    let html = '';

    if (a.doc_type) {
      html += createAnalysisItem('📄', 'Document Type', a.doc_type);
    }
    if (a.emails?.length) {
      html += createAnalysisItem('📧', 'Emails', a.emails.join(', '));
    }
    if (a.dates?.length) {
      html += createAnalysisItem('📅', 'Dates', a.dates.join(', '));
    }
    if (a.amounts?.length) {
      html += createAnalysisItem('💰', 'Amounts', a.amounts.join(', '));
    }
    if (a.urls?.length) {
      html += createAnalysisItem('🔗', 'URLs', a.urls.join(', '));
    }
    if (a.phones?.length) {
      html += createAnalysisItem('📞', 'Phone Numbers', a.phones.join(', '));
    }

    analysisContainer.innerHTML = html || '<p class="text-muted">No key entities found.</p>';
  }
}

function createAnalysisItem(icon, label, value) {
  return `
    <div class="analysis-item">
      <span style="font-size:18px;">${icon}</span>
      <div>
        <div class="label">${label}</div>
        <div class="value">${value}</div>
      </div>
    </div>`;
}

// ─── Benchmark ─────────────────────────────────────────────────────────────────
async function runBenchmark() {
  setProcessing(true, 'benchmark');
  try {
    const result = await API.postJson('/api/benchmark/yolov8n', {});
    displayBenchmarkResults(result);
    showToast(`Benchmark complete! Avg: ${result.avg_ms}ms, FPS: ${result.fps}`, 'success');
  } catch (err) {
    showToast(`Benchmark failed: ${err.message}`, 'error');
  } finally {
    setProcessing(false, 'benchmark');
  }
}

function displayBenchmarkResults(result) {
  updateStat('bench-provider', result.provider?.replace('ExecutionProvider', '') || 'N/A');
  updateStat('bench-avg', `${result.avg_ms}ms`);
  updateStat('bench-min', `${result.min_ms}ms`);
  updateStat('bench-max', `${result.max_ms}ms`);
  updateStat('bench-p95', `${result.p95_ms}ms`);
  updateStat('bench-fps', result.fps);
  updateStat('bench-iterations', result.iterations);

  // Animate bar chart
  const npuBar = $('#bench-bar-npu');
  const cpuBar = $('#bench-bar-cpu');
  if (npuBar && cpuBar) {
    // If running on NPU, show comparison
    const maxHeight = 180;
    const npuHeight = maxHeight;
    const cpuHeight = Math.min(maxHeight, (result.avg_ms / 50) * maxHeight); // Assume ~50ms is CPU baseline

    if (result.provider === 'QNNExecutionProvider') {
      npuBar.style.height = `${Math.max(20, maxHeight - (result.avg_ms / 50) * maxHeight)}px`;
      cpuBar.style.height = `${maxHeight}px`;
      $('#bench-npu-value').textContent = `${result.avg_ms}ms`;
      $('#bench-cpu-value').textContent = `~${(result.avg_ms * 3).toFixed(0)}ms (est.)`;
    } else {
      npuBar.style.height = '40px';
      cpuBar.style.height = `${Math.min(maxHeight, (result.avg_ms / 100) * maxHeight)}px`;
      $('#bench-npu-value').textContent = 'N/A';
      $('#bench-cpu-value').textContent = `${result.avg_ms}ms`;
    }
  }
}

// ─── Utility ───────────────────────────────────────────────────────────────────
function updateStat(id, value) {
  const el = $(`#stat-${id}`);
  if (el) el.textContent = value;
}

function setProcessing(isProcessing, section) {
  state.isProcessing = isProcessing;
  const overlay = $(`#${section}-loading`);
  if (overlay) {
    overlay.classList.toggle('hidden', !isProcessing);
  }
}

function copyText(elementId) {
  const el = $(`#${elementId}`);
  if (!el) return;
  navigator.clipboard.writeText(el.textContent).then(() => {
    showToast('Copied to clipboard!', 'success', 2000);
  });
}

function downloadResult(imageId) {
  const img = $(`#${imageId} img`);
  if (!img) {
    showToast('No result image to download.', 'warning');
    return;
  }
  const a = document.createElement('a');
  a.href = img.src;
  a.download = `neurolens_result_${Date.now()}.png`;
  a.click();
}

// ─── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  console.log('🧠 NeuroLens AI — Initializing...');

  initTabs();
  initDetectionUpload();
  initEnhancementUpload();
  initOCRUpload();

  // Load voices for TTS
  if ('speechSynthesis' in window) {
    window.speechSynthesis.getVoices();
  }

  // Fetch system status
  fetchSystemStatus();

  // Refresh status periodically
  setInterval(fetchSystemStatus, 30000);

  console.log('✅ NeuroLens AI — Ready!');
});
