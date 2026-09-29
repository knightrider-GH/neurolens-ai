"""
NeuroLens AI — NPU Manager
Handles Snapdragon NPU detection, ONNX Runtime session creation,
and execution provider management with automatic fallback.
"""

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("neurolens.npu")


class NPUManager:
    """Manages ONNX Runtime sessions with Snapdragon NPU (QNN) acceleration."""

    def __init__(self):
        self.available_providers: list[str] = []
        self.active_provider: str = "CPUExecutionProvider"
        self.npu_available: bool = False
        self.gpu_available: bool = False
        self._detect_hardware()

    def _detect_hardware(self):
        """Detect available execution providers."""
        try:
            import onnxruntime as ort

            self.available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {self.available_providers}")

            if "QNNExecutionProvider" in self.available_providers:
                self.npu_available = True
                self.active_provider = "QNNExecutionProvider"
                logger.info("✅ Snapdragon NPU (QNN) detected and available!")
            elif "DmlExecutionProvider" in self.available_providers:
                self.gpu_available = True
                self.active_provider = "DmlExecutionProvider"
                logger.info("✅ DirectML GPU acceleration available.")
            else:
                self.active_provider = "CPUExecutionProvider"
                logger.info("ℹ️  Running on CPU — NPU/GPU not detected.")

        except ImportError:
            logger.warning("ONNX Runtime not installed. Running in demo mode.")
            self.available_providers = []
            self.active_provider = "CPUExecutionProvider"

    def create_session(
        self,
        model_path: str,
        provider: Optional[str] = None,
        qnn_options: Optional[dict] = None,
    ):
        """
        Create an ONNX Runtime inference session with optimal provider.

        Args:
            model_path: Path to the ONNX model file
            provider: Force a specific provider (None = auto-select best)
            qnn_options: QNN-specific session options

        Returns:
            ort.InferenceSession or None if creation fails
        """
        import onnxruntime as ort

        target_provider = provider or self.active_provider
        providers_to_try = [target_provider, "CPUExecutionProvider"]
        # Remove duplicates while preserving order
        providers_to_try = list(dict.fromkeys(providers_to_try))

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True

        provider_options = []
        if target_provider == "QNNExecutionProvider" and qnn_options:
            provider_options = [qnn_options if p == "QNNExecutionProvider" else {} for p in providers_to_try]
        else:
            provider_options = [{} for _ in providers_to_try]

        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers_to_try,
                provider_options=provider_options,
            )
            actual_provider = session.get_providers()[0]
            logger.info(
                f"✅ Model loaded: {model_path} → Provider: {actual_provider}"
            )
            return session
        except Exception as e:
            logger.error(f"❌ Failed to create session: {e}")
            # Final fallback to CPU only
            try:
                session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
                logger.info(f"⚠️  Fallback to CPU for: {model_path}")
                return session
            except Exception as e2:
                logger.error(f"❌ CPU fallback also failed: {e2}")
                return None

    def benchmark_inference(
        self,
        session,
        input_data: dict,
        iterations: int = 50,
        warmup: int = 5,
    ) -> dict:
        """
        Benchmark model inference performance.

        Returns:
            Dictionary with avg_ms, min_ms, max_ms, fps, provider
        """
        # Warmup
        for _ in range(warmup):
            session.run(None, input_data)

        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            session.run(None, input_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_ms = np.mean(latencies)
        return {
            "provider": session.get_providers()[0],
            "avg_ms": round(avg_ms, 2),
            "min_ms": round(np.min(latencies), 2),
            "max_ms": round(np.max(latencies), 2),
            "p95_ms": round(np.percentile(latencies, 95), 2),
            "fps": round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0,
            "iterations": iterations,
        }

    def get_hardware_info(self) -> dict:
        """Get current hardware/provider status."""
        import platform
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "npu_available": self.npu_available,
            "gpu_available": self.gpu_available,
            "active_provider": self.active_provider,
            "all_providers": self.available_providers,
        }

        # Try to get more detailed info
        try:
            import psutil
            info["cpu_count"] = psutil.cpu_count()
            info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            pass

        return info


# Global singleton
npu_manager = NPUManager()
