"""
NeuroLens AI — Image Enhancer
AI-powered image super-resolution and background removal
optimized for Snapdragon NPU.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import MODELS, MODELS_DIR, QNN_OPTIONS
from utils.npu_manager import npu_manager

logger = logging.getLogger("neurolens.enhancer")


class ImageEnhancer:
    """AI image enhancement with super-resolution and background removal."""

    def __init__(self):
        self.sr_session = None
        self.bg_session = None
        self.sr_loaded: bool = False
        self.bg_loaded: bool = False

    # ── Super-Resolution ──────────────────────────────────────────────────────

    def load_super_resolution(self, model_path: Optional[str] = None) -> bool:
        """Load the Real-ESRGAN super-resolution model."""
        if model_path is None:
            model_path = str(MODELS_DIR / MODELS["real_esrgan_x4"]["filename"])

        if not Path(model_path).exists():
            logger.warning(f"SR model not found: {model_path}")
            return False

        try:
            self.sr_session = npu_manager.create_session(
                model_path, qnn_options=QNN_OPTIONS
            )
            if self.sr_session:
                self.sr_loaded = True
                logger.info("✅ Super-resolution model loaded.")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load SR model: {e}")
            return False

    def upscale(self, image: np.ndarray, tile_size: int = 256) -> tuple[np.ndarray, float]:
        """
        Upscale image using AI super-resolution with tiling for memory efficiency.

        Args:
            image: BGR input image
            tile_size: Tile size for processing (prevents OOM on large images)

        Returns:
            (upscaled_image, inference_time_ms)
        """
        if not self.sr_loaded:
            # Fallback: bicubic upscale
            h, w = image.shape[:2]
            upscaled = cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
            return upscaled, 0.0

        input_meta = self.sr_session.get_inputs()[0]
        input_name = input_meta.name

        h, w = image.shape[:2]

        # For small images, process directly
        if h <= tile_size and w <= tile_size:
            blob = self._preprocess_sr(image)
            start = time.perf_counter()
            output = self.sr_session.run(None, {input_name: blob})
            elapsed = (time.perf_counter() - start) * 1000
            result = self._postprocess_sr(output[0])
            return result, round(elapsed, 2)

        # Tiled processing for larger images
        scale = 4
        output_image = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
        overlap = 16
        total_time = 0.0

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Extract tile
                ye = min(y + tile_size, h)
                xe = min(x + tile_size, w)
                tile = image[y:ye, x:xe]

                blob = self._preprocess_sr(tile)
                start = time.perf_counter()
                output = self.sr_session.run(None, {input_name: blob})
                total_time += (time.perf_counter() - start) * 1000

                result_tile = self._postprocess_sr(output[0])
                th, tw = result_tile.shape[:2]
                output_image[
                    y * scale : y * scale + th,
                    x * scale : x * scale + tw,
                ] = result_tile

        return output_image, round(total_time, 2)

    def _preprocess_sr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for super-resolution model."""
        img = image[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalize
        img = img.transpose(2, 0, 1)  # HWC → CHW
        return np.expand_dims(img, axis=0)

    def _postprocess_sr(self, output: np.ndarray) -> np.ndarray:
        """Post-process super-resolution model output."""
        img = output.squeeze()
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)  # CHW → HWC
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = img[:, :, ::-1]  # RGB → BGR
        return img

    # ── Background Removal ────────────────────────────────────────────────────

    def load_background_removal(self, model_path: Optional[str] = None) -> bool:
        """Load U²-Net background removal model."""
        if model_path is None:
            model_path = str(MODELS_DIR / MODELS["u2net"]["filename"])

        if not Path(model_path).exists():
            logger.warning(f"BG removal model not found: {model_path}")
            return False

        try:
            self.bg_session = npu_manager.create_session(
                model_path, qnn_options=QNN_OPTIONS
            )
            if self.bg_session:
                self.bg_loaded = True
                logger.info("✅ Background removal model loaded.")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load BG removal model: {e}")
            return False

    def remove_background(
        self, image: np.ndarray, return_mask: bool = False
    ) -> tuple[np.ndarray, float]:
        """
        Remove background from image using U²-Net.

        Args:
            image: BGR input image
            return_mask: If True, return the mask instead of the masked image

        Returns:
            (result_image_with_alpha_or_mask, inference_time_ms)
        """
        if not self.bg_loaded:
            logger.warning("BG removal model not loaded.")
            return image, 0.0

        input_meta = self.bg_session.get_inputs()[0]
        input_name = input_meta.name
        target_size = (320, 320)

        h, w = image.shape[:2]
        # Preprocess
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        blob = (blob - mean) / std
        blob = blob.transpose(2, 0, 1).astype(np.float32)
        blob = np.expand_dims(blob, axis=0)

        # Inference
        start = time.perf_counter()
        outputs = self.bg_session.run(None, {input_name: blob})
        elapsed = (time.perf_counter() - start) * 1000

        # Post-process mask
        mask = outputs[0].squeeze()
        if mask.ndim == 3:
            mask = mask[0]  # Take first channel
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        if return_mask:
            return mask, round(elapsed, 2)

        # Apply mask to create transparent background
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask

        return rgba, round(elapsed, 2)

    def replace_background(
        self, image: np.ndarray, background_color: tuple = (255, 255, 255)
    ) -> tuple[np.ndarray, float]:
        """
        Replace image background with a solid color.

        Returns:
            (result_image, inference_time_ms)
        """
        mask, elapsed = self.remove_background(image, return_mask=True)
        if elapsed == 0.0 and not self.bg_loaded:
            return image, 0.0

        # Create background
        bg = np.full_like(image, background_color, dtype=np.uint8)
        mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

        # Blend
        result = (image.astype(np.float32) * mask_3ch + bg.astype(np.float32) * (1 - mask_3ch))
        return result.astype(np.uint8), elapsed


# Global instance
enhancer = ImageEnhancer()
