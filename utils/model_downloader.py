"""
NeuroLens AI — Model Downloader
Handles downloading and caching of ONNX models with progress tracking.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import requests
from tqdm import tqdm

from config import MODELS, MODELS_DIR

logger = logging.getLogger("neurolens.downloader")


class ModelDownloader:
    """Downloads and manages AI model files."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_key: str) -> Optional[Path]:
        """Get the local path for a model, or None if not downloaded."""
        if model_key not in MODELS:
            logger.error(f"Unknown model key: {model_key}")
            return None
        model_info = MODELS[model_key]
        path = self.models_dir / model_info["filename"]
        return path if path.exists() else None

    def is_downloaded(self, model_key: str) -> bool:
        """Check if a model is already downloaded."""
        return self.get_model_path(model_key) is not None

    def download_model(
        self,
        model_key: str,
        progress_callback: Optional[Callable] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Download a model file with progress tracking.

        Args:
            model_key: Key from MODELS config
            progress_callback: Optional callback(downloaded, total) for progress
            force: Force re-download even if file exists

        Returns:
            Path to downloaded model or None on failure
        """
        if model_key not in MODELS:
            logger.error(f"Unknown model: {model_key}")
            return None

        model_info = MODELS[model_key]
        target_path = self.models_dir / model_info["filename"]

        if target_path.exists() and not force:
            logger.info(f"Model already exists: {target_path}")
            return target_path

        url = model_info["url"]
        logger.info(f"Downloading {model_info['name']} from {url}")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            # Download with progress
            tmp_path = target_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_size)

            # Move temp file to final location
            tmp_path.rename(target_path)
            size_mb = target_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"✅ Downloaded {model_info['name']} ({size_mb:.1f} MB)"
            )
            return target_path

        except requests.RequestException as e:
            logger.error(f"❌ Download failed for {model_info['name']}: {e}")
            # Cleanup temp file
            tmp_path = target_path.with_suffix(".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error downloading {model_info['name']}: {e}")
            return None

    def download_all(self, progress_callback: Optional[Callable] = None) -> dict:
        """Download all configured models. Returns dict of {model_key: path_or_none}."""
        results = {}
        for key in MODELS:
            results[key] = self.download_model(key, progress_callback)
        return results

    def get_status(self) -> dict:
        """Get download status for all models."""
        status = {}
        for key, info in MODELS.items():
            path = self.models_dir / info["filename"]
            status[key] = {
                "name": info["name"],
                "description": info["description"],
                "downloaded": path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1)
                if path.exists()
                else None,
                "filename": info["filename"],
                "task": info["task"],
            }
        return status


# Global singleton
model_downloader = ModelDownloader()
