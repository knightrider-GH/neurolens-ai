"""
NeuroLens AI — Object Detector
YOLOv8-based real-time object detection optimized for Snapdragon NPU.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    COCO_CLASSES,
    CONFIDENCE_THRESHOLD,
    MAX_DETECTIONS,
    MODELS,
    MODELS_DIR,
    NMS_THRESHOLD,
    QNN_OPTIONS,
)
from utils.npu_manager import npu_manager

logger = logging.getLogger("neurolens.detector")


class ObjectDetector:
    """YOLOv8 object detection with Snapdragon NPU acceleration."""

    def __init__(self):
        self.session = None
        self.input_name: str = ""
        self.input_shape: tuple = (1, 3, 640, 640)
        self.loaded: bool = False
        self.class_names: list = COCO_CLASSES
        self.conf_threshold: float = CONFIDENCE_THRESHOLD
        self.nms_threshold: float = NMS_THRESHOLD

    def load(self, model_path: Optional[str] = None) -> bool:
        """Load the YOLOv8 ONNX model."""
        if model_path is None:
            model_path = str(MODELS_DIR / MODELS["yolov8n"]["filename"])

        if not Path(model_path).exists():
            logger.warning(f"Model not found: {model_path}. Run model download first.")
            return False

        try:
            self.session = npu_manager.create_session(
                model_path, qnn_options=QNN_OPTIONS
            )
            if self.session is None:
                return False

            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.input_shape = input_meta.shape
            self.loaded = True
            logger.info("✅ YOLOv8 detector loaded successfully.")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load detector: {e}")
            return False

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple]:
        """
        Preprocess image for YOLOv8 inference.

        Returns:
            (preprocessed_tensor, original_shape)
        """
        original_shape = image.shape[:2]  # (h, w)
        target_h, target_w = 640, 640

        # Letterbox resize
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # Normalize and transpose: HWC → CHW, BGR → RGB, scale to [0,1]
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        return blob, (scale, pad_w, pad_h, original_shape)

    def postprocess(
        self, outputs: np.ndarray, preprocess_info: tuple
    ) -> list[dict]:
        """
        Post-process YOLOv8 outputs to detection results.

        Returns:
            List of detection dicts with keys: class, confidence, bbox, label
        """
        scale, pad_w, pad_h, (orig_h, orig_w) = preprocess_info

        # YOLOv8 output shape: (1, 84, 8400) → transpose to (8400, 84)
        predictions = outputs[0].squeeze()
        if predictions.shape[0] == 84:
            predictions = predictions.T

        detections = []

        # Extract boxes and class scores
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        # Get best class for each detection
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes_xywh) == 0:
            return []

        # Convert xywh → xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Remove padding and rescale to original image
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale

        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        indices = indices.flatten()[:MAX_DETECTIONS]

        for i in indices:
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            cls_id = int(class_ids[i])
            conf = float(confidences[i])
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

            detections.append({
                "class_id": cls_id,
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })

        return detections

    def detect(self, image: np.ndarray) -> tuple[list[dict], float]:
        """
        Run object detection on an image.

        Args:
            image: BGR image as numpy array

        Returns:
            (detections_list, inference_time_ms)
        """
        if not self.loaded:
            logger.warning("Detector not loaded.")
            return [], 0.0

        blob, preprocess_info = self.preprocess(image)

        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})
        inference_ms = (time.perf_counter() - start) * 1000

        detections = self.postprocess(outputs, preprocess_info)

        return detections, round(inference_ms, 2)

    def draw_detections(
        self,
        image: np.ndarray,
        detections: list[dict],
        show_labels: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        annotated = image.copy()

        # Color palette (vibrant, distinguishable)
        colors = [
            (255, 107, 107), (78, 205, 196), (69, 183, 209), (150, 111, 214),
            (255, 159, 67), (46, 213, 115), (116, 185, 255), (223, 228, 234),
            (255, 71, 87), (55, 66, 250), (255, 195, 18), (196, 69, 105),
        ]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            color = colors[cls_id % len(colors)]
            conf = det["confidence"]
            label = det["label"]

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                text = f"{label} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                # Label background
                cv2.rectangle(
                    annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

        return annotated


# Global instance
detector = ObjectDetector()
