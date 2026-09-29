"""
NeuroLens AI — OCR Engine
Optical Character Recognition with intelligent document analysis.
Uses EasyOCR with Snapdragon-optimized processing pipeline.
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("neurolens.ocr")


class OCREngine:
    """OCR with document intelligence and text analysis."""

    def __init__(self):
        self.reader = None
        self.loaded: bool = False

    def load(self, languages: list[str] = None) -> bool:
        """Load the OCR engine."""
        if languages is None:
            languages = ["en"]
        try:
            import easyocr
            self.reader = easyocr.Reader(languages, gpu=True)
            self.loaded = True
            logger.info(f"✅ OCR engine loaded with languages: {languages}")
            return True
        except ImportError:
            logger.error("EasyOCR not installed. Install with: pip install easyocr")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load OCR: {e}")
            return False

    def extract_text(
        self, image: np.ndarray, detail: bool = True
    ) -> tuple[dict, float]:
        """
        Extract text from an image.

        Args:
            image: BGR image
            detail: If True, return detailed results with bounding boxes

        Returns:
            (result_dict, inference_time_ms)
        """
        if not self.loaded:
            return {"text": "", "blocks": [], "error": "OCR not loaded"}, 0.0

        start = time.perf_counter()

        # Convert BGR → RGB for EasyOCR
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb)

        elapsed = (time.perf_counter() - start) * 1000

        # Process results
        blocks = []
        full_text_parts = []

        for bbox, text, conf in results:
            # Convert bbox to simple format
            pts = np.array(bbox, dtype=int)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)

            block = {
                "text": text,
                "confidence": round(float(conf), 3),
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
            }
            blocks.append(block)
            full_text_parts.append(text)

        full_text = " ".join(full_text_parts)

        result = {
            "text": full_text,
            "blocks": blocks if detail else [],
            "block_count": len(blocks),
            "avg_confidence": round(
                sum(b["confidence"] for b in blocks) / len(blocks), 3
            )
            if blocks
            else 0,
            "analysis": self._analyze_text(full_text),
        }

        return result, round(elapsed, 2)

    def _analyze_text(self, text: str) -> dict:
        """
        Analyze extracted text to identify key information.

        Returns:
            Dictionary of identified patterns (emails, phones, dates, amounts, etc.)
        """
        analysis = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "emails": re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text),
            "phones": re.findall(r"[\+]?[\d\s\-\(\)]{7,15}", text),
            "dates": re.findall(
                r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}",
                text,
            ),
            "amounts": re.findall(
                r"(?:₹|\$|€|£|USD|INR|Rs\.?)\s*[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:₹|\$|€|£|USD|INR|Rs)",
                text,
            ),
            "urls": re.findall(r"https?://[^\s]+|www\.[^\s]+", text),
        }

        # Determine document type heuristic
        text_lower = text.lower()
        if any(w in text_lower for w in ["invoice", "bill", "receipt", "total", "amount due"]):
            analysis["doc_type"] = "invoice/receipt"
        elif any(w in text_lower for w in ["resume", "cv", "experience", "education", "skills"]):
            analysis["doc_type"] = "resume/cv"
        elif any(w in text_lower for w in ["agreement", "contract", "terms", "hereby"]):
            analysis["doc_type"] = "legal document"
        elif any(w in text_lower for w in ["dear", "sincerely", "regards", "subject:"]):
            analysis["doc_type"] = "letter/email"
        else:
            analysis["doc_type"] = "general text"

        return analysis

    def draw_ocr_results(
        self, image: np.ndarray, blocks: list[dict]
    ) -> np.ndarray:
        """Draw OCR bounding boxes and text on image."""
        annotated = image.copy()

        for block in blocks:
            x1, y1, x2, y2 = block["bbox"]
            conf = block["confidence"]

            # Color based on confidence
            if conf > 0.8:
                color = (78, 205, 196)  # Teal for high
            elif conf > 0.5:
                color = (255, 159, 67)  # Orange for medium
            else:
                color = (255, 107, 107)  # Red for low

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            text = f"{block['text'][:30]} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, text, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )

        return annotated


# Global instance
ocr_engine = OCREngine()
