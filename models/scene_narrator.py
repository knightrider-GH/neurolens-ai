"""
NeuroLens AI — Scene Narrator
Generates natural language descriptions of visual scenes for accessibility.
Combines object detection results with spatial reasoning to create
meaningful scene narratives.
"""

import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger("neurolens.narrator")


class SceneNarrator:
    """
    Converts object detection results into natural language scene descriptions.
    Designed for accessibility — helping visually impaired users understand their
    surroundings through audio narration.
    """

    def __init__(self):
        self.last_description: str = ""
        self.description_history: list[str] = []
        self.max_history: int = 10

    def narrate(
        self,
        detections: list[dict],
        image_width: int,
        image_height: int,
        verbose: bool = False,
    ) -> str:
        """
        Generate a natural language description from detection results.

        Args:
            detections: List of detection dicts from ObjectDetector
            image_width: Width of the source image
            image_height: Height of the source image
            verbose: If True, include more spatial detail

        Returns:
            Human-readable scene description string
        """
        if not detections:
            return "The scene appears clear with no recognizable objects detected."

        # Count objects
        object_counts = Counter(d["label"] for d in detections)
        total_objects = len(detections)

        # Build description parts
        parts = []

        # Opening summary
        unique_count = len(object_counts)
        if total_objects == 1:
            label = detections[0]["label"]
            parts.append(f"I can see {self._article(label)} {label}")
        elif unique_count == 1:
            label = list(object_counts.keys())[0]
            count = object_counts[label]
            parts.append(f"I can see {count} {self._pluralize(label, count)}")
        else:
            parts.append(f"I can see {total_objects} objects in the scene")

        # Detail each object type
        if unique_count > 1 or verbose:
            object_descriptions = []
            for label, count in object_counts.most_common():
                if count == 1:
                    object_descriptions.append(f"{self._article(label)} {label}")
                else:
                    object_descriptions.append(
                        f"{count} {self._pluralize(label, count)}"
                    )

            if len(object_descriptions) <= 2:
                detail = " and ".join(object_descriptions)
            else:
                detail = (
                    ", ".join(object_descriptions[:-1])
                    + f", and {object_descriptions[-1]}"
                )
            parts.append(f"including {detail}")

        # Spatial information
        if verbose and len(detections) > 0:
            spatial_info = self._describe_spatial(
                detections, image_width, image_height
            )
            if spatial_info:
                parts.append(spatial_info)

        # People-specific narration
        people_count = object_counts.get("person", 0)
        if people_count > 0:
            if people_count == 1:
                person_det = next(d for d in detections if d["label"] == "person")
                pos = self._get_position(
                    person_det["bbox"], image_width, image_height
                )
                parts.append(f"There is a person {pos}")
            elif people_count <= 3:
                parts.append(f"There are {people_count} people visible")
            else:
                parts.append(f"The scene is quite busy with {people_count} people")

        description = ". ".join(parts) + "."

        # Track history for change detection
        self.last_description = description
        self.description_history.append(description)
        if len(self.description_history) > self.max_history:
            self.description_history.pop(0)

        return description

    def get_scene_changes(
        self,
        current_detections: list[dict],
        previous_detections: list[dict],
    ) -> Optional[str]:
        """
        Describe what changed between two detection frames.
        Useful for alerting users to new objects entering/leaving the scene.
        """
        current_objects = Counter(d["label"] for d in current_detections)
        previous_objects = Counter(d["label"] for d in previous_detections)

        appeared = {}
        disappeared = {}

        all_labels = set(list(current_objects.keys()) + list(previous_objects.keys()))

        for label in all_labels:
            curr = current_objects.get(label, 0)
            prev = previous_objects.get(label, 0)
            if curr > prev:
                appeared[label] = curr - prev
            elif prev > curr:
                disappeared[label] = prev - curr

        changes = []
        for label, count in appeared.items():
            if count == 1:
                changes.append(f"{self._article(label)} {label} appeared")
            else:
                changes.append(f"{count} {self._pluralize(label, count)} appeared")

        for label, count in disappeared.items():
            if count == 1:
                changes.append(f"{self._article(label)} {label} left the scene")
            else:
                changes.append(
                    f"{count} {self._pluralize(label, count)} left the scene"
                )

        if not changes:
            return None

        return "Scene update: " + "; ".join(changes) + "."

    def _get_position(
        self, bbox: list[int], img_w: int, img_h: int
    ) -> str:
        """Get human-readable position description."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Horizontal
        if cx < img_w * 0.33:
            h_pos = "on the left"
        elif cx > img_w * 0.66:
            h_pos = "on the right"
        else:
            h_pos = "in the center"

        # Vertical
        if cy < img_h * 0.33:
            v_pos = "near the top"
        elif cy > img_h * 0.66:
            v_pos = "near the bottom"
        else:
            v_pos = ""

        return f"{h_pos} {v_pos}".strip()

    def _describe_spatial(
        self, detections: list[dict], img_w: int, img_h: int
    ) -> str:
        """Generate spatial relationship description."""
        positions = {"left": [], "center": [], "right": []}

        for det in detections:
            cx = (det["bbox"][0] + det["bbox"][2]) / 2
            if cx < img_w * 0.33:
                positions["left"].append(det["label"])
            elif cx > img_w * 0.66:
                positions["right"].append(det["label"])
            else:
                positions["center"].append(det["label"])

        parts = []
        for pos, labels in positions.items():
            if labels:
                counts = Counter(labels)
                items = [
                    f"{c} {self._pluralize(l, c)}" if c > 1 else l
                    for l, c in counts.items()
                ]
                parts.append(f"{', '.join(items)} {'is' if len(labels)==1 else 'are'} on the {pos}")

        return "; ".join(parts) if parts else ""

    @staticmethod
    def _article(word: str) -> str:
        """Get the correct indefinite article."""
        return "an" if word[0].lower() in "aeiou" else "a"

    @staticmethod
    def _pluralize(word: str, count: int) -> str:
        """Simple English pluralization."""
        if count == 1:
            return word
        if word.endswith(("s", "sh", "ch", "x", "z")):
            return word + "es"
        if word.endswith("y") and word[-2] not in "aeiou":
            return word[:-1] + "ies"
        return word + "s"


# Global instance
narrator = SceneNarrator()
