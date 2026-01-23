"""
Debug Panel Module

Displays intermediate processing stages for debugging and visualization.
Shows a grid of images at each step of the face recognition pipeline.
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class DebugImages:
    """Container for intermediate processing images."""

    original: Optional[np.ndarray] = None
    quality_overlay: Optional[np.ndarray] = None
    face_detection: Optional[np.ndarray] = None
    face_crop: Optional[np.ndarray] = None
    face_aligned: Optional[np.ndarray] = None
    face_padded: Optional[np.ndarray] = None
    landmarks_overlay: Optional[np.ndarray] = None
    embedding_input: Optional[np.ndarray] = None
    match_result: Optional[np.ndarray] = None
    extra: Dict[str, np.ndarray] = field(default_factory=dict)


class DebugPanel:
    """
    Debug visualization panel that shows intermediate processing stages.

    Creates a grid display of images from each pipeline step.
    """

    def __init__(
        self,
        panel_width: int = 1280,
        panel_height: int = 720,
        grid_cols: int = 3,
        grid_rows: int = 3,
        window_name: str = "Debug Panel",
    ):
        """
        Initialize debug panel.

        Args:
            panel_width: Total panel width in pixels
            panel_height: Total panel height in pixels
            grid_cols: Number of columns in the grid
            grid_rows: Number of rows in the grid
            window_name: Name of the debug window
        """
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.window_name = window_name

        # Calculate cell dimensions
        self.cell_width = panel_width // grid_cols
        self.cell_height = panel_height // grid_rows

        # Current debug images
        self.images = DebugImages()

        # Labels for each cell
        self.labels = [
            "Original",
            "Quality Check",
            "Face Detection",
            "Face Crop",
            "Aligned",
            "Padded/Resized",
            "Landmarks",
            "Embedding Input",
            "Match Result",
        ]

        self.enabled = False

    def enable(self):
        """Enable the debug panel."""
        self.enabled = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.panel_width, self.panel_height)

    def disable(self):
        """Disable the debug panel."""
        self.enabled = False
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass

    def toggle(self):
        """Toggle debug panel on/off."""
        if self.enabled:
            self.disable()
        else:
            self.enable()
        return self.enabled

    def clear(self):
        """Clear all debug images."""
        self.images = DebugImages()

    def set_original(self, frame: np.ndarray):
        """Set the original frame."""
        self.images.original = frame.copy()

    def set_quality_overlay(
        self,
        frame: np.ndarray,
        brightness: float,
        blur: float,
        is_acceptable: bool,
        issues: List[str],
    ):
        """Set quality check visualization."""
        img = frame.copy()

        # Draw quality metrics
        color = (0, 255, 0) if is_acceptable else (0, 0, 255)
        status = "PASS" if is_acceptable else "FAIL"

        cv2.putText(
            img, f"Quality: {status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        cv2.putText(
            img,
            f"Brightness: {brightness:.1f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            f"Blur: {blur:.1f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Draw issues
        for i, issue in enumerate(issues[:3]):
            cv2.putText(
                img,
                f"- {issue}",
                (10, 95 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )

        self.images.quality_overlay = img

    def set_face_detection(
        self,
        frame: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]],
        largest_idx: int = 0,
    ):
        """Set face detection visualization."""
        img = frame.copy()

        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Largest face in green, others in yellow
            if i == largest_idx:
                color = (0, 255, 0)
                thickness = 2
                label = "LARGEST"
            else:
                color = (0, 255, 255)
                thickness = 1
                label = f"Face {i+1}"

            cv2.rectangle(img, (left, top), (right, bottom), color, thickness)
            cv2.putText(
                img,
                label,
                (left, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        cv2.putText(
            img,
            f"Detected: {len(face_locations)} faces",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        self.images.face_detection = img

    def set_face_crop(self, face_crop: np.ndarray, label: str = ""):
        """Set the cropped face image."""
        img = face_crop.copy()
        if label:
            cv2.putText(
                img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        self.images.face_crop = img

    def set_face_aligned(self, aligned_face: np.ndarray, angle: float = 0.0):
        """Set the aligned face image."""
        img = aligned_face.copy()
        cv2.putText(
            img,
            f"Angle: {angle:.1f}°",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        self.images.face_aligned = img

    def set_face_padded(self, padded_face: np.ndarray, mode: str = ""):
        """Set the padded/resized face image."""
        img = padded_face.copy()
        if mode:
            cv2.putText(
                img,
                f"Mode: {mode}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        self.images.face_padded = img

    def set_landmarks(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        landmarks: Optional[dict],
    ):
        """Set landmarks visualization."""
        img = frame.copy()
        top, right, bottom, left = face_location

        # Draw face box
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)

        if landmarks:
            # Define colors for different facial features
            feature_colors = {
                "chin": (255, 0, 0),
                "left_eyebrow": (0, 255, 0),
                "right_eyebrow": (0, 255, 0),
                "nose_bridge": (255, 255, 0),
                "nose_tip": (255, 255, 0),
                "left_eye": (0, 255, 255),
                "right_eye": (0, 255, 255),
                "top_lip": (255, 0, 255),
                "bottom_lip": (255, 0, 255),
            }

            total_points = 0
            for feature, points in landmarks.items():
                color = feature_colors.get(feature, (255, 255, 255))
                for x, y in points:
                    cv2.circle(img, (x, y), 2, color, -1)
                    total_points += 1

            cv2.putText(
                img,
                f"Landmarks: {total_points}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                img,
                "No landmarks",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        self.images.landmarks_overlay = img

    def set_embedding_input(self, embedding_input: np.ndarray, size: int = 0):
        """Set the final image used for embedding extraction."""
        img = embedding_input.copy()
        if size > 0:
            cv2.putText(
                img,
                f"{size}x{size}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        self.images.embedding_input = img

    def set_match_result(
        self,
        frame: np.ndarray,
        face_location: Optional[Tuple[int, int, int, int]],
        name: Optional[str],
        distance: Optional[float],
        matched: bool,
    ):
        """Set match result visualization."""
        img = frame.copy()

        if face_location:
            top, right, bottom, left = face_location
            color = (0, 255, 0) if matched else (0, 0, 255)
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)

        if matched and name:
            cv2.putText(
                img,
                f"MATCH: {name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            if distance is not None:
                cv2.putText(
                    img,
                    f"Distance: {distance:.3f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
        else:
            cv2.putText(
                img,
                "NO MATCH",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        self.images.match_result = img

    def set_extra(self, name: str, image: np.ndarray):
        """Set an extra debug image."""
        self.images.extra[name] = image.copy()

    def _resize_to_cell(self, img: np.ndarray) -> np.ndarray:
        """Resize image to fit in a grid cell."""
        if img is None:
            # Return black placeholder
            return np.zeros((self.cell_height - 30, self.cell_width, 3), dtype=np.uint8)

        # Handle grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        target_h = self.cell_height - 30  # Leave room for label
        target_w = self.cell_width

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center in cell
        cell = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        cell[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return cell

    def _create_labeled_cell(self, img: np.ndarray, label: str) -> np.ndarray:
        """Create a cell with label bar at top."""
        cell_img = self._resize_to_cell(img)

        # Create label bar
        label_bar = np.zeros((30, self.cell_width, 3), dtype=np.uint8)
        label_bar[:] = (40, 40, 40)  # Dark gray background

        cv2.putText(
            label_bar,
            label,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Combine label and image
        return np.vstack([label_bar, cell_img])

    def render(self) -> np.ndarray:
        """Render the complete debug panel."""
        # Collect all images in order (3x3 grid = 9 cells)
        images_list = [
            self.images.original,
            self.images.quality_overlay,
            self.images.face_detection,
            self.images.face_crop,
            self.images.face_aligned,
            self.images.face_padded,
            self.images.landmarks_overlay,
            self.images.embedding_input,
            self.images.match_result,
        ]

        # Create grid
        rows = []
        for row in range(self.grid_rows):
            row_cells = []
            for col in range(self.grid_cols):
                idx = row * self.grid_cols + col
                if idx < len(images_list):
                    img = images_list[idx]
                    label = self.labels[idx] if idx < len(self.labels) else f"Image {idx}"
                else:
                    img = None
                    label = "Empty"

                cell = self._create_labeled_cell(img, label)
                row_cells.append(cell)

            rows.append(np.hstack(row_cells))

        panel = np.vstack(rows)
        return panel

    def show(self):
        """Display the debug panel."""
        if not self.enabled:
            return

        panel = self.render()
        cv2.imshow(self.window_name, panel)

    def update(self):
        """Update the display (call in main loop)."""
        if self.enabled:
            self.show()


# Global debug panel instance
_debug_panel: Optional[DebugPanel] = None


def get_debug_panel() -> DebugPanel:
    """Get the global debug panel instance."""
    global _debug_panel
    if _debug_panel is None:
        _debug_panel = DebugPanel()
    return _debug_panel


def debug_enabled() -> bool:
    """Check if debug panel is enabled."""
    return _debug_panel is not None and _debug_panel.enabled
