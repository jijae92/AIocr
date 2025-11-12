"""
Coordinate transformation utilities for PDF and image processing.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BoundingBox:
    """Bounding box representation."""

    x: float
    y: float
    width: float
    height: float
    page: int = 0
    confidence: float = 1.0

    @property
    def x2(self) -> float:
        """Right edge x-coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> float:
        """Bottom edge y-coordinate."""
        return self.y + self.height

    @property
    def area(self) -> float:
        """Bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Center point (x, y)."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'page': self.page,
            'confidence': self.confidence,
        }

    def to_corners(self) -> Tuple[float, float, float, float]:
        """Return as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x2, self.y2)

    @classmethod
    def from_corners(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        page: int = 0,
        confidence: float = 1.0,
    ) -> 'BoundingBox':
        """Create from corner coordinates."""
        return cls(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            page=page,
            confidence=confidence,
        )

    @classmethod
    def from_normalized(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
        page: int = 0,
        confidence: float = 1.0,
    ) -> 'BoundingBox':
        """Create from normalized coordinates (0-1)."""
        return cls(
            x=x * image_width,
            y=y * image_height,
            width=width * image_width,
            height=height * image_height,
            page=page,
            confidence=confidence,
        )

    def to_normalized(
        self,
        image_width: int,
        image_height: int,
    ) -> 'BoundingBox':
        """Convert to normalized coordinates (0-1)."""
        return BoundingBox(
            x=self.x / image_width,
            y=self.y / image_height,
            width=self.width / image_width,
            height=self.height / image_height,
            page=self.page,
            confidence=self.confidence,
        )


@dataclass
class TextBlock:
    """Text block with bounding box and content."""

    text: str
    bbox: BoundingBox
    confidence: float = 1.0
    block_type: str = 'text'  # text, table, image, etc.

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'bbox': self.bbox.to_dict(),
            'confidence': self.confidence,
            'block_type': self.block_type,
        }


def pdf_to_image_coords(
    x: float,
    y: float,
    pdf_width: float,
    pdf_height: float,
    image_width: int,
    image_height: int,
    pdf_origin: str = 'bottom-left',
) -> Tuple[int, int]:
    """
    Convert PDF coordinates to image pixel coordinates.

    Args:
        x: PDF x-coordinate
        y: PDF y-coordinate
        pdf_width: PDF page width in points
        pdf_height: PDF page height in points
        image_width: Image width in pixels
        image_height: Image height in pixels
        pdf_origin: PDF coordinate origin ('bottom-left' or 'top-left')

    Returns:
        Tuple of (image_x, image_y) in pixels
    """
    # Scale factor
    scale_x = image_width / pdf_width
    scale_y = image_height / pdf_height

    # Convert x
    image_x = int(x * scale_x)

    # Convert y (PDF typically has origin at bottom-left, images at top-left)
    if pdf_origin == 'bottom-left':
        image_y = int((pdf_height - y) * scale_y)
    else:
        image_y = int(y * scale_y)

    return (image_x, image_y)


def image_to_pdf_coords(
    x: int,
    y: int,
    pdf_width: float,
    pdf_height: float,
    image_width: int,
    image_height: int,
    pdf_origin: str = 'bottom-left',
) -> Tuple[float, float]:
    """
    Convert image pixel coordinates to PDF coordinates.

    Args:
        x: Image x-coordinate in pixels
        y: Image y-coordinate in pixels
        pdf_width: PDF page width in points
        pdf_height: PDF page height in points
        image_width: Image width in pixels
        image_height: Image height in pixels
        pdf_origin: PDF coordinate origin ('bottom-left' or 'top-left')

    Returns:
        Tuple of (pdf_x, pdf_y) in points
    """
    # Scale factor
    scale_x = pdf_width / image_width
    scale_y = pdf_height / image_height

    # Convert x
    pdf_x = x * scale_x

    # Convert y
    if pdf_origin == 'bottom-left':
        pdf_y = pdf_height - (y * scale_y)
    else:
        pdf_y = y * scale_y

    return (pdf_x, pdf_y)


def rotate_bbox(
    bbox: BoundingBox,
    angle: float,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """
    Rotate bounding box.

    Args:
        bbox: Original bounding box
        angle: Rotation angle in degrees (90, 180, 270)
        image_width: Image width
        image_height: Image height

    Returns:
        Rotated bounding box
    """
    if angle == 0:
        return bbox

    # Get corners
    x1, y1, x2, y2 = bbox.to_corners()

    if angle == 90:
        # Rotate 90 degrees clockwise
        new_x1 = y1
        new_y1 = image_width - x2
        new_x2 = y2
        new_y2 = image_width - x1
        return BoundingBox.from_corners(
            new_x1, new_y1, new_x2, new_y2, bbox.page, bbox.confidence
        )

    elif angle == 180:
        # Rotate 180 degrees
        new_x1 = image_width - x2
        new_y1 = image_height - y2
        new_x2 = image_width - x1
        new_y2 = image_height - y1
        return BoundingBox.from_corners(
            new_x1, new_y1, new_x2, new_y2, bbox.page, bbox.confidence
        )

    elif angle == 270:
        # Rotate 270 degrees clockwise (90 counter-clockwise)
        new_x1 = image_height - y2
        new_y1 = x1
        new_x2 = image_height - y1
        new_y2 = x2
        return BoundingBox.from_corners(
            new_x1, new_y1, new_x2, new_y2, bbox.page, bbox.confidence
        )

    else:
        raise ValueError(f"Unsupported rotation angle: {angle}. Use 0, 90, 180, or 270.")


def compute_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        IoU value (0-1)
    """
    # Get corners
    x1_1, y1_1, x2_1, y2_1 = bbox1.to_corners()
    x1_2, y1_2, x2_2, y2_2 = bbox2.to_corners()

    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    area1 = bbox1.area
    area2 = bbox2.area
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def merge_bboxes(bboxes: List[BoundingBox]) -> Optional[BoundingBox]:
    """
    Merge multiple bounding boxes into one.

    Args:
        bboxes: List of bounding boxes

    Returns:
        Merged bounding box or None if list is empty
    """
    if not bboxes:
        return None

    # Get all corners
    corners = [bbox.to_corners() for bbox in bboxes]

    # Find min/max
    x1 = min(c[0] for c in corners)
    y1 = min(c[1] for c in corners)
    x2 = max(c[2] for c in corners)
    y2 = max(c[3] for c in corners)

    # Use page and confidence from first bbox
    return BoundingBox.from_corners(
        x1, y1, x2, y2, bboxes[0].page, bboxes[0].confidence
    )


def sort_bboxes_reading_order(
    bboxes: List[BoundingBox],
    horizontal_threshold: float = 10.0,
) -> List[BoundingBox]:
    """
    Sort bounding boxes in reading order (top-to-bottom, left-to-right).

    Args:
        bboxes: List of bounding boxes
        horizontal_threshold: Threshold for considering boxes on same line

    Returns:
        Sorted list of bounding boxes
    """
    if not bboxes:
        return []

    # Sort by y-coordinate first
    sorted_boxes = sorted(bboxes, key=lambda b: b.y)

    # Group boxes by lines
    lines = []
    current_line = [sorted_boxes[0]]

    for bbox in sorted_boxes[1:]:
        # Check if on same line
        if abs(bbox.y - current_line[0].y) < horizontal_threshold:
            current_line.append(bbox)
        else:
            # Sort current line by x-coordinate
            lines.append(sorted(current_line, key=lambda b: b.x))
            current_line = [bbox]

    # Add last line
    if current_line:
        lines.append(sorted(current_line, key=lambda b: b.x))

    # Flatten
    return [bbox for line in lines for bbox in line]
