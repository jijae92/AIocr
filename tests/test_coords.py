"""
Tests for coordinate utilities.
"""

import pytest

from util.coords import (
    BoundingBox,
    compute_iou,
    image_to_pdf_coords,
    merge_bboxes,
    pdf_to_image_coords,
    sort_bboxes_reading_order,
)


def test_bounding_box_creation():
    """Test BoundingBox creation."""
    bbox = BoundingBox(x=10, y=20, width=100, height=50)

    assert bbox.x == 10
    assert bbox.y == 20
    assert bbox.width == 100
    assert bbox.height == 50
    assert bbox.x2 == 110
    assert bbox.y2 == 70
    assert bbox.area == 5000
    assert bbox.center == (60, 45)


def test_bounding_box_from_corners():
    """Test creating BoundingBox from corners."""
    bbox = BoundingBox.from_corners(10, 20, 110, 70)

    assert bbox.x == 10
    assert bbox.y == 20
    assert bbox.width == 100
    assert bbox.height == 50


def test_pdf_to_image_coords():
    """Test PDF to image coordinate conversion."""
    # PDF: 612x792 points, Image: 2550x3300 pixels @ 300 DPI
    x, y = pdf_to_image_coords(
        x=100, y=100, pdf_width=612, pdf_height=792, image_width=2550, image_height=3300
    )

    assert x == pytest.approx(416, abs=1)
    # Y is flipped: (792 - 100) * (3300/792) = 2883
    assert y == pytest.approx(2883, abs=1)


def test_image_to_pdf_coords():
    """Test image to PDF coordinate conversion."""
    x, y = image_to_pdf_coords(
        x=416, y=2883, pdf_width=612, pdf_height=792, image_width=2550, image_height=3300
    )

    assert x == pytest.approx(100, abs=1)
    assert y == pytest.approx(100, abs=1)


def test_compute_iou():
    """Test IoU computation."""
    bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
    bbox2 = BoundingBox(x=50, y=50, width=100, height=100)

    iou = compute_iou(bbox1, bbox2)

    # Intersection: 50x50 = 2500
    # Union: 10000 + 10000 - 2500 = 17500
    # IoU: 2500 / 17500 = 0.1428
    assert iou == pytest.approx(0.1428, abs=0.01)


def test_merge_bboxes():
    """Test merging multiple bounding boxes."""
    bboxes = [
        BoundingBox(x=10, y=10, width=50, height=30),
        BoundingBox(x=40, y=20, width=60, height=40),
        BoundingBox(x=20, y=50, width=40, height=20),
    ]

    merged = merge_bboxes(bboxes)

    assert merged.x == 10
    assert merged.y == 10
    assert merged.x2 == 100
    assert merged.y2 == 70


def test_sort_bboxes_reading_order():
    """Test sorting bounding boxes in reading order."""
    # Create boxes in random order
    bboxes = [
        BoundingBox(x=200, y=10, width=50, height=20),  # Top right
        BoundingBox(x=10, y=10, width=50, height=20),  # Top left
        BoundingBox(x=10, y=50, width=50, height=20),  # Bottom left
        BoundingBox(x=200, y=50, width=50, height=20),  # Bottom right
    ]

    sorted_boxes = sort_bboxes_reading_order(bboxes)

    # Should be: top-left, top-right, bottom-left, bottom-right
    assert sorted_boxes[0].x == 10 and sorted_boxes[0].y == 10
    assert sorted_boxes[1].x == 200 and sorted_boxes[1].y == 10
    assert sorted_boxes[2].x == 10 and sorted_boxes[2].y == 50
    assert sorted_boxes[3].x == 200 and sorted_boxes[3].y == 50
