"""
Tests for searchable PDF text layer generation.
"""

import pytest

from pdf.text_layer import SearchablePDFGenerator
from util.coords import BoundingBox, TextBlock


def test_searchable_pdf_generator_init():
    """Test SearchablePDFGenerator initialization."""
    generator = SearchablePDFGenerator(text_opacity=0)
    assert generator.text_opacity == 0

    generator = SearchablePDFGenerator(text_opacity=128)
    assert generator.text_opacity == 128


def test_text_block_creation():
    """Test TextBlock creation."""
    bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.95)
    block = TextBlock(text="Hello World", bbox=bbox, confidence=0.95)

    assert block.text == "Hello World"
    assert block.bbox == bbox
    assert block.confidence == 0.95


def test_text_block_to_dict():
    """Test TextBlock to_dict conversion."""
    bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.95)
    block = TextBlock(text="Test", bbox=bbox, confidence=0.90)

    data = block.to_dict()

    assert data['text'] == "Test"
    assert data['confidence'] == 0.90
    assert 'bbox' in data
    assert data['bbox']['x'] == 10
    assert data['bbox']['y'] == 20
