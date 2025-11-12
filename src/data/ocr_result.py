"""Data structures for OCR results."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .block_types import BlockType, ContentPattern


@dataclass
class BoundingBox:
    """Bounding box coordinates for detected content."""

    x: float
    y: float
    width: float
    height: float
    page: int = 0

    @property
    def x_max(self) -> float:
        """Right edge coordinate."""
        return self.x + self.width

    @property
    def y_max(self) -> float:
        """Bottom edge coordinate."""
        return self.y + self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this bounding box overlaps with another."""
        return not (
            self.x_max < other.x
            or self.x > other.x_max
            or self.y_max < other.y
            or self.y > other.y_max
        )

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another bounding box."""
        if not self.overlaps(other):
            return 0.0

        # Calculate intersection
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x_max, other.x_max)
        y_bottom = min(self.y_max, other.y_max)

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        union_area = self.area + other.area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0


@dataclass
class Block:
    """Individual content block detected in a document."""

    text: str
    block_type: BlockType
    confidence: float
    bbox: BoundingBox
    language: Optional[str] = None
    model_source: Optional[str] = None
    content_pattern: Optional[ContentPattern] = None
    child_blocks: List["Block"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reading_order: Optional[int] = None

    def __post_init__(self):
        """Validate block data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    @property
    def is_high_confidence(self, threshold: float = 0.85) -> bool:
        """Check if block has high confidence."""
        return self.confidence >= threshold

    @property
    def is_low_confidence(self, threshold: float = 0.5) -> bool:
        """Check if block has low confidence."""
        return self.confidence < threshold

    @property
    def has_children(self) -> bool:
        """Check if block has child blocks."""
        return len(self.child_blocks) > 0

    @property
    def is_special_content(self) -> bool:
        """Check if block contains special content (table, formula, etc.)."""
        return self.block_type in {
            BlockType.TABLE,
            BlockType.FORMULA,
            BlockType.CODE,
            BlockType.IMAGE,
        }


@dataclass
class PageResult:
    """OCR results for a single page."""

    page_number: int
    blocks: List[Block]
    width: float
    height: float
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all blocks."""
        if not self.blocks:
            return 0.0
        return sum(block.confidence for block in self.blocks) / len(self.blocks)

    @property
    def text(self) -> str:
        """Get all text from the page in reading order."""
        sorted_blocks = sorted(self.blocks, key=lambda b: b.reading_order or 0)
        return "\n".join(block.text for block in sorted_blocks if block.text)

    @property
    def has_tables(self) -> bool:
        """Check if page contains tables."""
        return any(block.block_type == BlockType.TABLE for block in self.blocks)

    @property
    def has_formulas(self) -> bool:
        """Check if page contains formulas."""
        return any(block.block_type == BlockType.FORMULA for block in self.blocks)

    @property
    def table_count(self) -> int:
        """Count number of tables in page."""
        return sum(1 for block in self.blocks if block.block_type == BlockType.TABLE)

    def get_blocks_by_type(self, block_type: BlockType) -> List[Block]:
        """Get all blocks of a specific type."""
        return [block for block in self.blocks if block.block_type == block_type]


@dataclass
class OCRResult:
    """Complete OCR result for a document."""

    pages: List[PageResult]
    doc_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        """Number of pages in document."""
        return len(self.pages)

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all pages."""
        if not self.pages:
            return 0.0
        confidences = [page.average_confidence for page in self.pages]
        return sum(confidences) / len(confidences)

    @property
    def text(self) -> str:
        """Get all text from the document."""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def total_blocks(self) -> int:
        """Total number of blocks across all pages."""
        return sum(len(page.blocks) for page in self.pages)

    @property
    def has_tables(self) -> bool:
        """Check if document contains tables."""
        return any(page.has_tables for page in self.pages)

    @property
    def has_formulas(self) -> bool:
        """Check if document contains formulas."""
        return any(page.has_formulas for page in self.pages)

    def get_page(self, page_number: int) -> Optional[PageResult]:
        """Get a specific page by number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None


@dataclass
class EnsembleResult:
    """Result from ensemble of multiple OCR models."""

    primary_result: OCRResult
    alternative_results: List[OCRResult] = field(default_factory=list)
    model_sources: Dict[str, str] = field(default_factory=dict)
    confidence_map: Dict[str, float] = field(default_factory=dict)
    merge_strategy: Optional[str] = None
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def all_results(self) -> List[OCRResult]:
        """Get all results including primary."""
        return [self.primary_result] + self.alternative_results

    @property
    def model_count(self) -> int:
        """Number of models used."""
        return len(self.model_sources)
