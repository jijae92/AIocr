"""Block type definitions for OCR results."""

from enum import Enum


class BlockType(str, Enum):
    """Types of content blocks detected in documents."""

    TEXT_BLOCK = "text_block"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    FORMULA = "formula"
    IMAGE = "image"
    LIST = "list"
    LIST_ITEM = "list_item"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    CODE = "code"
    DATE = "date"
    NUMBER = "number"
    SIGNATURE = "signature"
    BARCODE = "barcode"
    QR_CODE = "qr_code"
    UNKNOWN = "unknown"


class LayoutComplexity(str, Enum):
    """Document layout complexity levels."""

    SIMPLE = "simple"  # Single column, minimal structure
    MODERATE = "moderate"  # Multiple columns or some tables
    COMPLEX = "complex"  # Multi-column with many tables/formulas


class ContentPattern(str, Enum):
    """Content pattern types for specialized recognition."""

    NUMERIC = "numeric"  # Numbers, currencies, prices
    ALPHANUMERIC = "alphanumeric"  # Mixed text and numbers
    DATE = "date"  # Date patterns
    TIME = "time"  # Time patterns
    CODE = "code"  # Programming code, IDs
    FORMULA = "formula"  # Mathematical formulas
    MIXED = "mixed"  # Mixed content
