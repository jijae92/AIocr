"""
Routing and ensemble heuristics for OCR model selection.

This module implements intelligent routing logic to select the optimal OCR model(s)
based on document characteristics, content type, and confidence scores.
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path

from ..data.ocr_result import Block, PageResult, OCRResult, BoundingBox
from ..data.block_types import BlockType, LayoutComplexity, ContentPattern


@dataclass
class RoutingDecision:
    """Decision about which model(s) to use for OCR."""

    primary_model: str
    fallback_models: List[str]
    validation_models: List[str]
    confidence_threshold: float
    reason: str
    content_specific_routing: Dict[BlockType, str]
    use_ensemble: bool = False


@dataclass
class DocumentCharacteristics:
    """Characteristics of a document used for routing decisions."""

    page_count: int
    average_confidence: float
    layout_complexity: LayoutComplexity
    has_tables: bool
    table_count: int
    has_formulas: bool
    has_multi_column: bool
    detected_language: Optional[str]
    content_patterns: Set[ContentPattern]
    numeric_content_ratio: float
    image_quality_score: float


class HeuristicRouter:
    """
    Heuristic-based router for OCR model selection.

    Implements page/block-level routing decisions:
    - DocAI priority: avg_conf >= T_docai, simple tables, language matching
    - Uncertain/complex: avg_conf < T_docai or multi-column/many tables → Donut
    - Numeric/code/date: TrOCR ONNX cross-validation
    - Tables: TATR structure recovery
    - Formulas: pix2tex
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the router with configuration.

        Args:
            config_path: Path to configuration file (default: configs/app.yaml)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "app.yaml"

        self.config = self._load_config(config_path)
        self.thresholds = self.config["thresholds"]
        self.models = self.config["models"]
        self.routing = self.config["routing"]
        self.ensemble = self.config["ensemble"]

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def analyze_document(
        self, image_or_result, language: Optional[str] = None
    ) -> DocumentCharacteristics:
        """
        Analyze document characteristics for routing decisions.

        Args:
            image_or_result: Input image or preliminary OCR result
            language: Detected or specified language

        Returns:
            DocumentCharacteristics object
        """
        # If we have a preliminary result, analyze it
        if isinstance(image_or_result, OCRResult):
            return self._analyze_ocr_result(image_or_result)
        elif isinstance(image_or_result, PageResult):
            return self._analyze_page_result(image_or_result, language)
        else:
            # Analyze raw image
            return self._analyze_image(image_or_result, language)

    def _analyze_ocr_result(self, result: OCRResult) -> DocumentCharacteristics:
        """Analyze an OCR result to determine document characteristics."""
        total_tables = sum(page.table_count for page in result.pages)
        has_formulas = result.has_formulas
        avg_confidence = result.average_confidence

        # Detect layout complexity
        layout_complexity = self._detect_layout_complexity(result)

        # Detect content patterns
        content_patterns = self._detect_content_patterns(result)

        # Calculate numeric content ratio
        numeric_ratio = self._calculate_numeric_content_ratio(result)

        # Check for multi-column layout
        has_multi_column = self._detect_multi_column_layout(result)

        # Extract detected language
        detected_language = self._extract_primary_language(result)

        return DocumentCharacteristics(
            page_count=result.page_count,
            average_confidence=avg_confidence,
            layout_complexity=layout_complexity,
            has_tables=result.has_tables,
            table_count=total_tables,
            has_formulas=has_formulas,
            has_multi_column=has_multi_column,
            detected_language=detected_language,
            content_patterns=content_patterns,
            numeric_content_ratio=numeric_ratio,
            image_quality_score=0.8,  # Placeholder
        )

    def _analyze_page_result(
        self, page: PageResult, language: Optional[str]
    ) -> DocumentCharacteristics:
        """Analyze a single page result."""
        content_patterns = set()
        numeric_blocks = 0
        total_chars = 0

        for block in page.blocks:
            if block.content_pattern:
                content_patterns.add(block.content_pattern)
            if self._is_numeric_content(block.text):
                numeric_blocks += 1
            total_chars += len(block.text)

        numeric_ratio = numeric_blocks / len(page.blocks) if page.blocks else 0.0

        # Check for multi-column
        has_multi_column = self._detect_multi_column_page(page)

        return DocumentCharacteristics(
            page_count=1,
            average_confidence=page.average_confidence,
            layout_complexity=self._classify_page_complexity(page),
            has_tables=page.has_tables,
            table_count=page.table_count,
            has_formulas=page.has_formulas,
            has_multi_column=has_multi_column,
            detected_language=language or page.language,
            content_patterns=content_patterns,
            numeric_content_ratio=numeric_ratio,
            image_quality_score=0.8,  # Placeholder
        )

    def _analyze_image(self, image, language: Optional[str]) -> DocumentCharacteristics:
        """Analyze a raw image (placeholder for future implementation)."""
        # This would involve pre-processing analysis
        # For now, return conservative defaults
        return DocumentCharacteristics(
            page_count=1,
            average_confidence=0.0,
            layout_complexity=LayoutComplexity.MODERATE,
            has_tables=False,
            table_count=0,
            has_formulas=False,
            has_multi_column=False,
            detected_language=language,
            content_patterns=set(),
            numeric_content_ratio=0.0,
            image_quality_score=0.8,
        )

    def route(
        self,
        characteristics: DocumentCharacteristics,
        preliminary_result: Optional[OCRResult] = None,
    ) -> RoutingDecision:
        """
        Make routing decision based on document characteristics.

        Decision logic:
        1. DocAI priority: high confidence, simple layout, supported language
        2. Donut fallback: low confidence, complex layout, many tables
        3. Content-specific routing: tables→TATR, formulas→pix2tex, numeric→TrOCR

        Args:
            characteristics: Document characteristics
            preliminary_result: Optional preliminary OCR result (e.g., from DocAI)

        Returns:
            RoutingDecision object
        """
        # Extract thresholds
        t_docai = self.thresholds["docai_confidence"]
        t_low = self.thresholds["low_confidence"]
        t_high = self.thresholds["high_confidence"]

        # Check if DocAI is suitable
        docai_suitable = self._is_docai_suitable(characteristics)

        # Determine primary model
        if docai_suitable and characteristics.average_confidence >= t_docai:
            # DocAI priority path
            primary_model = "docai"
            fallback_models = ["donut"]
            validation_models = []
            reason = "High confidence DocAI result with suitable document characteristics"
            use_ensemble = False

        elif characteristics.layout_complexity == LayoutComplexity.COMPLEX or \
             characteristics.has_multi_column or \
             characteristics.table_count > self.thresholds.get("moderate_max_tables", 5):
            # Complex layout → Donut
            primary_model = "donut"
            fallback_models = ["docai", "trocr"]
            validation_models = []
            reason = "Complex layout or multi-column detected, using Donut for reinterpretation"
            use_ensemble = self.ensemble["enabled"]

        elif characteristics.average_confidence < t_low:
            # Low confidence → ensemble approach
            primary_model = "donut"
            fallback_models = ["docai", "trocr"]
            validation_models = ["trocr"]
            reason = "Low confidence score, using ensemble approach"
            use_ensemble = True

        elif characteristics.numeric_content_ratio > self.thresholds.get("numeric_content_ratio", 0.7):
            # High numeric content → TrOCR for validation
            primary_model = "docai"
            fallback_models = ["donut"]
            validation_models = ["trocr"]
            reason = "High numeric content ratio, using TrOCR for cross-validation"
            use_ensemble = True

        else:
            # Default: DocAI with possible fallback
            primary_model = "docai"
            fallback_models = ["donut"]
            validation_models = []
            reason = "Standard document, using DocAI as primary"
            use_ensemble = False

        # Content-specific routing
        content_routing = self._build_content_routing(characteristics)

        return RoutingDecision(
            primary_model=primary_model,
            fallback_models=fallback_models,
            validation_models=validation_models,
            confidence_threshold=t_docai,
            reason=reason,
            content_specific_routing=content_routing,
            use_ensemble=use_ensemble,
        )

    def route_block(
        self, block: Block, page_characteristics: DocumentCharacteristics
    ) -> str:
        """
        Route a single block to the appropriate model.

        Block-level routing:
        - Tables → TATR
        - Formulas → pix2tex
        - Numeric/code/dates → TrOCR ONNX
        - General text → based on page-level decision

        Args:
            block: Block to route
            page_characteristics: Characteristics of the page

        Returns:
            Model name to use for this block
        """
        # Special content routing
        if block.block_type == BlockType.TABLE:
            return self.routing["content_routing"]["tables"]["primary"]

        if block.block_type == BlockType.FORMULA:
            return self.routing["content_routing"]["formulas"]["primary"]

        if block.block_type in {BlockType.NUMBER, BlockType.CODE, BlockType.DATE}:
            return self.routing["content_routing"].get("numeric", {}).get("primary", "trocr")

        # Pattern-based routing
        if block.content_pattern == ContentPattern.NUMERIC:
            return self.routing["content_routing"].get("numeric", {}).get("primary", "trocr")

        if block.content_pattern == ContentPattern.CODE:
            return self.routing["content_routing"].get("code", {}).get("primary", "trocr")

        if block.content_pattern == ContentPattern.DATE:
            return self.routing["content_routing"].get("dates", {}).get("primary", "trocr")

        # Default to page-level decision
        page_decision = self.route(page_characteristics)
        return page_decision.primary_model

    def should_use_ensemble(self, characteristics: DocumentCharacteristics) -> bool:
        """
        Determine if ensemble mode should be used.

        Ensemble is used when:
        - Enabled in config
        - Low confidence detected
        - Complex layout
        - High numeric content (needs validation)

        Args:
            characteristics: Document characteristics

        Returns:
            True if ensemble should be used
        """
        if not self.ensemble["enabled"]:
            return False

        # Use ensemble for uncertain cases
        if characteristics.average_confidence < self.thresholds["low_confidence"]:
            return True

        # Use ensemble for complex layouts
        if characteristics.layout_complexity == LayoutComplexity.COMPLEX:
            return True

        # Use ensemble for high numeric content (cross-validation)
        if characteristics.numeric_content_ratio > self.thresholds.get("numeric_content_ratio", 0.7):
            return True

        return False

    def get_validation_models(self, block: Block) -> List[str]:
        """
        Get models to use for validation/cross-checking of a block.

        Args:
            block: Block to validate

        Returns:
            List of model names for validation
        """
        validation_models = []

        # Numeric content validation
        if self._is_numeric_content(block.text) or block.block_type == BlockType.NUMBER:
            validation_models.append("trocr")
            if "docai" != block.model_source:
                validation_models.append("docai")

        # Date validation
        if block.block_type == BlockType.DATE or self._is_date_pattern(block.text):
            validation_models.append("trocr")

        # Table structure validation
        if block.block_type == BlockType.TABLE:
            validation_models.append("donut")

        return validation_models

    def _is_docai_suitable(self, characteristics: DocumentCharacteristics) -> bool:
        """
        Check if DocAI is suitable for the document.

        DocAI is suitable when:
        - Language is supported
        - Layout is simple to moderate
        - Table count is reasonable

        Args:
            characteristics: Document characteristics

        Returns:
            True if DocAI is suitable
        """
        # Check language support
        supported_languages = self.config["language"]["docai_languages"]
        if characteristics.detected_language and \
           characteristics.detected_language not in supported_languages:
            return False

        # Check layout complexity
        if characteristics.layout_complexity == LayoutComplexity.COMPLEX:
            return False

        # Check table count
        if characteristics.table_count > self.thresholds.get("simple_table_cells", 50):
            return False

        return True

    def _build_content_routing(
        self, characteristics: DocumentCharacteristics
    ) -> Dict[BlockType, str]:
        """Build content-specific routing map."""
        routing_map = {}

        # Tables
        if characteristics.has_tables:
            routing_map[BlockType.TABLE] = self.routing["content_routing"]["tables"]["primary"]

        # Formulas
        if characteristics.has_formulas:
            routing_map[BlockType.FORMULA] = self.routing["content_routing"]["formulas"]["primary"]

        # Numeric/code/dates
        if ContentPattern.NUMERIC in characteristics.content_patterns:
            routing_map[BlockType.NUMBER] = self.routing["content_routing"]["numeric"]["primary"]

        if ContentPattern.CODE in characteristics.content_patterns:
            routing_map[BlockType.CODE] = self.routing["content_routing"]["code"]["primary"]

        if ContentPattern.DATE in characteristics.content_patterns:
            routing_map[BlockType.DATE] = self.routing["content_routing"]["dates"]["primary"]

        return routing_map

    # Helper methods for content analysis

    def _detect_layout_complexity(self, result: OCRResult) -> LayoutComplexity:
        """Detect document layout complexity."""
        avg_tables_per_page = sum(p.table_count for p in result.pages) / max(result.page_count, 1)

        # Check for multi-column layouts
        has_multi_column = any(self._detect_multi_column_page(page) for page in result.pages)

        if avg_tables_per_page > self.thresholds.get("moderate_max_tables", 5) or has_multi_column:
            return LayoutComplexity.COMPLEX
        elif avg_tables_per_page > self.thresholds.get("simple_max_tables", 2):
            return LayoutComplexity.MODERATE
        else:
            return LayoutComplexity.SIMPLE

    def _classify_page_complexity(self, page: PageResult) -> LayoutComplexity:
        """Classify single page complexity."""
        if page.table_count > self.thresholds.get("moderate_max_tables", 5):
            return LayoutComplexity.COMPLEX
        elif page.table_count > self.thresholds.get("simple_max_tables", 2):
            return LayoutComplexity.MODERATE
        else:
            return LayoutComplexity.SIMPLE

    def _detect_content_patterns(self, result: OCRResult) -> Set[ContentPattern]:
        """Detect content patterns in the document."""
        patterns = set()

        for page in result.pages:
            for block in page.blocks:
                if self._is_numeric_content(block.text):
                    patterns.add(ContentPattern.NUMERIC)
                if self._is_date_pattern(block.text):
                    patterns.add(ContentPattern.DATE)
                if self._is_code_pattern(block.text):
                    patterns.add(ContentPattern.CODE)
                if block.block_type == BlockType.FORMULA:
                    patterns.add(ContentPattern.FORMULA)

        return patterns

    def _calculate_numeric_content_ratio(self, result: OCRResult) -> float:
        """Calculate ratio of numeric content in document."""
        total_blocks = 0
        numeric_blocks = 0

        for page in result.pages:
            for block in page.blocks:
                total_blocks += 1
                if self._is_numeric_content(block.text):
                    numeric_blocks += 1

        return numeric_blocks / total_blocks if total_blocks > 0 else 0.0

    def _detect_multi_column_layout(self, result: OCRResult) -> bool:
        """Detect if document has multi-column layout."""
        return any(self._detect_multi_column_page(page) for page in result.pages)

    def _detect_multi_column_page(self, page: PageResult) -> bool:
        """Detect if a page has multi-column layout."""
        if not page.blocks:
            return False

        # Group blocks by vertical position
        blocks_sorted = sorted(page.blocks, key=lambda b: b.bbox.y)

        # Simple heuristic: check for significant horizontal gaps
        # This is a simplified version; production would be more sophisticated
        x_positions = [block.bbox.x for block in blocks_sorted]
        if len(set(x_positions)) > self.thresholds.get("multi_column_threshold", 2):
            return True

        return False

    def _extract_primary_language(self, result: OCRResult) -> Optional[str]:
        """Extract primary language from OCR result."""
        # Count language occurrences
        language_counts = {}

        for page in result.pages:
            for block in page.blocks:
                if block.language:
                    language_counts[block.language] = language_counts.get(block.language, 0) + 1

        if not language_counts:
            return None

        # Return most common language
        return max(language_counts, key=language_counts.get)

    @staticmethod
    def _is_numeric_content(text: str) -> bool:
        """Check if text is primarily numeric."""
        if not text:
            return False

        # Remove common punctuation
        cleaned = re.sub(r'[,.\s$€¥£₩%]', '', text)

        # Check if mostly digits
        digit_ratio = sum(c.isdigit() for c in cleaned) / len(cleaned) if cleaned else 0
        return digit_ratio > 0.7

    @staticmethod
    def _is_date_pattern(text: str) -> bool:
        """Check if text matches common date patterns."""
        date_patterns = [
            r'\d{4}[-/]\d{2}[-/]\d{2}',  # YYYY-MM-DD
            r'\d{2}[-/]\d{2}[-/]\d{4}',  # DD-MM-YYYY
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # Flexible date
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',  # Month DD, YYYY
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns)

    @staticmethod
    def _is_code_pattern(text: str) -> bool:
        """Check if text looks like code or IDs."""
        code_indicators = [
            r'[A-Z]{2,}\d{2,}',  # Pattern like AB1234
            r'\d{3,}-\d{3,}',  # Pattern like 123-456
            r'[A-Z0-9]{8,}',  # Long alphanumeric sequences
            r'(def|class|function|var|const|import)\s+\w+',  # Code keywords
        ]

        return any(re.search(pattern, text) for pattern in code_indicators)
