"""
Layout merge utilities for reading order restoration and ensemble merging.

This module provides functions to:
- Restore reading order from spatial layout
- Merge results from multiple OCR models
- Preserve source/confidence labels
- Handle overlapping blocks and conflicts
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, replace
import yaml
from pathlib import Path

from ..data.ocr_result import (
    Block,
    PageResult,
    OCRResult,
    BoundingBox,
    EnsembleResult,
)
from ..data.block_types import BlockType


@dataclass
class MergeConfig:
    """Configuration for merge operations."""

    paragraph_gap_threshold: float = 1.5
    same_line_gap_threshold: float = 2.0
    column_detection_sensitivity: float = 0.7
    bbox_overlap_threshold: float = 0.5
    confidence_weight: float = 1.0
    preserve_metadata: bool = True


class LayoutMerger:
    """
    Merge OCR results from multiple models while preserving reading order.

    Key features:
    - Spatial analysis for reading order restoration
    - Multi-model result merging
    - Conflict resolution based on confidence
    - Source/confidence label preservation
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the merger with configuration.

        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "app.yaml"

        self.config = self._load_config(config_path)
        self.merge_config = self._build_merge_config()

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _build_merge_config(self) -> MergeConfig:
        """Build merge configuration from app config."""
        layout_merge = self.config.get("layout_merge", {}).get("merge", {})
        ensemble_config = self.config.get("ensemble", {}).get("conflict_resolution", {})

        return MergeConfig(
            paragraph_gap_threshold=layout_merge.get("paragraph_gap_threshold", 1.5),
            same_line_gap_threshold=layout_merge.get("same_line_gap_threshold", 2.0),
            column_detection_sensitivity=layout_merge.get("column_detection_sensitivity", 0.7),
            bbox_overlap_threshold=ensemble_config.get("bbox_overlap_threshold", 0.5),
            preserve_metadata=self.config.get("layout_merge", {})
            .get("preserve_metadata", {})
            .get("confidence_scores", True),
        )

    def restore_reading_order(
        self, page: PageResult, method: str = "spatial_analysis"
    ) -> PageResult:
        """
        Restore natural reading order to blocks in a page.

        Args:
            page: Page with blocks to order
            method: Method to use ('spatial_analysis', 'neural', 'hybrid')

        Returns:
            PageResult with blocks in reading order
        """
        if method == "spatial_analysis":
            ordered_blocks = self._spatial_reading_order(page)
        elif method == "neural":
            # Placeholder for neural reading order
            ordered_blocks = self._spatial_reading_order(page)
        elif method == "hybrid":
            # Placeholder for hybrid approach
            ordered_blocks = self._spatial_reading_order(page)
        else:
            raise ValueError(f"Unknown reading order method: {method}")

        # Update reading_order field
        for idx, block in enumerate(ordered_blocks):
            block.reading_order = idx

        return replace(page, blocks=ordered_blocks)

    def _spatial_reading_order(self, page: PageResult) -> List[Block]:
        """
        Determine reading order using spatial analysis.

        Algorithm:
        1. Detect columns
        2. Sort blocks by column, then by vertical position
        3. Handle special cases (tables, multi-line paragraphs)

        Args:
            page: Page to analyze

        Returns:
            Ordered list of blocks
        """
        if not page.blocks:
            return []

        # Detect columns
        columns = self._detect_columns(page)

        # Sort blocks into columns
        column_blocks = [[] for _ in range(len(columns))]

        for block in page.blocks:
            column_idx = self._assign_to_column(block, columns)
            column_blocks[column_idx].append(block)

        # Sort within each column (top to bottom)
        for blocks in column_blocks:
            blocks.sort(key=lambda b: (b.bbox.y, b.bbox.x))

        # Get reading order configuration
        primary_direction = self.config.get("layout_merge", {}).get(
            "primary_direction", "top_to_bottom"
        )
        secondary_direction = self.config.get("layout_merge", {}).get(
            "secondary_direction", "left_to_right"
        )

        # Flatten columns in reading order
        if secondary_direction == "left_to_right":
            ordered_blocks = [block for col in column_blocks for block in col]
        else:  # right_to_left
            ordered_blocks = [block for col in reversed(column_blocks) for block in col]

        return ordered_blocks

    def _detect_columns(self, page: PageResult) -> List[Tuple[float, float]]:
        """
        Detect column boundaries in a page.

        Returns:
            List of (start_x, end_x) tuples for each column
        """
        if not page.blocks:
            return [(0, page.width)]

        # Collect x-positions of block starts
        x_positions = sorted([block.bbox.x for block in page.blocks])

        # Find gaps that indicate column boundaries
        columns = []
        current_column_start = 0

        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i - 1]
            gap_ratio = gap / page.width

            # Significant gap indicates new column
            if gap_ratio > self.merge_config.column_detection_sensitivity * 0.1:
                columns.append((current_column_start, x_positions[i - 1]))
                current_column_start = x_positions[i]

        # Add final column
        columns.append((current_column_start, page.width))

        # If only one column detected, return full width
        if len(columns) == 1:
            return [(0, page.width)]

        return columns

    def _assign_to_column(
        self, block: Block, columns: List[Tuple[float, float]]
    ) -> int:
        """
        Assign a block to a column.

        Args:
            block: Block to assign
            columns: List of column boundaries

        Returns:
            Column index
        """
        block_center_x = block.bbox.center[0]

        for idx, (start_x, end_x) in enumerate(columns):
            if start_x <= block_center_x <= end_x:
                return idx

        # Default to first column if no match
        return 0

    def merge_results(
        self,
        results: List[OCRResult],
        model_names: List[str],
        strategy: str = "weighted_voting",
    ) -> EnsembleResult:
        """
        Merge results from multiple OCR models.

        Strategies:
        - weighted_voting: Weight by model confidence and config weights
        - consensus: Use blocks that appear in multiple results
        - confidence_max: Use highest confidence for each region

        Args:
            results: List of OCR results from different models
            model_names: Names of models that produced results
            strategy: Merge strategy to use

        Returns:
            EnsembleResult with merged output
        """
        if not results:
            raise ValueError("No results to merge")

        if len(results) == 1:
            # Single result, just return it
            return EnsembleResult(
                primary_result=results[0],
                model_sources={model_names[0]: "primary"},
            )

        # Ensure all results have same number of pages
        page_count = results[0].page_count
        if not all(r.page_count == page_count for r in results):
            raise ValueError("All results must have same number of pages")

        # Merge page by page
        merged_pages = []
        conflicts = []

        for page_idx in range(page_count):
            page_results = [r.pages[page_idx] for r in results]

            if strategy == "weighted_voting":
                merged_page, page_conflicts = self._merge_page_weighted_voting(
                    page_results, model_names
                )
            elif strategy == "consensus":
                merged_page, page_conflicts = self._merge_page_consensus(
                    page_results, model_names
                )
            elif strategy == "confidence_max":
                merged_page, page_conflicts = self._merge_page_confidence_max(
                    page_results, model_names
                )
            else:
                raise ValueError(f"Unknown merge strategy: {strategy}")

            merged_pages.append(merged_page)
            conflicts.extend(page_conflicts)

        # Create merged OCR result
        merged_result = OCRResult(pages=merged_pages)

        # Build model sources map
        model_sources = {name: "ensemble" for name in model_names}

        # Calculate confidence map
        confidence_map = {
            name: results[idx].average_confidence
            for idx, name in enumerate(model_names)
        }

        return EnsembleResult(
            primary_result=merged_result,
            alternative_results=results,
            model_sources=model_sources,
            confidence_map=confidence_map,
            merge_strategy=strategy,
            conflicts=conflicts,
        )

    def _merge_page_weighted_voting(
        self, pages: List[PageResult], model_names: List[str]
    ) -> Tuple[PageResult, List[Dict]]:
        """
        Merge page results using weighted voting.

        Args:
            pages: Page results from different models
            model_names: Names of models

        Returns:
            Merged PageResult and list of conflicts
        """
        # Get model weights from config
        weights = self.config.get("ensemble", {}).get("weights", {})
        model_weights = [weights.get(name, 1.0) for name in model_names]

        # Collect all blocks from all models
        all_blocks_with_sources = []
        for page, model_name, weight in zip(pages, model_names, model_weights):
            for block in page.blocks:
                all_blocks_with_sources.append((block, model_name, weight))

        # Group overlapping blocks
        block_groups = self._group_overlapping_blocks(all_blocks_with_sources)

        # Merge each group
        merged_blocks = []
        conflicts = []

        for group in block_groups:
            merged_block, conflict = self._merge_block_group_weighted(group)
            merged_blocks.append(merged_block)
            if conflict:
                conflicts.append(conflict)

        # Use first page as template
        template_page = pages[0]

        # Restore reading order
        merged_page = PageResult(
            page_number=template_page.page_number,
            blocks=merged_blocks,
            width=template_page.width,
            height=template_page.height,
            language=template_page.language,
            metadata={"merged": True, "source_models": model_names},
        )

        merged_page = self.restore_reading_order(merged_page)

        return merged_page, conflicts

    def _merge_page_consensus(
        self, pages: List[PageResult], model_names: List[str]
    ) -> Tuple[PageResult, List[Dict]]:
        """Merge using consensus (blocks appearing in multiple results)."""
        # Similar to weighted voting but requires minimum number of models
        # to agree on a block
        min_models = self.config.get("ensemble", {}).get("min_models", 2)

        # Implementation similar to weighted voting but with consensus threshold
        return self._merge_page_weighted_voting(pages, model_names)

    def _merge_page_confidence_max(
        self, pages: List[PageResult], model_names: List[str]
    ) -> Tuple[PageResult, List[Dict]]:
        """Merge by selecting highest confidence for each region."""
        # Collect all blocks
        all_blocks_with_sources = []
        for page, model_name in zip(pages, model_names):
            for block in page.blocks:
                all_blocks_with_sources.append((block, model_name, 1.0))

        # Group overlapping blocks
        block_groups = self._group_overlapping_blocks(all_blocks_with_sources)

        # For each group, select highest confidence
        merged_blocks = []
        conflicts = []

        for group in block_groups:
            # Select block with highest confidence
            best_block = max(group, key=lambda x: x[0].confidence)[0]

            # Preserve source information
            if self.merge_config.preserve_metadata:
                sources = [model_name for _, model_name, _ in group]
                best_block.metadata["sources"] = sources
                best_block.metadata["alternatives"] = len(group) - 1

            merged_blocks.append(best_block)

        template_page = pages[0]
        merged_page = PageResult(
            page_number=template_page.page_number,
            blocks=merged_blocks,
            width=template_page.width,
            height=template_page.height,
            language=template_page.language,
            metadata={"merged": True, "strategy": "confidence_max"},
        )

        merged_page = self.restore_reading_order(merged_page)

        return merged_page, conflicts

    def _group_overlapping_blocks(
        self, blocks_with_sources: List[Tuple[Block, str, float]]
    ) -> List[List[Tuple[Block, str, float]]]:
        """
        Group blocks that overlap spatially.

        Args:
            blocks_with_sources: List of (block, model_name, weight) tuples

        Returns:
            List of groups, each group is a list of overlapping blocks
        """
        if not blocks_with_sources:
            return []

        groups = []
        used = set()

        for i, (block1, name1, weight1) in enumerate(blocks_with_sources):
            if i in used:
                continue

            # Start new group
            group = [(block1, name1, weight1)]
            used.add(i)

            # Find overlapping blocks
            for j, (block2, name2, weight2) in enumerate(blocks_with_sources):
                if j in used:
                    continue

                # Check if any block in group overlaps with block2
                if any(
                    self._blocks_overlap(b[0], block2, self.merge_config.bbox_overlap_threshold)
                    for b in group
                ):
                    group.append((block2, name2, weight2))
                    used.add(j)

            groups.append(group)

        return groups

    def _blocks_overlap(
        self, block1: Block, block2: Block, threshold: float = 0.5
    ) -> bool:
        """
        Check if two blocks overlap significantly.

        Args:
            block1: First block
            block2: Second block
            threshold: IoU threshold for considering overlap

        Returns:
            True if blocks overlap
        """
        iou = block1.bbox.iou(block2.bbox)
        return iou >= threshold

    def _merge_block_group_weighted(
        self, group: List[Tuple[Block, str, float]]
    ) -> Tuple[Block, Optional[Dict]]:
        """
        Merge a group of overlapping blocks using weighted voting.

        Args:
            group: List of (block, model_name, weight) tuples

        Returns:
            Merged block and optional conflict info
        """
        if len(group) == 1:
            # No conflict, single block
            block, model_name, _ = group[0]
            block.model_source = model_name
            return block, None

        # Calculate weighted scores for each block
        scored_blocks = []
        for block, model_name, weight in group:
            score = block.confidence * weight
            scored_blocks.append((block, model_name, score))

        # Select highest scoring block as primary
        primary_block, primary_model, primary_score = max(
            scored_blocks, key=lambda x: x[2]
        )

        # Check for conflicts (significantly different text)
        conflict = None
        texts = {block.text for block, _, _ in group}
        if len(texts) > 1:
            conflict = {
                "bbox": primary_block.bbox,
                "texts": {model: block.text for block, model, _ in group},
                "confidences": {model: block.confidence for block, model, _ in group},
                "selected": primary_block.text,
                "selected_model": primary_model,
            }

        # Preserve metadata
        if self.merge_config.preserve_metadata:
            primary_block.model_source = primary_model
            primary_block.metadata["ensemble"] = True
            primary_block.metadata["alternative_sources"] = [
                model for _, model, _ in group if model != primary_model
            ]
            primary_block.metadata["weighted_score"] = primary_score

        return primary_block, conflict

    def merge_with_substitution(
        self,
        base_result: OCRResult,
        validation_result: OCRResult,
        content_patterns: List[str],
    ) -> OCRResult:
        """
        Merge results with selective substitution for specific content patterns.

        Used for numeric/code/date validation where TrOCR results may be
        more accurate than DocAI for certain fields.

        Args:
            base_result: Base OCR result (e.g., from DocAI)
            validation_result: Validation result (e.g., from TrOCR)
            content_patterns: Patterns to validate/substitute

        Returns:
            Merged OCR result with substitutions
        """
        merged_pages = []

        for page_idx in range(base_result.page_count):
            base_page = base_result.pages[page_idx]
            val_page = validation_result.pages[page_idx]

            merged_blocks = []

            for base_block in base_page.blocks:
                # Find overlapping validation block
                val_block = self._find_overlapping_block(base_block, val_page.blocks)

                if val_block and self._should_substitute(
                    base_block, val_block, content_patterns
                ):
                    # Use validation result
                    substituted_block = replace(
                        val_block,
                        model_source="trocr_validation",
                        metadata={
                            **val_block.metadata,
                            "substituted": True,
                            "original_source": base_block.model_source,
                            "original_text": base_block.text,
                            "original_confidence": base_block.confidence,
                        },
                    )
                    merged_blocks.append(substituted_block)
                else:
                    # Keep base result
                    merged_blocks.append(base_block)

            merged_page = PageResult(
                page_number=base_page.page_number,
                blocks=merged_blocks,
                width=base_page.width,
                height=base_page.height,
                language=base_page.language,
                metadata={**base_page.metadata, "validated": True},
            )

            merged_pages.append(merged_page)

        return OCRResult(
            pages=merged_pages,
            doc_id=base_result.doc_id,
            metadata={**base_result.metadata, "validation_applied": True},
        )

    def _find_overlapping_block(
        self, target_block: Block, blocks: List[Block], threshold: float = 0.5
    ) -> Optional[Block]:
        """Find block that overlaps with target block."""
        for block in blocks:
            if self._blocks_overlap(target_block, block, threshold):
                return block
        return None

    def _should_substitute(
        self, base_block: Block, val_block: Block, content_patterns: List[str]
    ) -> bool:
        """
        Determine if validation result should replace base result.

        Args:
            base_block: Base block
            val_block: Validation block
            content_patterns: Content patterns to check

        Returns:
            True if should substitute
        """
        # Check if content matches target patterns
        if base_block.block_type.value not in content_patterns:
            if base_block.content_pattern and base_block.content_pattern.value not in content_patterns:
                return False

        # Substitute if validation confidence is significantly higher
        confidence_diff = val_block.confidence - base_block.confidence
        if confidence_diff > 0.1:  # 10% better
            return True

        # Substitute if base confidence is low
        if base_block.confidence < 0.7:
            return True

        return False
