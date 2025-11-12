"""
Layout merging and reading order detection.
"""

from typing import List

from util.coords import TextBlock, sort_bboxes_reading_order
from util.logging import get_logger

logger = get_logger(__name__)


class LayoutMerger:
    """Merge text blocks according to layout and reading order."""

    @staticmethod
    def merge_by_reading_order(
        blocks: List[TextBlock],
        preserve_layout: bool = True,
    ) -> str:
        """
        Merge text blocks in reading order.

        Args:
            blocks: List of text blocks
            preserve_layout: Preserve layout structure with spacing

        Returns:
            Merged text
        """
        if not blocks:
            return ""

        # Sort blocks by reading order
        sorted_blocks = sort_bboxes_reading_order(
            [block.bbox for block in blocks]
        )

        # Get original block texts in sorted order
        block_map = {id(block.bbox): block for block in blocks}
        sorted_texts = []

        for bbox in sorted_blocks:
            for block in blocks:
                if id(block.bbox) == id(bbox):
                    sorted_texts.append(block.text)
                    break

        if preserve_layout:
            # Add appropriate spacing based on vertical distances
            result = []
            prev_bbox = None

            for i, bbox in enumerate(sorted_blocks):
                text = sorted_texts[i]

                if prev_bbox is not None:
                    # Calculate vertical distance
                    vertical_dist = bbox.y - prev_bbox.y2

                    # Add spacing based on distance
                    if vertical_dist > prev_bbox.height:
                        # Large gap - add double newline
                        result.append('\n\n')
                    elif vertical_dist > prev_bbox.height * 0.5:
                        # Medium gap - add single newline
                        result.append('\n')
                    else:
                        # Small gap - add space
                        result.append(' ')

                result.append(text)
                prev_bbox = bbox

            return ''.join(result)
        else:
            # Simple join with newlines
            return '\n'.join(sorted_texts)

    @staticmethod
    def merge_blocks_by_proximity(
        blocks: List[TextBlock],
        horizontal_threshold: float = 10.0,
        vertical_threshold: float = 5.0,
    ) -> List[TextBlock]:
        """
        Merge nearby text blocks.

        Args:
            blocks: List of text blocks
            horizontal_threshold: Max horizontal distance for merging
            vertical_threshold: Max vertical distance for merging

        Returns:
            List of merged blocks
        """
        if not blocks:
            return []

        # Sort by reading order
        sorted_blocks = list(blocks)

        merged = []
        current_block = sorted_blocks[0]

        for next_block in sorted_blocks[1:]:
            # Check if blocks are on same line
            vertical_dist = abs(next_block.bbox.y - current_block.bbox.y)
            horizontal_dist = next_block.bbox.x - current_block.bbox.x2

            if (
                vertical_dist < vertical_threshold
                and 0 <= horizontal_dist <= horizontal_threshold
            ):
                # Merge blocks
                merged_text = current_block.text + ' ' + next_block.text

                # Merge bounding boxes
                from util.coords import merge_bboxes

                merged_bbox = merge_bboxes([current_block.bbox, next_block.bbox])

                # Average confidence
                merged_conf = (current_block.confidence + next_block.confidence) / 2

                current_block = TextBlock(
                    text=merged_text,
                    bbox=merged_bbox,
                    confidence=merged_conf,
                    block_type='text',
                )
            else:
                # Save current and start new
                merged.append(current_block)
                current_block = next_block

        # Add last block
        merged.append(current_block)

        return merged

    @staticmethod
    def group_into_paragraphs(
        blocks: List[TextBlock],
        vertical_threshold: float = None,
    ) -> List[List[TextBlock]]:
        """
        Group blocks into paragraphs.

        Args:
            blocks: List of text blocks
            vertical_threshold: Max vertical distance for same paragraph

        Returns:
            List of paragraph groups (each is a list of blocks)
        """
        if not blocks:
            return []

        # Auto-calculate threshold if not provided
        if vertical_threshold is None:
            avg_height = sum(b.bbox.height for b in blocks) / len(blocks)
            vertical_threshold = avg_height * 1.5

        paragraphs = []
        current_para = [blocks[0]]

        for block in blocks[1:]:
            prev_block = current_para[-1]
            vertical_dist = block.bbox.y - prev_block.bbox.y2

            if vertical_dist <= vertical_threshold:
                # Same paragraph
                current_para.append(block)
            else:
                # New paragraph
                paragraphs.append(current_para)
                current_para = [block]

        # Add last paragraph
        if current_para:
            paragraphs.append(current_para)

        return paragraphs
