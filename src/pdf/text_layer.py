"""
Searchable PDF generation with invisible text layer.
"""

from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from util.coords import TextBlock
from util.logging import get_logger

logger = get_logger(__name__)


class SearchablePDFGenerator:
    """Generate searchable PDFs with invisible text layer."""

    def __init__(self, text_opacity: int = 0):
        """
        Initialize generator.

        Args:
            text_opacity: Text opacity (0-255, 0=invisible)
        """
        self.text_opacity = text_opacity

    def create_searchable_pdf(
        self,
        input_pdf_path: Path,
        output_pdf_path: Path,
        text_blocks_per_page: List[List[TextBlock]],
    ):
        """
        Create searchable PDF from original PDF and OCR results.

        Args:
            input_pdf_path: Path to original PDF
            output_pdf_path: Path to output PDF
            text_blocks_per_page: List of text blocks for each page
        """
        try:
            logger.info(f"Creating searchable PDF: {output_pdf_path}")

            # Open original PDF
            doc = fitz.open(input_pdf_path)

            # Process each page
            for page_num, text_blocks in enumerate(text_blocks_per_page):
                if page_num >= len(doc):
                    logger.warning(f"Page {page_num} exceeds PDF page count")
                    break

                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height

                # Add text layer for each block
                for block in text_blocks:
                    # Get bounding box
                    bbox = block.bbox

                    # Convert coordinates (image -> PDF)
                    # Note: PDF coordinates have origin at bottom-left
                    x0 = bbox.x
                    y0 = page_height - bbox.y2  # Flip Y
                    x1 = bbox.x2
                    y1 = page_height - bbox.y  # Flip Y

                    # Create rectangle
                    rect = fitz.Rect(x0, y0, x1, y1)

                    # Calculate font size to fit text in bbox
                    text = block.text
                    if not text.strip():
                        continue

                    # Estimate font size
                    # This is approximate - may need adjustment
                    font_size = min(bbox.height * 0.8, 12)

                    # Insert text (invisible if opacity=0)
                    try:
                        page.insert_textbox(
                            rect,
                            text,
                            fontsize=font_size,
                            color=(0, 0, 0) if self.text_opacity > 0 else None,
                            fill=(1, 1, 1, self.text_opacity / 255),
                            align=0,  # Left align
                            overlay=True,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to insert text block: {e}")
                        continue

            # Save output PDF
            output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(output_pdf_path)
            doc.close()

            logger.info(f"Searchable PDF saved to {output_pdf_path}")

        except Exception as e:
            logger.error(f"Failed to create searchable PDF: {e}")
            raise

    def add_text_layer_simple(
        self,
        input_pdf_path: Path,
        output_pdf_path: Path,
        text_per_page: List[str],
    ):
        """
        Add simple text layer (full-page text without positioning).

        Args:
            input_pdf_path: Path to original PDF
            output_pdf_path: Path to output PDF
            text_per_page: List of text for each page
        """
        try:
            logger.info(f"Adding text layer to PDF: {output_pdf_path}")

            # Open original PDF
            doc = fitz.open(input_pdf_path)

            # Process each page
            for page_num, text in enumerate(text_per_page):
                if page_num >= len(doc):
                    break

                page = doc[page_num]

                # Add text as metadata (makes PDF searchable)
                # This doesn't position text, but allows search
                if text.strip():
                    # Insert invisible text overlay
                    rect = page.rect
                    try:
                        page.insert_textbox(
                            rect,
                            text,
                            fontsize=1,  # Very small
                            color=None,
                            fill=(1, 1, 1, 0),  # Transparent
                            overlay=True,
                        )
                    except Exception:
                        pass

            # Save
            output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(output_pdf_path)
            doc.close()

            logger.info(f"Text layer added to {output_pdf_path}")

        except Exception as e:
            logger.error(f"Failed to add text layer: {e}")
            raise
