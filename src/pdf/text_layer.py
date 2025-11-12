"""
Searchable PDF generation with invisible text layer.

Creates searchable PDFs by adding an invisible text layer over the original
image-based PDF, enabling text selection, search, and copy-paste functionality.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from util.coords import TextBlock, image_to_pdf_coords, BoundingBox
from util.logging import get_logger

logger = get_logger(__name__)


class SearchablePDFGenerator:
    """
    Generate searchable PDFs with invisible text layer.

    This class takes image-based PDFs and OCR results, then creates
    a searchable PDF by overlaying invisible text at the correct positions.
    """

    def __init__(
        self,
        text_opacity: int = 0,
        font_name: str = "helv",
        min_font_size: float = 1.0,
        max_font_size: float = 72.0,
    ):
        """
        Initialize generator.

        Args:
            text_opacity: Text opacity (0-255, 0=invisible for searchable PDF)
            font_name: Font name (helv, times, cour, etc.)
            min_font_size: Minimum font size in points
            max_font_size: Maximum font size in points
        """
        self.text_opacity = max(0, min(255, text_opacity))
        self.font_name = font_name
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size

    def create_searchable_pdf(
        self,
        input_pdf_path: Path,
        output_pdf_path: Optional[Path] = None,
        text_blocks_per_page: Optional[List[List[TextBlock]]] = None,
        image_size_per_page: Optional[List[Tuple[int, int]]] = None,
    ) -> Path:
        """
        Create searchable PDF from original PDF and OCR results.

        OCR results (text blocks) are converted from image coordinates to PDF
        coordinates and overlaid as invisible text, making the PDF searchable.

        Args:
            input_pdf_path: Path to original (image-based) PDF
            output_pdf_path: Path to output PDF (default: <name>_searchable.pdf)
            text_blocks_per_page: List of text blocks for each page
            image_size_per_page: List of (width, height) for each page's image

        Returns:
            Path to created searchable PDF
        """
        try:
            # Generate output path if not provided
            if output_pdf_path is None:
                output_pdf_path = self._generate_output_path(input_pdf_path)

            logger.info(f"Creating searchable PDF: {output_pdf_path}")

            # Open original PDF
            doc = fitz.open(input_pdf_path)

            if text_blocks_per_page is None:
                logger.warning("No text blocks provided, creating empty searchable PDF")
                text_blocks_per_page = [[] for _ in range(len(doc))]

            # Process each page
            for page_num, text_blocks in enumerate(text_blocks_per_page):
                if page_num >= len(doc):
                    logger.warning(f"Page {page_num} exceeds PDF page count")
                    break

                page = doc[page_num]
                pdf_width = page.rect.width
                pdf_height = page.rect.height

                # Get image size for this page
                if image_size_per_page and page_num < len(image_size_per_page):
                    img_width, img_height = image_size_per_page[page_num]
                else:
                    # Assume image size matches PDF size
                    img_width, img_height = int(pdf_width), int(pdf_height)

                logger.debug(
                    f"Page {page_num}: PDF size=({pdf_width:.1f}, {pdf_height:.1f}), "
                    f"Image size=({img_width}, {img_height})"
                )

                # Sort blocks by reading order (top-to-bottom, left-to-right)
                # This ensures PDF text extraction matches the TXT export order
                sorted_blocks = sorted(text_blocks, key=lambda b: (b.bbox.y, b.bbox.x))

                # Add text layer for each block
                for block in sorted_blocks:
                    self._add_text_block(
                        page,
                        block,
                        img_width,
                        img_height,
                        pdf_width,
                        pdf_height,
                    )

            # Save output PDF
            output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
            doc.close()

            logger.info(f"Searchable PDF saved to {output_pdf_path}")
            return output_pdf_path

        except Exception as e:
            logger.error(f"Failed to create searchable PDF: {e}")
            raise

    def _generate_output_path(self, input_path: Path) -> Path:
        """
        Generate output path with '_searchable' suffix.

        Args:
            input_path: Input PDF path

        Returns:
            Output path with _searchable suffix
        """
        stem = input_path.stem
        suffix = input_path.suffix
        return input_path.parent / f"{stem}_searchable{suffix}"

    def _add_text_block(
        self,
        page: fitz.Page,
        block: TextBlock,
        img_width: int,
        img_height: int,
        pdf_width: float,
        pdf_height: float,
    ):
        """
        Add a single text block to the page.

        Args:
            page: PyMuPDF page object
            block: Text block with OCR results
            img_width: Image width in pixels
            img_height: Image height in pixels
            pdf_width: PDF page width in points
            pdf_height: PDF page height in points
        """
        text = block.text
        if not text or not text.strip():
            return

        bbox = block.bbox

        # Convert image coordinates to PDF coordinates
        # image_to_pdf_coords handles the conversion including Y-axis flip
        pdf_x0, pdf_y0 = image_to_pdf_coords(
            bbox.x,
            bbox.y,
            img_width,
            img_height,
            pdf_width,
            pdf_height,
        )

        pdf_x1, pdf_y1 = image_to_pdf_coords(
            bbox.x2,
            bbox.y2,
            img_width,
            img_height,
            pdf_width,
            pdf_height,
        )

        # Ensure coordinates are in correct order (top-left to bottom-right)
        x0 = min(pdf_x0, pdf_x1)
        x1 = max(pdf_x0, pdf_x1)
        y0 = min(pdf_y0, pdf_y1)
        y1 = max(pdf_y0, pdf_y1)

        # Create rectangle
        rect = fitz.Rect(x0, y0, x1, y1)

        # Calculate appropriate font size
        font_size = self._calculate_font_size(rect, text)

        # Prepare text layer attributes
        if self.text_opacity == 0:
            # Invisible text (for searchable PDF)
            color = None
            fill_color = (1, 1, 1, 0)  # Transparent white
        else:
            # Semi-visible text (for debugging or highlighting)
            color = (0, 0, 0)  # Black text
            opacity = self.text_opacity / 255.0
            fill_color = (1, 1, 1, 1 - opacity)  # Semi-transparent white background

        # Insert text into PDF
        # Add explicit newline after each block to help PDF readers separate blocks correctly
        # This ensures that blocks on the same Y-level (like table cells) are extracted separately
        try:
            # Append newline to ensure block separation during text extraction
            # This helps prevent PyMuPDF from merging adjacent blocks into a single line
            text_with_newline = text + "\n"

            rc = page.insert_textbox(
                rect,
                text_with_newline,
                fontname=self.font_name,
                fontsize=font_size,
                color=color,
                fill=fill_color,
                align=fitz.TEXT_ALIGN_LEFT,
                overlay=True,
                rotate=0,
            )

            if rc < 0:
                logger.debug(
                    f"Text overflow for block: '{text[:50]}...' "
                    f"(font_size={font_size:.1f}, rect={rect})"
                )

        except Exception as e:
            logger.warning(f"Failed to insert text block '{text[:30]}...': {e}")

    def _calculate_font_size(self, rect: fitz.Rect, text: str) -> float:
        """
        Calculate appropriate font size to fit text in rectangle.

        Uses a heuristic based on rectangle height and text length.

        Args:
            rect: Bounding rectangle
            text: Text to fit

        Returns:
            Font size in points
        """
        if not text:
            return self.min_font_size

        # Estimate based on rectangle height
        # Typical ratio: font_size ≈ 0.7 * height for single-line text
        height_based_size = rect.height * 0.7

        # Adjust based on text length
        # Longer text needs smaller font to fit width
        rect_width = rect.width
        char_count = len(text)

        if char_count > 0 and rect_width > 0:
            # Estimate: each character takes about 0.6 * font_size in width
            width_based_size = (rect_width / char_count) / 0.6
        else:
            width_based_size = height_based_size

        # Use the more restrictive constraint
        font_size = min(height_based_size, width_based_size)

        # Clamp to min/max range
        font_size = max(self.min_font_size, min(self.max_font_size, font_size))

        return font_size

    def add_text_layer_simple(
        self,
        input_pdf_path: Path,
        output_pdf_path: Optional[Path] = None,
        text_per_page: Optional[List[str]] = None,
    ) -> Path:
        """
        Add simple text layer (full-page text without positioning).

        This method adds text as a single invisible block per page,
        making the PDF searchable but without precise positioning.
        Useful when only full-page text is available without coordinates.

        Args:
            input_pdf_path: Path to original PDF
            output_pdf_path: Path to output PDF (default: <name>_searchable.pdf)
            text_per_page: List of text for each page

        Returns:
            Path to created searchable PDF
        """
        try:
            # Generate output path if not provided
            if output_pdf_path is None:
                output_pdf_path = self._generate_output_path(input_pdf_path)

            logger.info(f"Adding simple text layer to PDF: {output_pdf_path}")

            # Open original PDF
            doc = fitz.open(input_pdf_path)

            if text_per_page is None:
                text_per_page = [""] * len(doc)

            # Process each page
            for page_num, text in enumerate(text_per_page):
                if page_num >= len(doc):
                    break

                page = doc[page_num]

                # Add text as invisible overlay
                if text and text.strip():
                    rect = page.rect
                    try:
                        page.insert_textbox(
                            rect,
                            text,
                            fontsize=1,  # Very small font
                            color=None,
                            fill=(1, 1, 1, 0),  # Fully transparent
                            overlay=True,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add text to page {page_num}: {e}")

            # Save
            output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
            doc.close()

            logger.info(f"Text layer added to {output_pdf_path}")
            return output_pdf_path

        except Exception as e:
            logger.error(f"Failed to add text layer: {e}")
            raise

    def create_from_ocr_result(
        self,
        input_pdf_path: Path,
        output_pdf_path: Optional[Path] = None,
        ocr_result = None,  # Type hint omitted to avoid circular import
    ) -> Path:
        """
        Create searchable PDF from OCR result object.

        This is a convenience method that accepts an OCRResult object
        and extracts the necessary information to create a searchable PDF.

        Args:
            input_pdf_path: Path to original PDF
            output_pdf_path: Path to output PDF
            ocr_result: OCRResult object from src.data.ocr_result

        Returns:
            Path to created searchable PDF
        """
        if ocr_result is None:
            raise ValueError("ocr_result cannot be None")

        # Extract text blocks and image sizes from OCR result
        text_blocks_per_page = []
        image_size_per_page = []

        for page_result in ocr_result.pages:
            # Convert Block objects to TextBlock objects
            text_blocks = []
            for block in page_result.blocks:
                text_block = TextBlock(
                    text=block.text,
                    bbox=BoundingBox(
                        x=block.bbox.x,
                        y=block.bbox.y,
                        width=block.bbox.width,
                        height=block.bbox.height,
                        confidence=block.confidence,
                    ),
                    confidence=block.confidence,
                )
                text_blocks.append(text_block)

            text_blocks_per_page.append(text_blocks)
            image_size_per_page.append((int(page_result.width), int(page_result.height)))

        return self.create_searchable_pdf(
            input_pdf_path,
            output_pdf_path,
            text_blocks_per_page,
            image_size_per_page,
        )

    @staticmethod
    def verify_searchable(pdf_path: Path) -> bool:
        """
        Verify if a PDF is searchable (has text layer).

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if PDF has text content
        """
        try:
            doc = fitz.open(pdf_path)
            has_text = False

            for page in doc:
                text = page.get_text().strip()
                if text:
                    has_text = True
                    break

            doc.close()
            return has_text

        except Exception as e:
            logger.error(f"Failed to verify PDF: {e}")
            return False

    @staticmethod
    def extract_text(pdf_path: Path) -> List[str]:
        """
        Extract text from PDF pages.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of text per page
        """
        try:
            doc = fitz.open(pdf_path)
            texts = []

            for page in doc:
                text = page.get_text()
                texts.append(text)

            doc.close()
            return texts

        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return []
