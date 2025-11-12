"""
PDF loading and image extraction utilities.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

from util.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PageImage:
    """Container for page image and metadata."""

    image: Image.Image
    page_num: int  # 0-indexed
    width: int
    height: int
    dpi: int

    @property
    def size(self) -> Tuple[int, int]:
        """Get image size as (width, height)."""
        return (self.width, self.height)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.image)


class PDFLoader:
    """Load and extract images from PDF files."""

    def __init__(
        self,
        dpi: int = 300,
        image_format: str = 'PNG',
        max_pages_per_batch: int = 10,
    ):
        """
        Initialize PDF loader.

        Args:
            dpi: DPI for image conversion
            image_format: Output image format (PNG, JPEG)
            max_pages_per_batch: Maximum pages to process per batch
        """
        self.dpi = dpi
        self.image_format = image_format
        self.max_pages_per_batch = max_pages_per_batch

    def get_page_count(self, pdf_path: Path) -> int:
        """
        Get number of pages in PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages
        """
        try:
            with fitz.open(pdf_path) as doc:
                return len(doc)
        except Exception as e:
            logger.error(f"Failed to get page count: {e}")
            raise

    def get_page_dimensions(
        self,
        pdf_path: Path,
        page_num: int = 0,
    ) -> Tuple[float, float]:
        """
        Get dimensions of a page in points.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            Tuple of (width, height) in points
        """
        try:
            with fitz.open(pdf_path) as doc:
                page = doc[page_num]
                rect = page.rect
                return (rect.width, rect.height)
        except Exception as e:
            logger.error(f"Failed to get page dimensions: {e}")
            raise

    def load_page(
        self,
        pdf_path: Path,
        page_num: int,
    ) -> PageImage:
        """
        Load a single page as image.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            PageImage object
        """
        try:
            # Convert single page
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=page_num + 1,  # pdf2image uses 1-indexed
                last_page=page_num + 1,
                fmt=self.image_format.lower(),
            )

            if not images:
                raise ValueError(f"Failed to convert page {page_num}")

            image = images[0]

            return PageImage(
                image=image,
                page_num=page_num,
                width=image.width,
                height=image.height,
                dpi=self.dpi,
            )

        except Exception as e:
            logger.error(f"Failed to load page {page_num}: {e}")
            raise

    def load_pages(
        self,
        pdf_path: Path,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List[PageImage]:
        """
        Load multiple pages as images.

        Args:
            pdf_path: Path to PDF file
            page_range: Optional (start, end) page range (0-indexed, inclusive)
                       None for all pages

        Returns:
            List of PageImage objects
        """
        try:
            page_count = self.get_page_count(pdf_path)

            # Determine page range
            if page_range is None:
                start_page = 0
                end_page = page_count - 1
            else:
                start_page, end_page = page_range
                start_page = max(0, start_page)
                end_page = min(page_count - 1, end_page)

            logger.info(
                f"Loading pages {start_page}-{end_page} from {pdf_path.name} "
                f"({end_page - start_page + 1} pages)"
            )

            # Convert pages (pdf2image uses 1-indexed)
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=start_page + 1,
                last_page=end_page + 1,
                fmt=self.image_format.lower(),
            )

            # Create PageImage objects
            page_images = []
            for i, image in enumerate(images):
                page_images.append(
                    PageImage(
                        image=image,
                        page_num=start_page + i,
                        width=image.width,
                        height=image.height,
                        dpi=self.dpi,
                    )
                )

            logger.info(f"Loaded {len(page_images)} pages")
            return page_images

        except Exception as e:
            logger.error(f"Failed to load pages: {e}")
            raise

    def extract_text(self, pdf_path: Path, page_num: int = 0) -> str:
        """
        Extract text from PDF page (if available).

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            Extracted text
        """
        try:
            with fitz.open(pdf_path) as doc:
                page = doc[page_num]
                text = page.get_text()
                return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from page {page_num}: {e}")
            return ""

    def has_text(self, pdf_path: Path, page_num: int = 0) -> bool:
        """
        Check if page has embedded text.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            True if page has text
        """
        text = self.extract_text(pdf_path, page_num)
        return len(text) > 50  # Arbitrary threshold

    def save_page_image(
        self,
        page_image: PageImage,
        output_path: Path,
        format: Optional[str] = None,
    ):
        """
        Save page image to file.

        Args:
            page_image: PageImage object
            output_path: Output file path
            format: Image format (default: from output_path extension)
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            page_image.image.save(output_path, format=format)
            logger.debug(f"Saved page image to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save page image: {e}")
            raise

    def load_image(self, image_path: Path) -> PageImage:
        """
        Load standalone image file.

        Args:
            image_path: Path to image file

        Returns:
            PageImage object
        """
        try:
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            return PageImage(
                image=image,
                page_num=0,
                width=image.width,
                height=image.height,
                dpi=self.dpi,
            )

        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def is_pdf(self, file_path: Path) -> bool:
        """Check if file is a PDF."""
        return file_path.suffix.lower() == '.pdf'

    def is_image(self, file_path: Path) -> bool:
        """Check if file is an image."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
        return file_path.suffix.lower() in image_extensions

    def load(
        self,
        file_path: Path,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List[PageImage]:
        """
        Load file (PDF or image) and return page images.

        Args:
            file_path: Path to file
            page_range: Optional page range for PDFs

        Returns:
            List of PageImage objects
        """
        if self.is_pdf(file_path):
            return self.load_pages(file_path, page_range)
        elif self.is_image(file_path):
            return [self.load_image(file_path)]
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
