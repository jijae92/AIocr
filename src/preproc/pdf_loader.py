"""
PDF loading and image extraction utilities.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import fitz  # PyMuPDF
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

try:
    from PyQt5.QtCore import QThreadPool, QRunnable, pyqtSlot, QObject, pyqtSignal
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

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

    def rotate_page(
        self,
        page_image: PageImage,
        angle: int,
    ) -> PageImage:
        """
        Rotate page image.

        Args:
            page_image: PageImage to rotate
            angle: Rotation angle in degrees (0, 90, 180, 270)

        Returns:
            Rotated PageImage
        """
        if angle not in {0, 90, 180, 270}:
            raise ValueError(f"Angle must be 0, 90, 180, or 270, got {angle}")

        if angle == 0:
            return page_image

        # Rotate image
        rotated = page_image.image.rotate(-angle, expand=True)

        return PageImage(
            image=rotated,
            page_num=page_image.page_num,
            width=rotated.width,
            height=rotated.height,
            dpi=page_image.dpi,
        )

    def crop_page(
        self,
        page_image: PageImage,
        box: Tuple[int, int, int, int],
    ) -> PageImage:
        """
        Crop page image.

        Args:
            page_image: PageImage to crop
            box: Crop box as (left, top, right, bottom)

        Returns:
            Cropped PageImage
        """
        cropped = page_image.image.crop(box)

        return PageImage(
            image=cropped,
            page_num=page_image.page_num,
            width=cropped.width,
            height=cropped.height,
            dpi=page_image.dpi,
        )

    def load_pages_parallel(
        self,
        pdf_path: Path,
        page_range: Optional[Tuple[int, int]] = None,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[PageImage]:
        """
        Load multiple pages in parallel using ThreadPoolExecutor.

        Args:
            pdf_path: Path to PDF file
            page_range: Optional (start, end) page range (0-indexed, inclusive)
            max_workers: Maximum number of worker threads (default: CPU count)
            progress_callback: Optional callback(completed, total)

        Returns:
            List of PageImage objects in order
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

            page_numbers = list(range(start_page, end_page + 1))
            total_pages = len(page_numbers)

            logger.info(
                f"Loading {total_pages} pages in parallel from {pdf_path.name}"
            )

            # Use ThreadPoolExecutor for parallel loading
            results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all page loading tasks
                future_to_page = {
                    executor.submit(self.load_page, pdf_path, page_num): page_num
                    for page_num in page_numbers
                }

                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        page_image = future.result()
                        results[page_num] = page_image
                        completed += 1

                        if progress_callback:
                            progress_callback(completed, total_pages)

                    except Exception as e:
                        logger.error(f"Failed to load page {page_num}: {e}")
                        raise

            # Return pages in order
            page_images = [results[page_num] for page_num in page_numbers]

            logger.info(f"Loaded {len(page_images)} pages in parallel")
            return page_images

        except Exception as e:
            logger.error(f"Failed to load pages in parallel: {e}")
            raise


# QThreadPool support for PyQt5 applications
if PYQT_AVAILABLE:

    class PageLoadSignals(QObject):
        """Signals for page loading worker."""

        finished = pyqtSignal(int, object)  # page_num, PageImage
        error = pyqtSignal(int, str)  # page_num, error_message
        progress = pyqtSignal(int, int)  # completed, total

    class PageLoadWorker(QRunnable):
        """Worker for loading a single page in QThreadPool."""

        def __init__(self, loader: 'PDFLoader', pdf_path: Path, page_num: int):
            super().__init__()
            self.loader = loader
            self.pdf_path = pdf_path
            self.page_num = page_num
            self.signals = PageLoadSignals()

        @pyqtSlot()
        def run(self):
            """Run page loading task."""
            try:
                page_image = self.loader.load_page(self.pdf_path, self.page_num)
                self.signals.finished.emit(self.page_num, page_image)
            except Exception as e:
                self.signals.error.emit(self.page_num, str(e))

    class QThreadPoolLoader:
        """
        PDF loader using QThreadPool for parallel page loading.

        Suitable for PyQt5 GUI applications.
        """

        def __init__(self, loader: PDFLoader):
            """
            Initialize QThreadPool loader.

            Args:
                loader: PDFLoader instance
            """
            self.loader = loader
            self.thread_pool = QThreadPool.globalInstance()
            self.results = {}
            self.total_pages = 0
            self.completed_pages = 0

        def load_pages_async(
            self,
            pdf_path: Path,
            page_range: Optional[Tuple[int, int]] = None,
            finished_callback: Optional[Callable[[List[PageImage]], None]] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None,
            error_callback: Optional[Callable[[int, str], None]] = None,
        ):
            """
            Load pages asynchronously using QThreadPool.

            Args:
                pdf_path: Path to PDF file
                page_range: Optional page range
                finished_callback: Called with list of PageImage when all done
                progress_callback: Called with (completed, total) on progress
                error_callback: Called with (page_num, error_msg) on error
            """
            page_count = self.loader.get_page_count(pdf_path)

            # Determine page range
            if page_range is None:
                start_page = 0
                end_page = page_count - 1
            else:
                start_page, end_page = page_range
                start_page = max(0, start_page)
                end_page = min(page_count - 1, end_page)

            page_numbers = list(range(start_page, end_page + 1))
            self.total_pages = len(page_numbers)
            self.completed_pages = 0
            self.results = {}

            logger.info(
                f"Loading {self.total_pages} pages with QThreadPool "
                f"(max threads: {self.thread_pool.maxThreadCount()})"
            )

            def on_page_finished(page_num: int, page_image: PageImage):
                """Handle page loading completion."""
                self.results[page_num] = page_image
                self.completed_pages += 1

                if progress_callback:
                    progress_callback(self.completed_pages, self.total_pages)

                # Check if all pages are done
                if self.completed_pages == self.total_pages:
                    # Return pages in order
                    page_images = [
                        self.results[pn] for pn in page_numbers if pn in self.results
                    ]
                    if finished_callback:
                        finished_callback(page_images)

            def on_page_error(page_num: int, error_msg: str):
                """Handle page loading error."""
                logger.error(f"Failed to load page {page_num}: {error_msg}")
                if error_callback:
                    error_callback(page_num, error_msg)

            # Submit workers for each page
            for page_num in page_numbers:
                worker = PageLoadWorker(self.loader, pdf_path, page_num)
                worker.signals.finished.connect(on_page_finished)
                worker.signals.error.connect(on_page_error)
                self.thread_pool.start(worker)

else:
    # Fallback when PyQt5 is not available
    class QThreadPoolLoader:
        """Fallback loader when PyQt5 is not available."""

        def __init__(self, loader: PDFLoader):
            raise ImportError("PyQt5 is required for QThreadPoolLoader")
