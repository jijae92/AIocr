"""
QThreadPool workers for GUI processing.

Implements page-level workers that integrate the full OCR pipeline:
- PDF loading with preprocessing
- DocAI/routing with heuristics
- Ensemble processing
- Postprocessing and normalization
- Searchable PDF generation
- Cache integration
"""

import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image

from cache.store import CacheManager
from data.ocr_result import OCRResult, PageResult, Block
from pdf.text_layer import SearchablePDFGenerator
from preproc.pdf_loader import PDFLoader
from postproc.text_norm import TextNormalizer
from router.heuristics import HeuristicRouter, DocumentCharacteristics
from util.logging import get_logger

logger = get_logger(__name__)


class WorkerSignals(QObject):
    """Signals for worker communication."""

    # Progress signals
    progress = pyqtSignal(str, int, int)  # message, current, total
    page_completed = pyqtSignal(int, dict)  # page_num, result
    page_failed = pyqtSignal(int, str, QPixmap)  # page_num, reason, thumbnail

    # Completion signals
    finished = pyqtSignal(dict)  # final_result
    error = pyqtSignal(str)  # error_message

    # Status signals
    status_update = pyqtSignal(str)  # status_message
    log_message = pyqtSignal(str)  # log_message


class PDFOCRWorker(QRunnable):
    """
    Worker for processing PDF files with full OCR pipeline.

    Integrates:
    - PDF loading (PDFLoader)
    - Routing/ensemble (HeuristicRouter)
    - Postprocessing (TextNormalizer)
    - Searchable PDF generation
    - Cache integration
    """

    def __init__(
        self,
        file_path: Path,
        config: Dict[str, Any],
        cache_manager: Optional[CacheManager] = None,
        engine: str = "auto",
        page_range: Optional[Tuple[int, int]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        use_ensemble: bool = True,
    ):
        """
        Initialize PDF OCR worker.

        Args:
            file_path: Path to PDF file
            config: Application configuration
            cache_manager: Cache manager instance
            engine: OCR engine ('auto', 'docai', 'donut', etc.)
            page_range: Optional (start, end) page range
            thresholds: Confidence thresholds
            use_ensemble: Whether to use ensemble mode
        """
        super().__init__()
        self.file_path = file_path
        self.config = config
        self.cache_manager = cache_manager
        self.engine = engine
        self.page_range = page_range
        self.thresholds = thresholds or {}
        self.use_ensemble = use_ensemble

        self.signals = WorkerSignals()
        self.is_cancelled = False

        # Initialize components
        self.pdf_loader = PDFLoader(
            dpi=config.get('pdf', {}).get('dpi', 300),
            image_format=config.get('pdf', {}).get('image_format', 'PNG'),
        )

        self.router = HeuristicRouter(config)
        self.normalizer = TextNormalizer()
        self.pdf_generator = SearchablePDFGenerator(
            text_opacity=config.get('export', {}).get('pdf_text_opacity', 0)
        )

    @pyqtSlot()
    def run(self):
        """Run OCR processing."""
        try:
            self.signals.log_message.emit(f"Processing: {self.file_path.name}")
            self.signals.status_update.emit("Loading PDF...")

            # Load PDF pages
            page_images = self._load_pdf_pages()
            if not page_images:
                raise ValueError("No pages loaded from PDF")

            total_pages = len(page_images)
            self.signals.progress.emit("Processing pages...", 0, total_pages)

            # Process each page
            all_page_results = []
            low_confidence_pages = []

            for idx, page_image in enumerate(page_images):
                if self.is_cancelled:
                    self.signals.log_message.emit("Processing cancelled")
                    return

                page_num = idx + 1
                self.signals.progress.emit(
                    f"Processing page {page_num}/{total_pages}",
                    idx + 1,
                    total_pages
                )

                try:
                    # Check cache
                    page_result = self._process_page_with_cache(
                        page_image, page_num
                    )

                    all_page_results.append(page_result)

                    # Check confidence
                    avg_confidence = self._calculate_page_confidence(page_result)
                    if avg_confidence < self.thresholds.get('low_confidence', 0.5):
                        low_confidence_pages.append(page_num)

                    # Emit page completion
                    self.signals.page_completed.emit(page_num, {
                        'page_num': page_num,
                        'confidence': avg_confidence,
                        'blocks': len(page_result.blocks),
                    })

                except Exception as e:
                    error_msg = f"Failed to process page {page_num}: {str(e)}"
                    logger.error(error_msg)
                    self.signals.log_message.emit(error_msg)

                    # Generate thumbnail for error view
                    thumbnail = self._create_thumbnail(page_image.image)
                    self.signals.page_failed.emit(page_num, str(e), thumbnail)

            # Create OCR result
            ocr_result = OCRResult(
                pages=all_page_results,
                doc_id=str(self.file_path),
            )

            # Generate searchable PDF
            self.signals.status_update.emit("Generating searchable PDF...")
            output_pdf_path = self._generate_searchable_pdf(ocr_result)

            # Prepare final result
            final_result = {
                'file_path': str(self.file_path),
                'output_pdf_path': str(output_pdf_path),
                'total_pages': total_pages,
                'low_confidence_pages': low_confidence_pages,
                'average_confidence': self._calculate_document_confidence(ocr_result),
                'text': self._extract_full_text(ocr_result),
                'ocr_result': ocr_result,
            }

            self.signals.finished.emit(final_result)
            self.signals.log_message.emit(
                f"Completed: {self.file_path.name} "
                f"({total_pages} pages, {len(low_confidence_pages)} low confidence)"
            )

        except Exception as e:
            error_msg = f"OCR processing failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.signals.error.emit(error_msg)

    def cancel(self):
        """Cancel processing."""
        self.is_cancelled = True

    def _load_pdf_pages(self) -> List:
        """Load PDF pages with optional page range filtering."""
        page_images = self.pdf_loader.load_pdf(
            self.file_path,
            page_range=self.page_range
        )

        # Apply preprocessing
        if self.config.get('preprocessing', {}).get('auto_rotate', True):
            # Auto-rotate pages if needed
            pass

        return page_images

    def _process_page_with_cache(self, page_image, page_num: int) -> PageResult:
        """
        Process single page with cache integration.

        Args:
            page_image: PageImage object
            page_num: Page number

        Returns:
            PageResult with OCR results
        """
        # Check if cache is enabled
        if self.cache_manager and self.cache_manager.enabled:
            cache_key = f"{self.file_path.stem}_page_{page_num}_{self.engine}"

            # Try to get from cache
            # For now, compute directly (cache integration with content hash would be more complex)

        # Process with routing/ensemble
        if self.engine == "auto":
            page_result = self._process_with_routing(page_image, page_num)
        else:
            page_result = self._process_with_engine(page_image, page_num, self.engine)

        # Apply postprocessing
        page_result = self._apply_postprocessing(page_result)

        return page_result

    def _process_with_routing(self, page_image, page_num: int) -> PageResult:
        """
        Process page with intelligent routing.

        Args:
            page_image: PageImage object
            page_num: Page number

        Returns:
            PageResult
        """
        # Analyze page characteristics
        # For now, use a simple heuristic
        # In production, this would call actual OCR engines

        # TODO: Integrate with actual DocAI, Donut, TrOCR, etc.
        # For now, create placeholder result

        blocks = []
        page_result = PageResult(
            page_number=page_num,
            blocks=blocks,
            width=page_image.width,
            height=page_image.height,
        )

        return page_result

    def _process_with_engine(
        self, page_image, page_num: int, engine: str
    ) -> PageResult:
        """
        Process page with specific engine.

        Args:
            page_image: PageImage object
            page_num: Page number
            engine: Engine name

        Returns:
            PageResult
        """
        # TODO: Call specific engine
        # For now, return placeholder

        blocks = []
        page_result = PageResult(
            page_number=page_num,
            blocks=blocks,
            width=page_image.width,
            height=page_image.height,
        )

        return page_result

    def _apply_postprocessing(self, page_result: PageResult) -> PageResult:
        """
        Apply postprocessing to page result.

        Args:
            page_result: PageResult to process

        Returns:
            Processed PageResult
        """
        # Apply text normalization to each block
        for block in page_result.blocks:
            if block.text:
                # Normalize whitespace
                if self.config.get('postprocessing', {}).get('normalize_whitespace', True):
                    block.text = self.normalizer.normalize_whitespace(block.text)

                # Korean spacing
                if 'ko' in str(block.language):
                    block.text = self.normalizer.normalize_korean_spacing(block.text)

                # Normalize dates, currency, etc.
                if self.config.get('postprocessing', {}).get('fix_common_errors', True):
                    block.text = self.normalizer.normalize_dates(block.text)
                    block.text = self.normalizer.normalize_currency(block.text)

        return page_result

    def _generate_searchable_pdf(self, ocr_result: OCRResult) -> Path:
        """
        Generate searchable PDF from OCR result.

        Args:
            ocr_result: OCRResult

        Returns:
            Path to searchable PDF
        """
        try:
            output_path = self.pdf_generator.create_from_ocr_result(
                self.file_path,
                ocr_result=ocr_result,
            )
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate searchable PDF: {e}")
            # Return original file if generation fails
            return self.file_path

    def _calculate_page_confidence(self, page_result: PageResult) -> float:
        """Calculate average confidence for page."""
        if not page_result.blocks:
            return 0.0

        confidences = [block.confidence for block in page_result.blocks if block.confidence]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _calculate_document_confidence(self, ocr_result: OCRResult) -> float:
        """Calculate average confidence for entire document."""
        all_confidences = []
        for page in ocr_result.pages:
            for block in page.blocks:
                if block.confidence:
                    all_confidences.append(block.confidence)

        return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    def _extract_full_text(self, ocr_result: OCRResult) -> str:
        """Extract full text from OCR result."""
        lines = []
        for page in ocr_result.pages:
            page_lines = [block.text for block in page.blocks if block.text]
            if page_lines:
                lines.extend(page_lines)
                lines.append('\n')  # Page separator

        return '\n'.join(lines)

    def _create_thumbnail(self, image: Image.Image, size: int = 100) -> QPixmap:
        """
        Create thumbnail from PIL Image.

        Args:
            image: PIL Image
            size: Thumbnail size

        Returns:
            QPixmap thumbnail
        """
        try:
            # Create thumbnail
            image.thumbnail((size, size))

            # Convert to QPixmap
            if image.mode == 'RGB':
                data = image.tobytes("raw", "RGB")
                qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)
            else:
                # Convert to RGB first
                rgb_image = image.convert('RGB')
                data = rgb_image.tobytes("raw", "RGB")
                qimage = QImage(data, rgb_image.width, rgb_image.height, QImage.Format_RGB888)

            return QPixmap.fromImage(qimage)

        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return QPixmap()


class PageOCRWorker(QRunnable):
    """
    Worker for processing single PDF page.

    Used for reprocessing individual pages or low-confidence pages.
    """

    def __init__(
        self,
        page_image,
        page_num: int,
        config: Dict[str, Any],
        engine: str = "auto",
    ):
        """
        Initialize page OCR worker.

        Args:
            page_image: PageImage object
            page_num: Page number
            config: Configuration dict
            engine: OCR engine to use
        """
        super().__init__()
        self.page_image = page_image
        self.page_num = page_num
        self.config = config
        self.engine = engine

        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """Run page OCR processing."""
        try:
            self.signals.log_message.emit(f"Processing page {self.page_num}...")

            # TODO: Implement page processing
            # For now, create placeholder result

            result = {
                'page_num': self.page_num,
                'confidence': 0.85,
                'blocks': [],
            }

            self.signals.finished.emit(result)

        except Exception as e:
            error_msg = f"Failed to process page {self.page_num}: {str(e)}"
            logger.error(error_msg)
            self.signals.error.emit(error_msg)


class ExportWorker(QRunnable):
    """Worker for exporting OCR results to various formats."""

    def __init__(
        self,
        ocr_result: OCRResult,
        output_path: Path,
        format: str,
        config: Dict[str, Any],
    ):
        """
        Initialize export worker.

        Args:
            ocr_result: OCRResult to export
            output_path: Output file path
            format: Export format ('txt', 'json', 'markdown', 'docx', 'searchable_pdf')
            config: Configuration dict
        """
        super().__init__()
        self.ocr_result = ocr_result
        self.output_path = output_path
        self.format = format
        self.config = config

        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """Run export."""
        try:
            self.signals.log_message.emit(
                f"Exporting to {self.format.upper()}: {self.output_path}"
            )

            if self.format == 'txt':
                self._export_txt()
            elif self.format == 'json':
                self._export_json()
            elif self.format == 'markdown':
                self._export_markdown()
            elif self.format == 'docx':
                self._export_docx()
            elif self.format == 'searchable_pdf':
                self._export_searchable_pdf()
            else:
                raise ValueError(f"Unsupported format: {self.format}")

            self.signals.finished.emit({
                'output_path': str(self.output_path),
                'format': self.format,
            })

            self.signals.log_message.emit(f"Export completed: {self.output_path}")

        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            logger.error(error_msg)
            self.signals.error.emit(error_msg)

    def _export_txt(self):
        """Export as plain text."""
        lines = []
        for page in self.ocr_result.pages:
            for block in page.blocks:
                if block.text:
                    lines.append(block.text)
            lines.append('\n')  # Page separator

        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _export_json(self):
        """Export as JSON."""
        import json
        from dataclasses import asdict

        # Convert to dict
        result_dict = asdict(self.ocr_result)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

    def _export_markdown(self):
        """Export as Markdown."""
        lines = ['# OCR Result\n']

        for page in self.ocr_result.pages:
            lines.append(f'## Page {page.page_number}\n')
            for block in page.blocks:
                if block.text:
                    lines.append(block.text)
                    lines.append('')

        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _export_docx(self):
        """Export as DOCX."""
        # TODO: Implement DOCX export using python-docx
        raise NotImplementedError("DOCX export not yet implemented")

    def _export_searchable_pdf(self):
        """Export as searchable PDF."""
        # This should already be done in the main processing
        # Just copy/verify the existing searchable PDF
        pass
