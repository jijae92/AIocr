"""
QThreadPool workers for GUI processing.

Implements page-level workers that integrate the full OCR pipeline:
- PDF loading with preprocessing
- DocAI processing (OCR + Form Parser + Layout Parser)
- Postprocessing and normalization
- Searchable PDF generation
- Cache integration
"""

import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import io

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image

from cache.store import CacheManager
from connectors.docai_client import DocumentAIClient
from data.ocr_result import OCRResult, PageResult, Block, BoundingBox
from data.block_types import BlockType
from pdf.text_layer import SearchablePDFGenerator
from preproc.pdf_loader import PDFLoader
from postproc.text_norm import TextNormalizer
from postproc.spell_checker import get_spell_corrector
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
        output_directory: Optional[Path] = None,
        export_formats: Optional[List[str]] = None,
        table_settings: Optional[Dict[str, Any]] = None,
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
            output_directory: Directory to save output files (None = same as input)
            export_formats: List of formats to export ('txt', 'json', 'markdown', 'searchable_pdf', 'tables_csv')
            table_settings: Table extraction settings (enabled, min_rows, min_cols)
        """
        super().__init__()
        self.file_path = file_path
        self.config = config
        self.cache_manager = cache_manager
        self.engine = engine
        self.page_range = page_range
        self.thresholds = thresholds or {}
        self.use_ensemble = use_ensemble
        self.output_directory = output_directory
        self.export_formats = export_formats or ['searchable_pdf']
        self.table_settings = table_settings or {'enabled': False, 'min_rows': 3, 'min_cols': 2}

        self.signals = WorkerSignals()
        self.is_cancelled = False

        # Initialize components
        self.pdf_loader = PDFLoader(
            dpi=config.get('pdf', {}).get('dpi', 300),
            image_format=config.get('pdf', {}).get('image_format', 'PNG'),
        )

        self.normalizer = TextNormalizer()
        self.spell_corrector = get_spell_corrector()
        self.pdf_generator = SearchablePDFGenerator(
            text_opacity=config.get('export', {}).get('pdf_text_opacity', 0)
        )

        # Initialize DocAI client (only OCR engine used)
        self.docai_client = None
        try:
            self.docai_client = DocumentAIClient()
            self.signals.log_message.emit("DocAI client initialized (OCR + Form + Layout)")
        except Exception as e:
            logger.error(f"Failed to initialize DocAI client: {e}")
            self.signals.log_message.emit(f"DocAI client init failed: {str(e)}")

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
                        'file_path': str(self.file_path),
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

            # Export results in selected formats
            self.signals.status_update.emit("Exporting results...")
            output_files = self._export_results(ocr_result)

            # Prepare final result
            final_result = {
                'file_path': str(self.file_path),
                'output_files': output_files,
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
            self.signals.log_message.emit(f"Output files: {', '.join(str(f) for f in output_files.values())}")

        except Exception as e:
            error_msg = f"OCR processing failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.signals.error.emit(error_msg)

    def cancel(self):
        """Cancel processing."""
        self.is_cancelled = True

    def _load_pdf_pages(self) -> List:
        """Load PDF pages with optional page range filtering."""
        page_images = self.pdf_loader.load_pages(
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

        # Process with DocAI (all three processors)
        page_result = self._process_with_docai_all_processors(page_image, page_num)

        # Apply postprocessing
        page_result = self._apply_postprocessing(page_result)

        return page_result


    def _process_with_docai_all_processors(self, page_image, page_num: int) -> PageResult:
        """
        Process with all available DocAI processors and merge results.

        Uses OCR, Form Parser, and Layout Parser for comprehensive extraction.

        Args:
            page_image: PageImage object
            page_num: Page number

        Returns:
            PageResult with merged blocks from all processors
        """
        self.signals.log_message.emit(f"Processing page {page_num} with all DocAI processors...")

        all_blocks = []
        processors_used = []

        # Try OCR Processor
        if self.docai_client.processor_id_ocr:
            try:
                self.signals.log_message.emit(f"  → OCR Processor...")
                ocr_blocks = self._process_with_docai(page_image, page_num, 'ocr')
                if ocr_blocks:
                    all_blocks.extend(ocr_blocks)
                    processors_used.append('OCR')
            except Exception as e:
                logger.warning(f"OCR processor failed: {e}")

        # Try Form Parser
        if self.docai_client.processor_id_form:
            try:
                self.signals.log_message.emit(f"  → Form Parser...")
                form_blocks = self._process_with_docai(page_image, page_num, 'form')
                if form_blocks:
                    all_blocks.extend(form_blocks)
                    processors_used.append('Form')
            except Exception as e:
                logger.warning(f"Form parser failed: {e}")

        # Try Layout Parser
        if self.docai_client.processor_id_layout:
            try:
                self.signals.log_message.emit(f"  → Layout Parser...")
                layout_blocks = self._process_with_docai(page_image, page_num, 'layout')
                if layout_blocks:
                    all_blocks.extend(layout_blocks)
                    processors_used.append('Layout')
            except Exception as e:
                logger.warning(f"Layout parser failed: {e}")

        # Deduplicate and merge blocks
        merged_blocks = self._merge_docai_blocks(all_blocks)

        self.signals.log_message.emit(
            f"DocAI processors used: {', '.join(processors_used)} → {len(merged_blocks)} blocks"
        )

        return PageResult(
            page_number=page_num,
            blocks=merged_blocks,
            width=page_image.width,
            height=page_image.height,
        )

    def _merge_docai_blocks(self, blocks: List[Block]) -> List[Block]:
        """
        Merge blocks from multiple processors, removing duplicates.

        Strategy:
        - Keep blocks with highest confidence for overlapping regions
        - Preserve unique blocks from each processor
        """
        if not blocks:
            return []

        # Simple approach: deduplicate by text and keep highest confidence
        unique_blocks = {}

        for block in blocks:
            key = (block.text.strip(), tuple(block.bbox.__dict__.values()))

            if key not in unique_blocks:
                unique_blocks[key] = block
            else:
                # Keep block with higher confidence
                if block.confidence > unique_blocks[key].confidence:
                    unique_blocks[key] = block

        return list(unique_blocks.values())


    def _postprocess_text(self, text: str) -> str:
        """
        Apply postprocessing to a single text string.

        Args:
            text: Text to process

        Returns:
            Processed text
        """
        if not text:
            return text

        # Get spell correction setting
        spell_correction = self.config.get('system', {}).get('spell_correction', 'Disabled')

        # Normalize whitespace
        if self.config.get('postprocessing', {}).get('normalize_whitespace', True):
            text = self.normalizer.normalize_whitespace(text)

        # Korean spacing (apply if text contains Korean characters)
        if any('\uac00' <= char <= '\ud7a3' for char in text):
            text = self.normalizer.normalize_korean_spacing(text)

        # Normalize dates, currency, etc.
        if self.config.get('postprocessing', {}).get('fix_common_errors', True):
            text = self.normalizer.normalize_dates(text)
            text = self.normalizer.normalize_currency(text)

        # Apply spell correction
        if spell_correction and spell_correction != 'Disabled':
            try:
                text = self.spell_corrector.correct(text, spell_correction)
            except Exception as e:
                logger.warning(f"Spell correction failed: {e}")

        return text

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
                block.text = self._postprocess_text(block.text)

        return page_result

    def _get_output_path(self, suffix: str) -> Path:
        """
        Get output file path with specified suffix.

        Args:
            suffix: File suffix (e.g., '_ocr.txt', '_ocr.pdf')

        Returns:
            Path for output file
        """
        if self.output_directory:
            # Use specified output directory
            output_dir = self.output_directory
        else:
            # Use same directory as input file
            output_dir = self.file_path.parent

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename
        base_name = self.file_path.stem
        return output_dir / f"{base_name}{suffix}"

    def _export_results(self, ocr_result: OCRResult) -> Dict[str, Path]:
        """
        Export OCR results in selected formats.

        Args:
            ocr_result: OCRResult to export

        Returns:
            Dictionary mapping format name to output file path
        """
        output_files = {}

        try:
            for format_type in self.export_formats:
                if format_type == 'txt':
                    output_path = self._export_txt(ocr_result)
                    output_files['txt'] = output_path
                    self.signals.log_message.emit(f"Saved TXT: {output_path}")

                elif format_type == 'json':
                    output_path = self._export_json(ocr_result)
                    output_files['json'] = output_path
                    self.signals.log_message.emit(f"Saved JSON: {output_path}")

                elif format_type == 'markdown':
                    output_path = self._export_markdown(ocr_result)
                    output_files['markdown'] = output_path
                    self.signals.log_message.emit(f"Saved Markdown: {output_path}")

                elif format_type == 'searchable_pdf':
                    output_path = self._generate_searchable_pdf(ocr_result)
                    output_files['searchable_pdf'] = output_path
                    self.signals.log_message.emit(f"Saved Searchable PDF: {output_path}")

                elif format_type == 'tables_csv':
                    table_files = self._export_tables_csv(ocr_result)
                    if table_files:
                        output_files['tables_csv'] = table_files
                        self.signals.log_message.emit(f"Saved {len(table_files)} table(s) as CSV")

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            self.signals.log_message.emit(f"Export error: {str(e)}")

        return output_files

    def _export_txt(self, ocr_result: OCRResult) -> Path:
        """Export as plain text."""
        try:
            output_path = self._get_output_path('_ocr.txt')

            lines = []
            for page in ocr_result.pages:
                for block in page.blocks:
                    if block.text:
                        lines.append(block.text)
                lines.append('\n')  # Page separator

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            logger.info(f"Saved TXT file: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export TXT: {e}")
            raise

    def _export_json(self, ocr_result: OCRResult) -> Path:
        """Export as JSON."""
        try:
            import json
            from dataclasses import asdict

            output_path = self._get_output_path('_ocr.json')

            # Convert to dict
            result_dict = asdict(ocr_result)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved JSON file: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            raise

    def _export_markdown(self, ocr_result: OCRResult) -> Path:
        """Export as Markdown."""
        try:
            output_path = self._get_output_path('_ocr.md')

            lines = ['# OCR Result\n']
            for page in ocr_result.pages:
                lines.append(f'## Page {page.page_number}\n')
                for block in page.blocks:
                    if block.text:
                        lines.append(block.text)
                        lines.append('')

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            logger.info(f"Saved Markdown file: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export Markdown: {e}")
            raise

    def _generate_searchable_pdf(self, ocr_result: OCRResult) -> Path:
        """
        Generate searchable PDF from OCR result.

        Args:
            ocr_result: OCRResult

        Returns:
            Path to searchable PDF
        """
        try:
            output_path = self._get_output_path('_searchable.pdf')

            logger.info(f"Generating searchable PDF: {output_path}")

            # Use SearchablePDFGenerator to create searchable PDF
            actual_output_path = self.pdf_generator.create_from_ocr_result(
                input_pdf_path=self.file_path,
                output_pdf_path=output_path,
                ocr_result=ocr_result,
            )

            logger.info(f"Saved Searchable PDF: {actual_output_path}")
            return actual_output_path
        except Exception as e:
            logger.error(f"Failed to generate searchable PDF: {e}")
            self.signals.log_message.emit(f"Warning: Searchable PDF generation failed: {str(e)}")
            # Return original file if generation fails
            return self.file_path

    def _export_tables_csv(self, ocr_result: OCRResult) -> List[Path]:
        """
        Export detected tables as CSV files.

        Args:
            ocr_result: OCRResult containing table data

        Returns:
            List of paths to exported CSV files
        """
        import csv

        output_files = []
        min_rows = self.table_settings.get('min_rows', 3)
        min_cols = self.table_settings.get('min_cols', 2)

        try:
            table_count = 0

            for page_idx, page in enumerate(ocr_result.pages, start=1):
                # Check if page has tables
                if not hasattr(page, 'tables') or not page.tables:
                    continue

                for table_idx, table in enumerate(page.tables, start=1):
                    # Check table meets minimum requirements
                    structure = table.get('structure', {})
                    num_rows = structure.get('num_rows', 0)
                    num_cols = structure.get('num_cols', 0)

                    if num_rows < min_rows or num_cols < min_cols:
                        logger.debug(f"Skipping table (page {page_idx}, table {table_idx}): "
                                   f"{num_rows}x{num_cols} < minimum {min_rows}x{min_cols}")
                        continue

                    # Build table as 2D array
                    cells = table.get('cells', [])
                    if not cells:
                        continue

                    # Initialize 2D array
                    table_array = [['' for _ in range(num_cols)] for _ in range(num_rows)]

                    # Fill cells (handling spans)
                    for cell in cells:
                        row = cell.get('row', 0)
                        col = cell.get('col', 0)
                        text = cell.get('text', '').strip()
                        row_span = cell.get('row_span', 1)
                        col_span = cell.get('col_span', 1)

                        # Apply postprocessing to table cell text
                        if text:
                            text = self._postprocess_text(text)

                        # Fill cell and handle spans
                        if row < num_rows and col < num_cols:
                            table_array[row][col] = text

                            # Mark spanned cells (optional: repeat value or leave empty)
                            for r in range(row, min(row + row_span, num_rows)):
                                for c in range(col, min(col + col_span, num_cols)):
                                    if r == row and c == col:
                                        continue  # Skip original cell
                                    table_array[r][c] = ''  # Leave spanned cells empty

                    # Export to CSV
                    table_count += 1
                    output_path = self._get_output_path(f'_table_p{page_idx}_t{table_idx}.csv')

                    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(table_array)

                    output_files.append(output_path)
                    logger.info(f"Saved table CSV: {output_path} ({num_rows}x{num_cols})")

            if table_count > 0:
                self.signals.log_message.emit(f"Extracted {table_count} table(s) to CSV")
            else:
                self.signals.log_message.emit("No tables found meeting minimum criteria")

        except Exception as e:
            logger.error(f"Failed to export tables: {e}")
            self.signals.log_message.emit(f"Warning: Table export failed: {str(e)}")

        return output_files

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

    def _process_with_docai(self, page_image, page_num: int, processor_type: str = 'ocr') -> List[Block]:
        """
        Process page with Google Cloud Document AI.

        Args:
            page_image: PageImage object
            page_num: Page number
            processor_type: 'ocr', 'form', or 'layout'

        Returns:
            List of Block objects
        """
        if not self.docai_client:
            raise RuntimeError("DocAI client not initialized")

        self.signals.log_message.emit(f"Processing page {page_num} with DocAI ({processor_type})...")

        # Convert PIL image to bytes
        img_bytes = io.BytesIO()
        page_image.image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Process with DocAI
        result = self.docai_client.process_and_extract(
            input_data=img_bytes.read(),
            processor_type=processor_type,
            mime_type='image/png'
        )

        # Check for errors
        if 'error' in result:
            raise RuntimeError(f"DocAI error: {result['error']}")

        # Convert to Block objects
        blocks = []
        if result.get('pages'):
            page_data = result['pages'][0]  # Single page

            for block_data in page_data.get('blocks', []):
                # Convert bbox list to BoundingBox object
                bbox_norm = block_data.get('bbox_norm', [0, 0, 0, 0])
                if isinstance(bbox_norm, list) and len(bbox_norm) >= 4:
                    bbox = BoundingBox(
                        x=bbox_norm[0],
                        y=bbox_norm[1],
                        width=bbox_norm[2],
                        height=bbox_norm[3],
                        page=page_num - 1  # 0-indexed
                    )
                else:
                    bbox = BoundingBox(x=0, y=0, width=0, height=0, page=page_num - 1)

                # Map block type string to BlockType enum
                block_type_str = block_data.get('type', 'paragraph')
                if block_type_str == 'paragraph':
                    block_type = BlockType.PARAGRAPH
                elif block_type_str == 'line':
                    block_type = BlockType.TEXT_BLOCK
                else:
                    block_type = BlockType.TEXT_BLOCK

                block = Block(
                    text=block_data.get('text', ''),
                    bbox=bbox,
                    confidence=block_data.get('conf', 1.0),
                    block_type=block_type,
                )
                blocks.append(block)

        self.signals.log_message.emit(
            f"DocAI processed page {page_num}: {len(blocks)} blocks, "
            f"avg confidence: {result.get('pages', [{}])[0].get('avg_conf', 0):.2f}"
        )

        return blocks


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
