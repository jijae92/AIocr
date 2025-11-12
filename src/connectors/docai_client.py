"""
Google Cloud Document AI client for OCR and document parsing.

Supports single/multi-page processing with standardized output schema.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPIError, RetryError
from google.cloud import documentai_v1 as documentai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from util.coords import BoundingBox, TextBlock
from util.logging import get_logger
from util.timing import Timer

logger = get_logger(__name__)


class DocumentAIClient:
    """
    Client for Google Cloud Document AI.

    Features:
    - Single/multi-page PDF processing
    - Bytes or file path input
    - Standardized output schema
    - Automatic retry with exponential backoff
    - Table extraction
    - Language detection
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        processor_id_ocr: Optional[str] = None,
        processor_id_form: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """
        Initialize Document AI client.

        Args:
            project_id: GCP project ID
            location: GCP location (e.g., 'us', 'eu')
            processor_id_ocr: OCR processor ID
            processor_id_form: Form parser processor ID
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        # Get from environment if not provided
        self.project_id = project_id or os.getenv('GCP_PROJECT')
        self.location = location or os.getenv('GCP_LOCATION', 'us')
        self.processor_id_ocr = processor_id_ocr or os.getenv('DOCAI_PROCESSOR_ID_OCR')
        self.processor_id_form = processor_id_form or os.getenv('DOCAI_PROCESSOR_ID_FORM')
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.project_id:
            raise ValueError("GCP project ID not provided")

        # Initialize client
        opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)

        logger.info(
            f"Initialized Document AI client: project={self.project_id}, "
            f"location={self.location}"
        )

    def _get_processor_name(self, processor_type: str = 'ocr') -> str:
        """Get processor resource name."""
        if processor_type == 'ocr':
            processor_id = self.processor_id_ocr
        elif processor_type == 'form':
            processor_id = self.processor_id_form
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")

        if not processor_id:
            raise ValueError(f"Processor ID not configured for type: {processor_type}")

        return self.client.processor_path(self.project_id, self.location, processor_id)

    @retry(
        retry=retry_if_exception_type((GoogleAPIError, RetryError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _process_raw_document(
        self,
        content: bytes,
        mime_type: str,
        processor_type: str = 'ocr',
    ) -> documentai.Document:
        """
        Process raw document content.

        Args:
            content: Document bytes
            mime_type: MIME type
            processor_type: Processor type

        Returns:
            Document AI Document object
        """
        processor_name = self._get_processor_name(processor_type)
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)

        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=raw_document,
        )

        try:
            result = self.client.process_document(request=request, timeout=self.timeout)
            return result.document
        except GoogleAPIError as e:
            logger.error(f"Document AI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during processing: {e}")
            raise

    def process_document(
        self,
        input_data: Union[Path, bytes],
        processor_type: str = 'ocr',
        mime_type: Optional[str] = None,
    ) -> documentai.Document:
        """
        Process document from file path or bytes.

        Args:
            input_data: Path to file or document bytes
            processor_type: Processor type ('ocr' or 'form')
            mime_type: MIME type (auto-detected for Path)

        Returns:
            Document AI Document object
        """
        # Handle file path
        if isinstance(input_data, (Path, str)):
            file_path = Path(input_data)
            logger.info(f"Processing file: {file_path.name}")

            with open(file_path, 'rb') as f:
                content = f.read()

            # Auto-detect MIME type
            if mime_type is None:
                mime_type = self._detect_mime_type(file_path)

        # Handle bytes
        elif isinstance(input_data, bytes):
            content = input_data
            if mime_type is None:
                mime_type = 'application/pdf'  # Default
            logger.info(f"Processing {len(content)} bytes")

        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        # Process
        return self._process_raw_document(content, mime_type, processor_type)

    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type from file extension."""
        mime_types = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
        }
        suffix = file_path.suffix.lower()
        return mime_types.get(suffix, 'application/pdf')

    def extract_page_data(
        self,
        document: documentai.Document,
    ) -> List[Dict]:
        """
        Extract page-level data with standardized schema.

        Returns:
            List of page dictionaries with schema:
            {
                "page_index": 0,
                "text": "full page text",
                "blocks": [{"text", "bbox_norm", "conf", "type"}],
                "tables": [{"cells", "structure", "conf"}],
                "avg_conf": 0.91,
                "lang": ["ko", "en"]
            }
        """
        pages_data = []

        for page_idx, page in enumerate(document.pages):
            # Page dimensions
            page_width = page.dimension.width
            page_height = page.dimension.height

            # Extract blocks
            blocks = self._extract_page_blocks(page, page_width, page_height, document.text)

            # Extract tables
            tables = self._extract_page_tables(page, page_width, page_height, document.text)

            # Extract page text
            page_text = self._extract_page_text(page, document.text)

            # Calculate average confidence
            avg_conf = self._calculate_page_confidence(page)

            # Detect languages
            languages = self._extract_page_languages(page)

            page_data = {
                'page_index': page_idx,
                'text': page_text,
                'blocks': blocks,
                'tables': tables,
                'avg_conf': avg_conf,
                'lang': languages,
            }

            pages_data.append(page_data)

        return pages_data

    def _extract_page_blocks(
        self,
        page: documentai.Document.Page,
        page_width: float,
        page_height: float,
        document_text: str,
    ) -> List[Dict]:
        """Extract blocks from page."""
        blocks = []

        # Process paragraphs
        for para in page.paragraphs:
            text = self._get_text_from_layout(para.layout, document_text)
            bbox_norm = self._extract_normalized_bbox(para.layout)
            conf = para.layout.confidence if para.layout.confidence else 1.0

            blocks.append({
                'text': text,
                'bbox_norm': bbox_norm,
                'conf': conf,
                'type': 'paragraph',
            })

        # Process lines (if no paragraphs)
        if not blocks:
            for line in page.lines:
                text = self._get_text_from_layout(line.layout, document_text)
                bbox_norm = self._extract_normalized_bbox(line.layout)
                conf = line.layout.confidence if line.layout.confidence else 1.0

                blocks.append({
                    'text': text,
                    'bbox_norm': bbox_norm,
                    'conf': conf,
                    'type': 'line',
                })

        return blocks

    def _extract_page_tables(
        self,
        page: documentai.Document.Page,
        page_width: float,
        page_height: float,
        document_text: str,
    ) -> List[Dict]:
        """Extract tables from page."""
        tables = []

        for table in page.tables:
            # Extract table structure
            rows = []
            for row_idx, row in enumerate(table.body_rows):
                cells = []
                for cell_idx, cell in enumerate(row.cells):
                    cell_text = self._get_text_from_layout(cell.layout, document_text)
                    cell_bbox = self._extract_normalized_bbox(cell.layout)
                    cell_conf = cell.layout.confidence if cell.layout.confidence else 1.0

                    cells.append({
                        'text': cell_text,
                        'bbox_norm': cell_bbox,
                        'conf': cell_conf,
                        'row': row_idx,
                        'col': cell_idx,
                        'row_span': cell.row_span if cell.row_span else 1,
                        'col_span': cell.col_span if cell.col_span else 1,
                    })
                rows.append(cells)

            # Table confidence (average of cell confidences)
            cell_confs = [
                cell['conf'] for row in rows for cell in row if cell['conf'] > 0
            ]
            table_conf = sum(cell_confs) / len(cell_confs) if cell_confs else 1.0

            tables.append({
                'cells': [cell for row in rows for cell in row],
                'structure': {
                    'num_rows': len(rows),
                    'num_cols': max(len(row) for row in rows) if rows else 0,
                },
                'conf': table_conf,
            })

        return tables

    def _extract_page_text(self, page: documentai.Document.Page, document_text: str) -> str:
        """Extract full text from page."""
        if not page.layout or not page.layout.text_anchor:
            return ""

        return self._get_text_from_layout(page.layout, document_text)

    def _calculate_page_confidence(self, page: documentai.Document.Page) -> float:
        """Calculate average confidence for page."""
        confidences = []

        # Collect from paragraphs
        for para in page.paragraphs:
            if para.layout.confidence:
                confidences.append(para.layout.confidence)

        # Collect from lines if no paragraphs
        if not confidences:
            for line in page.lines:
                if line.layout.confidence:
                    confidences.append(line.layout.confidence)

        # Collect from tokens as fallback
        if not confidences:
            for token in page.tokens:
                if token.layout.confidence:
                    confidences.append(token.layout.confidence)

        return sum(confidences) / len(confidences) if confidences else 1.0

    def _extract_page_languages(self, page: documentai.Document.Page) -> List[str]:
        """Extract detected languages from page."""
        languages = set()

        # Check page-level language
        if hasattr(page, 'detected_languages'):
            for lang in page.detected_languages:
                if lang.language_code:
                    languages.add(lang.language_code)

        return sorted(list(languages)) if languages else []

    def _get_text_from_layout(self, layout, document_text: str) -> str:
        """Extract text from layout object."""
        if not layout or not layout.text_anchor:
            return ""

        text_segments = []
        for segment in layout.text_anchor.text_segments:
            start_idx = int(segment.start_index) if segment.start_index else 0
            end_idx = int(segment.end_index) if segment.end_index else len(document_text)
            text_segments.append(document_text[start_idx:end_idx])

        return ''.join(text_segments)

    def _extract_normalized_bbox(self, layout) -> List[float]:
        """
        Extract normalized bounding box [x, y, w, h].

        Returns coordinates normalized to 0-1 range.
        """
        if not layout or not layout.bounding_poly or not layout.bounding_poly.normalized_vertices:
            return [0.0, 0.0, 0.0, 0.0]

        vertices = layout.bounding_poly.normalized_vertices

        # Get min/max coordinates
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def process_and_extract(
        self,
        input_data: Union[Path, bytes],
        processor_type: str = 'ocr',
        mime_type: Optional[str] = None,
    ) -> Dict:
        """
        Process document and extract structured data with standardized schema.

        Args:
            input_data: Path to file or document bytes
            processor_type: Processor type
            mime_type: MIME type (auto-detected for files)

        Returns:
            Standardized output schema:
            {
                "pages": [
                    {
                        "page_index": 0,
                        "text": "...",
                        "blocks": [{...}],
                        "tables": [{...}],
                        "avg_conf": 0.91,
                        "lang": ["ko", "en"]
                    }
                ],
                "elapsed_ms": 1234,
                "processor_type": "ocr",
                "page_count": 5
            }
        """
        timer = Timer()
        timer.start()

        try:
            # Process document
            document = self.process_document(input_data, processor_type, mime_type)

            # Extract page data
            pages_data = self.extract_page_data(document)

            # Calculate elapsed time
            elapsed_ms = int(timer.stop() * 1000)

            return {
                'pages': pages_data,
                'elapsed_ms': elapsed_ms,
                'processor_type': processor_type,
                'page_count': len(pages_data),
            }

        except Exception as e:
            elapsed_ms = int(timer.stop() * 1000) if timer.start_time else 0
            logger.error(f"Document processing failed: {e}")

            return {
                'pages': [],
                'elapsed_ms': elapsed_ms,
                'processor_type': processor_type,
                'page_count': 0,
                'error': str(e),
            }

    # Legacy methods for backward compatibility
    def extract_text_blocks(self, document: documentai.Document) -> List[TextBlock]:
        """
        Extract text blocks from Document AI result (legacy format).

        Args:
            document: Document AI Document

        Returns:
            List of TextBlock objects
        """
        blocks = []

        if not document.pages:
            return blocks

        for page_idx, page in enumerate(document.pages):
            page_width = page.dimension.width
            page_height = page.dimension.height

            # Extract paragraphs
            for para in page.paragraphs:
                text = self._get_text_from_layout(para.layout, document.text)
                bbox_norm = self._extract_normalized_bbox(para.layout)
                conf = para.layout.confidence if para.layout.confidence else 1.0

                # Convert normalized bbox to absolute
                bbox = BoundingBox(
                    x=bbox_norm[0] * page_width,
                    y=bbox_norm[1] * page_height,
                    width=bbox_norm[2] * page_width,
                    height=bbox_norm[3] * page_height,
                    page=page_idx,
                    confidence=conf,
                )

                blocks.append(
                    TextBlock(
                        text=text,
                        bbox=bbox,
                        confidence=conf,
                        block_type='paragraph',
                    )
                )

        return blocks

    def extract_all_text(self, document: documentai.Document) -> str:
        """Extract all text from document."""
        return document.text

    def extract_confidence(self, document: documentai.Document) -> float:
        """Extract average confidence from document."""
        pages_data = self.extract_page_data(document)
        if not pages_data:
            return 1.0

        avg_confs = [p['avg_conf'] for p in pages_data]
        return sum(avg_confs) / len(avg_confs)
