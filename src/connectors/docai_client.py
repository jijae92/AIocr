"""
Google Cloud Document AI client for OCR and document parsing.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPIError
from google.cloud import documentai_v1 as documentai
from tenacity import retry, stop_after_attempt, wait_exponential

from util.coords import BoundingBox, TextBlock
from util.logging import get_logger
from util.timing import timeit

logger = get_logger(__name__)


class DocumentAIClient:
    """Client for Google Cloud Document AI."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        processor_id_ocr: Optional[str] = None,
        processor_id_form: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Initialize Document AI client.

        Args:
            project_id: GCP project ID
            location: GCP location (e.g., 'us', 'eu')
            processor_id_ocr: OCR processor ID
            processor_id_form: Form parser processor ID
            timeout: Request timeout in seconds
        """
        # Get from environment if not provided
        self.project_id = project_id or os.getenv('GCP_PROJECT')
        self.location = location or os.getenv('GCP_LOCATION', 'us')
        self.processor_id_ocr = processor_id_ocr or os.getenv('DOCAI_PROCESSOR_ID_OCR')
        self.processor_id_form = processor_id_form or os.getenv('DOCAI_PROCESSOR_ID_FORM')
        self.timeout = timeout

        if not self.project_id:
            raise ValueError("GCP project ID not provided")

        # Initialize client
        opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)

        logger.info(f"Initialized Document AI client for project {self.project_id}")

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
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    @timeit(name="Document AI OCR")
    def process_document(
        self,
        file_path: Path,
        processor_type: str = 'ocr',
        mime_type: Optional[str] = None,
    ) -> documentai.Document:
        """
        Process document with Document AI.

        Args:
            file_path: Path to document file
            processor_type: Processor type ('ocr' or 'form')
            mime_type: MIME type (auto-detected if None)

        Returns:
            Document AI Document object
        """
        # Read file
        with open(file_path, 'rb') as f:
            content = f.read()

        # Detect MIME type if not provided
        if mime_type is None:
            suffix = file_path.suffix.lower()
            mime_types = {
                '.pdf': 'application/pdf',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.tiff': 'image/tiff',
                '.tif': 'image/tiff',
            }
            mime_type = mime_types.get(suffix, 'application/pdf')

        # Create request
        processor_name = self._get_processor_name(processor_type)
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)

        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=raw_document,
        )

        # Process document
        try:
            logger.info(f"Processing {file_path.name} with Document AI ({processor_type})")
            result = self.client.process_document(request=request, timeout=self.timeout)
            return result.document

        except GoogleAPIError as e:
            logger.error(f"Document AI processing failed: {e}")
            raise

    def extract_text_blocks(self, document: documentai.Document) -> List[TextBlock]:
        """
        Extract text blocks from Document AI result.

        Args:
            document: Document AI Document

        Returns:
            List of TextBlock objects
        """
        blocks = []

        # Get page dimensions
        if not document.pages:
            return blocks

        for page_idx, page in enumerate(document.pages):
            page_width = page.dimension.width
            page_height = page.dimension.height

            # Extract blocks
            for block in page.blocks:
                # Get text
                text = self._get_text_from_layout(block.layout, document.text)

                # Get bounding box
                bbox = self._extract_bbox(block.layout, page_width, page_height, page_idx)

                # Get confidence
                confidence = block.layout.confidence if block.layout.confidence else 1.0

                blocks.append(
                    TextBlock(
                        text=text,
                        bbox=bbox,
                        confidence=confidence,
                        block_type='text',
                    )
                )

        return blocks

    def extract_all_text(self, document: documentai.Document) -> str:
        """
        Extract all text from document.

        Args:
            document: Document AI Document

        Returns:
            Extracted text
        """
        return document.text

    def extract_confidence(self, document: documentai.Document) -> float:
        """
        Extract average confidence from document.

        Args:
            document: Document AI Document

        Returns:
            Average confidence score (0-1)
        """
        confidences = []

        for page in document.pages:
            for block in page.blocks:
                if block.layout.confidence:
                    confidences.append(block.layout.confidence)

        if not confidences:
            return 1.0

        return sum(confidences) / len(confidences)

    def _get_text_from_layout(self, layout, document_text: str) -> str:
        """Extract text from layout object."""
        text_segments = []

        for segment in layout.text_anchor.text_segments:
            start_idx = int(segment.start_index) if segment.start_index else 0
            end_idx = int(segment.end_index) if segment.end_index else len(document_text)
            text_segments.append(document_text[start_idx:end_idx])

        return ''.join(text_segments)

    def _extract_bbox(
        self,
        layout,
        page_width: float,
        page_height: float,
        page_num: int,
    ) -> BoundingBox:
        """Extract bounding box from layout."""
        if not layout.bounding_poly.normalized_vertices:
            return BoundingBox(x=0, y=0, width=0, height=0, page=page_num)

        vertices = layout.bounding_poly.normalized_vertices

        # Get min/max coordinates
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Convert from normalized to absolute coordinates
        return BoundingBox(
            x=x_min * page_width,
            y=y_min * page_height,
            width=(x_max - x_min) * page_width,
            height=(y_max - y_min) * page_height,
            page=page_num,
            confidence=layout.confidence if layout.confidence else 1.0,
        )

    def process_and_extract(
        self,
        file_path: Path,
        processor_type: str = 'ocr',
    ) -> Dict:
        """
        Process document and extract structured data.

        Args:
            file_path: Path to document
            processor_type: Processor type

        Returns:
            Dictionary with extracted data
        """
        # Process document
        document = self.process_document(file_path, processor_type)

        # Extract data
        text_blocks = self.extract_text_blocks(document)
        full_text = self.extract_all_text(document)
        confidence = self.extract_confidence(document)

        return {
            'text': full_text,
            'blocks': [block.to_dict() for block in text_blocks],
            'confidence': confidence,
            'page_count': len(document.pages),
            'engine': 'document_ai',
            'processor_type': processor_type,
        }
