"""Google Cloud Document AI model wrapper."""

from typing import List, Optional
from PIL import Image

from .base_model import BaseOCRModel
from ..data.ocr_result import PageResult, Block, BoundingBox
from ..data.block_types import BlockType


class DocAIModel(BaseOCRModel):
    """
    Google Cloud Document AI wrapper.

    Provides high-quality OCR with good support for:
    - Multiple languages
    - Tables and forms
    - Structured documents
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize DocAI model."""
        super().__init__(config)
        self.model_name = "docai"
        self.project_id = config.get("project_id") if config else None
        self.processor_id = config.get("processor_id") if config else None

    def load_model(self) -> None:
        """Load DocAI client."""
        # TODO: Initialize Google Cloud Document AI client
        # from google.cloud import documentai_v1 as documentai
        # self._model = documentai.DocumentProcessorServiceClient()
        self._is_loaded = True
        print(f"[{self.model_name}] Model loaded (stub)")

    def predict(
        self, image: Image.Image, language: Optional[str] = None
    ) -> PageResult:
        """
        Run Document AI OCR.

        Args:
            image: Input image
            language: Optional language hint

        Returns:
            PageResult with OCR output
        """
        if not self._is_loaded:
            self.load_model()

        # TODO: Implement actual Document AI API call
        # For now, return stub result
        print(f"[{self.model_name}] Processing image (stub)")

        return PageResult(
            page_number=0,
            blocks=[],
            width=image.width,
            height=image.height,
            language=language,
            metadata={"model": self.model_name, "stub": True},
        )

    def get_confidence(self) -> float:
        """Get overall confidence."""
        # TODO: Return actual confidence from last prediction
        return 0.9

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return ["en", "ko", "ja", "zh", "es", "fr", "de"]

    def get_model_size(self) -> int:
        """Get model size (API-based, minimal local footprint)."""
        return 10  # MB

    def get_inference_speed(self) -> float:
        """Get inference speed (API latency dependent)."""
        return 0.8
