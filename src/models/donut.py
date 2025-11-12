"""Donut (Document Understanding Transformer) model wrapper."""

from typing import List, Optional
from PIL import Image

from .base_model import BaseOCRModel
from ..data.ocr_result import PageResult


class DonutModel(BaseOCRModel):
    """
    Donut model wrapper for document understanding.

    Good for:
    - Complex layouts
    - Multi-column documents
    - Document reinterpretation
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize Donut model."""
        super().__init__(config)
        self.model_name = "donut"

    def load_model(self) -> None:
        """Load Donut model."""
        # TODO: Load Donut model from transformers
        # from transformers import DonutProcessor, VisionEncoderDecoderModel
        # self._processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        # self._model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        self._is_loaded = True
        print(f"[{self.model_name}] Model loaded (stub)")

    def predict(
        self, image: Image.Image, language: Optional[str] = None
    ) -> PageResult:
        """Run Donut prediction."""
        if not self._is_loaded:
            self.load_model()

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
        """Get confidence."""
        return 0.85

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return ["en", "ko", "ja", "zh"]

    def get_model_size(self) -> int:
        """Get model size."""
        return 500  # MB

    def get_inference_speed(self) -> float:
        """Get inference speed."""
        return 0.6  # Slower than DocAI but more thorough
