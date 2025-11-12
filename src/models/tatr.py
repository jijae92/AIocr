"""TATR (Table Transformer) model wrapper."""

from typing import List, Optional
from PIL import Image

from .base_model import BaseOCRModel
from ..data.ocr_result import PageResult


class TATRModel(BaseOCRModel):
    """
    Table Transformer (TATR) model wrapper.

    Specialized for:
    - Table detection
    - Table structure recovery
    - Cell extraction and parsing
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize TATR model."""
        super().__init__(config)
        self.model_name = "tatr"
        self.min_confidence = config.get("min_table_confidence", 0.6) if config else 0.6

    def load_model(self) -> None:
        """Load TATR model."""
        # TODO: Load Table Transformer model
        # from transformers import AutoModelForObjectDetection
        # self._model = AutoModelForObjectDetection.from_pretrained(
        #     "microsoft/table-transformer-detection"
        # )
        self._is_loaded = True
        print(f"[{self.model_name}] Model loaded (stub)")

    def predict(
        self, image: Image.Image, language: Optional[str] = None
    ) -> PageResult:
        """Run table detection and extraction."""
        if not self._is_loaded:
            self.load_model()

        print(f"[{self.model_name}] Processing image for tables (stub)")

        return PageResult(
            page_number=0,
            blocks=[],
            width=image.width,
            height=image.height,
            language=language,
            metadata={
                "model": self.model_name,
                "min_confidence": self.min_confidence,
                "stub": True,
            },
        )

    def get_confidence(self) -> float:
        """Get confidence."""
        return 0.82

    def get_supported_languages(self) -> List[str]:
        """Get supported languages (language-agnostic for structure)."""
        return ["en", "ko", "ja", "zh", "all"]

    def get_model_size(self) -> int:
        """Get model size."""
        return 250  # MB

    def get_inference_speed(self) -> float:
        """Get inference speed."""
        return 0.7  # Slower, focused on structure
