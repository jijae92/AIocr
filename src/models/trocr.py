"""TrOCR (Transformer-based OCR) model wrapper with ONNX support."""

from typing import List, Optional
from PIL import Image

from .base_model import BaseOCRModel
from ..data.ocr_result import PageResult


class TrOCRModel(BaseOCRModel):
    """
    Microsoft TrOCR model wrapper.

    Specialized for:
    - Numeric content
    - Dates and codes
    - Cross-validation of critical fields
    - ONNX inference for speed
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize TrOCR model."""
        super().__init__(config)
        self.model_name = "trocr"
        self.use_onnx = config.get("use_onnx", True) if config else True

    def load_model(self) -> None:
        """Load TrOCR model."""
        # TODO: Load TrOCR model
        if self.use_onnx:
            # Load ONNX version
            # import onnxruntime as ort
            # self._model = ort.InferenceSession("trocr_model.onnx")
            print(f"[{self.model_name}] ONNX model loaded (stub)")
        else:
            # Load PyTorch version
            # from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            # self._processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            # self._model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            print(f"[{self.model_name}] PyTorch model loaded (stub)")

        self._is_loaded = True

    def predict(
        self, image: Image.Image, language: Optional[str] = None
    ) -> PageResult:
        """Run TrOCR prediction."""
        if not self._is_loaded:
            self.load_model()

        print(f"[{self.model_name}] Processing image (stub)")

        return PageResult(
            page_number=0,
            blocks=[],
            width=image.width,
            height=image.height,
            language=language,
            metadata={
                "model": self.model_name,
                "onnx": self.use_onnx,
                "stub": True,
            },
        )

    def get_confidence(self) -> float:
        """Get confidence."""
        return 0.88

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return ["en", "zh", "ja", "ko"]

    def get_model_size(self) -> int:
        """Get model size."""
        if self.use_onnx:
            return 150  # MB (ONNX is more compact)
        return 300  # MB (PyTorch)

    def get_inference_speed(self) -> float:
        """Get inference speed."""
        if self.use_onnx:
            return 1.5  # Faster with ONNX
        return 0.9
