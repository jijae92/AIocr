"""Base interface for OCR models."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from PIL import Image
import numpy as np

from ..data.ocr_result import OCRResult, PageResult


class BaseOCRModel(ABC):
    """
    Abstract base class for OCR models.

    All OCR model implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OCR model.

        Args:
            config: Model-specific configuration
        """
        self.config = config or {}
        self.model_name = self.__class__.__name__
        self._model = None
        self._is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def predict(
        self, image: Image.Image, language: Optional[str] = None
    ) -> PageResult:
        """
        Run OCR prediction on a single image/page.

        Args:
            image: Input image
            language: Optional language hint

        Returns:
            PageResult with detected text and metadata
        """
        pass

    def predict_batch(
        self, images: List[Image.Image], language: Optional[str] = None
    ) -> OCRResult:
        """
        Run OCR prediction on multiple images/pages.

        Args:
            images: List of input images
            language: Optional language hint

        Returns:
            OCRResult with all pages
        """
        pages = []
        for idx, image in enumerate(images):
            page = self.predict(image, language)
            page.page_number = idx
            pages.append(page)

        return OCRResult(
            pages=pages,
            metadata={"model": self.model_name, "language": language},
        )

    @abstractmethod
    def get_confidence(self) -> float:
        """
        Get overall confidence score for last prediction.

        Returns:
            Confidence score between 0 and 1
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.

        Returns:
            List of language codes (e.g., ['en', 'ko', 'ja'])
        """
        pass

    def get_model_size(self) -> int:
        """
        Get model memory footprint in MB.

        Returns:
            Model size in megabytes
        """
        return 0

    def get_inference_speed(self) -> float:
        """
        Get relative inference speed (higher is faster).

        Returns:
            Relative speed score
        """
        return 1.0

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def unload_model(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._is_loaded = False

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image before OCR.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Default: no preprocessing
        return image

    def supports_language(self, language: str) -> bool:
        """
        Check if model supports a specific language.

        Args:
            language: Language code

        Returns:
            True if language is supported
        """
        return language in self.get_supported_languages()

    def __str__(self) -> str:
        """String representation."""
        return f"{self.model_name}(loaded={self._is_loaded})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"{self.model_name}("
            f"loaded={self._is_loaded}, "
            f"languages={len(self.get_supported_languages())}, "
            f"size={self.get_model_size()}MB)"
        )
