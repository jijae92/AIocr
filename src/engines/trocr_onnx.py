"""
TrOCR engine with ONNX Runtime support for INT8 quantization.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from util.device import get_device_manager
from util.logging import get_logger
from util.timing import timeit

logger = get_logger(__name__)


class TrOCREngine:
    """TrOCR OCR engine with optional ONNX backend."""

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        onnx_model_path: Optional[Path] = None,
        use_onnx: bool = False,
        max_length: int = 512,
        batch_size: int = 4,
    ):
        """
        Initialize TrOCR engine.

        Args:
            model_name: Model name from HuggingFace
            onnx_model_path: Path to ONNX model file
            use_onnx: Whether to use ONNX backend
            max_length: Maximum generation length
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.onnx_model_path = onnx_model_path
        self.use_onnx = use_onnx
        self.max_length = max_length
        self.batch_size = batch_size

        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Load model and processor
        self.processor = None
        self.model = None
        self.onnx_session = None
        self._load_model()

        logger.info(f"Initialized TrOCR engine with {model_name}")

    def _load_model(self):
        """Load model and processor."""
        try:
            logger.info(f"Loading TrOCR model: {self.model_name}")

            # Load processor
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)

            if self.use_onnx and self.onnx_model_path:
                # Load ONNX model
                logger.info(f"Loading ONNX model from {self.onnx_model_path}")
                self.onnx_session = ort.InferenceSession(
                    str(self.onnx_model_path),
                    providers=['CPUExecutionProvider'],
                )
            else:
                # Load PyTorch model
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                self.model = self.device_manager.move_to_device(self.model)
                self.model.eval()

            logger.info("TrOCR model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load TrOCR model: {e}")
            raise

    @timeit(name="TrOCR OCR")
    def process_image(self, image: Image.Image) -> Dict:
        """
        Process image with TrOCR.

        Args:
            image: PIL Image

        Returns:
            Dictionary with OCR results
        """
        try:
            # Prepare inputs
            pixel_values = self.processor(
                image, return_tensors="pt"
            ).pixel_values

            if self.use_onnx and self.onnx_session:
                # ONNX inference
                pixel_values_np = pixel_values.numpy()
                onnx_inputs = {self.onnx_session.get_inputs()[0].name: pixel_values_np}
                outputs = self.onnx_session.run(None, onnx_inputs)
                generated_ids = outputs[0]
            else:
                # PyTorch inference
                pixel_values = self.device_manager.move_to_device(pixel_values)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=self.max_length,
                    )

                generated_ids = generated_ids.cpu().numpy()

            # Decode
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return {
                'text': text,
                'confidence': 0.9,  # TrOCR doesn't provide confidence scores
                'engine': 'trocr',
            }

        except Exception as e:
            logger.error(f"TrOCR processing failed: {e}")
            raise

    def process_batch(self, images: List[Image.Image]) -> List[Dict]:
        """
        Process batch of images.

        Args:
            images: List of PIL Images

        Returns:
            List of results
        """
        results = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]

            for image in batch:
                result = self.process_image(image)
                results.append(result)

        return results

    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        if self.onnx_session is not None:
            del self.onnx_session
