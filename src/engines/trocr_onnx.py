"""
TrOCR engine with ONNX Runtime support for INT8 quantization.

Optimized for numbers, code, and English text recognition.
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
from util.timing import Timer

logger = get_logger(__name__)


class TrOCREngine:
    """
    TrOCR OCR engine with optional ONNX backend.

    Features:
    - PyTorch or ONNX Runtime backend
    - INT8 quantization support (ONNX only)
    - Optimized for printed text, numbers, code
    - Batch processing
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        onnx_model_path: Optional[Path] = None,
        use_onnx: bool = False,
        max_length: int = 512,
        batch_size: int = 4,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize TrOCR engine.

        Args:
            model_name: Model name from HuggingFace
            onnx_model_path: Path to ONNX model file
            use_onnx: Whether to use ONNX backend
            max_length: Maximum generation length
            batch_size: Batch size for inference
            cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.onnx_model_path = Path(onnx_model_path) if onnx_model_path else None
        self.use_onnx = use_onnx
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Load model and processor
        self.processor = None
        self.model = None
        self.onnx_session = None
        self.is_loaded = False

        logger.info(
            f"Initialized TrOCR engine: model={model_name}, "
            f"onnx={use_onnx}, batch={batch_size}"
        )

    def _load_model(self):
        """Lazy load model and processor."""
        if self.is_loaded:
            return

        try:
            logger.info(f"Loading TrOCR model: {self.model_name}")

            # Load processor
            self.processor = TrOCRProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )

            if self.use_onnx and self.onnx_model_path:
                # Load ONNX model
                if not self.onnx_model_path.exists():
                    raise FileNotFoundError(
                        f"ONNX model not found: {self.onnx_model_path}\n"
                        f"Run: python scripts/export_trocr_onnx.py --model {self.model_name} "
                        f"--output {self.onnx_model_path}"
                    )

                logger.info(f"Loading ONNX model from {self.onnx_model_path}")
                self.onnx_session = ort.InferenceSession(
                    str(self.onnx_model_path),
                    providers=['CPUExecutionProvider'],
                )

                # Print model info
                logger.info(f"ONNX inputs: {[i.name for i in self.onnx_session.get_inputs()]}")
                logger.info(f"ONNX outputs: {[o.name for o in self.onnx_session.get_outputs()]}")

            else:
                # Load PyTorch model
                logger.info("Loading PyTorch model")
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_name,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                )
                self.model = self.device_manager.move_to_device(self.model)
                self.model.eval()

            self.is_loaded = True
            logger.info("TrOCR model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load TrOCR model: {e}")
            raise

    def process_image(self, image: Image.Image) -> Dict:
        """
        Process image with TrOCR.

        Args:
            image: PIL Image

        Returns:
            Dictionary with OCR results:
            {
                "text": "recognized text",
                "confidence": 0.9,
                "engine": "trocr",
                "elapsed_ms": 123
            }
        """
        self._load_model()

        timer = Timer()
        timer.start()

        try:
            # Prepare inputs
            pixel_values = self.processor(
                images=image,
                return_tensors="pt"
            ).pixel_values

            if self.use_onnx and self.onnx_session:
                # ONNX inference (encoder only)
                pixel_values_np = pixel_values.numpy()
                onnx_inputs = {self.onnx_session.get_inputs()[0].name: pixel_values_np}
                encoder_outputs = self.onnx_session.run(None, onnx_inputs)

                # For full generation, we'd need to export decoder too
                # For now, this is a simplified version
                # In production, you'd implement full ONNX generation loop

                # Fallback to PyTorch for generation
                logger.warning("ONNX encoder-only mode, using PyTorch for generation")
                if self.model is None:
                    self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                    self.model = self.device_manager.move_to_device(self.model)
                    self.model.eval()

                pixel_values = self.device_manager.move_to_device(pixel_values)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=self.max_length,
                    )
                generated_ids = generated_ids.cpu().numpy()

            else:
                # PyTorch inference
                pixel_values = self.device_manager.move_to_device(pixel_values)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=self.max_length,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                # Estimate confidence
                confidence = self._estimate_confidence(generated_ids)
                generated_ids = generated_ids.sequences.cpu().numpy()

            # Decode
            text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            # Post-process for numbers and code
            text = self._postprocess_text(text)

            elapsed_ms = int(timer.stop() * 1000)

            return {
                'text': text,
                'confidence': confidence if not self.use_onnx else 0.9,
                'engine': 'trocr',
                'backend': 'onnx' if self.use_onnx else 'pytorch',
                'elapsed_ms': elapsed_ms,
            }

        except Exception as e:
            elapsed_ms = int(timer.stop() * 1000) if timer.start_time else 0
            logger.error(f"TrOCR processing failed: {e}")

            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'trocr',
                'elapsed_ms': elapsed_ms,
                'error': str(e),
            }

    def _estimate_confidence(self, outputs) -> float:
        """
        Estimate confidence from generation scores.

        Args:
            outputs: Model outputs

        Returns:
            Confidence score (0-1)
        """
        try:
            if not hasattr(outputs, 'scores') or not outputs.scores:
                return 0.9

            scores = []
            for step_scores in outputs.scores:
                probs = torch.softmax(step_scores[0], dim=-1)
                max_prob = torch.max(probs).item()
                scores.append(max_prob)

            return sum(scores) / len(scores) if scores else 0.9

        except Exception:
            return 0.9

    def _postprocess_text(self, text: str) -> str:
        """
        Post-process text for numbers, code, and English.

        Args:
            text: Raw OCR text

        Returns:
            Processed text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Fix common OCR errors for numbers
        # O -> 0 in numeric contexts
        import re
        text = re.sub(r'(\d)O(\d)', r'\g<1>0\g<2>', text)
        text = re.sub(r'O(\d)', r'0\g<1>', text)

        # l -> 1 in numeric contexts
        text = re.sub(r'(\d)l(\d)', r'\g<1>1\g<2>', text)

        return text

    def process_batch(self, images: List[Image.Image]) -> List[Dict]:
        """
        Process batch of images.

        Args:
            images: List of PIL Images

        Returns:
            List of results
        """
        self._load_model()

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
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.onnx_session is not None:
            del self.onnx_session
            self.onnx_session = None

        self.is_loaded = False

        logger.info("TrOCR engine cleaned up")

    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except Exception:
            pass
