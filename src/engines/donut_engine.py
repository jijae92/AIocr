"""
Donut model engine with optional LoRA fine-tuning support.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from util.coords import TextBlock, BoundingBox
from util.device import get_device_manager
from util.logging import get_logger
from util.timing import timeit

logger = get_logger(__name__)


class DonutEngine:
    """Donut OCR engine."""

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        lora_adapter_path: Optional[Path] = None,
        use_lora: bool = False,
        max_length: int = 1024,
        batch_size: int = 1,
        quantization: str = "fp16",
    ):
        """
        Initialize Donut engine.

        Args:
            model_name: Base model name from HuggingFace
            lora_adapter_path: Path to LoRA adapter weights
            use_lora: Whether to use LoRA adapter
            max_length: Maximum generation length
            batch_size: Batch size for inference
            quantization: Quantization mode ('none', 'int8', 'fp16')
        """
        self.model_name = model_name
        self.lora_adapter_path = lora_adapter_path
        self.use_lora = use_lora
        self.max_length = max_length
        self.batch_size = batch_size
        self.quantization = quantization

        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Load model and processor
        self.processor = None
        self.model = None
        self._load_model()

        logger.info(f"Initialized Donut engine with {model_name}")

    def _load_model(self):
        """Load model and processor."""
        try:
            logger.info(f"Loading Donut model: {self.model_name}")

            # Load processor
            self.processor = DonutProcessor.from_pretrained(self.model_name)

            # Load model
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

            # Load LoRA adapter if specified
            if self.use_lora and self.lora_adapter_path:
                logger.info(f"Loading LoRA adapter from {self.lora_adapter_path}")
                try:
                    from peft import PeftModel

                    self.model = PeftModel.from_pretrained(
                        self.model, str(self.lora_adapter_path)
                    )
                except Exception as e:
                    logger.warning(f"Failed to load LoRA adapter: {e}")

            # Apply quantization
            if self.quantization == "fp16":
                self.model = self.model.half()
            elif self.quantization == "int8":
                logger.warning("INT8 quantization not yet implemented for Donut")

            # Move to device
            self.model = self.device_manager.move_to_device(self.model)
            self.model.eval()

            logger.info("Donut model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Donut model: {e}")
            raise

    @timeit(name="Donut OCR")
    def process_image(
        self,
        image: Image.Image,
        task_prompt: str = "<s_ocr>",
    ) -> Dict:
        """
        Process image with Donut.

        Args:
            image: PIL Image
            task_prompt: Task prompt for Donut

        Returns:
            Dictionary with OCR results
        """
        try:
            # Prepare inputs
            pixel_values = self.processor(
                image, return_tensors="pt"
            ).pixel_values
            pixel_values = self.device_manager.move_to_device(pixel_values)

            # Prepare decoder input
            decoder_input_ids = self.processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids
            decoder_input_ids = self.device_manager.move_to_device(decoder_input_ids)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.max_length,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

            # Decode
            sequence = outputs.sequences[0]
            sequence = sequence.replace(self.processor.tokenizer.pad_token_id, 0)
            text = self.processor.batch_decode([sequence])[0]

            # Remove special tokens
            text = text.replace(task_prompt, "").replace("</s_ocr>", "").strip()

            return {
                'text': text,
                'confidence': 0.9,  # Donut doesn't provide confidence scores
                'engine': 'donut',
            }

        except Exception as e:
            logger.error(f"Donut processing failed: {e}")
            raise

    def process_batch(
        self,
        images: List[Image.Image],
        task_prompt: str = "<s_ocr>",
    ) -> List[Dict]:
        """
        Process batch of images.

        Args:
            images: List of PIL Images
            task_prompt: Task prompt

        Returns:
            List of results
        """
        results = []

        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]

            for image in batch:
                result = self.process_image(image, task_prompt)
                results.append(result)

        return results

    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor

        # Clear cache
        if self.device_manager.is_mps():
            try:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except Exception:
                pass
