"""
Donut model engine with optional LoRA fine-tuning support.

Supports OCR, document parsing, and structured extraction.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from util.device import get_device_manager
from util.logging import get_logger
from util.timing import Timer

logger = get_logger(__name__)


class DonutEngine:
    """
    Donut OCR engine with LoRA support.

    Features:
    - HuggingFace Donut base model
    - Optional LoRA adapter loading
    - Multiple task types (OCR, DocVQA, parsing)
    - JSON output parsing
    - Confidence estimation
    """

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        lora_adapter_path: Optional[Path] = None,
        use_lora: bool = False,
        max_length: int = 1024,
        batch_size: int = 1,
        quantization: str = "fp16",
        cache_dir: Optional[Path] = None,
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
            cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.lora_adapter_path = Path(lora_adapter_path) if lora_adapter_path else None
        self.use_lora = use_lora
        self.max_length = max_length
        self.batch_size = batch_size
        self.quantization = quantization
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Load model and processor
        self.processor = None
        self.model = None
        self.is_loaded = False

        logger.info(
            f"Initialized Donut engine: model={model_name}, "
            f"lora={use_lora}, quant={quantization}"
        )

    def _load_model(self):
        """Lazy load model and processor."""
        if self.is_loaded:
            return

        try:
            logger.info(f"Loading Donut model: {self.model_name}")

            # Load processor
            self.processor = DonutProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )

            # Load model
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )

            # Load LoRA adapter if specified
            if self.use_lora and self.lora_adapter_path:
                if self.lora_adapter_path.exists():
                    logger.info(f"Loading LoRA adapter from {self.lora_adapter_path}")
                    try:
                        from peft import PeftModel

                        self.model = PeftModel.from_pretrained(
                            self.model, str(self.lora_adapter_path)
                        )
                        logger.info("LoRA adapter loaded successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load LoRA adapter: {e}")
                else:
                    logger.warning(f"LoRA adapter path not found: {self.lora_adapter_path}")

            # Apply quantization
            if self.quantization == "fp16":
                self.model = self.model.half()
                logger.info("Applied FP16 quantization")
            elif self.quantization == "int8":
                # Placeholder for INT8 quantization
                logger.warning("INT8 quantization not yet implemented for Donut")

            # Move to device
            self.model = self.device_manager.move_to_device(self.model)
            self.model.eval()

            self.is_loaded = True
            logger.info("Donut model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Donut model: {e}")
            raise

    def process_image(
        self,
        image: Image.Image,
        task_prompt: str = "<s_ocr>",
        return_json: bool = False,
    ) -> Dict:
        """
        Process image with Donut.

        Args:
            image: PIL Image
            task_prompt: Task prompt for Donut
                        - "<s_ocr>": OCR
                        - "<s_docvqa>": Document VQA
                        - "<s_cord>": Document parsing
            return_json: Whether to parse output as JSON

        Returns:
            Dictionary with OCR results:
            {
                "text": "extracted text",
                "confidence": 0.9,
                "parsed": {...},  # if return_json=True
                "engine": "donut",
                "elapsed_ms": 1234
            }
        """
        self._load_model()

        timer = Timer()
        timer.start()

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
                    output_scores=True,
                )

            # Decode
            sequence = outputs.sequences[0]
            # Remove padding
            sequence = sequence[sequence != self.processor.tokenizer.pad_token_id]
            text = self.processor.batch_decode([sequence], skip_special_tokens=False)[0]

            # Remove task prompt and end token
            text = text.replace(task_prompt, "").replace("</s>", "").strip()

            # Estimate confidence from output scores
            confidence = self._estimate_confidence(outputs)

            elapsed_ms = int(timer.stop() * 1000)

            result = {
                'text': text,
                'confidence': confidence,
                'engine': 'donut',
                'elapsed_ms': elapsed_ms,
            }

            # Parse JSON if requested
            if return_json:
                parsed = self._parse_json_output(text)
                result['parsed'] = parsed

            return result

        except Exception as e:
            elapsed_ms = int(timer.stop() * 1000) if timer.start_time else 0
            logger.error(f"Donut processing failed: {e}")

            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'donut',
                'elapsed_ms': elapsed_ms,
                'error': str(e),
            }

    def _estimate_confidence(self, outputs) -> float:
        """
        Estimate confidence from generation scores.

        Args:
            outputs: Model outputs with scores

        Returns:
            Confidence score (0-1)
        """
        try:
            if not hasattr(outputs, 'scores') or not outputs.scores:
                return 0.9  # Default confidence

            # Calculate average probability of generated tokens
            scores = []
            for step_scores in outputs.scores:
                probs = torch.softmax(step_scores[0], dim=-1)
                max_prob = torch.max(probs).item()
                scores.append(max_prob)

            return sum(scores) / len(scores) if scores else 0.9

        except Exception as e:
            logger.warning(f"Failed to estimate confidence: {e}")
            return 0.9

    def _parse_json_output(self, text: str) -> Optional[Dict]:
        """
        Parse JSON from Donut output.

        Args:
            text: Generated text

        Returns:
            Parsed JSON or None
        """
        try:
            # Try direct JSON parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text
        json_patterns = [
            r'\{[^{}]*\}',  # Simple object
            r'\{.*?\}',  # Non-greedy object
            r'\[.*?\]',  # Array
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        logger.warning("Failed to parse JSON from output")
        return None

    def process_batch(
        self,
        images: List[Image.Image],
        task_prompt: str = "<s_ocr>",
        return_json: bool = False,
    ) -> List[Dict]:
        """
        Process batch of images.

        Args:
            images: List of PIL Images
            task_prompt: Task prompt
            return_json: Parse output as JSON

        Returns:
            List of results
        """
        self._load_model()

        results = []

        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]

            for image in batch:
                result = self.process_image(image, task_prompt, return_json)
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

        self.is_loaded = False

        # Clear cache
        if self.device_manager.is_mps():
            try:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except Exception:
                pass

        logger.info("Donut engine cleaned up")

    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except Exception:
            pass
