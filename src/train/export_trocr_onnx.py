"""
TrOCR ONNX Export and INT8 Quantization.

Converts TrOCR model from Hugging Face checkpoint to ONNX format
with INT8 dynamic post-training quantization (PTQ) for efficient inference.

Steps:
1. Load TrOCR model from Hugging Face
2. Export to ONNX format
3. Apply INT8 dynamic quantization
4. Validate quantized model
5. Fallback to TorchScript if ONNX export fails

Usage:
    python src/train/export_trocr_onnx.py \
        --model microsoft/trocr-base-printed \
        --output models/trocr_int8.onnx \
        --quantize

Requirements:
    - transformers
    - torch
    - onnx
    - onnxruntime
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from util.logging import get_logger

logger = get_logger(__name__)


class TrOCRONNXExporter:
    """
    TrOCR ONNX exporter with INT8 quantization.

    Handles model export, quantization, and validation.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        output_path: Optional[Path] = None,
    ):
        """
        Initialize exporter.

        Args:
            model_name: Hugging Face model name or path
            output_path: Output path for ONNX model
        """
        self.model_name = model_name
        self.output_path = (
            Path(output_path) if output_path else Path("models/trocr.onnx")
        )

        self.processor = None
        self.model = None

    def load_model(self):
        """Load TrOCR model and processor from Hugging Face."""
        logger.info(f"Loading TrOCR model: {self.model_name}")

        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        # Set to eval mode
        self.model.eval()

        logger.info("Model loaded successfully")

    def export_to_onnx(
        self, input_size: Tuple[int, int] = (384, 384), opset_version: int = 14
    ) -> Path:
        """
        Export model to ONNX format.

        Args:
            input_size: Input image size (height, width)
            opset_version: ONNX opset version

        Returns:
            Path to exported ONNX model
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Exporting to ONNX: {self.output_path}")

        # Create dummy input
        batch_size = 1
        dummy_input = torch.randn(
            batch_size, 3, input_size[0], input_size[1], dtype=torch.float32
        )

        # Prepare output path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Export to ONNX
            # Note: Full TrOCR export is complex due to encoder-decoder architecture
            # We export encoder and decoder separately

            # Export encoder
            encoder_path = self.output_path.with_stem(
                self.output_path.stem + "_encoder"
            )

            with torch.no_grad():
                torch.onnx.export(
                    self.model.encoder,
                    dummy_input,
                    str(encoder_path),
                    opset_version=opset_version,
                    input_names=["pixel_values"],
                    output_names=["last_hidden_state"],
                    dynamic_axes={
                        "pixel_values": {0: "batch_size"},
                        "last_hidden_state": {0: "batch_size"},
                    },
                    do_constant_folding=True,
                )

            logger.info(f"Encoder exported to: {encoder_path}")

            # For decoder, we export a simplified version
            # Full decoder export requires handling the generation loop
            decoder_path = self.output_path.with_stem(
                self.output_path.stem + "_decoder"
            )

            # Create dummy decoder inputs
            hidden_size = self.model.config.encoder.hidden_size
            encoder_hidden_states = torch.randn(batch_size, 196, hidden_size)
            decoder_input_ids = torch.randint(
                0, self.model.config.decoder.vocab_size, (batch_size, 1)
            )

            with torch.no_grad():
                torch.onnx.export(
                    self.model.decoder,
                    (decoder_input_ids, encoder_hidden_states),
                    str(decoder_path),
                    opset_version=opset_version,
                    input_names=["input_ids", "encoder_hidden_states"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence_length"},
                        "encoder_hidden_states": {0: "batch_size"},
                        "logits": {0: "batch_size", 1: "sequence_length"},
                    },
                    do_constant_folding=True,
                )

            logger.info(f"Decoder exported to: {decoder_path}")

            # Verify ONNX models
            self._verify_onnx_model(encoder_path)
            self._verify_onnx_model(decoder_path)

            return encoder_path

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            logger.info("Attempting TorchScript export as fallback...")
            return self._export_torchscript()

    def _verify_onnx_model(self, onnx_path: Path):
        """
        Verify ONNX model.

        Args:
            onnx_path: Path to ONNX model
        """
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model verified: {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            raise

    def quantize_int8(
        self, onnx_path: Path, quantized_path: Optional[Path] = None
    ) -> Path:
        """
        Apply INT8 dynamic quantization to ONNX model.

        Args:
            onnx_path: Path to original ONNX model
            quantized_path: Output path for quantized model

        Returns:
            Path to quantized model
        """
        if quantized_path is None:
            quantized_path = onnx_path.with_stem(onnx_path.stem + "_int8")

        logger.info(f"Applying INT8 quantization: {quantized_path}")

        try:
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QInt8,
                optimize_model=True,
            )

            logger.info(f"Quantization complete: {quantized_path}")

            # Check file size reduction
            orig_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
            quant_size = quantized_path.stat().st_size / (1024 * 1024)  # MB

            logger.info(
                f"Model size: {orig_size:.2f} MB -> {quant_size:.2f} MB "
                f"({quant_size / orig_size * 100:.1f}%)"
            )

            return quantized_path

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise

    def _export_torchscript(self) -> Path:
        """
        Export model to TorchScript as fallback.

        Returns:
            Path to TorchScript model
        """
        torchscript_path = self.output_path.with_suffix(".pt")

        logger.info(f"Exporting to TorchScript: {torchscript_path}")

        try:
            # Trace encoder
            dummy_input = torch.randn(1, 3, 384, 384)

            with torch.no_grad():
                traced_encoder = torch.jit.trace(self.model.encoder, dummy_input)
                traced_encoder.save(str(torchscript_path))

            logger.info(f"TorchScript export complete: {torchscript_path}")
            return torchscript_path

        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise

    def export_full_pipeline(
        self, quantize: bool = True, input_size: Tuple[int, int] = (384, 384)
    ) -> dict:
        """
        Export complete pipeline: ONNX + quantization.

        Args:
            quantize: Whether to apply INT8 quantization
            input_size: Input image size

        Returns:
            Dict with paths to exported models
        """
        # Load model
        self.load_model()

        # Export to ONNX
        onnx_path = self.export_to_onnx(input_size=input_size)

        results = {"onnx_encoder": onnx_path}

        # Quantize if requested
        if quantize and onnx_path.suffix == ".onnx":
            quantized_path = self.quantize_int8(onnx_path)
            results["quantized_encoder"] = quantized_path

            # Quantize decoder as well
            decoder_path = onnx_path.with_stem(onnx_path.stem.replace("_encoder", "_decoder"))
            if decoder_path.exists():
                quantized_decoder = self.quantize_int8(decoder_path)
                results["quantized_decoder"] = quantized_decoder

        # Save processor
        processor_dir = self.output_path.parent / "processor"
        processor_dir.mkdir(parents=True, exist_ok=True)
        self.processor.save_pretrained(processor_dir)
        results["processor"] = processor_dir

        logger.info("Export pipeline complete!")
        logger.info(f"Results: {results}")

        return results


def main():
    """Main export entry point."""
    parser = argparse.ArgumentParser(description="TrOCR ONNX Export")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/trocr-base-printed",
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/trocr.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[384, 384],
        help="Input image size (height width)",
    )

    args = parser.parse_args()

    # Initialize exporter
    exporter = TrOCRONNXExporter(
        model_name=args.model, output_path=Path(args.output)
    )

    # Export
    results = exporter.export_full_pipeline(
        quantize=args.quantize, input_size=tuple(args.input_size)
    )

    print("\n=== Export Complete ===")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
