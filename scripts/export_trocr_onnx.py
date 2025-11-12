"""
Export TrOCR model to ONNX format with INT8 quantization.

Usage:
    python scripts/export_trocr_onnx.py \
        --model microsoft/trocr-base-printed \
        --output models/trocr_int8.onnx \
        --quantize int8
"""

import argparse
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def export_trocr_to_onnx(
    model_name: str,
    output_path: Path,
    quantize: str = "none",
    opset_version: int = 14,
):
    """
    Export TrOCR model to ONNX format.

    Args:
        model_name: HuggingFace model name
        output_path: Output ONNX file path
        quantize: Quantization mode ('none', 'int8', 'uint8')
        opset_version: ONNX opset version
    """
    print(f"Loading TrOCR model: {model_name}")

    # Load model and processor
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()

    # Create dummy input
    print("Creating dummy inputs...")
    dummy_image = torch.randn(1, 3, 384, 384)  # TrOCR input size

    # Prepare encoder input
    pixel_values = processor(
        images=dummy_image,
        return_tensors="pt"
    ).pixel_values

    # Export to ONNX (encoder only for simplicity)
    print(f"Exporting to ONNX (opset {opset_version})...")
    temp_path = output_path.with_suffix('.temp.onnx')

    with torch.no_grad():
        torch.onnx.export(
            model.encoder,
            pixel_values,
            str(temp_path),
            input_names=['pixel_values'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'last_hidden_state': {0: 'batch_size'},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    print(f"Model exported to {temp_path}")

    # Quantize if requested
    if quantize in ['int8', 'uint8']:
        print(f"Quantizing to {quantize.upper()}...")

        quant_type = QuantType.QInt8 if quantize == 'int8' else QuantType.QUInt8

        quantize_dynamic(
            str(temp_path),
            str(output_path),
            weight_type=quant_type,
        )

        print(f"Quantized model saved to {output_path}")

        # Remove temp file
        temp_path.unlink()

        # Verify quantized model
        model_onnx = onnx.load(str(output_path))
        onnx.checker.check_model(model_onnx)
        print("✓ Quantized model verified")

    else:
        # Just rename temp file
        temp_path.rename(output_path)
        print(f"Model saved to {output_path}")

    # Print model info
    model_onnx = onnx.load(str(output_path))
    print(f"\nModel info:")
    print(f"  Inputs: {[i.name for i in model_onnx.graph.input]}")
    print(f"  Outputs: {[o.name for o in model_onnx.graph.output]}")
    print(f"  IR version: {model_onnx.ir_version}")
    print(f"  Opset: {model_onnx.opset_import[0].version}")

    # File size
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Size: {size_mb:.2f} MB")

    print("\n✓ Export complete!")


def main():
    parser = argparse.ArgumentParser(description="Export TrOCR to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/trocr-base-printed",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="models/trocr_int8.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "int8", "uint8"],
        default="int8",
        help="Quantization mode",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Export
    export_trocr_to_onnx(
        model_name=args.model,
        output_path=args.output,
        quantize=args.quantize,
        opset_version=args.opset,
    )


if __name__ == '__main__':
    main()
