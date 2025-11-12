# Hybrid PDF OCR for macOS

A production-grade hybrid PDF OCR system combining Google Document AI, Donut, TrOCR, and Table Transformer (TATR) with a PyQt5 GUI for macOS.

## Features

- **Multiple OCR Engines**:
  - Google Cloud Document AI (primary, production-grade)
  - Donut with LoRA fine-tuning support
  - TrOCR with ONNX INT8 quantization
  - Table Transformer (TATR) for table structure recognition
  - Pix2Tex for mathematical formula OCR (placeholder)

- **Intelligent Routing**:
  - Confidence-based routing between engines
  - Ensemble methods for combining results
  - Customizable thresholds and strategies

- **Advanced Processing**:
  - Document preprocessing (deskew, contrast enhancement, denoising)
  - Text normalization and layout preservation
  - Searchable PDF generation with invisible text layer

- **macOS Optimized**:
  - Apple Silicon (M1/M2) MPS acceleration
  - Native PyQt5 GUI
  - Optimized for macOS 13+

- **Production Features**:
  - Content-addressable caching
  - PII filtering in logs
  - Audit trail logging (JSONL)
  - Comprehensive evaluation metrics

## Requirements

- macOS 13+ (Apple Silicon recommended)
- Python 3.12+
- Google Cloud Platform account (for Document AI)

## Installation

### 1. Bootstrap Script

```bash
chmod +x scripts/bootstrap_mac.sh
./scripts/bootstrap_mac.sh
```

### 2. Manual Installation

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# For table support
pip install -e ".[table]"

# For math OCR support
pip install -e ".[math]"
```

### 3. Google Cloud Setup

1. Create a GCP project
2. Enable Document AI API
3. Create Document AI processors (OCR and Form Parser)
4. Download service account JSON key
5. Set environment variables:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
export GCP_PROJECT="your-project-id"
export GCP_LOCATION="us"
export DOCAI_PROCESSOR_ID_OCR="your-ocr-processor-id"
export DOCAI_PROCESSOR_ID_FORM="your-form-processor-id"
```

## Usage

### GUI Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run GUI
python src/gui/desktop_app.py
```

### Python API

```python
from pathlib import Path
from connectors.docai_client import DocumentAIClient
from preproc.pdf_loader import PDFLoader
from postproc.exporters import Exporter

# Initialize
client = DocumentAIClient()
loader = PDFLoader(dpi=300)

# Load PDF
pages = loader.load_pages(Path("document.pdf"))

# Process with Document AI
result = client.process_and_extract(Path("document.pdf"))

# Export results
Exporter.export_all(
    text=result['text'],
    data=result,
    base_path=Path("output"),
    formats=['txt', 'json', 'searchable_pdf']
)
```

## Configuration

Edit `configs/app.yaml` to customize:

- Routing strategy and thresholds
- Engine settings (Donut, TrOCR, TATR)
- Preprocessing options
- Export formats
- Cache configuration

## Project Structure

```
hybrid-pdf-ocr-macgui/
├── configs/               # Configuration files
│   ├── app.yaml          # Application settings
│   └── train_donut.yaml  # Donut LoRA training config
├── src/
│   ├── gui/              # PyQt5 GUI application
│   ├── connectors/       # Document AI client
│   ├── engines/          # OCR engines (Donut, TrOCR, TATR)
│   ├── router/           # Routing and ensemble logic
│   ├── preproc/          # PDF loading and preprocessing
│   ├── postproc/         # Text normalization and export
│   ├── pdf/              # Searchable PDF generation
│   ├── cache/            # Caching system
│   ├── eval/             # Evaluation and benchmarking
│   └── util/             # Utilities (logging, device, coords)
├── models/               # Model storage
├── tests/                # Unit tests
└── scripts/              # Setup scripts
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Evaluation

```python
from eval.bench import OCREvaluator

evaluator = OCREvaluator()
metrics = evaluator.evaluate(predicted_text, ground_truth)

print(f"CER: {metrics['cer']:.3f}")
print(f"WER: {metrics['wer']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

## Fine-tuning Donut

See `configs/train_donut.yaml` for LoRA fine-tuning configuration.

```python
# Training script (to be implemented)
python scripts/train_donut.py \
    --config configs/train_donut.yaml \
    --data-dir data/train
```

## Troubleshooting

### MPS Not Available

If MPS is not available, the system will fall back to CPU. Ensure you're running on Apple Silicon with macOS 13+.

### Document AI Errors

- Verify your service account key has Document AI permissions
- Check processor IDs are correct
- Ensure Document AI API is enabled in your GCP project

### Memory Issues

For large PDFs:
- Reduce `max_pages_per_batch` in `configs/app.yaml`
- Enable cache to avoid reprocessing
- Process pages individually

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Acknowledgments

- Google Cloud Document AI
- Naver Clova Donut
- Microsoft TrOCR
- Microsoft Table Transformer
