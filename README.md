# AIocr - Intelligent OCR Routing and Ensemble System

An intelligent OCR system that uses heuristic-based routing and ensemble methods to select and combine results from multiple OCR models for optimal accuracy.

## Overview

AIocr implements a sophisticated routing and ensemble system that automatically selects the best OCR model(s) based on document characteristics, content type, and confidence scores. The system supports multiple state-of-the-art OCR models and intelligently combines their results.

## Features

### 🎯 Intelligent Routing
- **DocAI Priority**: Automatically uses Google Cloud Document AI for high-confidence, simple documents with supported languages
- **Complex Layout Handling**: Falls back to Donut for multi-column layouts, complex structures, or low-confidence results
- **Content-Specific Routing**:
  - Tables → TATR (Table Transformer)
  - Mathematical formulas → pix2tex
  - Numeric content, dates, codes → TrOCR ONNX

### 🔄 Ensemble Methods
- **Weighted Voting**: Combines results based on model confidence and configured weights
- **Consensus**: Uses blocks that appear in multiple model results
- **Confidence Max**: Selects highest confidence result for each region
- **Cross-Validation**: Validates numeric fields across multiple models

### 📐 Layout Analysis
- **Reading Order Restoration**: Spatial analysis to restore natural reading order
- **Multi-Column Detection**: Automatically detects and handles multi-column layouts
- **Table Structure Recovery**: Preserves table structure and cell relationships
- **Source Labeling**: Maintains provenance and confidence metadata

### ⚙️ Configuration-Driven
- All thresholds, priorities, and timeouts externalized in `configs/app.yaml`
- Easy tuning without code changes
- Per-model and per-content-type settings

## Architecture

```
AIocr/
├── src/
│   ├── data/               # Data structures
│   │   ├── block_types.py  # Block type definitions
│   │   └── ocr_result.py   # OCR result classes
│   ├── models/             # OCR model wrappers
│   │   ├── base_model.py   # Base interface
│   │   ├── docai.py        # Google Cloud Document AI
│   │   ├── donut.py        # Donut (NAVER)
│   │   ├── trocr.py        # TrOCR (Microsoft)
│   │   ├── tatr.py         # Table Transformer
│   │   └── pix2tex.py      # Formula extraction
│   ├── router/             # Routing logic
│   │   └── heuristics.py   # Heuristic-based router
│   └── utils/              # Utilities
│       └── layout_merge.py # Layout and merge utilities
├── configs/
│   └── app.yaml            # Configuration
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
└── setup.py               # Package setup
```

## Installation

```bash
# Clone the repository
git clone https://github.com/jijae92/AIocr.git
cd AIocr

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .

# For GPU support
pip install -e ".[gpu]"

# For development
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from src.router.heuristics import HeuristicRouter, DocumentCharacteristics
from src.data.block_types import LayoutComplexity, ContentPattern

# Initialize router
router = HeuristicRouter()

# Analyze document and get routing decision
characteristics = DocumentCharacteristics(
    page_count=1,
    average_confidence=0.82,
    layout_complexity=LayoutComplexity.SIMPLE,
    has_tables=True,
    table_count=2,
    has_formulas=False,
    has_multi_column=False,
    detected_language="en",
    content_patterns={ContentPattern.NUMERIC},
    numeric_content_ratio=0.4,
    image_quality_score=0.9
)

# Get routing decision
decision = router.route(characteristics)
print(f"Primary model: {decision.primary_model}")
print(f"Reason: {decision.reason}")
print(f"Use ensemble: {decision.use_ensemble}")
```

### Merging Results

```python
from src.utils.layout_merge import LayoutMerger
from src.data.ocr_result import OCRResult

# Initialize merger
merger = LayoutMerger()

# Restore reading order
ordered_page = merger.restore_reading_order(page_result)

# Merge multiple model results
ensemble_result = merger.merge_results(
    results=[docai_result, donut_result, trocr_result],
    model_names=["docai", "donut", "trocr"],
    strategy="weighted_voting"
)

# Access merged result
print(f"Text: {ensemble_result.primary_result.text}")
print(f"Confidence: {ensemble_result.primary_result.average_confidence}")
print(f"Conflicts: {len(ensemble_result.conflicts)}")
```

## Configuration

Key configuration options in `configs/app.yaml`:

### Thresholds
```yaml
thresholds:
  docai_confidence: 0.85      # DocAI acceptance threshold
  low_confidence: 0.5         # Triggers ensemble mode
  high_confidence: 0.95       # Accept without validation
  simple_table_cells: 50      # Max cells for "simple" table
  numeric_content_ratio: 0.7  # Threshold for numeric routing
```

### Model Weights
```yaml
ensemble:
  weights:
    docai: 1.5
    donut: 1.2
    trocr: 1.0
    tatr: 1.3
    pix2tex: 1.1
```

### Content Routing
```yaml
routing:
  content_routing:
    tables:
      primary: tatr
      fallback: docai
    formulas:
      primary: pix2tex
    numeric:
      primary: trocr
      validation: docai
```

## Routing Decision Logic

### 1. DocAI Priority Path
Used when:
- Confidence ≥ 0.85
- Simple to moderate layout
- Supported language
- Table count ≤ threshold

### 2. Donut Reinterpretation
Used when:
- Confidence < 0.85
- Complex/multi-column layout
- High table count (> 5)

### 3. TrOCR Cross-Validation
Used for:
- Numeric content (ratio > 0.7)
- Date/time fields
- Code/ID patterns
- Validation of critical fields

### 4. Specialized Models
- **TATR**: Table structure recovery
- **pix2tex**: Mathematical formula extraction

## Data Structures

### Block
```python
@dataclass
class Block:
    text: str
    block_type: BlockType
    confidence: float
    bbox: BoundingBox
    language: Optional[str]
    model_source: Optional[str]
    content_pattern: Optional[ContentPattern]
    reading_order: Optional[int]
```

### OCRResult
```python
@dataclass
class OCRResult:
    pages: List[PageResult]
    doc_id: Optional[str]
    processing_time: Optional[float]
    metadata: Dict[str, Any]
```

### EnsembleResult
```python
@dataclass
class EnsembleResult:
    primary_result: OCRResult
    alternative_results: List[OCRResult]
    model_sources: Dict[str, str]
    confidence_map: Dict[str, float]
    conflicts: List[Dict[str, Any]]
```

## Model Support

| Model | Purpose | Strengths |
|-------|---------|-----------|
| **DocAI** | General OCR | Multi-language, high accuracy, tables |
| **Donut** | Complex layouts | Multi-column, document understanding |
| **TrOCR** | Text recognition | Numeric, dates, ONNX support |
| **TATR** | Table extraction | Structure recovery, cell detection |
| **pix2tex** | Formula extraction | LaTeX generation, scientific docs |

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Type Checking
```bash
mypy src/
```

## Roadmap

- [ ] Implement actual model integrations (currently stubs)
- [ ] Add neural reading order detection
- [ ] Implement confidence calibration
- [ ] Add support for more languages
- [ ] Performance optimization
- [ ] Web API interface
- [ ] CLI tool
- [ ] Benchmarking suite

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[License details to be added]

## Citation

If you use this project in your research, please cite:

```bibtex
@software{aiocr2024,
  title={AIocr: Intelligent OCR Routing and Ensemble System},
  author={AIocr Team},
  year={2024},
  url={https://github.com/jijae92/AIocr}
}
```

## Acknowledgments

- Google Cloud Document AI
- NAVER Clova Donut
- Microsoft TrOCR
- Table Transformer (TATR)
- pix2tex project
