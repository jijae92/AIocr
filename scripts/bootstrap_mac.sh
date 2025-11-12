#!/usr/bin/env bash

# Bootstrap script for macOS setup - Hybrid PDF OCR

set -euo pipefail

echo "🚀 Hybrid PDF OCR - macOS Setup"
echo "================================"
echo ""

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Install/Setup Homebrew
echo "📦 Setting up Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
    echo "⚠️  Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH
    if [[ -f ~/.zprofile ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    fi
    eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true
else
    echo "✅ Homebrew already installed"
fi

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
brew install python@3.12 cmake pkg-config git ffmpeg poppler tesseract

# Set Python path
PYTHON_BIN="/opt/homebrew/opt/python@3.12/bin/python3.12"
if [[ ! -f "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3.12"
fi

# Verify Python version
PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
echo "✅ Using Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "🐍 Creating virtual environment (.venv)..."
if [[ -d ".venv" ]]; then
    echo "⚠️  .venv already exists, skipping..."
else
    $PYTHON_BIN -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip, wheel, setuptools..."
python -m pip install -U pip wheel setuptools

# Install core packages
echo ""
echo "📦 Installing core ML/OCR packages..."
python -m pip install \
    torch torchvision torchaudio \
    transformers accelerate peft \
    onnx onnxruntime \
    google-cloud-documentai google-cloud-storage \
    pymupdf pdf2image pillow opencv-python scikit-image \
    python-docx markdown \
    jiwer editdistance python-Levenshtein \
    PyQt5 \
    pyyaml python-dotenv click tenacity \
    aiofiles httpx tqdm rich

# Install development packages
echo ""
echo "📦 Installing development packages..."
python -m pip install \
    pytest pytest-cov pytest-asyncio \
    black ruff mypy \
    pre-commit

# Optional: Install table support (commented out by default)
# echo ""
# echo "📦 Installing table support (optional)..."
# python -m pip install table-transformer

# Check MPS availability
echo ""
echo "🔍 Checking MPS (Metal Performance Shaders) availability..."
python - <<'PY'
import torch
import sys

mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()

print(f"✅ MPS available: {mps_available}")
print(f"✅ MPS built: {mps_built}")

if mps_available:
    print("🎉 Your Mac supports GPU acceleration via MPS!")
else:
    print("⚠️  MPS not available. Will use CPU (slower).")
    if not mps_built:
        print("   PyTorch was not built with MPS support.")

sys.exit(0)
PY

# Create cache directories
echo ""
echo "📁 Creating cache directories..."
mkdir -p ~/.cache/hybrid-pdf-ocr/logs
mkdir -p ~/.cache/hybrid-pdf-ocr/cache
mkdir -p /tmp/hybrid-pdf-ocr

# Create model directories
echo "📁 Creating model directories..."
mkdir -p models/donut_base
mkdir -p models/donut_lora

# Check for GCP credentials
echo ""
echo "🔑 Google Cloud Setup"
echo "===================="

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
    echo "⚠️  GOOGLE_APPLICATION_CREDENTIALS not set"
    echo ""
    echo "To use Document AI, complete these steps:"
    echo ""
    echo "1. Create a GCP project"
    echo "2. Enable Document AI API"
    echo "3. Create Document AI processors (OCR + Form Parser)"
    echo "4. Download service account JSON key"
    echo "5. Set environment variables in ~/.zshrc or ~/.bashrc:"
    echo ""
    echo "   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'"
    echo "   export GCP_PROJECT='your-project-id'"
    echo "   export GCP_LOCATION='us'"
    echo "   export DOCAI_PROCESSOR_ID_OCR='your-ocr-processor-id'"
    echo "   export DOCAI_PROCESSOR_ID_FORM='your-form-processor-id'"
    echo ""
else
    echo "✅ GOOGLE_APPLICATION_CREDENTIALS is set"
    if [[ -n "${GCP_PROJECT:-}" ]]; then
        echo "✅ GCP_PROJECT: $GCP_PROJECT"
    fi
fi

# Summary
echo ""
echo "✅ Setup complete!"
echo "================================"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Set up GCP credentials (if not done):"
echo "   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'"
echo ""
echo "3. Run the GUI application:"
echo "   python src/gui/desktop_app.py"
echo ""
echo "4. Or use the Python API:"
echo "   python -c 'from connectors.docai_client import DocumentAIClient; print(DocumentAIClient)'"
echo ""
echo "5. Run tests:"
echo "   pytest tests/"
echo ""
echo "📖 For more information, see README.md"
echo ""
