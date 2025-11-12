#!/bin/bash

# Bootstrap script for macOS setup

set -e

echo "🚀 Hybrid PDF OCR - macOS Setup"
echo "================================"

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "📦 Detected Python version: $PYTHON_VERSION"

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 12 ]]; then
    echo "❌ Python 3.12+ required. Please install from https://www.python.org"
    exit 1
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "⚠️  Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
brew install poppler  # For pdf2image
brew install tesseract  # For pytesseract

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install project
echo "📦 Installing project dependencies..."
pip install -e ".[dev]"

# Create cache directories
echo "📁 Creating cache directories..."
mkdir -p ~/.cache/hybrid-pdf-ocr/logs
mkdir -p ~/.cache/hybrid-pdf-ocr/cache
mkdir -p /tmp/hybrid-pdf-ocr

# Check for GCP credentials
echo ""
echo "🔑 Google Cloud Setup"
echo "===================="

if [[ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
    echo "⚠️  GOOGLE_APPLICATION_CREDENTIALS not set"
    echo ""
    echo "To use Document AI, set up:"
    echo "1. Create GCP project"
    echo "2. Enable Document AI API"
    echo "3. Create Document AI processors"
    echo "4. Download service account key"
    echo "5. Set environment variables:"
    echo ""
    echo "export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'"
    echo "export GCP_PROJECT='your-project-id'"
    echo "export GCP_LOCATION='us'"
    echo "export DOCAI_PROCESSOR_ID_OCR='your-ocr-processor-id'"
    echo "export DOCAI_PROCESSOR_ID_FORM='your-form-processor-id'"
else
    echo "✅ GOOGLE_APPLICATION_CREDENTIALS is set"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the GUI:"
echo "  python src/gui/desktop_app.py"
echo ""
echo "For more information, see README.md"
