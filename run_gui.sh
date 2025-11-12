#!/bin/bash
# Run Hybrid OCR GUI Application

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include src directory
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Activate virtual environment and run
source "${SCRIPT_DIR}/venv/bin/activate"
python "${SCRIPT_DIR}/src/gui/desktop_app.py" "$@"
