"""Setup script for AIocr package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="aiocr",
    version="0.1.0",
    description="Intelligent OCR routing and ensemble system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AIocr Team",
    author_email="",
    url="https://github.com/jijae92/AIocr",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "onnxruntime>=1.15.0",
        "google-cloud-documentai>=2.16.0",
        "opencv-python>=4.7.0",
        "scikit-image>=0.19.0",
        "pydantic>=2.0.0",
        "timm>=0.9.0",
        "tqdm>=4.65.0",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.15.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="ocr, document-ai, ensemble, routing, ml",
)
