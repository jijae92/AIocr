"""
SynthDoG Dataset Loader for Donut Training.

Loads SynthDoG (Synthetic Document Generator) dataset or custom OCR datasets
for Donut fine-tuning. Supports:
- Image-text pairs in JSON format
- Image preprocessing and augmentation
- Token encoding for sequence-to-sequence training

Dataset Structure:
    data/train/
        ├── images/
        │   ├── image_0001.png
        │   ├── image_0002.png
        │   └── ...
        └── metadata.jsonl  # One JSON per line: {"file_name": "image_0001.png", "text": "..."}

Alternative structure:
    data/train/
        ├── image_0001.png
        ├── image_0001.json  # {"text": "..."}
        └── ...
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from util.logging import get_logger

logger = get_logger(__name__)


class SynthDogDataset(Dataset):
    """
    Dataset for Donut training with SynthDoG or custom data.

    Supports two formats:
    1. JSONL format: metadata.jsonl with image paths and texts
    2. Paired format: image.png + image.json with text
    """

    def __init__(
        self,
        data_dir: Path,
        processor,
        max_length: int = 1024,
        image_size: Tuple[int, int] = (1280, 960),
        augment: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing images and annotations
            processor: Donut processor (handles tokenization and image processing)
            max_length: Maximum sequence length
            image_size: Target image size (width, height)
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.augment = augment

        # Load dataset
        self.samples = self._load_samples()

        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")

    def _load_samples(self) -> List[Dict]:
        """
        Load dataset samples.

        Returns:
            List of dicts with 'image_path' and 'text'
        """
        samples = []

        # Try JSONL format first
        metadata_path = self.data_dir / "metadata.jsonl"
        if metadata_path.exists():
            samples = self._load_jsonl_format(metadata_path)
        else:
            # Try paired format
            samples = self._load_paired_format()

        if not samples:
            raise ValueError(f"No valid samples found in {self.data_dir}")

        return samples

    def _load_jsonl_format(self, metadata_path: Path) -> List[Dict]:
        """Load samples from JSONL metadata file."""
        samples = []
        images_dir = self.data_dir / "images"

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    file_name = data.get("file_name")
                    text = data.get("text", data.get("ground_truth", ""))

                    if not file_name or not text:
                        continue

                    image_path = images_dir / file_name
                    if not image_path.exists():
                        image_path = self.data_dir / file_name

                    if image_path.exists():
                        samples.append({"image_path": image_path, "text": text})

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON line: {line[:50]}...")

        return samples

    def _load_paired_format(self) -> List[Dict]:
        """Load samples from paired image+json files."""
        samples = []

        # Find all image files
        image_extensions = [".png", ".jpg", ".jpeg", ".tiff"]
        for image_path in self.data_dir.iterdir():
            if image_path.suffix.lower() not in image_extensions:
                continue

            # Look for corresponding JSON file
            json_path = image_path.with_suffix(".json")
            if not json_path.exists():
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    text = data.get("text", data.get("ground_truth", ""))

                    if text:
                        samples.append({"image_path": image_path, "text": text})

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON: {json_path}")

        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get single sample.

        Args:
            idx: Sample index

        Returns:
            Dict with processed image and labels
        """
        sample = self.samples[idx]
        image_path = sample["image_path"]
        text = sample["text"]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", self.image_size, color="white")

        # Apply augmentation if enabled
        if self.augment:
            image = self._augment_image(image)

        # Process with Donut processor
        # The processor handles both image preprocessing and text tokenization
        pixel_values = self.processor(
            image, return_tensors="pt", size=self.image_size
        ).pixel_values.squeeze(0)

        # Tokenize text
        # Format: "<s_docvqa><s_question>What is the text?</s_question><s_answer>{text}</s_answer>"
        # For general OCR, we simplify:
        prompt = f"<s_ocr>{text}</s_ocr>"

        labels = self.processor.tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Set padding tokens to -100 (ignored in loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

    def _augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply data augmentation to image.

        Args:
            image: PIL Image

        Returns:
            Augmented image
        """
        augment_transform = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(degrees=2, translate=(0.02, 0.02)),
            ]
        )

        try:
            image = augment_transform(image)
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")

        return image


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dict with stacked tensors
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


class CustomOCRDataset(Dataset):
    """
    Custom OCR dataset for arbitrary image-text pairs.

    Use this when you have your own OCR dataset with different structure.
    """

    def __init__(
        self,
        image_paths: List[Path],
        texts: List[str],
        processor,
        max_length: int = 1024,
        image_size: Tuple[int, int] = (1280, 960),
    ):
        """
        Initialize custom dataset.

        Args:
            image_paths: List of image file paths
            texts: Corresponding ground truth texts
            processor: Donut processor
            max_length: Maximum sequence length
            image_size: Target image size
        """
        assert len(image_paths) == len(
            texts
        ), "Number of images and texts must match"

        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        """Get single sample."""
        image_path = self.image_paths[idx]
        text = self.texts[idx]

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            image = Image.new("RGB", self.image_size, color="white")

        # Process
        pixel_values = self.processor(
            image, return_tensors="pt", size=self.image_size
        ).pixel_values.squeeze(0)

        prompt = f"<s_ocr>{text}</s_ocr>"
        labels = self.processor.tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }
