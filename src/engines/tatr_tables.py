"""
Table Transformer (TATR) engine for table detection and structure recognition.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from util.coords import BoundingBox
from util.device import get_device_manager
from util.logging import get_logger
from util.timing import timeit

logger = get_logger(__name__)


class TATREngine:
    """Table Transformer engine for table detection and structure."""

    def __init__(
        self,
        detection_model: str = "microsoft/table-transformer-detection",
        structure_model: str = "microsoft/table-transformer-structure-recognition",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize TATR engine.

        Args:
            detection_model: Table detection model name
            structure_model: Table structure recognition model name
            confidence_threshold: Minimum confidence threshold
        """
        self.detection_model_name = detection_model
        self.structure_model_name = structure_model
        self.confidence_threshold = confidence_threshold

        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Models will be loaded on demand
        self.detection_model = None
        self.structure_model = None
        self.feature_extractor = None

        logger.info("Initialized TATR engine (models will load on first use)")

    def _load_models(self):
        """Load detection and structure models."""
        if self.detection_model is not None:
            return

        try:
            logger.info("Loading Table Transformer models...")

            from transformers import AutoFeatureExtractor, TableTransformerForObjectDetection

            # Load feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.detection_model_name
            )

            # Load detection model
            self.detection_model = TableTransformerForObjectDetection.from_pretrained(
                self.detection_model_name
            )
            self.detection_model = self.device_manager.move_to_device(self.detection_model)
            self.detection_model.eval()

            # Load structure model
            self.structure_model = TableTransformerForObjectDetection.from_pretrained(
                self.structure_model_name
            )
            self.structure_model = self.device_manager.move_to_device(self.structure_model)
            self.structure_model.eval()

            logger.info("TATR models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load TATR models: {e}")
            raise

    @timeit(name="TATR Table Detection")
    def detect_tables(self, image: Image.Image) -> List[BoundingBox]:
        """
        Detect tables in image.

        Args:
            image: PIL Image

        Returns:
            List of table bounding boxes
        """
        self._load_models()

        try:
            # Prepare inputs
            encoding = self.feature_extractor(image, return_tensors="pt")
            pixel_values = self.device_manager.move_to_device(encoding.pixel_values)

            # Detect tables
            with torch.no_grad():
                outputs = self.detection_model(pixel_values)

            # Post-process
            target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
            results = self.feature_extractor.post_process_object_detection(
                outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
            )[0]

            # Extract table bounding boxes
            tables = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                if score >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.tolist()
                    tables.append(
                        BoundingBox.from_corners(
                            x1, y1, x2, y2, confidence=float(score)
                        )
                    )

            logger.info(f"Detected {len(tables)} tables")
            return tables

        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            raise

    @timeit(name="TATR Table Structure")
    def recognize_structure(self, table_image: Image.Image) -> Dict:
        """
        Recognize table structure.

        Args:
            table_image: PIL Image of table region

        Returns:
            Dictionary with table structure
        """
        self._load_models()

        try:
            # Prepare inputs
            encoding = self.feature_extractor(table_image, return_tensors="pt")
            pixel_values = self.device_manager.move_to_device(encoding.pixel_values)

            # Recognize structure
            with torch.no_grad():
                outputs = self.structure_model(pixel_values)

            # Post-process
            target_sizes = torch.tensor([table_image.size[::-1]])
            results = self.feature_extractor.post_process_object_detection(
                outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
            )[0]

            # Extract cells and structure
            rows = []
            columns = []
            cells = []

            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                if score >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.tolist()
                    bbox = BoundingBox.from_corners(
                        x1, y1, x2, y2, confidence=float(score)
                    )

                    label_name = self.structure_model.config.id2label[label.item()]

                    if label_name == "table row":
                        rows.append(bbox)
                    elif label_name == "table column":
                        columns.append(bbox)
                    elif label_name == "table cell":
                        cells.append(bbox)

            return {
                'rows': [bbox.to_dict() for bbox in rows],
                'columns': [bbox.to_dict() for bbox in columns],
                'cells': [bbox.to_dict() for bbox in cells],
                'num_rows': len(rows),
                'num_columns': len(columns),
            }

        except Exception as e:
            logger.error(f"Table structure recognition failed: {e}")
            raise

    def process_image(self, image: Image.Image) -> Dict:
        """
        Detect and recognize tables in image.

        Args:
            image: PIL Image

        Returns:
            Dictionary with table detection and structure results
        """
        # Detect tables
        tables = self.detect_tables(image)

        # Process each table
        table_results = []
        for i, table_bbox in enumerate(tables):
            # Crop table region
            x1, y1, x2, y2 = table_bbox.to_corners()
            table_image = image.crop((x1, y1, x2, y2))

            # Recognize structure
            structure = self.recognize_structure(table_image)
            structure['bbox'] = table_bbox.to_dict()

            table_results.append(structure)

        return {
            'tables': table_results,
            'num_tables': len(tables),
            'engine': 'tatr',
        }

    def cleanup(self):
        """Cleanup resources."""
        if self.detection_model is not None:
            del self.detection_model
        if self.structure_model is not None:
            del self.structure_model
        if self.feature_extractor is not None:
            del self.feature_extractor
