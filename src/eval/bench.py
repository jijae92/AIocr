"""
Evaluation and benchmarking utilities.
"""

from pathlib import Path
from typing import Dict, List

import editdistance
from jiwer import cer, wer

from util.logging import get_logger

logger = get_logger(__name__)


class OCREvaluator:
    """Evaluate OCR results against ground truth."""

    @staticmethod
    def compute_cer(predicted: str, ground_truth: str) -> float:
        """
        Compute Character Error Rate.

        Args:
            predicted: Predicted text
            ground_truth: Ground truth text

        Returns:
            CER (0-1)
        """
        try:
            return cer(ground_truth, predicted)
        except Exception as e:
            logger.error(f"Failed to compute CER: {e}")
            return 1.0

    @staticmethod
    def compute_wer(predicted: str, ground_truth: str) -> float:
        """
        Compute Word Error Rate.

        Args:
            predicted: Predicted text
            ground_truth: Ground truth text

        Returns:
            WER (0-1)
        """
        try:
            return wer(ground_truth, predicted)
        except Exception as e:
            logger.error(f"Failed to compute WER: {e}")
            return 1.0

    @staticmethod
    def compute_accuracy(predicted: str, ground_truth: str) -> float:
        """
        Compute character-level accuracy.

        Args:
            predicted: Predicted text
            ground_truth: Ground truth text

        Returns:
            Accuracy (0-1)
        """
        if not ground_truth:
            return 1.0 if not predicted else 0.0

        # Use edit distance
        distance = editdistance.eval(predicted, ground_truth)
        max_length = max(len(predicted), len(ground_truth))

        if max_length == 0:
            return 1.0

        accuracy = 1.0 - (distance / max_length)
        return max(0.0, accuracy)

    @staticmethod
    def evaluate(predicted: str, ground_truth: str) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            predicted: Predicted text
            ground_truth: Ground truth text

        Returns:
            Dictionary with metrics
        """
        return {
            'cer': OCREvaluator.compute_cer(predicted, ground_truth),
            'wer': OCREvaluator.compute_wer(predicted, ground_truth),
            'accuracy': OCREvaluator.compute_accuracy(predicted, ground_truth),
        }

    @staticmethod
    def evaluate_batch(
        predictions: List[str],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate batch of predictions.

        Args:
            predictions: List of predicted texts
            ground_truths: List of ground truth texts

        Returns:
            Average metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        cers = []
        wers = []
        accuracies = []

        for pred, gt in zip(predictions, ground_truths):
            metrics = OCREvaluator.evaluate(pred, gt)
            cers.append(metrics['cer'])
            wers.append(metrics['wer'])
            accuracies.append(metrics['accuracy'])

        return {
            'cer': sum(cers) / len(cers),
            'wer': sum(wers) / len(wers),
            'accuracy': sum(accuracies) / len(accuracies),
            'cer_std': (sum((c - sum(cers) / len(cers)) ** 2 for c in cers) / len(cers)) ** 0.5,
            'wer_std': (sum((w - sum(wers) / len(wers)) ** 2 for w in wers) / len(wers)) ** 0.5,
        }


class BenchmarkRunner:
    """Run benchmarks on OCR system."""

    def __init__(self, ground_truth_dir: Path):
        """
        Initialize benchmark runner.

        Args:
            ground_truth_dir: Directory with ground truth files
        """
        self.ground_truth_dir = Path(ground_truth_dir)

    def load_ground_truth(self, file_name: str) -> str:
        """
        Load ground truth for file.

        Args:
            file_name: Name of file (without extension)

        Returns:
            Ground truth text
        """
        gt_path = self.ground_truth_dir / f"{file_name}.txt"

        if not gt_path.exists():
            logger.warning(f"Ground truth not found: {gt_path}")
            return ""

        with open(gt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def benchmark(
        self,
        ocr_function,
        test_files: List[Path],
    ) -> Dict:
        """
        Run benchmark on test files.

        Args:
            ocr_function: Function that takes file path and returns OCR text
            test_files: List of test file paths

        Returns:
            Benchmark results
        """
        results = []

        for file_path in test_files:
            logger.info(f"Benchmarking: {file_path.name}")

            # Get OCR result
            try:
                predicted = ocr_function(file_path)
            except Exception as e:
                logger.error(f"OCR failed for {file_path.name}: {e}")
                continue

            # Load ground truth
            file_stem = file_path.stem
            ground_truth = self.load_ground_truth(file_stem)

            if not ground_truth:
                logger.warning(f"Skipping {file_path.name} - no ground truth")
                continue

            # Evaluate
            metrics = OCREvaluator.evaluate(predicted, ground_truth)
            metrics['file'] = file_path.name

            results.append(metrics)

        # Compute average metrics
        if results:
            avg_metrics = {
                'cer': sum(r['cer'] for r in results) / len(results),
                'wer': sum(r['wer'] for r in results) / len(results),
                'accuracy': sum(r['accuracy'] for r in results) / len(results),
            }
        else:
            avg_metrics = {}

        return {
            'results': results,
            'average': avg_metrics,
            'num_files': len(results),
        }
