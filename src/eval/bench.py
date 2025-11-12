"""
Evaluation and benchmarking utilities.

Comprehensive evaluation system with:
- OCR metrics: CER, WER, Accuracy
- Table metrics: TEDS, Cell F1
- Performance metrics: P95 latency, throughput, engine hit rate
- Error analysis: Thumbnails, diff visualization, routing reasons
- Report generation: CSV, Markdown, JSON
"""

import csv
import difflib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import editdistance
import numpy as np
from PIL import Image
from jiwer import cer, wer

from data.ocr_result import OCRResult, PageResult
from util.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # OCR metrics
    cer: float
    wer: float
    accuracy: float

    # Performance metrics
    latency_ms: float
    throughput_pps: Optional[float] = None  # Pages per second

    # Model info
    engine_used: Optional[str] = None
    routing_reason: Optional[str] = None

    # Additional info
    file_name: Optional[str] = None
    page_count: Optional[int] = None


@dataclass
class TableMetrics:
    """Container for table-specific metrics."""

    teds: float  # Tree Edit Distance based Similarity
    cell_f1: float  # Cell-level F1 score
    cell_precision: float
    cell_recall: float
    structure_similarity: float


@dataclass
class ErrorAnalysis:
    """Container for error analysis data."""

    file_name: str
    page_num: int
    predicted: str
    ground_truth: str
    diff_html: str
    cer: float
    wer: float
    routing_reason: Optional[str] = None
    thumbnail_path: Optional[Path] = None


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
        Compute all OCR metrics.

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
            Average metrics with standard deviation
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
            'cer': np.mean(cers),
            'wer': np.mean(wers),
            'accuracy': np.mean(accuracies),
            'cer_std': np.std(cers),
            'wer_std': np.std(wers),
            'accuracy_std': np.std(accuracies),
        }

    @staticmethod
    def create_diff_html(predicted: str, ground_truth: str) -> str:
        """
        Create HTML diff visualization.

        Args:
            predicted: Predicted text
            ground_truth: Ground truth text

        Returns:
            HTML string with diff
        """
        differ = difflib.HtmlDiff()
        html = differ.make_table(
            ground_truth.splitlines(),
            predicted.splitlines(),
            fromdesc='Ground Truth',
            todesc='Predicted',
            context=True,
            numlines=2
        )
        return html


class TableEvaluator:
    """Evaluate table extraction results."""

    @staticmethod
    def compute_teds(predicted_table: List[List[str]], ground_truth_table: List[List[str]]) -> float:
        """
        Compute TEDS (Tree Edit Distance based Similarity).

        Simplified implementation - full TEDS requires tree structure comparison.

        Args:
            predicted_table: Predicted table (list of rows)
            ground_truth_table: Ground truth table (list of rows)

        Returns:
            TEDS score (0-1, higher is better)
        """
        # Simplified: Use string edit distance on flattened tables
        pred_flat = " ".join(" ".join(row) for row in predicted_table)
        gt_flat = " ".join(" ".join(row) for row in ground_truth_table)

        distance = editdistance.eval(pred_flat, gt_flat)
        max_length = max(len(pred_flat), len(gt_flat))

        if max_length == 0:
            return 1.0

        similarity = 1.0 - (distance / max_length)
        return max(0.0, similarity)

    @staticmethod
    def compute_cell_f1(
        predicted_cells: List[Tuple[int, int, str]],
        ground_truth_cells: List[Tuple[int, int, str]]
    ) -> Tuple[float, float, float]:
        """
        Compute cell-level F1 score.

        Args:
            predicted_cells: List of (row, col, text) tuples
            ground_truth_cells: List of (row, col, text) tuples

        Returns:
            (F1, precision, recall)
        """
        pred_set = set(predicted_cells)
        gt_set = set(ground_truth_cells)

        true_positives = len(pred_set & gt_set)
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1, precision, recall

    @staticmethod
    def evaluate_table(
        predicted_table: List[List[str]],
        ground_truth_table: List[List[str]]
    ) -> TableMetrics:
        """
        Evaluate table extraction.

        Args:
            predicted_table: Predicted table
            ground_truth_table: Ground truth table

        Returns:
            TableMetrics object
        """
        # Compute TEDS
        teds = TableEvaluator.compute_teds(predicted_table, ground_truth_table)

        # Convert to cell format
        pred_cells = [
            (i, j, cell)
            for i, row in enumerate(predicted_table)
            for j, cell in enumerate(row)
        ]
        gt_cells = [
            (i, j, cell)
            for i, row in enumerate(ground_truth_table)
            for j, cell in enumerate(row)
        ]

        # Compute Cell F1
        cell_f1, cell_precision, cell_recall = TableEvaluator.compute_cell_f1(pred_cells, gt_cells)

        # Structure similarity (compare dimensions)
        pred_rows, pred_cols = len(predicted_table), len(predicted_table[0]) if predicted_table else 0
        gt_rows, gt_cols = len(ground_truth_table), len(ground_truth_table[0]) if ground_truth_table else 0

        row_match = pred_rows == gt_rows
        col_match = pred_cols == gt_cols
        structure_similarity = (row_match + col_match) / 2.0

        return TableMetrics(
            teds=teds,
            cell_f1=cell_f1,
            cell_precision=cell_precision,
            cell_recall=cell_recall,
            structure_similarity=structure_similarity,
        )


class BenchmarkRunner:
    """Run comprehensive benchmarks on OCR system."""

    def __init__(
        self,
        ground_truth_dir: Path,
        reports_dir: Path,
        config: Optional[Dict] = None
    ):
        """
        Initialize benchmark runner.

        Args:
            ground_truth_dir: Directory with ground truth files
            reports_dir: Directory for output reports
            config: Configuration dict
        """
        self.ground_truth_dir = Path(ground_truth_dir)
        self.reports_dir = Path(reports_dir)
        self.config = config or {}

        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.latencies = []
        self.engine_usage = {}

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

    def generate_thumbnail(
        self,
        image_path: Path,
        output_path: Path,
        size: int = 200
    ) -> Path:
        """
        Generate thumbnail for error analysis.

        Args:
            image_path: Path to source image
            output_path: Path to save thumbnail
            size: Thumbnail size in pixels

        Returns:
            Path to saved thumbnail
        """
        try:
            image = Image.open(image_path)
            image.thumbnail((size, size))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            return None

    def benchmark(
        self,
        ocr_function,
        test_files: List[Path],
        generate_error_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark on test files.

        Args:
            ocr_function: Function that takes file path and returns (text, metadata)
            test_files: List of test file paths
            generate_error_analysis: Whether to generate error analysis

        Returns:
            Comprehensive benchmark results
        """
        results = []
        errors = []
        latencies = []

        for file_path in test_files:
            logger.info(f"Benchmarking: {file_path.name}")

            # Measure latency
            start_time = time.time()

            try:
                # Get OCR result (expect tuple: text, metadata)
                ocr_output = ocr_function(file_path)

                if isinstance(ocr_output, tuple):
                    predicted, metadata = ocr_output
                else:
                    predicted = ocr_output
                    metadata = {}

                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)

                # Track engine usage
                engine = metadata.get('engine_used', 'unknown')
                self.engine_usage[engine] = self.engine_usage.get(engine, 0) + 1

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

            # Create result
            result = EvaluationMetrics(
                cer=metrics['cer'],
                wer=metrics['wer'],
                accuracy=metrics['accuracy'],
                latency_ms=latency_ms,
                engine_used=metadata.get('engine_used'),
                routing_reason=metadata.get('routing_reason'),
                file_name=file_path.name,
                page_count=metadata.get('page_count', 1),
            )

            results.append(result)

            # Error analysis for failures
            if generate_error_analysis and (metrics['cer'] > 0.1 or metrics['wer'] > 0.1):
                diff_html = OCREvaluator.create_diff_html(predicted, ground_truth)

                error = ErrorAnalysis(
                    file_name=file_path.name,
                    page_num=1,  # TODO: Handle multi-page
                    predicted=predicted,
                    ground_truth=ground_truth,
                    diff_html=diff_html,
                    cer=metrics['cer'],
                    wer=metrics['wer'],
                    routing_reason=metadata.get('routing_reason'),
                )

                errors.append(error)

        # Compute aggregate metrics
        if results:
            cers = [r.cer for r in results]
            wers = [r.wer for r in results]
            accs = [r.accuracy for r in results]

            avg_metrics = {
                'cer': np.mean(cers),
                'wer': np.mean(wers),
                'accuracy': np.mean(accs),
                'cer_std': np.std(cers),
                'wer_std': np.std(wers),
                'accuracy_std': np.std(accs),
            }

            # Performance metrics
            perf_metrics = {
                'avg_latency_ms': np.mean(latencies),
                'p50_latency_ms': np.percentile(latencies, 50),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'throughput_files_per_sec': len(results) / (sum(latencies) / 1000) if latencies else 0,
            }

            # Engine hit rate
            total_files = sum(self.engine_usage.values())
            engine_hit_rate = {
                engine: count / total_files
                for engine, count in self.engine_usage.items()
            }

        else:
            avg_metrics = {}
            perf_metrics = {}
            engine_hit_rate = {}

        benchmark_results = {
            'results': [asdict(r) for r in results],
            'errors': errors,
            'average_metrics': avg_metrics,
            'performance_metrics': perf_metrics,
            'engine_hit_rate': engine_hit_rate,
            'num_files': len(results),
        }

        # Generate reports
        self._generate_reports(benchmark_results)

        return benchmark_results

    def _generate_reports(self, results: Dict[str, Any]):
        """
        Generate CSV and Markdown reports.

        Args:
            results: Benchmark results dictionary
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Generate CSV report
        csv_path = self.reports_dir / f"benchmark_{timestamp}.csv"
        self._generate_csv_report(results, csv_path)

        # Generate Markdown report
        md_path = self.reports_dir / f"benchmark_{timestamp}.md"
        self._generate_markdown_report(results, md_path)

        # Generate JSON report
        json_path = self.reports_dir / f"benchmark_{timestamp}.json"
        self._generate_json_report(results, json_path)

        logger.info(f"Reports generated in {self.reports_dir}")

    def _generate_csv_report(self, results: Dict[str, Any], output_path: Path):
        """Generate CSV report."""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'file_name', 'cer', 'wer', 'accuracy', 'latency_ms',
                    'engine_used', 'routing_reason'
                ])
                writer.writeheader()

                for result in results['results']:
                    writer.writerow(result)

            logger.info(f"CSV report saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate CSV report: {e}")

    def _generate_markdown_report(self, results: Dict[str, Any], output_path: Path):
        """Generate Markdown report."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# OCR Benchmark Report\n\n")

                # Summary
                f.write("## Summary\n\n")
                f.write(f"- **Files Evaluated**: {results['num_files']}\n")

                avg = results.get('average_metrics', {})
                f.write(f"- **Average CER**: {avg.get('cer', 0):.4f} ± {avg.get('cer_std', 0):.4f}\n")
                f.write(f"- **Average WER**: {avg.get('wer', 0):.4f} ± {avg.get('wer_std', 0):.4f}\n")
                f.write(f"- **Average Accuracy**: {avg.get('accuracy', 0):.4f} ± {avg.get('accuracy_std', 0):.4f}\n\n")

                # Performance
                perf = results.get('performance_metrics', {})
                f.write("## Performance\n\n")
                f.write(f"- **Average Latency**: {perf.get('avg_latency_ms', 0):.2f} ms\n")
                f.write(f"- **P95 Latency**: {perf.get('p95_latency_ms', 0):.2f} ms\n")
                f.write(f"- **P99 Latency**: {perf.get('p99_latency_ms', 0):.2f} ms\n")
                f.write(f"- **Throughput**: {perf.get('throughput_files_per_sec', 0):.2f} files/sec\n\n")

                # Engine hit rate
                hit_rate = results.get('engine_hit_rate', {})
                f.write("## Engine Hit Rate\n\n")
                for engine, rate in hit_rate.items():
                    f.write(f"- **{engine}**: {rate:.2%}\n")
                f.write("\n")

                # Error analysis
                errors = results.get('errors', [])
                if errors:
                    f.write("## Error Analysis\n\n")
                    f.write(f"**Top {min(len(errors), 10)} Errors:**\n\n")

                    for i, error in enumerate(errors[:10], 1):
                        f.write(f"### {i}. {error.file_name} (Page {error.page_num})\n\n")
                        f.write(f"- **CER**: {error.cer:.4f}\n")
                        f.write(f"- **WER**: {error.wer:.4f}\n")
                        if error.routing_reason:
                            f.write(f"- **Routing**: {error.routing_reason}\n")
                        f.write("\n")

            logger.info(f"Markdown report saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate Markdown report: {e}")

    def _generate_json_report(self, results: Dict[str, Any], output_path: Path):
        """Generate JSON report."""
        try:
            # Convert ErrorAnalysis objects to dicts (exclude diff_html for size)
            results_copy = results.copy()
            results_copy['errors'] = [
                {
                    'file_name': e.file_name,
                    'page_num': e.page_num,
                    'cer': e.cer,
                    'wer': e.wer,
                    'routing_reason': e.routing_reason,
                }
                for e in results.get('errors', [])
            ]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, indent=2, ensure_ascii=False)

            logger.info(f"JSON report saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
