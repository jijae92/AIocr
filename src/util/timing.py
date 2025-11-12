"""
Timing utilities for performance measurement.
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

from util.logging import get_logger

logger = get_logger(__name__)


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self, name: Optional[str] = None, logger_func: Optional[Callable] = None):
        """
        Initialize timer.

        Args:
            name: Timer name for logging
            logger_func: Logger function to use (default: logger.info)
        """
        self.name = name
        self.logger_func = logger_func or logger.info
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def start(self):
        """Start timer."""
        self.start_time = time.perf_counter()
        if self.name:
            self.logger_func(f"Started: {self.name}")

    def stop(self) -> float:
        """
        Stop timer and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

        if self.name:
            self.logger_func(f"Finished: {self.name} (took {self.elapsed:.3f}s)")

        return self.elapsed

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


@contextmanager
def timed_operation(name: str, logger_func: Optional[Callable] = None):
    """
    Context manager for timing operations.

    Args:
        name: Operation name
        logger_func: Logger function to use

    Example:
        with timed_operation("OCR processing"):
            # ... do work ...
    """
    timer = Timer(name=name, logger_func=logger_func)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


def timeit(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator for timing function execution.

    Args:
        func: Function to decorate
        name: Custom name for logging (default: function name)

    Example:
        @timeit
        def process_image(img):
            # ... do work ...

        @timeit(name="Custom operation")
        def another_func():
            # ... do work ...
    """

    def decorator(f: Callable) -> Callable:
        operation_name = name or f.__name__

        @wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            timer = Timer(name=operation_name)
            timer.start()
            try:
                result = f(*args, **kwargs)
                timer.stop()
                return result
            except Exception as e:
                timer.stop()
                logger.error(f"Error in {operation_name}: {e}")
                raise

        return wrapper

    # Support both @timeit and @timeit()
    if func is None:
        return decorator
    return decorator(func)


class PerformanceTracker:
    """Track performance metrics across multiple operations."""

    def __init__(self):
        self.operations: Dict[str, list] = {}

    def record(self, operation: str, duration: float):
        """
        Record operation duration.

        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        if operation not in self.operations:
            self.operations[operation] = []
        self.operations[operation].append(duration)

    @contextmanager
    def track(self, operation: str):
        """
        Context manager for tracking operation.

        Args:
            operation: Operation name

        Example:
            tracker = PerformanceTracker()
            with tracker.track("ocr"):
                # ... do work ...
        """
        timer = Timer()
        timer.start()
        try:
            yield timer
        finally:
            duration = timer.stop()
            self.record(operation, duration)

    def get_stats(self, operation: str) -> Dict[str, float]:
        """
        Get statistics for operation.

        Args:
            operation: Operation name

        Returns:
            Dictionary with min, max, mean, median, p95, p99
        """
        if operation not in self.operations or not self.operations[operation]:
            return {}

        durations = sorted(self.operations[operation])
        count = len(durations)

        stats = {
            'count': count,
            'min': durations[0],
            'max': durations[-1],
            'mean': sum(durations) / count,
            'median': durations[count // 2],
        }

        # Percentiles
        if count > 0:
            stats['p95'] = durations[int(count * 0.95)]
            stats['p99'] = durations[int(count * 0.99)]

        return stats

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.operations}

    def reset(self):
        """Reset all tracked operations."""
        self.operations.clear()

    def print_summary(self):
        """Print summary of all tracked operations."""
        logger.info("Performance Summary:")
        for operation, stats in self.get_all_stats().items():
            logger.info(f"\n{operation}:")
            for key, value in stats.items():
                if key == 'count':
                    logger.info(f"  {key}: {value}")
                else:
                    logger.info(f"  {key}: {value:.3f}s")
