"""
Device management utilities for macOS (MPS/CPU).
"""

import os
import platform
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Literal, Optional

import torch

from util.logging import get_logger

logger = get_logger(__name__)

DeviceType = Literal['mps', 'cpu']


class DeviceManager:
    """Manage device selection and thread/process pools."""

    def __init__(
        self,
        preferred_device: DeviceType = 'mps',
        num_workers: int = 4,
        num_processes: int = 2,
    ):
        """
        Initialize device manager.

        Args:
            preferred_device: Preferred device type ('mps' or 'cpu')
            num_workers: Number of threads for I/O-bound tasks
            num_processes: Number of processes for CPU-bound tasks
        """
        self.preferred_device = preferred_device
        self.num_workers = num_workers
        self.num_processes = num_processes

        # Detect available device
        self.device = self._detect_device()
        logger.info(f"Using device: {self.device}")

        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)

        # Initialize process pool
        self.process_pool = ProcessPoolExecutor(max_workers=num_processes)

    def _detect_device(self) -> torch.device:
        """Detect available device (MPS or CPU)."""
        # Check if macOS
        if platform.system() != 'Darwin':
            logger.warning("Not running on macOS, falling back to CPU")
            return torch.device('cpu')

        # Check if MPS is available
        if self.preferred_device == 'mps':
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    logger.warning(
                        "MPS not available because PyTorch was not built with MPS enabled, using CPU"
                    )
                else:
                    logger.warning(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device, using CPU"
                    )
                return torch.device('cpu')

            logger.info("MPS backend is available and will be used")
            return torch.device('mps')

        return torch.device('cpu')

    def get_device(self) -> torch.device:
        """Get current device."""
        return self.device

    def get_device_name(self) -> str:
        """Get device name as string."""
        return str(self.device)

    def is_mps(self) -> bool:
        """Check if using MPS."""
        return self.device.type == 'mps'

    def move_to_device(self, tensor_or_model):
        """Move tensor or model to device."""
        try:
            return tensor_or_model.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to move to device {self.device}: {e}, using CPU")
            return tensor_or_model.to('cpu')

    def get_torch_dtype(self) -> torch.dtype:
        """Get appropriate dtype for device."""
        if self.is_mps():
            # MPS works best with float32
            return torch.float32
        return torch.float32

    def optimize_for_device(self, model):
        """Apply device-specific optimizations."""
        if self.is_mps():
            # MPS-specific optimizations
            logger.info("Applying MPS optimizations")
            # Set optimal number of threads
            torch.set_num_threads(self.num_workers)

        return model

    def get_optimal_batch_size(self, base_batch_size: int = 1) -> int:
        """
        Get optimal batch size for device.

        Args:
            base_batch_size: Base batch size

        Returns:
            Optimal batch size for current device
        """
        if self.is_mps():
            # MPS can handle larger batches
            return base_batch_size * 2
        return base_batch_size

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up device manager resources")

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        # Shutdown process pool
        self.process_pool.shutdown(wait=True)

        # Clear MPS cache if using MPS
        if self.is_mps():
            try:
                # Note: MPS doesn't have empty_cache in older versions
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to clear MPS cache: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Global device manager instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager(
    preferred_device: DeviceType = 'mps',
    num_workers: Optional[int] = None,
    num_processes: Optional[int] = None,
) -> DeviceManager:
    """
    Get global device manager instance.

    Args:
        preferred_device: Preferred device type
        num_workers: Number of worker threads
        num_processes: Number of worker processes

    Returns:
        DeviceManager instance
    """
    global _device_manager

    if _device_manager is None:
        # Get from environment or use defaults
        num_workers = num_workers or int(os.getenv('NUM_WORKERS', '4'))
        num_processes = num_processes or int(os.getenv('NUM_PROCESSES', '2'))

        _device_manager = DeviceManager(
            preferred_device=preferred_device,
            num_workers=num_workers,
            num_processes=num_processes,
        )

    return _device_manager


def get_device() -> torch.device:
    """Get current device."""
    return get_device_manager().get_device()


def move_to_device(tensor_or_model):
    """Move tensor or model to device."""
    return get_device_manager().move_to_device(tensor_or_model)
