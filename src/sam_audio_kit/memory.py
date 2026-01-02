"""GPU memory management utilities."""

import gc
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GPUMemoryInfo:
    """GPU memory information."""

    device_name: str
    total_memory_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    peak_allocated_gb: float

    def __str__(self) -> str:
        return (
            f"GPU: {self.device_name}\n"
            f"  Total: {self.total_memory_gb:.2f} GB\n"
            f"  Allocated: {self.allocated_gb:.2f} GB\n"
            f"  Reserved: {self.reserved_gb:.2f} GB\n"
            f"  Free: {self.free_gb:.2f} GB\n"
            f"  Peak: {self.peak_allocated_gb:.2f} GB"
        )


def cleanup_gpu_memory() -> None:
    """
    Clean up GPU memory by running garbage collection and emptying CUDA cache.

    This should be called after processing large batches or when switching models
    to free up VRAM.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info(device_index: int = 0) -> Optional[GPUMemoryInfo]:
    """
    Get current GPU memory usage information.

    Args:
        device_index: GPU device index (default: 0)

    Returns:
        GPUMemoryInfo object with memory statistics, or None if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.device(f"cuda:{device_index}")
        props = torch.cuda.get_device_properties(device)

        total = props.total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        return GPUMemoryInfo(
            device_name=props.name,
            total_memory_gb=total,
            allocated_gb=allocated,
            reserved_gb=reserved,
            free_gb=total - reserved,
            peak_allocated_gb=peak,
        )
    except Exception:
        return None


def reset_peak_memory_stats(device_index: int = 0) -> None:
    """Reset the peak memory statistics for tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_index)


def log_memory_usage(label: str = "", device_index: int = 0) -> None:
    """
    Log current GPU memory usage with an optional label.

    Args:
        label: Optional label to identify the log entry
        device_index: GPU device index (default: 0)
    """
    info = get_gpu_memory_info(device_index)
    if info:
        prefix = f"[{label}] " if label else ""
        print(
            f"{prefix}GPU Memory - "
            f"Allocated: {info.allocated_gb:.2f}GB | "
            f"Reserved: {info.reserved_gb:.2f}GB | "
            f"Peak: {info.peak_allocated_gb:.2f}GB"
        )


class MemoryTracker:
    """
    Context manager for tracking GPU memory usage during operations.

    Example:
        >>> with MemoryTracker("Model Loading"):
        ...     model = load_model()
        GPU Memory [Model Loading] - Before: 0.00GB, After: 5.23GB, Delta: +5.23GB
    """

    def __init__(self, label: str = "Operation", device_index: int = 0):
        self.label = label
        self.device_index = device_index
        self.start_allocated: float = 0.0
        self.end_allocated: float = 0.0

    def __enter__(self) -> "MemoryTracker":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_allocated = (
                torch.cuda.memory_allocated(self.device_index) / (1024 ** 3)
            )
        return self

    def __exit__(self, *args) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.end_allocated = (
                torch.cuda.memory_allocated(self.device_index) / (1024 ** 3)
            )
            delta = self.end_allocated - self.start_allocated
            sign = "+" if delta >= 0 else ""
            print(
                f"GPU Memory [{self.label}] - "
                f"Before: {self.start_allocated:.2f}GB, "
                f"After: {self.end_allocated:.2f}GB, "
                f"Delta: {sign}{delta:.2f}GB"
            )

    @property
    def delta_gb(self) -> float:
        """Get the memory delta in GB."""
        return self.end_allocated - self.start_allocated
