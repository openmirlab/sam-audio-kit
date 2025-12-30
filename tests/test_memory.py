"""Tests for memory management utilities."""

import pytest

from sam_audio_infer.memory import (
    GPUMemoryInfo,
    cleanup_gpu_memory,
    get_gpu_memory_info,
    MemoryTracker,
)


class TestGPUMemoryInfo:
    """Tests for GPUMemoryInfo."""

    def test_str_representation(self):
        info = GPUMemoryInfo(
            device_name="Test GPU",
            total_memory_gb=8.0,
            allocated_gb=2.0,
            reserved_gb=3.0,
            free_gb=5.0,
            peak_allocated_gb=2.5,
        )
        output = str(info)
        assert "Test GPU" in output
        assert "8.00 GB" in output
        assert "2.00 GB" in output


class TestMemoryFunctions:
    """Tests for memory utility functions."""

    def test_cleanup_gpu_memory_no_error(self):
        # Should not raise even without CUDA
        cleanup_gpu_memory()

    def test_get_gpu_memory_info_returns_none_without_cuda(self):
        # May return None if CUDA is not available
        info = get_gpu_memory_info()
        # Just check it doesn't raise
        assert info is None or isinstance(info, GPUMemoryInfo)


class TestMemoryTracker:
    """Tests for MemoryTracker context manager."""

    def test_memory_tracker_context(self):
        with MemoryTracker("Test") as tracker:
            # Do nothing
            pass

        # Check delta is computed (may be 0 without CUDA)
        assert hasattr(tracker, "delta_gb")

    def test_memory_tracker_label(self):
        tracker = MemoryTracker("Custom Label")
        assert tracker.label == "Custom Label"
