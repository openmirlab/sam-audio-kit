"""
Lite mode demonstration for sam-audio-kit.

This example shows VRAM usage for different model configurations.
Lite mode is always enabled - you control which optional features to add.
"""

import torch

from sam_audio_kit import (
    SamAudio,
    cleanup_gpu_memory,
    get_gpu_memory_info,
    estimate_lite_savings,
)


def print_gpu_memory(label: str):
    """Print current GPU memory usage."""
    info = get_gpu_memory_info()
    if info:
        print(f"  {label}: {info.allocated_gb:.2f} GB allocated, {info.free_gb:.2f} GB free")
    else:
        print(f"  {label}: CUDA not available")


def main():
    print("=" * 60)
    print("SAM-Audio VRAM Usage Demonstration")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nWarning: CUDA not available. Memory comparisons won't be accurate.")

    # Show estimated savings
    print("\nEstimated VRAM savings with lite mode:")
    for size in ["base", "large"]:
        savings = estimate_lite_savings(size)
        print(f"\n  {size.upper()} model:")
        for component, gb in savings.items():
            print(f"    {component}: ~{gb:.1f} GB")

    print("\n" + "-" * 60)
    print("Loading models to compare memory usage...")
    print("-" * 60)

    # Cleanup before starting
    cleanup_gpu_memory()
    print_gpu_memory("Initial state")

    # Configuration 1: Minimal (default) - most VRAM efficient
    print("\n[1] Loading MINIMAL config (default, most efficient)...")
    print("    enable_text_ranker=False, enable_span_predictor=False")

    try:
        model = SamAudio.from_pretrained(
            "base",
            dtype="bfloat16",
            verbose=False,
        )
        print_gpu_memory("After load")

        minimal_memory = get_gpu_memory_info()
        minimal_allocated = minimal_memory.allocated_gb if minimal_memory else 0

        model.unload()
        cleanup_gpu_memory()
        print_gpu_memory("After unload")

    except Exception as e:
        print(f"    Error: {e}")
        minimal_allocated = 0

    # Configuration 2: With text ranker (better quality, +3GB)
    print("\n[2] Loading WITH TEXT RANKER (better quality, +3GB)...")
    print("    enable_text_ranker=True")

    try:
        model = SamAudio.from_pretrained(
            "base",
            dtype="bfloat16",
            enable_text_ranker=True,
            verbose=False,
        )
        print_gpu_memory("After load")

        ranker_memory = get_gpu_memory_info()
        ranker_allocated = ranker_memory.allocated_gb if ranker_memory else 0

        model.unload()
        cleanup_gpu_memory()
        print_gpu_memory("After unload")

    except Exception as e:
        print(f"    Error: {e}")
        ranker_allocated = 0

    print("\n" + "-" * 60)

    # Summary
    print("\nSummary (base model, bfloat16):")
    print(f"  Minimal config:      ~{minimal_allocated:.2f} GB")
    print(f"  With text ranker:    ~{ranker_allocated:.2f} GB")
    print(f"  Recommended minimum: 4 GB")
    print(f"  Comfortable:         8+ GB")

    print("\n" + "=" * 60)
    print("Configuration Options:")
    print("=" * 60)

    print("""
1. Minimal (default) - Maximum VRAM savings (~3 GB):
   model = SamAudio.from_pretrained("base", dtype="bfloat16")

2. With text ranker - Better separation quality (~6 GB):
   model = SamAudio.from_pretrained(
       "base",
       dtype="bfloat16",
       enable_text_ranker=True,
   )

3. With span predictor - Time segment detection (~6 GB):
   model = SamAudio.from_pretrained(
       "base",
       dtype="bfloat16",
       enable_span_predictor=True,
   )

4. All features enabled (~9 GB):
   model = SamAudio.from_pretrained(
       "base",
       dtype="bfloat16",
       enable_text_ranker=True,
       enable_span_predictor=True,
   )
    """)


if __name__ == "__main__":
    main()
