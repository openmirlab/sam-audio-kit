"""
Lite mode demonstration for sam-audio-infer.

This example shows the VRAM savings achieved by lite mode
compared to the full model.
"""

import torch

from sam_audio_infer import (
    SamAudioInfer,
    LiteModelConfig,
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
    print("SAM-Audio Lite Mode Demonstration")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nWarning: CUDA not available. Memory comparisons won't be accurate.")

    # Show estimated savings
    print("\nEstimated VRAM savings with lite mode:")
    for size in ["small", "base", "large"]:
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

    # Load lite model
    print("\n[1] Loading LITE model (aggressive config)...")
    lite_config = LiteModelConfig.aggressive()
    print(f"    Removing: {lite_config.components_to_remove}")

    try:
        model_lite = SamAudioInfer.from_pretrained(
            "base",
            lite_mode=True,
            lite_config=lite_config,
            dtype="bfloat16",
            verbose=False,
        )
        print_gpu_memory("After lite model load")

        # Get actual memory usage
        lite_memory = get_gpu_memory_info()
        lite_allocated = lite_memory.allocated_gb if lite_memory else 0

        # Cleanup
        model_lite.unload()
        cleanup_gpu_memory()
        print_gpu_memory("After unload")

    except Exception as e:
        print(f"    Error: {e}")
        lite_allocated = 0

    print("\n" + "-" * 60)

    # Summary
    print("\nSummary:")
    print(f"  Lite model (bfloat16): ~{lite_allocated:.2f} GB")
    print(f"  Recommended minimum VRAM: 6 GB")
    print(f"  Comfortable operation: 8+ GB")

    print("\n" + "=" * 60)
    print("Lite Mode Configuration Options:")
    print("=" * 60)

    # Show configuration options
    print("\n1. Aggressive (default) - Maximum VRAM savings:")
    aggressive = LiteModelConfig.aggressive()
    print(f"   Removes: {aggressive.components_to_remove}")

    print("\n2. Conservative - Keep some features:")
    conservative = LiteModelConfig.conservative()
    print(f"   Removes: {conservative.components_to_remove}")

    print("\n3. Custom configuration:")
    print("""
   from sam_audio_infer import LiteModelConfig

   config = LiteModelConfig(
       remove_vision_encoder=True,   # ~2GB savings
       remove_visual_ranker=True,    # ~2GB savings
       remove_text_ranker=False,     # Keep for better results
       remove_span_predictor=True,   # ~1-2GB savings
   )
   model = SamAudioInfer.from_pretrained("base", lite_config=config)
    """)


if __name__ == "__main__":
    main()
