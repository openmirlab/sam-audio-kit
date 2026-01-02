"""
Basic usage example for sam-audio-kit.

This example demonstrates how to:
1. Load a SAM-Audio model (lite mode is always enabled for efficiency)
2. Separate vocals from a song
3. Save the results

Note: Long audio files (>25s) are automatically chunked and processed.
"""

from pathlib import Path

from sam_audio_kit import SamAudio


def main():
    # Path to your audio file
    audio_path = "./assets/newjeas_supershy.wav"

    # Check if file exists
    if not Path(audio_path).exists():
        print("Please update 'audio_path' to point to an actual audio file")
        print("Example: audio_path = '/home/user/music/song.wav'")
        return

    # Load model (lite mode is always enabled, ~3 GB VRAM with bfloat16)
    print("Loading SAM-Audio model...")
    model = SamAudio.from_pretrained(
        "base",                   # Model size: "small", "base", or "large"
        dtype="bfloat16",         # Use bfloat16 for ~50% VRAM savings
        enable_text_ranker=False, # Keep VRAM low (+3GB if enabled)
        enable_span_predictor=False, # Keep VRAM low (+3GB if enabled)
        device="cuda",            # Use GPU (or "cpu" for CPU-only)
        verbose=True,             # Print loading progress
    )

    # What to extract (change this to separate different elements)
    description = "synthesizer"

    # Separate audio
    print(f"\nSeparating: {description}...")
    result = model.separate(
        audio_path,
        description=description,
        verbose=True,
    )

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Use description for filename (replace spaces with underscores)
    safe_name = description.replace(" ", "_")
    target_path = output_dir / f"{safe_name}.wav"
    residual_path = output_dir / f"{safe_name}_residual.wav"

    result.save(target_path, residual_path)

    print(f"\nResults saved:")
    print(f"  Target ({description}): {target_path}")
    print(f"  Residual: {residual_path}")
    print(f"  Processing time: {result.processing_time:.1f}s")
    print(f"  Chunks processed: {result.num_chunks}")


if __name__ == "__main__":
    main()
