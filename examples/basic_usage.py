"""
Basic usage example for sam-audio-infer.

This example demonstrates how to:
1. Load a SAM-Audio model with lite mode
2. Separate vocals from a song
3. Save the results
"""

from pathlib import Path

from sam_audio_infer import SamAudioInfer


def main():
    # Path to your audio file
    audio_path = "path/to/your/song.wav"

    # Check if file exists
    if not Path(audio_path).exists():
        print(f"Please update 'audio_path' to point to an actual audio file")
        print("Example: audio_path = '/home/user/music/song.wav'")
        return

    # Load model with lite mode (reduced VRAM)
    print("Loading SAM-Audio model...")
    model = SamAudioInfer.from_pretrained(
        "base",              # Model size: "small", "base", or "large"
        lite_mode=True,      # Enable lite mode for ~40% VRAM reduction
        dtype="bfloat16",    # Use bfloat16 for ~50% additional savings
        device="cuda",       # Use GPU (or "cpu" for CPU-only)
        verbose=True,        # Print loading progress
    )

    # Separate vocals
    print("\nSeparating vocals...")
    result = model.separate(
        audio_path,
        description="vocals",  # What to extract
        verbose=True,
    )

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    vocals_path = output_dir / "vocals.wav"
    accompaniment_path = output_dir / "accompaniment.wav"

    result.save(vocals_path, accompaniment_path)

    print(f"\nResults saved:")
    print(f"  Vocals: {vocals_path}")
    print(f"  Accompaniment: {accompaniment_path}")
    print(f"  Processing time: {result.processing_time:.1f}s")


if __name__ == "__main__":
    main()
