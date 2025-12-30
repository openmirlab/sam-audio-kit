"""
Batch separation example for sam-audio-infer.

This example demonstrates how to separate multiple stems
from a single audio file.
"""

from pathlib import Path

from sam_audio_infer import SamAudioInfer, cleanup_gpu_memory


def main():
    # Path to your audio file
    audio_path = "path/to/your/song.wav"

    # Check if file exists
    if not Path(audio_path).exists():
        print(f"Please update 'audio_path' to point to an actual audio file")
        return

    # Define stems to extract
    stems = [
        "vocals",
        "drums and percussion",
        "bass",
        "piano and keyboards",
        "guitar",
    ]

    # Load model
    print("Loading SAM-Audio model...")
    model = SamAudioInfer.from_pretrained(
        "base",
        lite_mode=True,
        dtype="bfloat16",
        verbose=True,
    )

    # Create output directory
    output_dir = Path("output/stems")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate each stem
    print(f"\nSeparating {len(stems)} stems...")

    for i, stem in enumerate(stems):
        print(f"\n[{i+1}/{len(stems)}] Extracting: {stem}")

        result = model.separate(
            audio_path,
            description=stem,
            verbose=True,
        )

        # Save with sanitized filename
        safe_name = stem.replace(" ", "_").replace("/", "-")
        output_path = output_dir / f"{safe_name}.wav"
        result.save(output_path)

        print(f"  Saved: {output_path}")
        print(f"  Time: {result.processing_time:.1f}s")

        # Cleanup between stems to manage memory
        cleanup_gpu_memory()

    print(f"\n{'='*50}")
    print(f"All stems saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
