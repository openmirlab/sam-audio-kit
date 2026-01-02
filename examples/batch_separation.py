"""
Batch separation example for sam-audio-kit.

This example demonstrates two approaches for separating multiple stems:
1. Manual loop with cleanup between stems
2. Using the built-in separate_batch() method
"""

from pathlib import Path

from sam_audio_kit import SamAudio, cleanup_gpu_memory


def main():
    # Path to your audio file
    audio_path = "path/to/your/song.wav"

    # Check if file exists
    if not Path(audio_path).exists():
        print("Please update 'audio_path' to point to an actual audio file")
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
    model = SamAudio.from_pretrained(
        "base",
        dtype="bfloat16",
        verbose=True,
    )

    # Create output directory
    output_dir = Path("output/stems")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Option 1: Manual loop (more control over each separation)
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

    # Option 2: Using separate_batch() (simpler, handles cleanup automatically)
    # results = model.separate_batch(audio_path, stems, verbose=True)
    # for result, stem in zip(results, stems):
    #     safe_name = stem.replace(" ", "_").replace("/", "-")
    #     result.save(output_dir / f"{safe_name}.wav")


if __name__ == "__main__":
    main()
