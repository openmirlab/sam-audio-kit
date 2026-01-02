"""
Granular synthesis example for sam-audio-kit.

This example demonstrates latent granular synthesis with SOURCE and TARGET concept:

    SOURCE (grain database)     TARGET (guide audio)
    ━━━━━━━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━━━━━━
    Contains grains from:      Provides structure to follow:
    - drums                    - rhythm
    - bass                     - timing
    - any audio samples        - envelope shape

    The synthesizer finds the best matching grain from SOURCE
    for each segment of TARGET, creating a new hybrid audio.

Workflow:
1. Build SOURCE database from audio (drums, bass, samples, etc.)
2. Use TARGET audio as guide - its structure drives grain selection
3. Output: TARGET's rhythm/structure played with SOURCE's sounds

The approach is inspired by Naotokui's latent granular synthesis work.
"""

from pathlib import Path

import torchaudio

from sam_audio_kit import SamAudio, cleanup_gpu_memory
from sam_audio_kit.synth import LatentSynthesizer


# =============================================================================
# CONFIGURATION - Set your audio file paths
# =============================================================================

# SOURCE audio files (will become grains in the database)
SOURCE_FILES = {
    "drums": "path/to/drums.wav",
    "bass": "path/to/bass.wav",
}

# TARGET audio file (guide - its structure drives grain selection)
TARGET_FILE = "path/to/vocals.wav"


def main():
    # Validate paths
    missing = [k for k, v in SOURCE_FILES.items() if not Path(v).exists()]
    if missing:
        print("Please update 'SOURCE_FILES' paths. Missing:")
        for k in missing:
            print(f"  {k}: {SOURCE_FILES[k]}")
        return
    if not Path(TARGET_FILE).exists():
        print(f"Please update 'TARGET_FILE' path: {TARGET_FILE}")
        return

    # Load model (needed for latent encoding)
    print("Loading SAM-Audio model...")
    model = SamAudio.from_pretrained(
        "base",
        dtype="bfloat16",
        verbose=True,
    )

    # Create synthesizer
    synth = LatentSynthesizer(model)

    # Create output directory
    output_dir = Path("output/granular")
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load audio files
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 1: Loading audio files")
    print("=" * 50)

    # Load SOURCE audio (grain material)
    sources = {}
    for name, path in SOURCE_FILES.items():
        audio, _ = torchaudio.load(path)
        sources[name] = audio
        print(f"  SOURCE '{name}': {path}")

    # Load TARGET audio (guide)
    target, _ = torchaudio.load(TARGET_FILE)
    print(f"  TARGET: {TARGET_FILE}")

    # =========================================================================
    # Step 2: Build SOURCE grain database
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 2: Building SOURCE grain database")
    print("=" * 50)

    # Build database from all SOURCE audio
    source_names = list(sources.keys())
    source_audio = list(sources.values())

    db_info = synth.build_database(
        sources=source_audio,
        names=source_names,
    )

    print(f"  Total grains: {db_info['n_grains']}")
    print(f"  Sources: {db_info['sources']}")

    # =========================================================================
    # Step 3: Basic granular synthesis (TARGET → SOURCE grains)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 3: Granular synthesis (TARGET → SOURCE grains)")
    print("=" * 50)
    print("  Using TARGET as guide, replacing with SOURCE grains")

    # Temperature controls randomness:
    # 0.0 = always pick the most similar grain (deterministic)
    # 1.0 = completely random grain selection
    for temp in [0.0, 0.3, 0.7]:
        result = synth.granular_synthesize(target, temperature=temp)
        output_path = output_dir / f"granular_temp_{int(temp*10)}.wav"
        synth.save(output_path, result)
        print(f"  Saved: {output_path} (temperature={temp})")

    # =========================================================================
    # Step 4: Blending TARGET with granular
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 4: Blend TARGET with granular")
    print("=" * 50)

    # Keep some of the original TARGET while adding granular texture
    for blend in [0.3, 0.5, 0.7]:
        result = synth.granular_synthesize(target, temperature=0.3, blend_original=blend)
        output_path = output_dir / f"granular_blend_{int(blend*10)}.wav"
        synth.save(output_path, result)
        print(f"  Saved: {output_path} (blend_original={blend})")

    # =========================================================================
    # Step 5: Weighted SOURCE remix
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 5: Weighted SOURCE remix")
    print("=" * 50)

    # Control the mix of different sources in the granular output
    # (Only works if you have multiple sources)
    if len(sources) >= 2:
        names = list(sources.keys())
        result_first_heavy = synth.granular_remix(
            target,
            source_weights={names[0]: 0.8, names[1]: 0.2},
            temperature=0.3,
        )
        synth.save(output_dir / f"remix_{names[0]}_heavy.wav", result_first_heavy)
        print(f"  Saved: remix_{names[0]}_heavy.wav (80% {names[0]}, 20% {names[1]})")

        result_second_heavy = synth.granular_remix(
            target,
            source_weights={names[0]: 0.2, names[1]: 0.8},
            temperature=0.3,
        )
        synth.save(output_dir / f"remix_{names[1]}_heavy.wav", result_second_heavy)
        print(f"  Saved: remix_{names[1]}_heavy.wav (20% {names[0]}, 80% {names[1]})")

    # =========================================================================
    # Step 6: Random collage (no TARGET needed)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 6: Random collage")
    print("=" * 50)

    # Generate a random collage of grains (no guide needed)
    collage = synth.random_collage(duration_seconds=10.0)
    synth.save(output_dir / "random_collage.wav", collage)
    print("  Saved: random_collage.wav (10 seconds)")

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n" + "=" * 50)
    print(f"All granular outputs saved to: {output_dir}")
    print("=" * 50)

    synth.clear_database()
    cleanup_gpu_memory()


if __name__ == "__main__":
    main()
