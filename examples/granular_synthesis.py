"""
Granular synthesis example for sam-audio-kit.

This example demonstrates latent granular synthesis:
1. Build a grain database from source audio
2. Synthesize new audio using a guide's rhythm/structure
3. Remix with weighted sources
4. Create random collages

The approach is inspired by Naotokui's latent granular synthesis work.
"""

from pathlib import Path

from sam_audio_kit import SamAudio, cleanup_gpu_memory
from sam_audio_kit.synth import LatentSynthesizer


def main():
    # Path to your audio file
    audio_path = "path/to/your/song.wav"

    # Check if file exists
    if not Path(audio_path).exists():
        print("Please update 'audio_path' to point to an actual audio file")
        print("Example: audio_path = '/home/user/music/song.wav'")
        return

    # Load model
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
    # Step 1: Separate stems from the song
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 1: Separating stems")
    print("=" * 50)

    vocals = model.separate(audio_path, "vocals", verbose=True)
    drums = model.separate(audio_path, "drums and percussion", verbose=True)
    bass = model.separate(audio_path, "bass", verbose=True)

    # Save original stems for reference
    vocals.save(output_dir / "original_vocals.wav")
    drums.save(output_dir / "original_drums.wav")
    bass.save(output_dir / "original_bass.wav")
    print("  Original stems saved")

    # =========================================================================
    # Step 2: Build grain database
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 2: Building grain database")
    print("=" * 50)

    # Add stems to the grain database
    db_info = synth.build_database(
        sources=[drums, bass],
        names=["drums", "bass"],
    )

    print(f"  Total grains: {db_info['n_grains']}")
    print(f"  Sources: {db_info['sources']}")

    # =========================================================================
    # Step 3: Basic granular synthesis
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 3: Granular synthesis (vocals → drums/bass)")
    print("=" * 50)
    print("  Using vocals as guide, replacing with drum/bass grains")

    # Temperature controls randomness:
    # 0.0 = always pick the most similar grain (deterministic)
    # 1.0 = completely random grain selection
    for temp in [0.0, 0.3, 0.7]:
        result = synth.granular_synthesize(vocals, temperature=temp)
        output_path = output_dir / f"granular_temp_{int(temp*10)}.wav"
        synth.save(output_path, result)
        print(f"  Saved: {output_path} (temperature={temp})")

    # =========================================================================
    # Step 4: Blending original with granular
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 4: Blend original with granular")
    print("=" * 50)

    # Keep some of the original vocals while adding granular texture
    for blend in [0.3, 0.5, 0.7]:
        result = synth.granular_synthesize(vocals, temperature=0.3, blend_original=blend)
        output_path = output_dir / f"granular_blend_{int(blend*10)}.wav"
        synth.save(output_path, result)
        print(f"  Saved: {output_path} (blend_original={blend})")

    # =========================================================================
    # Step 5: Weighted source remix
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 5: Weighted source remix")
    print("=" * 50)

    # Control the mix of drums vs bass in the granular output
    result_drums_heavy = synth.granular_remix(
        vocals,
        source_weights={"drums": 0.8, "bass": 0.2},
        temperature=0.3,
    )
    synth.save(output_dir / "remix_drums_heavy.wav", result_drums_heavy)
    print("  Saved: remix_drums_heavy.wav (80% drums, 20% bass)")

    result_bass_heavy = synth.granular_remix(
        vocals,
        source_weights={"drums": 0.2, "bass": 0.8},
        temperature=0.3,
    )
    synth.save(output_dir / "remix_bass_heavy.wav", result_bass_heavy)
    print("  Saved: remix_bass_heavy.wav (20% drums, 80% bass)")

    # =========================================================================
    # Step 6: Random collage
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 6: Random collage")
    print("=" * 50)

    # Generate a random collage of grains (no guide needed)
    collage = synth.random_collage(duration_seconds=10.0)
    synth.save(output_dir / "random_collage.wav", collage)
    print("  Saved: random_collage.wav (10 seconds)")

    # Weighted collage
    collage_weighted = synth.random_collage(
        duration_seconds=10.0,
        source_weights={"drums": 0.7, "bass": 0.3},
    )
    synth.save(output_dir / "random_collage_weighted.wav", collage_weighted)
    print("  Saved: random_collage_weighted.wav (70% drums, 30% bass)")

    # =========================================================================
    # Step 7: Adding more sources to database
    # =========================================================================
    print("\n" + "=" * 50)
    print("Step 7: Expanding the database")
    print("=" * 50)

    # Add vocals to the existing database
    grains_added = synth.add_to_database(vocals, name="vocals")
    print(f"  Added {grains_added} vocal grains")

    # Check database info
    db_info = synth.database_info()
    print(f"  Total grains now: {db_info['n_grains']}")
    print(f"  Sources: {db_info['sources']}")

    # Now we can remix with all three sources
    result_all = synth.granular_remix(
        drums,  # Use drums as guide this time
        source_weights={"vocals": 0.5, "drums": 0.3, "bass": 0.2},
        temperature=0.3,
    )
    synth.save(output_dir / "remix_all_sources.wav", result_all)
    print("  Saved: remix_all_sources.wav")

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
