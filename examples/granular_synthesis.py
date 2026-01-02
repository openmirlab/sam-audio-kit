"""
Granular synthesis example for sam-audio-kit.

This example demonstrates latent granular synthesis:
1. Build a grain database from source audio
2. Synthesize new audio using a guide's rhythm/structure
3. Remix with weighted sources
4. Create random collages

Two modes are supported:
- WITH SAM-Audio: Separate stems from a song, then use them for granular synthesis
- WITHOUT SAM-Audio: Use existing audio files directly (pre-separated stems, samples, etc.)

The approach is inspired by Naotokui's latent granular synthesis work.
"""

from pathlib import Path

from sam_audio_kit import SamAudio, cleanup_gpu_memory
from sam_audio_kit.synth import LatentSynthesizer


# =============================================================================
# CONFIGURATION - Choose your mode
# =============================================================================

# Set to True to use SAM-Audio separation, False to use existing audio files
USE_SAM_AUDIO_SEPARATION = True

# For USE_SAM_AUDIO_SEPARATION = True: path to song to separate
SONG_PATH = "path/to/your/song.wav"

# For USE_SAM_AUDIO_SEPARATION = False: paths to existing audio files
AUDIO_FILES = {
    "drums": "path/to/drums.wav",
    "bass": "path/to/bass.wav",
    "vocals": "path/to/vocals.wav",  # Used as guide
}


def main():
    import torchaudio

    # Validate paths based on mode
    if USE_SAM_AUDIO_SEPARATION:
        if not Path(SONG_PATH).exists():
            print("Please update 'SONG_PATH' to point to an actual audio file")
            print("Example: SONG_PATH = '/home/user/music/song.wav'")
            return
    else:
        missing = [k for k, v in AUDIO_FILES.items() if not Path(v).exists()]
        if missing:
            print("Please update 'AUDIO_FILES' paths. Missing:")
            for k in missing:
                print(f"  {k}: {AUDIO_FILES[k]}")
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
    # Step 1: Get audio sources (either separate or load existing)
    # =========================================================================
    print("\n" + "=" * 50)
    if USE_SAM_AUDIO_SEPARATION:
        print("Step 1: Separating stems with SAM-Audio")
        print("=" * 50)

        vocals = model.separate(SONG_PATH, "vocals", verbose=True)
        drums = model.separate(SONG_PATH, "drums and percussion", verbose=True)
        bass = model.separate(SONG_PATH, "bass", verbose=True)

        # Save separated stems for reference
        vocals.save(output_dir / "original_vocals.wav")
        drums.save(output_dir / "original_drums.wav")
        bass.save(output_dir / "original_bass.wav")
        print("  Separated stems saved")
    else:
        print("Step 1: Loading existing audio files")
        print("=" * 50)

        # Load audio files directly as tensors
        drums, _ = torchaudio.load(AUDIO_FILES["drums"])
        bass, _ = torchaudio.load(AUDIO_FILES["bass"])
        vocals, _ = torchaudio.load(AUDIO_FILES["vocals"])

        print(f"  Loaded drums: {AUDIO_FILES['drums']}")
        print(f"  Loaded bass: {AUDIO_FILES['bass']}")
        print(f"  Loaded vocals: {AUDIO_FILES['vocals']}")

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
