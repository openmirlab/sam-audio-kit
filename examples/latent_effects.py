"""
Latent space effects example for sam-audio-kit.

This example demonstrates creative audio manipulation in latent space:
1. Interpolation between two audio sources
2. Crossfade morphing over time
3. Texture transfer (apply one sound's texture to another)
4. Tonal enhancement (brightness, warmth, presence)
5. Time stretching and reversing
6. Multi-source blending
"""

from pathlib import Path

import torchaudio

from sam_audio_kit import SamAudio, cleanup_gpu_memory
from sam_audio_kit.synth import LatentSynthesizer


# =============================================================================
# CONFIGURATION - Set your audio file paths
# =============================================================================

AUDIO_PATH_1 = "path/to/audio1.wav"
AUDIO_PATH_2 = "path/to/audio2.wav"


def main():
    # Validate paths
    if not Path(AUDIO_PATH_1).exists() or not Path(AUDIO_PATH_2).exists():
        print("Please update audio paths to point to actual audio files")
        print("Example:")
        print("  AUDIO_PATH_1 = '/home/user/music/vocals.wav'")
        print("  AUDIO_PATH_2 = '/home/user/music/drums.wav'")
        return

    # Load model
    print("Loading SAM-Audio model...")
    model = SamAudio.from_pretrained(
        "base",
        dtype="bfloat16",
        verbose=True,
    )

    # Create synthesizer for latent manipulation
    synth = LatentSynthesizer(model)

    # Create output directory
    output_dir = Path("output/effects")
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Load audio files
    # =========================================================================
    print("\n" + "=" * 50)
    print("Loading audio files")
    print("=" * 50)

    audio_1, _ = torchaudio.load(AUDIO_PATH_1)
    audio_2, _ = torchaudio.load(AUDIO_PATH_2)

    print(f"  Audio 1: {AUDIO_PATH_1}")
    print(f"  Audio 2: {AUDIO_PATH_2}")

    # =========================================================================
    # Example 1: Interpolation
    # =========================================================================
    print("\n" + "=" * 50)
    print("Example 1: Interpolation between two sources")
    print("=" * 50)

    # Encode both audio sources to latent space
    latent_1 = synth.encode(audio_1)
    latent_2 = synth.encode(audio_2)

    # Create interpolated versions (0.0 = 100% source 1, 1.0 = 100% source 2)
    for alpha in [0.25, 0.5, 0.75]:
        interpolated = synth.interpolate(latent_1, latent_2, alpha=alpha)
        output_path = output_dir / f"interpolate_{int(alpha*100)}.wav"
        synth.save(output_path, interpolated)
        print(f"  Saved: {output_path} (alpha={alpha})")

    # =========================================================================
    # Example 2: Crossfade (time-varying morph)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Example 2: Crossfade (morphs over time)")
    print("=" * 50)

    for curve in ["linear", "cosine", "exponential"]:
        crossfaded = synth.crossfade(latent_1, latent_2, curve=curve)
        output_path = output_dir / f"crossfade_{curve}.wav"
        synth.save(output_path, crossfaded)
        print(f"  Saved: {output_path} (curve={curve})")

    # =========================================================================
    # Example 3: Texture Transfer
    # =========================================================================
    print("\n" + "=" * 50)
    print("Example 3: Texture Transfer")
    print("=" * 50)
    print("  Applies texture/timbre from source 2 to source 1")

    for strength in [0.3, 0.5, 0.7]:
        textured = synth.apply_texture(latent_1, latent_2, strength=strength)
        output_path = output_dir / f"texture_{int(strength*100)}.wav"
        synth.save(output_path, textured)
        print(f"  Saved: {output_path} (strength={strength})")

    # =========================================================================
    # Example 4: Tonal Enhancement
    # =========================================================================
    print("\n" + "=" * 50)
    print("Example 4: Tonal Enhancement")
    print("=" * 50)

    # Brighten (boost highs)
    brightened = synth.enhance(latent_1, brightness=0.5)
    synth.save(output_dir / "enhance_bright.wav", brightened)
    print("  Saved: enhance_bright.wav (brightness=0.5)")

    # Warm (boost lows)
    warmed = synth.enhance(latent_1, warmth=0.5)
    synth.save(output_dir / "enhance_warm.wav", warmed)
    print("  Saved: enhance_warm.wav (warmth=0.5)")

    # Add presence (boost mids)
    present = synth.enhance(latent_1, presence=0.5)
    synth.save(output_dir / "enhance_presence.wav", present)
    print("  Saved: enhance_presence.wav (presence=0.5)")

    # Combined enhancement
    enhanced = synth.enhance(latent_1, brightness=0.3, warmth=0.2, presence=0.4)
    synth.save(output_dir / "enhance_combined.wav", enhanced)
    print("  Saved: enhance_combined.wav (combined)")

    # =========================================================================
    # Example 5: Time Effects
    # =========================================================================
    print("\n" + "=" * 50)
    print("Example 5: Time Effects")
    print("=" * 50)

    # Time stretch (slow down)
    stretched = synth.time_stretch(latent_1, factor=1.5)
    synth.save(output_dir / "time_slow.wav", stretched)
    print("  Saved: time_slow.wav (factor=1.5, slower)")

    # Time compress (speed up)
    compressed = synth.time_stretch(latent_1, factor=0.7)
    synth.save(output_dir / "time_fast.wav", compressed)
    print("  Saved: time_fast.wav (factor=0.7, faster)")

    # Reverse
    reversed_audio = synth.reverse(latent_1)
    synth.save(output_dir / "reversed.wav", reversed_audio)
    print("  Saved: reversed.wav")

    # =========================================================================
    # Example 6: Multi-source Blending
    # =========================================================================
    print("\n" + "=" * 50)
    print("Example 6: Multi-source Blending")
    print("=" * 50)

    # Equal blend
    blended = synth.blend([latent_1, latent_2])
    synth.save(output_dir / "blend_equal.wav", blended)
    print("  Saved: blend_equal.wav (50/50 mix)")

    # Weighted blend
    blended_weighted = synth.blend([latent_1, latent_2], weights=[0.7, 0.3])
    synth.save(output_dir / "blend_weighted.wav", blended_weighted)
    print("  Saved: blend_weighted.wav (70/30 mix)")

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n" + "=" * 50)
    print(f"All effects saved to: {output_dir}")
    print("=" * 50)

    cleanup_gpu_memory()


if __name__ == "__main__":
    main()
