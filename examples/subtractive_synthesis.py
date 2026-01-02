"""
Iterative Subtractive Synthesis Example for SAM-Audio-Kit

This example demonstrates how to use SAM-Audio as a semantic subtractive synthesizer,
iteratively removing unwanted components from audio to sculpt desired sounds.

The concept:
1. Start with any audio source (noise, recording, etc.)
2. Use text descriptions to remove specific components
3. Keep the residual as input for the next iteration
4. Build complex sounds through semantic filtering

This approach transforms SAM-Audio from a separation tool into a creative
semantic synthesizer where natural language guides the filtering process.
"""

from pathlib import Path
import torch
import torchaudio

from sam_audio_kit import SamAudio, cleanup_gpu_memory


# =============================================================================
# CONFIGURATION
# =============================================================================

# Input audio file (or None to use generated white noise)
AUDIO_PATH = "/home/worzpro/Desktop/dev/patched_modules/sam-audio-kit/assets/guitar_loop.wav"

# Duration in seconds (None = full length, or trim to this duration)
DURATION = None

# What to iteratively remove (each step uses the previous residual)
# SAM-Audio works best with musical/audio element descriptions like:
#   - "vocals", "drums", "bass", "guitar", "piano", "strings"
#   - "high frequencies", "low frequencies"
#   - "reverb", "room ambience"
#
REMOVAL_DESCRIPTIONS = [
    "pick attack and string noise",
    "high frequency harmonics",
    "room reverb and ambience",
]

# Save intermediate results for each removal step
SAVE_INTERMEDIATE = True


def subtractive_synthesis(
    model,
    initial_audio,
    removal_descriptions,
    save_intermediate=False,
    output_dir="output/subtractive"
):
    """
    Iterative subtractive synthesis using SAM-Audio.
    
    Args:
        model: SamAudio instance
        initial_audio: Starting audio (tensor, array, or file path)
        removal_descriptions: List of what to remove in each iteration
        save_intermediate: Whether to save each iteration's results
        output_dir: Directory to save results
        
    Returns:
        Final sculpted audio tensor
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load initial audio if it's a file path
    if isinstance(initial_audio, (str, Path)):
        current_audio, sr = torchaudio.load(str(initial_audio))
        print(f"Loaded initial audio: {initial_audio} ({sr}Hz)")
    else:
        current_audio = initial_audio
        sr = model.sample_rate
    
    print(f"\n{'='*60}")
    print("ITERATIVE SUBTRACTIVE SYNTHESIS")
    print(f"{'='*60}")
    print(f"Starting with audio shape: {current_audio.shape}")
    print(f"Planned iterations: {len(removal_descriptions)}")
    
    for i, description in enumerate(removal_descriptions):
        print(f"\n--- Iteration {i+1}/{len(removal_descriptions)} ---")
        print(f"Removing: '{description}'")
        
        # Separate unwanted components
        result = model.separate(current_audio, description, verbose=False)
        
        # Keep the residual (what remains after removal)
        current_audio = result.residual
        
        print(f"  Target RMS: {torch.norm(result.target):.6f}")
        print(f"  Residual RMS: {torch.norm(result.residual):.6f}")
        print(f"  Quality ratio: {torch.norm(result.residual)/torch.norm(result.target):.3f}")
        
        if save_intermediate:
            target_path = output_path / f"removed_{i+1:02d}_{description.replace(' ', '_')}.wav"
            residual_path = output_path / f"remaining_{i+1:02d}_{description.replace(' ', '_')}.wav"
            
            result.save(target_path, residual_path)
            print(f"  Saved: {target_path.name}")
            print(f"  Saved: {residual_path.name}")
    
    # Save final result
    final_path = output_path / "final_result.wav"
    if current_audio.dim() == 1:
        current_audio = current_audio.unsqueeze(0)
    torchaudio.save(str(final_path), current_audio.cpu(), sr)
    print(f"\nFinal result saved: {final_path}")
    
    return current_audio


def create_white_noise(duration_seconds=10.0, sample_rate=48000):
    """Generate white noise as starting material."""
    num_samples = int(duration_seconds * sample_rate)
    return torch.randn(1, num_samples) * 0.1  # Lower amplitude for safety


def main():
    """Run subtractive synthesis with configuration."""
    print("=" * 60)
    print("SAM-Audio Iterative Subtractive Synthesis")
    print("=" * 60)

    # Load model
    print("\nLoading SAM-Audio model...")
    model = SamAudio.from_pretrained("base", dtype="bfloat16")

    # Get or generate input audio
    if AUDIO_PATH is None:
        print(f"\nGenerating {DURATION}s of white noise as starting material...")
        audio = create_white_noise(duration_seconds=DURATION)
    else:
        if not Path(AUDIO_PATH).exists():
            print(f"Audio file not found: {AUDIO_PATH}")
            return
        print(f"\nLoading audio: {AUDIO_PATH}")
        audio, sr = torchaudio.load(AUDIO_PATH)
        # Trim to DURATION if specified
        if DURATION is not None:
            max_samples = int(DURATION * sr)
            if audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]
                print(f"  Trimmed to {DURATION}s")

    # Run subtractive synthesis
    final_audio = subtractive_synthesis(
        model,
        audio,
        REMOVAL_DESCRIPTIONS,
        save_intermediate=SAVE_INTERMEDIATE,
        output_dir="output/subtractive"
    )

    cleanup_gpu_memory()

    print("\n" + "=" * 60)
    print("Done! Check 'output/subtractive/' for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
