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
from sam_audio_kit.synth import DACVAECodec


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


def example_1_noise_to_texture():
    """Example 1: Transform white noise into textured sound."""
    print("\n" + "="*60)
    print("EXAMPLE 1: White Noise to Wind Texture")
    print("="*60)
    
    # Load model
    model = SamAudio.from_pretrained("base", dtype="bfloat16")
    
    # Create white noise
    white_noise = create_white_noise(duration_seconds=8.0)
    
    # Define removal sequence
    removal_descriptions = [
        "sharp transient sounds and clicks",
        "low frequency rumble below 80Hz", 
        "high frequency hiss above 12kHz",
        "periodic and repetitive components",
        "harmonic content and musical tones"
    ]
    
    # Apply subtractive synthesis
    final_sound = subtractive_synthesis(
        model,
        white_noise,
        removal_descriptions,
        save_intermediate=True,
        output_dir="output/wind_texture"
    )
    
    cleanup_gpu_memory()
    return final_sound


def example_2_vintage_effect():
    """Example 2: Create vintage radio effect from clean audio."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Vintage Radio Effect")
    print("="*60)
    
    # Load model
    model = SamAudio.from_pretrained("base", dtype="bfloat16")
    
    # You need to provide your own audio file
    input_audio = "path/to/your/clean_recording.wav"
    
    if not Path(input_audio).exists():
        print(f"Please update input_audio path to point to an actual audio file")
        print(f"Current: {input_audio}")
        return None
    
    # Define vintage character removal
    removal_descriptions = [
        "deep bass frequencies below 100Hz",
        "bright high frequencies above 8kHz", 
        "stereo width and spatial imaging",
        "modern digital clarity",
        "precise transients and attacks"
    ]
    
    # Apply subtractive synthesis
    vintage_sound = subtractive_synthesis(
        model,
        input_audio,
        removal_descriptions,
        save_intermediate=True,
        output_dir="output/vintage_effect"
    )
    
    cleanup_gpu_memory()
    return vintage_sound


def example_3_sound_design():
    """Example 3: Create sci-fi creature sound from animal recording."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Sci-Fi Creature Sound Design")
    print("="*60)
    
    # Load model
    model = SamAudio.from_pretrained("base", dtype="bfloat16")
    
    # You need to provide an animal recording
    input_audio = "path/to/animal_recording.wav"
    
    if not Path(input_audio).exists():
        print(f"Please update input_audio path to point to an actual audio file")
        print(f"Current: {input_audio}")
        return None
    
    # Transform into alien creature
    removal_descriptions = [
        "natural animal characteristics",
        "familiar mammal sounds",
        "earthly resonances and tones",
        "organic warmth and body",
        "recognizable animal vocalizations"
    ]
    
    # Apply subtractive synthesis
    creature_sound = subtractive_synthesis(
        model,
        input_audio,
        removal_descriptions,
        save_intermediate=True,
        output_dir="output/creature_sound"
    )
    
    cleanup_gpu_memory()
    return creature_sound


def example_4_audio_restoration():
    """Example 4: Multi-stage audio restoration."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Audio Restoration")
    print("="*60)
    
    # Load model
    model = SamAudio.from_pretrained("base", dtype="bfloat16")
    
    # You need to provide a noisy recording
    input_audio = "path/to/noisy_recording.wav"
    
    if not Path(input_audio).exists():
        print(f"Please update input_audio path to point to an actual audio file")
        print(f"Current: {input_audio}")
        return None
    
    # Remove various types of noise
    removal_descriptions = [
        "vinyl crackle and pops",
        "tape hiss and surface noise",
        "60Hz electrical hum and buzz",
        "room reverb and ambience",
        "background noise and artifacts"
    ]
    
    # Apply subtractive synthesis
    restored_audio = subtractive_synthesis(
        model,
        input_audio,
        removal_descriptions,
        save_intermediate=True,
        output_dir="output/restored_audio"
    )
    
    cleanup_gpu_memory()
    return restored_audio


def example_5_semantic_eq():
    """Example 5: Semantic EQ with character shaping."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Semantic EQ and Character Shaping")
    print("="*60)
    
    # Load model
    model = SamAudio.from_pretrained("base", dtype="bfloat16")
    
    # You need to provide a mix
    input_audio = "path/to/your/mix.wav"
    
    if not Path(input_audio).exists():
        print(f"Please update input_audio path to point to an actual audio file")
        print(f"Current: {input_audio}")
        return None
    
    # Shape character through semantic filtering
    removal_descriptions = [
        "harsh sibilance and ess sounds",
        "muddy low-mid buildup around 300Hz",
        "boxy resonances in the 1kHz range",
        "excessive brightness and air",
        "digital harshness and edginess"
    ]
    
    # Apply subtractive synthesis
    shaped_audio = subtractive_synthesis(
        model,
        input_audio,
        removal_descriptions,
        save_intermediate=True,
        output_dir="output/semantic_eq"
    )
    
    cleanup_gpu_memory()
    return shaped_audio


def main():
    """Run all examples."""
    print("SAM-Audio Iterative Subtractive Synthesis Examples")
    print("=" * 60)
    print("\nThis demo shows how to use SAM-Audio as a semantic subtractive synthesizer.")
    print("Each example transforms audio by iteratively removing unwanted components.")
    
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    try:
        # Example 1: Noise to texture (always works)
        example_1_noise_to_texture()
        
        # Examples 2-5 require user-provided audio files
        print("\n" + "="*60)
        print("NOTE: Examples 2-5 require you to update audio file paths")
        print("Please edit the file paths in the example functions above")
        print("="*60)
        
        # Uncomment these examples after updating file paths:
        # example_2_vintage_effect()
        # example_3_sound_design()
        # example_4_audio_restoration()
        # example_5_semantic_eq()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Installed sam-audio-kit: pip install sam-audio-kit")
        print("2. Authenticated with HuggingFace: huggingface-cli login")
        print("3. CUDA-capable GPU with sufficient VRAM")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check the 'output/' directory for results.")
    print("="*60)


if __name__ == "__main__":
    main()
