"""
Creative Audio Effects with SAM-Audio - Content-Preserving Manipulation

This example demonstrates techniques for manipulating audio character
while preserving the original content and structure.

Key approaches:
1. Residual Blending - Keep core content, add manipulated character
2. Parallel Processing - Extract and blend specific characteristics  
3. Character Transfer - Apply timbre/texture from one sound to another
4. Semantic Morphing - Gradually transform between characters
"""

from pathlib import Path
import torch
import torchaudio

from sam_audio_kit import SamAudio, cleanup_gpu_memory
from sam_audio_kit.synth import LatentSynthesizer


def residual_blend_effect(model, audio, description, blend_amount=0.3):
    """
    Apply effect while preserving content through residual blending.
    
    Instead of just removing components, we:
    1. Separate the unwanted component
    2. Keep the residual (main content)
    3. Blend back a controlled amount of the separated component
    
    This preserves the core structure while adding character.
    """
    result = model.separate(audio, description)
    
    # Blend original with residual to maintain content
    # Higher blend_amount = more original content preserved
    processed = (1 - blend_amount) * result.residual + blend_amount * audio
    
    return processed, result


def parallel_character_extraction(model, audio, characteristics):
    """
    Extract multiple characteristics in parallel and blend them.
    
    This separates different aspects of the sound and allows
    independent control over each characteristic.
    """
    extracted = {}
    
    for char_name, description in characteristics.items():
        result = model.separate(audio, description)
        extracted[char_name] = {
            'target': result.target,
            'residual': result.residual,
            'description': description
        }
    
    return extracted


def character_transfer(model, source_audio, target_audio, characteristic):
    """
    Transfer a specific characteristic from source to target.
    
    1. Extract characteristic from source sound
    2. Apply it to target sound while preserving target's content
    """
    # Extract characteristic from source
    source_result = model.separate(source_audio, characteristic)
    source_char = source_result.target
    
    # Extract same characteristic from target (to be replaced)
    target_result = model.separate(target_audio, characteristic)
    target_residual = target_result.residual  # Target without this characteristic
    
    # Blend source characteristic into target residual
    # This preserves target's content but adds source's character
    transfer_amount = 0.4  # How much of source character to apply
    processed = (1 - transfer_amount) * target_residual + transfer_amount * source_char
    
    return processed, source_result, target_result


def semantic_morphing(model, audio, from_description, to_description, steps=10):
    """
    Gradually morph from one character to another.
    
    Creates a smooth transition between different sonic characters
    while maintaining the underlying content.
    """
    from_result = model.separate(audio, from_description)
    to_result = model.separate(audio, to_description)
    
    morphed_sequence = []
    
    for i in range(steps + 1):
        alpha = i / steps  # 0.0 = fully 'from', 1.0 = fully 'to'
        
        # Morph between the two extracted characteristics
        morphed_char = (1 - alpha) * from_result.target + alpha * to_result.target
        
        # Blend with original to maintain content
        content_preservation = 0.6  # Keep 60% original content
        morphed = content_preservation * audio + (1 - content_preservation) * morphed_char
        
        morphed_sequence.append(morphed)
    
    return morphed_sequence, from_result, to_result


def adaptive_enhancement(model, audio, target_characteristics):
    """
    Enhance specific characteristics based on what's missing or weak.
    
    Analyzes the audio and enhances characteristics that need improvement
    while preserving the overall content.
    """
    enhancements = {}
    
    for char_name, (description, target_level) in target_characteristics.items():
        result = model.separate(audio, description)
        
        # Simple analysis: compare extracted level to target
        current_level = torch.norm(result.target)
        
        if current_level < target_level:
            # Need to enhance this characteristic
            enhancement_factor = target_level / current_level
            enhanced_char = result.target * enhancement_factor
            
            # Blend enhanced characteristic back into original
            blend_amount = 0.3  # Subtle enhancement
            enhanced_audio = (1 - blend_amount) * audio + blend_amount * enhanced_char
            
            enhancements[char_name] = {
                'enhanced_audio': enhanced_audio,
                'original_level': current_level,
                'target_level': target_level,
                'enhancement_factor': enhancement_factor
            }
        else:
            enhancements[char_name] = {
                'enhanced_audio': audio,  # No change needed
                'original_level': current_level,
                'target_level': target_level,
                'enhancement_factor': 1.0
            }
    
    return enhancements


def create_vocal_chain_effects(model, vocal_audio):
    """
    Demonstrate a complete vocal processing chain using content-preserving effects.
    """
    print("\n" + "="*60)
    print("VOCAL PROCESSING CHAIN")
    print("="*60)
    
    output_dir = Path("output/creative_effects")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Gentle De-essing (preserve vocal content, reduce harsh sibilance)
    print("\n1. Gentle De-essing...")
    deessed, deess_result = residual_blend_effect(
        model, vocal_audio, "harsh sibilance and ess sounds", blend_amount=0.8
    )
    torchaudio.save(str(output_dir / "01_deessed.wav"), deessed.cpu(), 48000)
    print(f"  Original RMS: {torch.norm(vocal_audio):.6f}")
    print(f"  De-essed RMS: {torch.norm(deessed):.6f}")
    
    # 2. Presence Enhancement (add vocal presence without changing content)
    print("\n2. Presence Enhancement...")
    presence_char = ["warm presence", "body", "clarity"]
    extracted = parallel_character_extraction(model, deessed, {
        'warmth': 'warm low-mid frequencies',
        'presence': 'vocal presence around 2kHz', 
        'clarity': 'high frequency clarity and air'
    })
    
    # Blend characteristics back in
    enhanced = deessed.clone()
    for char_name, data in extracted.items():
        blend_amount = 0.15  # Subtle enhancement
        enhanced = (1 - blend_amount) * enhanced + blend_amount * data['target']
    
    torchaudio.save(str(output_dir / "02_enhanced.wav"), enhanced.cpu(), 48000)
    print(f"  Enhanced RMS: {torch.norm(enhanced):.6f}")
    
    # 3. Character Transfer (add tube warmth character)
    print("\n3. Tube Warmth Character Transfer...")
    # Use a reference with tube character (you'd need this file)
    # For demo, we'll extract warmth from the vocal itself and boost it
    tube_reference = vocal_audio  # In practice, use actual tube amp recording
    
    warmed, source_res, target_res = character_transfer(
        model, tube_reference, enhanced, "warm tube saturation character"
    )
    
    torchaudio.save(str(output_dir / "03_warmed.wav"), warmed.cpu(), 48000)
    print(f"  Warmed RMS: {torch.norm(warmed):.6f}")
    
    # 4. Semantic Morphing (create variation)
    print("\n4. Character Morphing...")
    morphed_sequence, from_res, to_res = semantic_morphing(
        model, warmed, 
        "intimate close-miked sound",
        "spacious ambient sound", 
        steps=5
    )
    
    for i, morphed in enumerate(morphed_sequence):
        torchaudio.save(str(output_dir / f"04_morph_{i:02d}.wav"), morphed.cpu(), 48000)
    
    print(f"  Created {len(morphed_sequence)} morph variations")
    
    return warmed, morphed_sequence


def create_instrumental_effects(model, instrumental_audio):
    """
    Demonstrate effects for instrumental music.
    """
    print("\n" + "="*60)
    print("INSTRUMENTAL EFFECTS")
    print("="*60)
    
    output_dir = Path("output/creative_effects")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Rhythm Enhancement (preserve melody, enhance rhythmic elements)
    print("\n1. Rhythm Enhancement...")
    rhythm_result = model.separate(instrumental_audio, "rhythmic percussive elements")
    
    # Boost rhythmic elements while preserving harmonic content
    rhythm_boost = 1.3  # Boost rhythm by 30%
    enhanced_rhythm = rhythm_result.target * rhythm_boost
    
    # Blend back with original (preserve harmonic content)
    blend_amount = 0.4
    rhythm_enhanced = (1 - blend_amount) * instrumental_audio + blend_amount * enhanced_rhythm
    
    torchaudio.save(str(output_dir / "rhythm_enhanced.wav"), rhythm_enhanced.cpu(), 48000)
    
    # 2. Harmonic Coloration (add harmonic richness)
    print("\n2. Harmonic Coloration...")
    harmonic_result = model.separate(instrumental_audio, "upper harmonics and overtones")
    
    # Add harmonic richness
    harmonic_boost = 1.2
    enriched_harmonics = harmonic_result.target * harmonic_boost
    
    # Subtle blend to maintain natural sound
    harmonic_enhanced = 0.85 * instrumental_audio + 0.15 * enriched_harmonics
    
    torchaudio.save(str(output_dir / "harmonic_enhanced.wav"), harmonic_enhanced.cpu(), 48000)
    
    return rhythm_enhanced, harmonic_enhanced


def main():
    """Demonstrate content-preserving creative audio effects."""
    print("=" * 60)
    print("CREATIVE AUDIO EFFECTS - CONTENT PRESERVING")
    print("=" * 60)
    
    # Load model
    print("\nLoading SAM-Audio model...")
    model = SamAudio.from_pretrained("base", dtype="bfloat16")
    
    # Test with different audio types
    test_files = [
        "./assets/guitar_loop.wav",  # Update with your files
        "./assets/vocal_sample.wav",  # Update with your files
    ]
    
    for audio_path in test_files:
        if not Path(audio_path).exists():
            print(f"\nSkipping {audio_path} (file not found)")
            continue
        
        print(f"\nProcessing: {audio_path}")
        audio, sr = torchaudio.load(audio_path)
        
        # Trim to 30 seconds for faster processing
        max_samples = int(30 * sr)
        if audio.shape[-1] > max_samples:
            audio = audio[..., :max_samples]
        
        # Determine audio type and apply appropriate effects
        if "vocal" in audio_path.lower():
            final_audio, variations = create_vocal_chain_effects(model, audio)
        else:
            final_audio, variations = create_instrumental_effects(model, audio)
    
    cleanup_gpu_memory()
    
    print("\n" + "=" * 60)
    print("Creative effects completed!")
    print("Check 'output/creative_effects/' for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
