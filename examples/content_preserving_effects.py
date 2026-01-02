"""
Content-Preserving Audio Effects with SAM-Audio

This example focuses on the key challenge: maintaining audio content
while manipulating character and timbre.

Core Techniques:
1. Latent Space Enhancement - Direct manipulation in encoded space
2. Controlled Blending - Preserve core content while adding effects
3. Character Transfer - Apply timbre from one sound to another
4. Parallel Processing - Independent control over different characteristics
"""

from pathlib import Path
import torch
import torchaudio

from sam_audio_kit import SamAudio, cleanup_gpu_memory
from sam_audio_kit.synth import LatentSynthesizer


def match_length(tensor_a, tensor_b):
    """Match tensor lengths by trimming the longer one."""
    len_a = tensor_a.shape[-1]
    len_b = tensor_b.shape[-1]
    min_len = min(len_a, len_b)
    return tensor_a[..., :min_len], tensor_b[..., :min_len]


def latent_enhancement_chain(model, audio):
    """
    Apply multiple enhancements in latent space.
    
    This is the most content-preserving approach because:
    - All information is preserved in the latent representation
    - Changes are applied to character, not content
    - Decoding reconstructs the full audio with new character
    """
    synth = LatentSynthesizer(model)
    
    # Encode to latent space (preserves all musical content)
    print("  Encoding to latent space...")
    latent = synth.encode(audio)
    print(f"  Latent shape: {latent.shape}")
    
    # Apply enhancements in latent space
    print("  Applying character enhancements...")
    
    # 1. Add warmth (low-mid enhancement)
    warm_latent = synth.enhance(latent, warmth=0.3)
    
    # 2. Add presence (mid-range enhancement) 
    present_latent = synth.enhance(warm_latent, presence=0.2)
    
    # 3. Add subtle brightness (high-frequency enhancement)
    bright_latent = synth.enhance(present_latent, brightness=0.15)
    
    # 4. Add air (very high-frequency enhancement)
    final_latent = synth.enhance(bright_latent, brightness=0.1)
    
    # Decode back to audio (content preserved, character enhanced)
    print("  Decoding enhanced audio...")
    enhanced_audio = synth.decode(final_latent)
    
    return enhanced_audio, latent, final_latent


def controlled_character_blending(model, audio, characteristics):
    """
    Extract and blend specific characteristics while preserving content.
    
    This allows independent control over different aspects of the sound.
    """
    processed_audio = audio.clone()
    blend_log = []
    
    for char_name, (description, intensity) in characteristics.items():
        print(f"  Processing {char_name}...")
        
        # Extract the characteristic
        result = model.separate(processed_audio, description)
        
        # Create enhanced version
        if intensity > 0:
            # Boost the characteristic
            enhanced_char = result.target * (1 + intensity)
            
            # Blend back with current audio (preserves other content)
            blend_amount = 0.2  # How much of the enhanced characteristic to add
            processed_audio = (1 - blend_amount) * processed_audio + blend_amount * enhanced_char
            
            blend_log.append(f"{char_name}: +{intensity*100:.0f}% boost")
        else:
            # Reduce the characteristic
            reduced_char = result.target * (1 + intensity)  # intensity is negative
            processed_audio = (1 - abs(intensity) * 0.3) * result.residual + abs(intensity) * 0.3 * reduced_char
            blend_log.append(f"{char_name}: {intensity*100:.0f}% reduction")
    
    return processed_audio, blend_log


def timbre_transfer(model, source_audio, target_audio, character_desc):
    """
    Transfer timbre/character from source to target while preserving target's content.
    
    This is like saying "make this guitar sound like that piano" 
    while keeping the guitar's notes and timing.
    """
    print(f"  Transferring '{character_desc}' from source to target...")
    
    # Extract character from source
    source_result = model.separate(source_audio, character_desc)
    source_character = source_result.target
    
    # Extract content from target (remove the character we want to replace)
    target_result = model.separate(target_audio, character_desc)
    target_content = target_result.residual  # Target without this character
    
    # Apply source character to target content
    transfer_strength = 0.5  # How strongly to apply the new character
    hybrid_audio = (1 - transfer_strength) * target_content + transfer_strength * source_character
    
    return hybrid_audio, source_result, target_result


def morph_between_characters(model, audio, from_char, to_char, steps=5):
    """
    Gradually morph from one character to another while preserving content.
    
    Useful for creating variations or transitions.
    """
    print(f"  Morphing from '{from_char}' to '{to_char}'...")
    
    # Extract both characters
    from_result = model.separate(audio, from_char)
    to_result = model.separate(audio, to_char)
    
    morph_sequence = []
    
    for i in range(steps + 1):
        alpha = i / steps  # 0.0 = fully 'from', 1.0 = fully 'to'
        
        # Morph between characters (match lengths first)
        from_t, to_t = match_length(from_result.target, to_result.target)
        morphed_char = (1 - alpha) * from_t + alpha * to_t

        # Preserve content by blending with original
        content_preservation = 0.7  # Keep 70% original content
        orig, morphed = match_length(audio, morphed_char)
        morphed_audio = content_preservation * orig + (1 - content_preservation) * morphed
        
        morph_sequence.append(morphed_audio)
    
    return morph_sequence


def adaptive_content_preservation(model, audio, target_profile):
    """
    Intelligently enhance audio based on a target profile while preserving content.
    
    This analyzes what the audio needs and applies just the right amount of processing.
    """
    print("  Analyzing and adaptively processing...")
    
    processed_audio = audio.clone()
    adjustments = []
    
    for aspect, (description, target_level) in target_profile.items():
        # Extract current level of this aspect
        result = model.separate(processed_audio, description)
        current_level = torch.norm(result.target)
        
        # Calculate needed adjustment
        if current_level < target_level:
            # Need to boost
            boost_factor = target_level / current_level
            enhanced_char = result.target * boost_factor
            
            # Apply subtle boost
            blend_amount = 0.25
            p, e = match_length(processed_audio, enhanced_char)
            processed_audio = (1 - blend_amount) * p + blend_amount * e
            adjustments.append(f"{aspect}: +{(boost_factor-1)*100:.0f}%")
        elif current_level > target_level:
            # Need to reduce
            reduction_factor = target_level / current_level
            res, tgt = match_length(result.residual, result.target)
            processed_audio = (1 - 0.3) * res + 0.3 * tgt * reduction_factor
            adjustments.append(f"{aspect}: -{(1-reduction_factor)*100:.0f}%")
        else:
            adjustments.append(f"{aspect}: no change")
    
    return processed_audio, adjustments


def demonstrate_vocal_processing(model, vocal_audio):
    """Demonstrate content-preserving effects on vocals."""
    print("\n" + "="*50)
    print("VOCAL PROCESSING - CONTENT PRESERVING")
    print("="*50)
    
    output_dir = Path("output/content_preserving")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Latent Space Enhancement (most content-preserving)
    print("\n1. Latent Space Enhancement...")
    enhanced, orig_latent, final_latent = latent_enhancement_chain(model, vocal_audio)
    torchaudio.save(str(output_dir / "01_latent_enhanced.wav"), enhanced.cpu(), 48000)
    
    # 2. Controlled Character Blending
    print("\n2. Controlled Character Blending...")
    characteristics = {
        'warmth': ('warm low-mid body', 0.3),
        'presence': ('vocal presence', 0.2), 
        'clarity': ('high frequency clarity', 0.15),
        'air': ('air and space', 0.1)
    }
    
    blended, blend_log = controlled_character_blending(model, vocal_audio, characteristics)
    torchaudio.save(str(output_dir / "02_character_blended.wav"), blended.cpu(), 48000)
    print("  Adjustments:", ", ".join(blend_log))
    
    # 3. Timbre Transfer (if you have reference audio)
    print("\n3. Timbre Transfer...")
    # For demo, use the same audio as source (in practice, use different source)
    source_ref = vocal_audio  # Would be different audio in real use
    
    transferred, source_res, target_res = timbre_transfer(
        model, source_ref, vocal_audio, "warm tube character"
    )
    torchaudio.save(str(output_dir / "03_timbre_transferred.wav"), transferred.cpu(), 48000)
    
    # 4. Character Morphing
    print("\n4. Character Morphing...")
    morph_sequence = morph_between_characters(
        model, vocal_audio,
        "intimate close sound",
        "spacious ambient sound",
        steps=3
    )
    
    for i, morphed in enumerate(morph_sequence):
        torchaudio.save(str(output_dir / f"04_morph_{i:02d}.wav"), morphed.cpu(), 48000)
    
    # 5. Adaptive Processing
    print("\n5. Adaptive Content Preservation...")
    target_profile = {
        'body': ('warm body and weight', torch.norm(vocal_audio) * 0.3),
        'presence': ('vocal presence', torch.norm(vocal_audio) * 0.2),
        'clarity': ('clear high frequencies', torch.norm(vocal_audio) * 0.15)
    }
    
    adaptive, adjustments = adaptive_content_preservation(model, vocal_audio, target_profile)
    torchaudio.save(str(output_dir / "05_adaptive.wav"), adaptive.cpu(), 48000)
    print("  Adaptive adjustments:", ", ".join(adjustments))
    
    return enhanced, blended, transferred


def demonstrate_instrumental_processing(model, instrumental_audio):
    """Demonstrate content-preserving effects on instruments."""
    print("\n" + "="*50)
    print("INSTRUMENTAL PROCESSING - CONTENT PRESERVING")
    print("="*50)
    
    output_dir = Path("output/content_preserving")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Rhythmic Enhancement (preserve melody, enhance rhythm)
    print("\n1. Rhythmic Enhancement...")
    rhythm_result = model.separate(instrumental_audio, "rhythmic attack and transients")
    
    # Boost rhythmic elements
    rhythm_boost = 1.4
    enhanced_rhythm = rhythm_result.target * rhythm_boost
    
    # Blend with original to preserve harmonic content
    orig, enhanced = match_length(instrumental_audio, enhanced_rhythm)
    rhythm_enhanced = 0.7 * orig + 0.3 * enhanced
    torchaudio.save(str(output_dir / "06_rhythm_enhanced.wav"), rhythm_enhanced.cpu(), 48000)
    
    # 2. Harmonic Enrichment (preserve rhythm, enrich harmonics)
    print("\n2. Harmonic Enrichment...")
    harmonic_result = model.separate(instrumental_audio, "upper harmonics and overtones")
    
    # Enrich harmonics
    harmonic_boost = 1.3
    enriched_harmonics = harmonic_result.target * harmonic_boost
    
    # Subtle blend to maintain natural sound
    orig, enhanced = match_length(instrumental_audio, enriched_harmonics)
    harmonic_enhanced = 0.8 * orig + 0.2 * enhanced
    torchaudio.save(str(output_dir / "07_harmonic_enhanced.wav"), harmonic_enhanced.cpu(), 48000)
    
    # 3. Spatial Enhancement (add space without changing content)
    print("\n3. Spatial Enhancement...")
    spatial_result = model.separate(instrumental_audio, "room reverb and space")
    
    # Add spatial character
    spatial_boost = 1.5
    enhanced_space = spatial_result.target * spatial_boost
    
    # Blend to add space
    orig, enhanced = match_length(instrumental_audio, enhanced_space)
    spatial_enhanced = 0.85 * orig + 0.15 * enhanced
    torchaudio.save(str(output_dir / "08_spatial_enhanced.wav"), spatial_enhanced.cpu(), 48000)
    
    return rhythm_enhanced, harmonic_enhanced, spatial_enhanced


def main():
    """Demonstrate content-preserving audio effects."""
    print("=" * 60)
    print("CONTENT-PRESERVING AUDIO EFFECTS")
    print("=" * 60)
    print("\nThe key principle: preserve musical content while manipulating character.")
    
    # Load model
    print("\nLoading SAM-Audio model...")
    model = SamAudio.from_pretrained("base", dtype="bfloat16")
    
    # Test with available audio
    audio_files = [
        ("./assets/guitar_loop.wav", "instrumental"),
        ("./assets/vocal_sample.wav", "vocal"),
    ]
    
    for audio_path, audio_type in audio_files:
        if not Path(audio_path).exists():
            print(f"\nSkipping {audio_path} (file not found)")
            continue
        
        print(f"\nProcessing {audio_type}: {audio_path}")
        audio, sr = torchaudio.load(audio_path)
        
        # Trim to 30 seconds for faster processing
        max_samples = int(30 * sr)
        if audio.shape[-1] > max_samples:
            audio = audio[..., :max_samples]
            print(f"  Trimmed to 30 seconds")
        
        if audio_type == "vocal":
            demonstrate_vocal_processing(model, audio)
        else:
            demonstrate_instrumental_processing(model, audio)
    
    cleanup_gpu_memory()
    
    print("\n" + "=" * 60)
    print("Content-preserving effects completed!")
    print("Check 'output/content_preserving/' for results.")
    print("\nKey takeaways:")
    print("1. Latent space manipulation preserves the most content")
    print("2. Controlled blending maintains core structure")
    print("3. Character transfer applies new timbre while preserving notes")
    print("4. Adaptive processing intelligently enhances what's needed")
    print("=" * 60)


if __name__ == "__main__":
    main()
