# Iterative Subtractive Synthesis

## Overview

SAM-Audio can be creatively repurposed as an **iterative subtractive synthesizer** - a novel approach that uses the model's separation capabilities to sculpt sound by repeatedly removing unwanted components from an audio source.

This technique transforms SAM-Audio from a source separation tool into a **semantic subtractive synthesizer**, where instead of traditional technical parameters (cutoff frequency, Q factor, etc.), you use natural language descriptions to guide the filtering process.

## Core Concept

### Traditional vs Semantic Subtractive Synthesis

| Traditional Subtractive Synthesis | Semantic Subtractive Synthesis (SAM-Audio) |
|---------------------------------|-------------------------------------------|
| Technical parameters (cutoff, Q, filter type) | Natural language descriptions ("harsh highs", "muddy bass") |
| Fixed filter algorithms | Context-aware, intelligent filtering |
| Manual parameter tweaking | Semantic control |
| Limited to predefined filter shapes | Learns from audio context |

### The Workflow

```
Initial Audio → Separate "unwanted" → Keep Residual → Repeat
```

1. **Start**: Begin with any audio source (white noise, field recording, complex sound)
2. **Iterate**: Remove specific components using text descriptions
3. **Refine**: Use the residual as input for the next iteration
4. **Result**: Sculpted sound through semantic filtering

## Implementation

### Basic Iterative Function

```python
from sam_audio_kit import SamAudio
import torch

def subtractive_synthesis(
    model, 
    initial_audio, 
    removal_descriptions,
    save_intermediate=False
):
    """
    Iterative subtractive synthesis using SAM-Audio.
    
    Args:
        model: SamAudio instance
        initial_audio: Starting audio (tensor, array, or file path)
        removal_descriptions: List of what to remove in each iteration
        save_intermediate: Whether to save each iteration's results
        
    Returns:
        Final sculpted audio tensor
    """
    current_audio = initial_audio
    
    for i, description in enumerate(removal_descriptions):
        print(f"Iteration {i+1}: Removing '{description}'")
        
        # Separate unwanted components
        result = model.separate(current_audio, description)
        
        # Keep the residual (what remains after removal)
        current_audio = result.residual
        
        if save_intermediate:
            result.save(f"removed_{i+1}.wav", f"remaining_{i+1}.wav")
    
    return current_audio

# Usage example
model = SamAudio.from_pretrained("base", dtype="bfloat16")

# Start with white noise
white_noise = torch.randn(1, 48000 * 10)  # 10 seconds at 48kHz

# Iteratively sculpt the sound
final_sound = subtractive_synthesis(
    model, 
    white_noise,
    [
        "harsh high frequencies above 8kHz",
        "muddy sub-bass below 60Hz", 
        "mid-range harshness",
        "noisy artifacts",
        "unpleasant resonances"
    ],
    save_intermediate=True
)
```

### Advanced Implementation with Quality Control

```python
def advanced_subtractive_synthesis(
    model,
    initial_audio,
    removal_descriptions,
    quality_threshold=0.8,
    max_iterations=5
):
    """
    Advanced subtractive synthesis with quality monitoring.
    
    Args:
        model: SamAudio instance
        initial_audio: Starting audio
        removal_descriptions: List of removal descriptions
        quality_threshold: Stop if quality drops below this
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (final_audio, quality_history)
    """
    current_audio = initial_audio
    quality_history = []
    
    for i, description in enumerate(removal_descriptions[:max_iterations]):
        result = model.separate(current_audio, description)
        
        # Simple quality metric (can be enhanced)
        current_rms = torch.norm(result.residual)
        target_rms = torch.norm(initial_audio) * 0.7  # Target 70% of original
        quality_score = min(1.0, current_rms / target_rms)
        quality_history.append(quality_score)
        
        print(f"Iteration {i+1}: Quality = {quality_score:.3f}")
        
        if quality_score < quality_threshold:
            print(f"Quality dropped below threshold. Stopping.")
            break
            
        current_audio = result.residual
    
    return current_audio, quality_history
```

## Practical Applications

### 1. Sound Design

#### Creating Textures from Noise
```python
# Create organic wind texture from white noise
wind_texture = subtractive_synthesis(
    model,
    white_noise,
    [
        "sharp transient sounds",
        "low frequency rumble below 100Hz",
        "high frequency hiss above 12kHz",
        "periodic components"
    ]
)
```

#### Field Recording Cleanup
```python
# Clean up outdoor recording
clean_nature = subtractive_synthesis(
    model,
    "forest_recording.wav",
    [
        "airplane noise",
        "distant traffic",
        "wind gusts",
        "electronic hum"
    ]
)
```

### 2. Musical Effects

#### Semantic EQ
```python
# "Vintage radio" effect
vintage_sound = subtractive_synthesis(
    model,
    "full_mix.wav",
    [
        "deep bass below 100Hz",
        "bright highs above 8kHz",
        "stereo width",
        "modern clarity"
    ]
)
```

#### Character Filtering
```python
# "Lo-fi" character
lofi_character = subtractive_synthesis(
    model,
    "clean_recording.wav",
    [
        "high frequency detail",
        "precise transients",
        "clean high end",
        "digital clarity"
    ]
)
```

### 3. Audio Restoration

#### Multi-stage Noise Removal
```python
# Restore old recording
restored_audio = subtractive_synthesis(
    model,
    "old_recording.wav",
    [
        "vinyl crackle and pops",
        "tape hiss",
        "60Hz electrical hum",
        "room reverb tail"
    ]
)
```

## Best Practices

### 1. Description Engineering

**Effective descriptions:**
- ✅ "harsh sibilance above 6kHz"
- ✅ "muddy low-mid buildup around 300Hz"
- ✅ "boxy resonances in the 1kHz range"
- ✅ "noisy artifacts and digital distortion"

**Less effective descriptions:**
- ❌ "bad sounds"
- ❌ "remove noise"
- ❌ "make it better"

### 2. Iteration Strategy

**Recommended approach:**
1. Start with broad removals (obvious problems)
2. Progress to specific issues (fine-tuning)
3. Limit to 3-5 iterations to avoid quality degradation
4. Save intermediate results for comparison

**Example progression:**
```python
descriptions = [
    "obvious noise and distortion",      # Broad first pass
    "harsh high frequencies",            # Specific problem areas
    "muddy low end",                    # Frequency-specific
    "unpleasant resonances"              # Fine-tuning
]
```

### 3. Quality Monitoring

**Watch for:**
- Cumulative quality degradation
- Loss of desired musical content
- Introduction of artifacts
- Excessive volume reduction

**Quality metrics:**
```python
def assess_quality(original, processed):
    """Simple quality assessment."""
    # RMS level
    original_rms = torch.norm(original)
    processed_rms = torch.norm(processed)
    
    # Spectral centroid (brightness)
    orig_centroid = spectral_centroid(original)
    proc_centroid = spectral_centroid(processed)
    
    return {
        'level_ratio': processed_rms / original_rms,
        'brightness_change': proc_centroid - orig_centroid,
        'overall_quality': min(1.0, processed_rms / (original_rms * 0.8))
    }
```

## Limitations and Considerations

### 1. Computational Cost
- Each iteration requires full model inference
- Processing time scales with iterations
- GPU memory usage per iteration

### 2. Quality Degradation
- Cumulative artifacts with multiple iterations
- Potential loss of desired content
- Non-linear quality degradation

### 3. Semantic Ambiguity
- Model interpretation of descriptions
- Context-dependent results
- Need for precise language

### 4. Audio Dependencies
- Results vary with input material
- Some sounds may not separate cleanly
- Frequency range limitations

## Advanced Techniques

### 1. Parallel Processing
```python
def parallel_subtractive_synthesis(model, audio, description_groups):
    """Process multiple removal paths in parallel."""
    results = []
    
    for group in description_groups:
        current = audio
        for desc in group:
            result = model.separate(current, desc)
            current = result.residual
        results.append(current)
    
    return results

# Compare different filtering approaches
paths = [
    ["highs", "mids"],           # Conservative approach
    ["harsh", "muddy", "bright"], # Aggressive approach
    ["subtle", "gentle"]          # Minimal approach
]

results = parallel_subtractive_synthesis(model, audio, paths)
```

### 2. Adaptive Description Generation
```python
def adaptive_descriptions(model, audio, target_characteristics):
    """Generate descriptions based on current audio state."""
    descriptions = []
    
    for target in target_characteristics:
        # Analyze current audio
        analysis = analyze_audio(audio)
        
        # Generate appropriate description
        if target == "brighter":
            if analysis["highs"] < target_level:
                descriptions.append("dark low frequencies")
        elif target == "cleaner":
            if analysis["noise"] > threshold:
                descriptions.append("noisy artifacts")
    
    return descriptions
```

### 3. Hybrid Approaches
```python
def hybrid_synthesis(model, audio, traditional_filters, semantic_descriptions):
    """Combine traditional EQ with semantic filtering."""
    
    # Apply traditional filters first
    filtered = apply_traditional_filters(audio, traditional_filters)
    
    # Then apply semantic refinement
    result = subtractive_synthesis(model, filtered, semantic_descriptions)
    
    return result
```

## Research Opportunities

This technique opens several research directions:

1. **Semantic Audio Processing**: Natural language as audio control interface
2. **Iterative Neural Filtering**: Multi-pass neural audio processing
3. **Quality-Aware Iteration**: Automatic stopping criteria
4. **Description Optimization**: Learning optimal descriptions
5. **Real-time Applications**: Live performance possibilities

## Examples and Case Studies

### Case Study 1: Sound Design for Film
*Creating creature vocalizations from animal recordings*

```python
# Start with animal recording
creature_base = load_audio("lion_roar.wav")

# Transform into alien creature
alien_creature = subtractive_synthesis(
    model,
    creature_base,
    [
        "natural animal characteristics",
        "familiar mammal sounds", 
        "earthly resonances",
        "organic warmth"
    ]
)
```

### Case Study 2: Musical Production
*Creating vintage vinyl effect*

```python
# Start with clean digital recording
vinyl_effect = subtractive_synthesis(
    model,
    "digital_master.wav",
    [
        "digital clarity and precision",
        "extended high frequencies",
        "clean low end",
        "modern stereo imaging"
    ]
)
```

## Conclusion

Iterative subtractive synthesis with SAM-Audio represents a novel approach to audio processing that bridges the gap between technical precision and semantic understanding. By leveraging the model's ability to understand and separate audio components based on natural language descriptions, this technique opens up new creative possibilities for sound designers, musicians, and audio engineers.

The key advantages are:
- **Intuitive control** through natural language
- **Context-aware processing** that understands musical content
- **Creative workflows** not possible with traditional tools
- **Semantic precision** in describing desired changes

As this technique evolves, it has the potential to transform how we think about audio processing, moving from technical parameters to semantic control and opening up new creative possibilities in audio production and sound design.
