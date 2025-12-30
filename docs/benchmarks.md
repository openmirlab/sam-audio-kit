# Benchmarks

## VRAM Usage by Configuration

Tested on RTX 4090 (24 GB VRAM).

### Base Model

| Configuration | bfloat16 | float16 | float32 |
|--------------|----------|---------|---------|
| **aggressive** (lite) | **2.84 GB** | 2.91 GB | 5.21 GB |
| with_text_ranker | 6.19 GB | 6.19 GB | 11.94 GB |
| with_span_predictor | 5.96 GB | 5.96 GB | 11.29 GB |
| with_all_features | 9.31 GB | 9.31 GB | OOM |
| **no_lite** (full) | 12.73 GB | 12.73 GB | OOM |

### Large Model

| Configuration | bfloat16 |
|--------------|----------|
| **aggressive** (lite) | **6.15 GB** |
| with_text_ranker | 9.57 GB |
| with_span_predictor | 9.28 GB |
| with_all_features | 12.62 GB |
| **no_lite** (full) | 16.18 GB |

## Key Findings

- **bfloat16 vs float16**: Nearly identical VRAM usage
- **bfloat16 vs float32**: ~2x reduction
- **Lite mode savings**: ~62-78% reduction
- **Recommended**: `aggressive` + `bfloat16` for most use cases

## VRAM Summary by Model Size

| Model | Full (bfloat16) | Lite (bfloat16) | Reduction |
|-------|-----------------|-----------------|-----------|
| Base | 12.73 GB | **2.84 GB** | **78%** |
| Large | 16.18 GB | **6.15 GB** | **62%** |

## GPU Compatibility

| GPU | VRAM | Small | Base | Large |
|-----|------|-------|------|-------|
| RTX 3060 | 6 GB | OK | Tight | No |
| RTX 3070 | 8 GB | OK | OK | No |
| RTX 3080 | 10 GB | OK | OK | Tight |
| RTX 4070 | 12 GB | OK | OK | OK |
| RTX 4090 | 24 GB | OK | OK | OK |

- **OK**: Comfortable headroom
- **Tight**: Use smaller chunks
- **No**: Not recommended
