"""Command-line interface for sam-audio-infer."""

import argparse
import sys
from pathlib import Path


def cmd_separate(args):
    """Handle the separate command."""
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Handle lite mode flag
    lite_mode = args.lite and not args.no_lite

    # Import here to avoid slow startup
    from .model import SamAudioInfer
    from .precision import PrecisionConfig

    # Build precision config from explicit flags
    allow_tf32 = True  # default
    if args.no_tf32:
        allow_tf32 = False
    elif args.tf32:
        allow_tf32 = True

    cudnn_benchmark = True  # default
    if args.no_cudnn_benchmark:
        cudnn_benchmark = False
    elif args.cudnn_benchmark:
        cudnn_benchmark = True

    precision_config = PrecisionConfig(
        matmul_precision=args.matmul_precision,
        allow_tf32=allow_tf32,
        cudnn_allow_tf32=allow_tf32,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_deterministic=args.deterministic,
    )

    try:
        # Load model
        if args.verbose:
            print(f"Loading SAM-Audio ({args.model})...")

        model = SamAudioInfer.from_pretrained(
            args.model,
            lite_mode=lite_mode,
            device=args.device,
            dtype=args.dtype,
            precision_config=precision_config,
            chunk_duration=args.chunk_duration,
            hf_token=args.hf_token,
            cache_dir=args.cache_dir,
            verbose=args.verbose,
        )

        # Run warmup if requested
        if args.warmup:
            from .download import warmup_model
            warmup_model(model, verbose=args.verbose)

        # Run separation
        if args.verbose:
            print(f"Separating: '{args.description}'...")

        result = model.separate(
            args.input,
            description=args.description,
            verbose=args.verbose,
        )

        # Save results
        result.save(args.output, args.residual)

        if args.verbose:
            print(f"Saved: {args.output}")
            if args.residual:
                print(f"Saved: {args.residual}")
            print(f"Processing time: {result.processing_time:.1f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_download(args):
    """Handle the download command."""
    from .download import download_model, download_and_warmup, get_cache_dir

    cache_dir = args.cache_dir or get_cache_dir()

    try:
        if args.warmup:
            # Download and warmup
            model = download_and_warmup(
                model_size=args.model,
                lite_mode=not args.no_lite,
                device=args.device,
                dtype=args.dtype,
                cache_dir=cache_dir,
                hf_token=args.hf_token,
                warmup_duration=args.warmup_duration,
                verbose=True,
            )
            # Unload model after warmup
            model.unload()
        else:
            # Just download
            download_model(
                model_size=args.model,
                cache_dir=cache_dir,
                hf_token=args.hf_token,
                include_judge=args.include_judge,
                verbose=True,
            )

        print(f"\nModel cached at: {cache_dir}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_list(args):
    """Handle the list command."""
    from .download import list_cached_models, get_cache_dir

    cache_dir = args.cache_dir or get_cache_dir()
    models = list_cached_models(cache_dir)

    if not models:
        print(f"No models cached in: {cache_dir}")
        return

    print(f"Cached models in: {cache_dir}\n")
    for m in models:
        print(f"  {m['model_id']}")
        print(f"    Snapshot: {m['snapshot']}")
        print(f"    Size: {m['size_gb']:.2f} GB")
        print()


def cmd_clear(args):
    """Handle the clear command."""
    from .download import clear_cache, get_cache_dir

    cache_dir = args.cache_dir or get_cache_dir()

    if not args.yes:
        if args.model:
            prompt = f"Clear {args.model} from cache? [y/N] "
        else:
            prompt = f"Clear all models from {cache_dir}? [y/N] "
        response = input(prompt).strip().lower()
        if response not in ("y", "yes"):
            print("Cancelled.")
            return

    clear_cache(cache_dir=cache_dir, model_size=args.model, verbose=True)
    print("Cache cleared.")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="SAM-Audio Inference - Optimized audio separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="sam-audio-infer 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ============ SEPARATE COMMAND ============
    sep_parser = subparsers.add_parser(
        "separate",
        help="Separate audio based on text description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic separation
  sam-audio-infer separate input.wav -d "vocals" -o vocals.wav

  # With residual output
  sam-audio-infer separate input.wav -d "drums" -o drums.wav --residual other.wav

  # Full quality mode
  sam-audio-infer separate input.wav -d "piano" -o piano.wav --no-lite --dtype float32
        """,
    )
    sep_parser.add_argument("input", type=str, help="Input audio file path")
    sep_parser.add_argument(
        "-d", "--description",
        type=str,
        required=True,
        help="Description of audio to extract (e.g., 'vocals', 'drums')",
    )
    sep_parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path for extracted audio",
    )
    sep_parser.add_argument(
        "--residual",
        type=str,
        default=None,
        help="Output file path for residual audio (optional)",
    )
    sep_parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["small", "base", "large"],
        help="Model size (default: base)",
    )
    sep_parser.add_argument(
        "--lite",
        action="store_true",
        default=True,
        help="Enable lite mode for reduced VRAM (default: enabled)",
    )
    sep_parser.add_argument(
        "--no-lite",
        action="store_true",
        help="Disable lite mode",
    )
    sep_parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for inference (default: bfloat16)",
    )
    sep_parser.add_argument(
        "--tf32",
        action="store_true",
        default=None,
        help="Enable TF32 for ~3x speedup on Ampere+ GPUs (default: enabled)",
    )
    sep_parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 for maximum precision",
    )
    sep_parser.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
        help="Matrix multiplication precision: highest (slowest), high (balanced), medium (fastest)",
    )
    sep_parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        default=None,
        help="Enable cuDNN auto-tuner for faster inference (default: enabled)",
    )
    sep_parser.add_argument(
        "--no-cudnn-benchmark",
        action="store_true",
        help="Disable cuDNN auto-tuner",
    )
    sep_parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic operations for reproducible results (slower)",
    )
    sep_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run on (default: cuda)",
    )
    sep_parser.add_argument(
        "--chunk-duration",
        type=float,
        default=25.0,
        help="Chunk duration in seconds for long audio (default: 25)",
    )
    sep_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached models (default: $SAM_AUDIO_CACHE_DIR or ~/.cache/sam-audio-infer)",
    )
    sep_parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token for gated models",
    )
    sep_parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run warmup inference before processing",
    )
    sep_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    # ============ DOWNLOAD COMMAND ============
    dl_parser = subparsers.add_parser(
        "download",
        help="Download and cache model files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download base model
  sam-audio-infer download --model base

  # Download and warmup for production
  sam-audio-infer download --model base --warmup

  # Download to custom directory
  sam-audio-infer download --model base --cache-dir /models/sam-audio

  # Using environment variable
  export SAM_AUDIO_CACHE_DIR=/models/sam-audio
  sam-audio-infer download --model base
        """,
    )
    dl_parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["small", "base", "large"],
        help="Model size to download (default: base)",
    )
    dl_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache models (default: $SAM_AUDIO_CACHE_DIR or ~/.cache/sam-audio-infer)",
    )
    dl_parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token",
    )
    dl_parser.add_argument(
        "--warmup",
        action="store_true",
        help="Also load model and run warmup inference",
    )
    dl_parser.add_argument(
        "--warmup-duration",
        type=float,
        default=1.0,
        help="Duration of warmup audio in seconds (default: 1.0)",
    )
    dl_parser.add_argument(
        "--no-lite",
        action="store_true",
        help="Disable lite mode during warmup",
    )
    dl_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for warmup (default: cuda)",
    )
    dl_parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for warmup (default: bfloat16)",
    )
    dl_parser.add_argument(
        "--include-judge",
        action="store_true",
        help="Also download the sam-audio-judge model",
    )
    dl_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output (default: enabled)",
    )

    # ============ LIST COMMAND ============
    list_parser = subparsers.add_parser(
        "list",
        help="List cached models",
    )
    list_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to check (default: $SAM_AUDIO_CACHE_DIR or ~/.cache/sam-audio-infer)",
    )

    # ============ CLEAR COMMAND ============
    clear_parser = subparsers.add_parser(
        "clear",
        help="Clear cached models",
    )
    clear_parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["small", "base", "large"],
        help="Specific model to clear (default: all)",
    )
    clear_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to clear (default: $SAM_AUDIO_CACHE_DIR or ~/.cache/sam-audio-infer)",
    )
    clear_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # Parse args
    args = parser.parse_args()

    # Handle commands
    if args.command == "separate":
        cmd_separate(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "clear":
        cmd_clear(args)
    else:
        # No command - show help
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
