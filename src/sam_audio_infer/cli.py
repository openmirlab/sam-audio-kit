"""Command-line interface for sam-audio-infer."""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="SAM-Audio Inference - Optimized audio separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic separation
  sam-audio-infer input.wav --description "vocals" --output vocals.wav

  # Lite mode with bfloat16 (recommended)
  sam-audio-infer input.wav -d "drums" -o drums.wav --lite --dtype bfloat16

  # Full quality mode
  sam-audio-infer input.wav -d "piano" -o piano.wav --no-lite --dtype float32

  # Process long audio with chunking
  sam-audio-infer long_song.wav -d "vocals" -o vocals.wav --chunk-duration 30
        """,
    )

    parser.add_argument("input", type=str, help="Input audio file path")
    parser.add_argument(
        "-d", "--description",
        type=str,
        required=True,
        help="Description of audio to extract (e.g., 'vocals', 'drums')",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path for extracted audio",
    )
    parser.add_argument(
        "--residual",
        type=str,
        default=None,
        help="Output file path for residual audio (optional)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["small", "base", "large"],
        help="Model size (default: base)",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        default=True,
        help="Enable lite mode for reduced VRAM (default: enabled)",
    )
    parser.add_argument(
        "--no-lite",
        action="store_true",
        help="Disable lite mode",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for inference (default: bfloat16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=25.0,
        help="Chunk duration in seconds for long audio (default: 25)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token for gated models",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="sam-audio-infer 0.1.0",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Handle lite mode flag
    lite_mode = args.lite and not args.no_lite

    # Import here to avoid slow startup
    from .model import SamAudioInfer

    try:
        # Load model
        if args.verbose:
            print(f"Loading SAM-Audio ({args.model})...")

        model = SamAudioInfer.from_pretrained(
            args.model,
            lite_mode=lite_mode,
            device=args.device,
            dtype=args.dtype,
            chunk_duration=args.chunk_duration,
            hf_token=args.hf_token,
            verbose=args.verbose,
        )

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


if __name__ == "__main__":
    main()
