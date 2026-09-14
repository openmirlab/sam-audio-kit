"""
Model download and warmup utilities for SAM-Audio.

This module provides standalone functions to:
- Download model files to a configurable cache directory, pinned to the
  commit `checkpoints.toml` records for that model (never a floating "main")
  and sha256-verified against that same catalog entry after download
- Warmup the model with a dummy inference to cache CUDA kernels

Reads: .checkpoints (checkpoint_info, ChecksumMismatchError); read by: .model
(SamAudio.from_pretrained), .session (cache_info), sam_audio.model.base
(BaseModel._from_pretrained's live-load-time digest verification, gated on
a subclass naming a `catalog_key` class attribute)

Environment Variables:
    SAM_AUDIO_CACHE_DIR: Directory to cache downloaded models
                         Default: ~/.cache/sam-audio-kit
    HF_TOKEN: HuggingFace API token for gated models
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Optional, Union

import torch

from .checkpoints import checkpoint_info, ChecksumMismatchError
from .types import (
    DeviceType,
    DType,
    ModelSize,
    get_model_name,
    get_model_size,
    MODEL_NAME_MAP,
)


# Environment variable names
ENV_CACHE_DIR = "SAM_AUDIO_CACHE_DIR"
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HF_TOKEN_ALT = "HUGGINGFACE_TOKEN"

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sam-audio-kit"


def get_cache_dir() -> Path:
    """
    Get the model cache directory.

    Priority:
    1. SAM_AUDIO_CACHE_DIR environment variable
    2. Default: ~/.cache/sam-audio-kit

    Returns:
        Path to the cache directory
    """
    cache_dir = os.getenv(ENV_CACHE_DIR)
    if cache_dir:
        return Path(cache_dir)
    return DEFAULT_CACHE_DIR


def resolve_model_cache_path(
    model_size: Union[str, ModelSize] = "base",
    cache_dir: Optional[Union[str, Path]] = None,
    revision: Optional[str] = None,
) -> Path:
    """Return Hugging Face's local repository path without creating it.

    With `revision` given, appends `snapshots/<revision>` so the result names
    that specific pinned snapshot rather than just the repo's cache root --
    lets a caller tell whether the *pinned* revision is cached, not merely
    whether some (possibly stale) snapshot is.
    """
    root = Path(cache_dir) if cache_dir is not None else get_cache_dir()
    model_name = get_model_name(model_size)
    path = root / f"models--{model_name.replace('/', '--')}"
    if revision is not None:
        path = path / "snapshots" / revision
    return path


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(model_key: str, snapshot_dir: Union[str, Path], verbose: bool = True) -> None:
    """Verify the downloaded artifact named by `[models.<model_key>]` in checkpoints.toml.

    Skips verification (with a `verbose`-gated warning) when the catalog entry
    marks its integrity as unavailable; raises `ChecksumMismatchError` on a
    digest mismatch. No-op (silent) for models absent from the registry
    (e.g. a user-supplied HuggingFace ID or local path).
    """
    info = checkpoint_info(model_key)
    artifact = info.get("artifact")
    expected_sha256 = info.get("sha256")
    if artifact is None or (expected_sha256 is None and info.get("integrity") != "unavailable"):
        # Not a registry-tracked model (or no digest recorded at all) -- nothing to verify.
        return
    if info.get("integrity") == "unavailable":
        if verbose:
            print(f"  Warning: no verified checksum for {model_key!r} artifact "
                  f"{artifact!r} (integrity=unavailable) -- skipping verification")
        return
    artifact_path = Path(snapshot_dir) / artifact
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"expected artifact {artifact!r} not found in downloaded snapshot {snapshot_dir}"
        )
    if verbose:
        print(f"  Verifying checksum of {artifact}...")
    actual_sha256 = _sha256_file(artifact_path)
    if actual_sha256 != expected_sha256:
        raise ChecksumMismatchError(model_key, artifact, expected_sha256, actual_sha256)
    if verbose:
        print(f"  Checksum OK ({actual_sha256})")


def get_hf_token() -> Optional[str]:
    """
    Get HuggingFace token from environment.

    Returns:
        HuggingFace token or None
    """
    return os.getenv(ENV_HF_TOKEN) or os.getenv(ENV_HF_TOKEN_ALT)


def download_model(
    model_size: Union[str, ModelSize] = "base",
    cache_dir: Optional[Union[str, Path]] = None,
    hf_token: Optional[str] = None,
    include_judge: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download SAM-Audio model files to cache.

    This downloads the model weights without loading them into memory,
    useful for pre-caching models before deployment.

    Args:
        model_size: Model size ("small", "base", "large") or HuggingFace model ID
        cache_dir: Directory to cache models (default: from env or ~/.cache/sam-audio-kit)
        hf_token: HuggingFace API token (default: from HF_TOKEN env var)
        include_judge: Also download the sam-audio-judge model for quality assessment
        verbose: Print download progress

    Returns:
        Path to the cache directory

    Example:
        >>> from sam_audio_kit import download_model
        >>> # Download to default cache
        >>> download_model("base")

        >>> # Download to custom directory
        >>> download_model("base", cache_dir="/models/sam-audio")

        >>> # Using environment variable
        >>> import os
        >>> os.environ["SAM_AUDIO_CACHE_DIR"] = "/models/sam-audio"
        >>> download_model("base")
    """
    from huggingface_hub import snapshot_download

    # Resolve cache directory
    if cache_dir is None:
        cache_dir = get_cache_dir()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve HF token
    if hf_token is None:
        hf_token = get_hf_token()

    # Resolve model name
    model_name = get_model_name(model_size)
    info = checkpoint_info(model_size)
    revision = info.get("source_revision")

    if verbose:
        print(f"Downloading {model_name} to {cache_dir}"
              + (f" (revision={revision})" if revision else ""))

    # Download main model, pinned to its catalog revision (never floats on "main")
    start_time = time.time()
    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=str(cache_dir),
        token=hf_token,
        revision=revision,
    )

    if verbose:
        elapsed = time.time() - start_time
        print(f"  Downloaded {model_name} in {elapsed:.1f}s")

    _verify_artifact(model_size, model_path, verbose=verbose)

    # Optionally download judge model
    if include_judge:
        judge_info = checkpoint_info("judge")
        judge_revision = judge_info.get("source_revision")
        if verbose:
            print("Downloading facebook/sam-audio-judge..."
                  + (f" (revision={judge_revision})" if judge_revision else ""))
        judge_path = snapshot_download(
            repo_id=judge_info.get("model_id", "facebook/sam-audio-judge"),
            cache_dir=str(cache_dir),
            token=hf_token,
            revision=judge_revision,
        )
        _verify_artifact("judge", judge_path, verbose=verbose)
        if verbose:
            print("  Downloaded sam-audio-judge")

    return cache_dir


def warmup_model(
    model: "SamAudio",  # type: ignore  # noqa: F821
    duration_seconds: float = 1.0,
    verbose: bool = True,
) -> float:
    """
    Warmup the model with a dummy inference.

    This runs a short audio through the model to:
    - Compile CUDA kernels (PyTorch JIT)
    - Cache memory allocations
    - Initialize any lazy-loaded components

    After warmup, subsequent inferences will be faster.

    Args:
        model: Loaded SamAudio instance
        duration_seconds: Duration of dummy audio to process
        verbose: Print warmup progress

    Returns:
        Warmup time in seconds

    Example:
        >>> model = SamAudio.from_pretrained("base")
        >>> warmup_time = warmup_model(model)
        >>> print(f"Warmup completed in {warmup_time:.2f}s")
    """
    if verbose:
        print("Warming up model...")

    # Get sample rate from model
    sample_rate = model.sample_rate

    # Create dummy audio (silence + small noise to avoid edge cases)
    num_samples = int(duration_seconds * sample_rate)
    dummy_audio = torch.randn(1, num_samples) * 0.001

    # Move to model device
    dummy_audio = dummy_audio.to(model.device)

    # Run warmup inference
    start_time = time.time()
    try:
        with torch.inference_mode():
            _ = model.separate(
                dummy_audio,
                description="warmup",
                verbose=False,
            )
    except Exception as e:
        if verbose:
            print(f"  Warmup warning: {e}")
        # Warmup might fail but that's okay - the CUDA kernels are still cached

    warmup_time = time.time() - start_time

    # Cleanup
    del dummy_audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"  Warmup completed in {warmup_time:.2f}s")

    return warmup_time


def download_and_warmup(
    model_size: Union[str, ModelSize] = "base",
    lite_mode: bool = True,
    device: DeviceType = "cuda",
    dtype: DType = "bfloat16",
    cache_dir: Optional[Union[str, Path]] = None,
    hf_token: Optional[str] = None,
    warmup_duration: float = 1.0,
    verbose: bool = True,
) -> "SamAudio":  # type: ignore  # noqa: F821
    """
    Download model, load it, and run warmup inference.

    This is the recommended way to prepare a model for production use.
    It ensures all files are cached and CUDA kernels are compiled.

    Args:
        model_size: Model size ("small", "base", "large")
        lite_mode: Enable lite mode for reduced VRAM
        device: Device to run on ("cuda", "cpu", "mps")
        dtype: Data type ("float32", "float16", "bfloat16")
        cache_dir: Directory to cache models
        hf_token: HuggingFace API token
        warmup_duration: Duration of warmup audio in seconds
        verbose: Print progress

    Returns:
        Loaded and warmed-up SamAudio instance

    Example:
        >>> # Prepare model for production
        >>> model = download_and_warmup("base", verbose=True)
        >>> # Model is now ready for fast inference
        >>> result = model.separate("audio.wav", "vocals")
    """
    from .model import SamAudio

    # Resolve cache directory
    if cache_dir is None:
        cache_dir = get_cache_dir()

    # Download model files first
    download_model(
        model_size=model_size,
        cache_dir=cache_dir,
        hf_token=hf_token,
        verbose=verbose,
    )

    # Load model
    if verbose:
        print("Loading model...")

    model = SamAudio.from_pretrained(
        model_size,
        lite_mode=lite_mode,
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        hf_token=hf_token,
        verbose=verbose,
    )

    # Run warmup
    warmup_model(model, duration_seconds=warmup_duration, verbose=verbose)

    if verbose:
        print("Model ready for inference!")

    return model


def list_cached_models(cache_dir: Optional[Union[str, Path]] = None) -> list[dict]:
    """
    List models that are cached locally.

    Args:
        cache_dir: Directory to check (default: from env or ~/.cache/sam-audio-kit)

    Returns:
        List of dicts with model info
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return []

    models = []
    # Check for HuggingFace hub cache structure
    for model_dir in cache_dir.glob("models--*"):
        model_id = model_dir.name.replace("models--", "").replace("--", "/")
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    # Calculate size
                    size_bytes = sum(f.stat().st_size for f in snapshot.rglob("*") if f.is_file())
                    size_gb = size_bytes / (1024**3)
                    models.append({
                        "model_id": model_id,
                        "snapshot": snapshot.name[:8],
                        "size_gb": round(size_gb, 2),
                        "path": str(snapshot),
                    })

    return models


def clear_cache(
    cache_dir: Optional[Union[str, Path]] = None,
    model_size: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Clear cached model files.

    Args:
        cache_dir: Directory to clear (default: from env or ~/.cache/sam-audio-kit)
        model_size: Only clear specific model size (None = clear all)
        verbose: Print progress
    """
    import shutil

    if cache_dir is None:
        cache_dir = get_cache_dir()
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        if verbose:
            print(f"Cache directory does not exist: {cache_dir}")
        return

    if model_size:
        # Clear specific model
        model_name = get_model_name(model_size)
        model_dir_name = f"models--{model_name.replace('/', '--')}"
        model_dir = cache_dir / model_dir_name
        if model_dir.exists():
            if verbose:
                print(f"Removing {model_name}...")
            shutil.rmtree(model_dir)
        else:
            if verbose:
                print(f"Model not cached: {model_name}")
    else:
        # Clear all
        if verbose:
            print(f"Clearing cache: {cache_dir}")
        for item in cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                if verbose:
                    print(f"  Removing {item.name}...")
                shutil.rmtree(item)


# Type hint import for runtime
if False:  # TYPE_CHECKING equivalent that works at runtime
    from .model import SamAudio
