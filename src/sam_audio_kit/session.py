"""Independent lifecycle session for SAM-Audio inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .checkpoints import checkpoint_info
from .types import AudioInput, DeviceType, DType, resolve_device


class SamAudioSession:
    """Own one loaded SAM-Audio runtime; disk cache is shared, live models are not."""

    def __init__(self, model: str = "base", *, device: DeviceType = "cuda",
                 dtype: DType = "bfloat16", chunk_duration: float = 25.0,
                 cache_dir: str | Path | None = None, hf_token: str | None = None,
                 verbose: bool = False, checkpoint_overrides: Mapping[str, Any] | None = None,
                 config_path: str | Path | None = None, **load_kwargs: Any):
        self.model = model
        self.device, self.dtype = device, dtype
        self.chunk_duration = chunk_duration
        self.cache_dir, self.hf_token = cache_dir, hf_token
        self.verbose, self.load_kwargs = verbose, load_kwargs
        self._checkpoint = checkpoint_info(model, overrides=checkpoint_overrides, config_path=config_path)
        self._runtime = None
        self._state = "created"

    @property
    def status(self) -> str:
        return self._state

    @property
    def checkpoint(self) -> dict[str, Any]:
        return dict(self._checkpoint)

    def load(self) -> "SamAudioSession":
        if self._state == "closed":
            raise RuntimeError("session is closed")
        if self._state == "ready":
            return self
        self._state = "loading"
        try:
            from .model import SamAudio

            self.device = resolve_device(self.device)
            self._runtime = SamAudio.from_pretrained(
                self._checkpoint.get("model_id", self.model), dtype=self.dtype,
                device=self.device, chunk_duration=self.chunk_duration,
                cache_dir=self.cache_dir, hf_token=self.hf_token,
                verbose=self.verbose, **self.load_kwargs)
            self._state = "ready"
            return self
        except Exception:
            self._state = "failed"
            raise

    def infer(self, audio: AudioInput, description: str, **kwargs: Any):
        if self._state != "ready" or self._runtime is None:
            raise RuntimeError("call load() before infer()")
        return self._runtime.separate(audio, description, **kwargs)

    def release(self) -> "SamAudioSession":
        if self._runtime is not None:
            self._runtime.unload()
            self._runtime = None
        if self._state != "closed":
            self._state = "released"
        return self

    def close(self) -> "SamAudioSession":
        if self._state != "closed":
            self.release()
            self._state = "closed"
        return self

    def cache_info(self) -> dict[str, Any]:
        from .download import get_cache_dir, list_cached_models, resolve_model_cache_path

        model_id = self._checkpoint.get("model_id", self.model)
        root = Path(self.cache_dir) if self.cache_dir is not None else get_cache_dir()
        path = resolve_model_cache_path(model_id, root)
        return {"path": str(path), "exists": path.exists(), "model": model_id,
                "checkpoints": list_cached_models(root), "status": self.status,
                "loaded": self._runtime is not None}

    def __enter__(self) -> "SamAudioSession":
        return self.load()

    def __exit__(self, *_: Any) -> None:
        self.close()
