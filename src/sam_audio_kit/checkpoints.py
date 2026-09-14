"""Read the package-owned, license-aware checkpoint registry.

Each registry entry pins `source_revision` (a commit sha, not a floating
branch) plus the primary artifact's `sha256`/`size_bytes` for post-download
verification (see `.download`'s `_verify_artifact`); a model that genuinely
has no available digest must say so explicitly (`integrity = "unavailable"`)
rather than omit the field silently.
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import copy
try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


class ChecksumMismatchError(Exception):
    """A downloaded checkpoint artifact's sha256 does not match the catalog entry."""

    def __init__(self, model: str, artifact: str, expected: str, actual: str):
        self.model, self.artifact = model, artifact
        self.expected, self.actual = expected, actual
        super().__init__(
            f"checksum mismatch for {model!r} artifact {artifact!r}: "
            f"expected sha256={expected}, got sha256={actual}"
        )


def load_checkpoint_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load checkpoint metadata without downloading or authenticating anything."""
    if path is None:
        resource = files("sam_audio_kit").joinpath("config/checkpoints.toml")
        return tomllib.loads(resource.read_text(encoding="utf-8"))
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def checkpoint_info(model: str, *, overrides: Mapping[str, Any] | None = None,
                    config_path: str | Path | None = None) -> dict[str, Any]:
    """Return one model's metadata with explicit caller overrides."""
    config = load_checkpoint_config(config_path)
    try:
        info = copy.deepcopy(config["models"][model])
    except KeyError:
        # Full Hugging Face IDs and local paths are valid explicit models even
        # when they are not part of the package's small/base/large registry.
        info = {"model_id": model, "url": model if model.startswith("https://") else "",
                "license": "user-supplied", "provenance": "user-supplied"}
    if overrides:
        info.update(dict(overrides))
    return info
