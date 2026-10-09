"""Serialize cold imports reached while deserializing serving artifacts."""

from functools import wraps
import logging
from threading import RLock


_ARTIFACT_LOAD_LOCK = RLock()


class ArtifactLoadError(RuntimeError):
    """An artifact could not be read, including the original failure detail."""


def serialized_artifact_load(loader):
    """Avoid concurrent sklearn imports through different joblib class paths."""
    @wraps(loader)
    def load(*args, **kwargs):
        with _ARTIFACT_LOAD_LOCK:
            try:
                return loader(*args, **kwargs)
            except Exception as exc:
                logging.getLogger(__name__).exception("Artifact loader %s failed", loader.__module__)
                raise ArtifactLoadError(f"{type(exc).__name__}: {exc}") from exc
    return load
