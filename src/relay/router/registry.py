"""P3 model registry: versioned LightGBM models with an active pointer.

The registry is a small JSON file (``models/registry.json``) that tracks every
trained model version plus which one is currently serving. It exists so the
self-learning loop can deploy a newly trained model **without restarting the
server** (hot reload) and **roll back** to a previous version with a one-line
pointer flip.

Layout on disk::

    models/
    ├── registry.json            # {"schema_version":1,"active":"v...","versions":[...]}
    ├── p3_lightgbm.txt          # legacy default model (pre-registry)
    └── versions/
        ├── v20260720-181500.txt
        └── v20260720-181500.meta.json

Each version entry records the training/eval metrics so the gate and roll-back
decisions can compare models without re-reading the LightGBM artifacts.

This module is pure stdlib (no LightGBM import) so it can be used by both the
online runtime (``ml_head``) and the offline training/self-learn scripts.
Writes are atomic (tmp file + os.replace) so a crash mid-write never leaves a
half-written registry.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("relay.router.registry")

_SCHEMA_VERSION = 1
_REGISTRY_FILENAME = "registry.json"
_VERSIONS_DIRNAME = "versions"


@dataclass
class ModelVersion:
    """One trained model artifact + its metrics."""

    version: str               # e.g. "v20260720-181500"
    model_path: str            # absolute or repo-relative path to the .txt model
    meta_path: str = ""
    trained_on: list[str] = field(default_factory=list)
    num_samples: int = 0
    tier_distribution: dict = field(default_factory=dict)
    val_accuracy: float = 0.0
    holdout_accuracy: float = 0.0
    trained_ts_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model_path": self.model_path,
            "meta_path": self.meta_path,
            "trained_on": self.trained_on,
            "num_samples": self.num_samples,
            "tier_distribution": self.tier_distribution,
            "val_accuracy": self.val_accuracy,
            "holdout_accuracy": self.holdout_accuracy,
            "trained_ts_ms": self.trained_ts_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelVersion":
        return cls(
            version=str(d.get("version", "")),
            model_path=str(d.get("model_path", "")),
            meta_path=str(d.get("meta_path", "")),
            trained_on=list(d.get("trained_on") or []),
            num_samples=int(d.get("num_samples") or 0),
            tier_distribution=dict(d.get("tier_distribution") or {}),
            val_accuracy=float(d.get("val_accuracy") or 0.0),
            holdout_accuracy=float(d.get("holdout_accuracy") or 0.0),
            trained_ts_ms=int(d.get("trained_ts_ms") or 0),
        )


class ModelRegistry:
    """JSON-backed registry of model versions with an active pointer."""

    def __init__(self, registry_path: str) -> None:
        self._path = registry_path
        self._versions: dict[str, ModelVersion] = {}
        self._active: str | None = None
        self._load()

    # ── persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("registry: failed to load %s (%s); starting empty", self._path, exc)
            return
        self._active = data.get("active")
        for vd in data.get("versions") or []:
            v = ModelVersion.from_dict(vd)
            if v.version:
                self._versions[v.version] = v

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "active": self._active,
            "versions": [self._versions[k].to_dict() for k in self._versions],
        }
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._path)  # atomic on POSIX + Windows (same volume)

    # ── queries ────────────────────────────────────────────────────────

    @property
    def active(self) -> ModelVersion | None:
        if self._active and self._active in self._versions:
            return self._versions[self._active]
        return None

    def get(self, version: str) -> ModelVersion | None:
        return self._versions.get(version)

    def list(self) -> list[ModelVersion]:
        """All versions, newest first by trained_ts_ms."""
        return sorted(self._versions.values(), key=lambda v: v.trained_ts_ms, reverse=True)

    def to_dict(self) -> dict:
        return {
            "schema_version": _SCHEMA_VERSION,
            "active": self._active,
            "versions": [v.to_dict() for v in self.list()],
        }

    # ── mutations ──────────────────────────────────────────────────────

    def add(self, version: ModelVersion, activate: bool = False) -> None:
        """Register a new version. Does not activate unless ``activate=True``."""
        self._versions[version.version] = version
        if activate:
            self._active = version.version
        self._save()

    def activate(self, version: str) -> bool:
        """Point the active pointer at an existing version (deploy / rollback)."""
        if version not in self._versions:
            logger.warning("registry: cannot activate unknown version %s", version)
            return False
        self._active = version
        self._save()
        return True


def registry_path_for(models_dir: str) -> str:
    """Canonical registry path for a models directory."""
    return os.path.join(models_dir, _REGISTRY_FILENAME)


def versions_dir_for(models_dir: str) -> str:
    return os.path.join(models_dir, _VERSIONS_DIRNAME)


def new_version_id(ts_ms: int | None = None) -> str:
    """Generate a sortable, human-readable version id like v20260720-181500."""
    ms = ts_ms if ts_ms is not None else int(time.time() * 1000)
    lt = time.localtime(ms / 1000)
    return time.strftime("v%Y%m%d-%H%M%S", lt)


def load_registry(models_dir: str) -> ModelRegistry:
    """Open (or create-in-memory) the registry for a models directory."""
    return ModelRegistry(registry_path_for(models_dir))


def active_model_path(models_dir: str) -> str | None:
    """Return the active model's path, or None if the registry has no active model.

    This is the hot-reload read path used by ``ml_head``: cheap (one small JSON
    read) and safe to call on every routing decision.
    """
    reg = load_registry(models_dir)
    active = reg.active
    if active is None:
        return None
    return active.model_path or None
