"""Stable hashes for the numerical implementation used by benchmark runs."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_IMPLEMENTATION_FILES = (
    "dpmeta/classifiers.py",
    "dpmeta/external_confirmation.py",
    "dpmeta/families.py",
    "dpmeta/mechanisms.py",
    "dpmeta/meta_features.py",
    "dpmeta/model.py",
    "dpmeta/openml_benchmark.py",
    "dpmeta/portfolio_benchmark.py",
    "dpmeta/privacy.py",
    "dpmeta/provenance.py",
    "dpmeta/schema.py",
    "dpmeta/selector.py",
)
RUNTIME_DISTRIBUTIONS = ("numpy", "pandas", "scipy", "scikit-learn", "openml", "threadpoolctl")


def numerical_implementation_sha256() -> str:
    """Hash numerical/evaluation code, excluding figures and report rendering."""
    digest = hashlib.sha256()
    for relative_path in NUMERICAL_IMPLEMENTATION_FILES:
        path = PROJECT_ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in RUNTIME_DISTRIBUTIONS},
    }


def save_source_snapshot(output_dir: Path, *, extra_files: tuple[str, ...] = ()) -> dict:
    """Save the exact source and runtime versions used by one new analysis."""
    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    files = tuple(dict.fromkeys((*NUMERICAL_IMPLEMENTATION_FILES, *extra_files,
                                 "requirements.txt", "pyproject.toml")))
    hashes = {}
    for name in files:
        source = PROJECT_ROOT / name
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = file_sha256(target)
    payload = {"files_sha256": hashes, **runtime_fingerprint()}
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload
