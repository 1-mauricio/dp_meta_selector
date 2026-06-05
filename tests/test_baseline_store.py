"""Testes para BaselineStore — persistência SQLite de baselines."""

import threading
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dp_meta_selector.baseline_store import BaselineStore
from dp_meta_selector.utility import META_FAST_PROFILE


@pytest.fixture
def store(tmp_path):
    return BaselineStore(db_path=tmp_path / "test_baselines.sqlite", enabled=True)


_FP = "abc123fingerprint"


def test_set_and_get(store):
    store.set("iris", "meta_logreg", _FP, META_FAST_PROFILE, 0.92)
    result = store.get("iris", "meta_logreg", _FP)
    assert result is not None
    assert abs(result - 0.92) < 1e-9


def test_has(store):
    assert not store.has("iris", "meta_logreg", _FP)
    store.set("iris", "meta_logreg", _FP, META_FAST_PROFILE, 0.92)
    assert store.has("iris", "meta_logreg", _FP)


def test_get_missing_returns_none(store):
    assert store.get("nonexistent", "algo", _FP) is None


def test_hits_and_misses(store):
    store.set("iris", "meta_logreg", _FP, META_FAST_PROFILE, 0.92)
    store.get("iris", "meta_logreg", _FP)   # hit
    store.get("nonexistent", "algo", _FP)   # miss
    summary = store.summary()
    assert summary  # deve retornar algo não-vazio


def test_disabled_store():
    with tempfile.TemporaryDirectory() as td:
        s = BaselineStore(db_path=Path(td) / "b.sqlite", enabled=False)
        s.set("x", "y", _FP, META_FAST_PROFILE, 0.5)  # deve ser no-op
        assert s.get("x", "y", _FP) is None


def test_thread_safety(store):
    """Múltiplas threads escrevendo entradas distintas sem corrupção."""
    errors = []

    def writer(i):
        try:
            fp = f"fp_{i}"
            store.set(f"ds_{i}", "meta_logreg", fp, META_FAST_PROFILE, float(i) / 100)
            result = store.get(f"ds_{i}", "meta_logreg", fp)
            assert result is not None
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Erros em threads: {errors}"
