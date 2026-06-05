"""
Armazenamento incremental de baselines (acurácia sem DP).

Cada entrada é identificada por (dataset_id, baseline_id, schema_version).
Novos datasets ou algoritmos acrescentam linhas sem invalidar as existentes.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .calibration import DELTA_DEFAULT
from .config import BASELINE_SCHEMA_VERSION, DEFAULT_CACHE_DIR
from .types import DatasetTuple
from .utility import (
    EVAL_FAST_PROFILE,
    EVAL_FULL_PROFILE,
    META_FAST_PROFILE,
    DPUtilityEvaluator,
    UtilityProfile,
    _data_fingerprint,
)

_log = logging.getLogger(__name__)

DEFAULT_BASELINE_DB = DEFAULT_CACHE_DIR / "baselines.sqlite"


@dataclass(frozen=True)
class BaselineEntry:
    dataset_id: str
    baseline_id: str
    fingerprint: str
    profile_key: str
    accuracy: float
    schema_version: str
    computed_at: str


class BaselineRegistry:
    """Catálogo de algoritmos de baseline (baseline_id → UtilityProfile)."""

    def __init__(self, profiles: Optional[Dict[str, UtilityProfile]] = None):
        self._profiles: Dict[str, UtilityProfile] = dict(profiles or {})

    def copy(self) -> BaselineRegistry:
        return BaselineRegistry(dict(self._profiles))

    def register(self, baseline_id: str, profile: UtilityProfile) -> None:
        if baseline_id in self._profiles:
            raise ValueError(f"baseline_id já registrado: {baseline_id}")
        self._profiles[baseline_id] = profile

    def get(self, baseline_id: str) -> UtilityProfile:
        if baseline_id not in self._profiles:
            raise KeyError(
                f"baseline_id desconhecido: {baseline_id}. "
                f"Disponíveis: {sorted(self._profiles)}"
            )
        return self._profiles[baseline_id]

    def ids(self) -> List[str]:
        return sorted(self._profiles)

    def items(self):
        return self._profiles.items()

    def resolve_id(self, profile: UtilityProfile) -> str:
        key = profile.cache_key()
        for bid, p in self._profiles.items():
            if p.cache_key() == key:
                return bid
        return f"custom_{profile.name}"


DEFAULT_BASELINE_REGISTRY = BaselineRegistry({
    "meta_logreg": META_FAST_PROFILE,
    "eval_rf_fast": EVAL_FAST_PROFILE,
    "eval_rf_full": EVAL_FULL_PROFILE,
})


class BaselineStore:
    """SQLite append-only por (dataset_id, baseline_id, schema_version).

    Q4: usa thread-local connections para reutilizar a conexão por thread,
    evitando o custo de abrir/fechar a cada operação.
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_BASELINE_DB,
        enabled: bool = True,
        schema_version: str = BASELINE_SCHEMA_VERSION,
    ):
        self.enabled = enabled
        self.db_path = Path(db_path)
        self.schema_version = schema_version
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._local = threading.local()  # Q4: thread-local state
        if enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Q4: retorna/cria conexão reutilizável por thread."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS baselines (
                dataset_id TEXT NOT NULL,
                baseline_id TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                profile_key TEXT NOT NULL,
                accuracy REAL NOT NULL,
                computed_at TEXT NOT NULL,
                PRIMARY KEY (dataset_id, baseline_id, schema_version)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_baselines_baseline_id "
            "ON baselines (baseline_id)"
        )
        conn.commit()

    def _inc_hits(self) -> None:
        """Thread-safe increment of the hit counter."""
        with self._lock:
            self.hits += 1

    def _inc_misses(self) -> None:
        """Thread-safe increment of the miss counter."""
        with self._lock:
            self.misses += 1

    def get(
        self,
        dataset_id: str,
        baseline_id: str,
        fingerprint: Optional[str] = None,
    ) -> Optional[float]:
        if not self.enabled:
            return None
        with self._lock:
            row = self._connect().execute(
                """
                SELECT fingerprint, accuracy FROM baselines
                WHERE dataset_id = ? AND baseline_id = ? AND schema_version = ?
                """,
                (dataset_id, baseline_id, self.schema_version),
            ).fetchone()
            if row is None:
                return None
            if fingerprint is not None and row["fingerprint"] != fingerprint:
                return None
            self.hits += 1
            return float(row["accuracy"])

    def set(
        self,
        dataset_id: str,
        baseline_id: str,
        fingerprint: str,
        profile: UtilityProfile,
        accuracy: float,
    ) -> None:
        if not self.enabled:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            conn.execute(
                """
                INSERT INTO baselines (
                    dataset_id, baseline_id, schema_version,
                    fingerprint, profile_key, accuracy, computed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id, baseline_id, schema_version) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    profile_key = excluded.profile_key,
                    accuracy = excluded.accuracy,
                    computed_at = excluded.computed_at
                """,
                (
                    dataset_id,
                    baseline_id,
                    self.schema_version,
                    fingerprint,
                    profile.cache_key(),
                    float(accuracy),
                    now,
                ),
            )
            conn.commit()

    def has(self, dataset_id: str, baseline_id: str, fingerprint: str) -> bool:
        return self.get(dataset_id, baseline_id, fingerprint) is not None

    def list_entries(
        self,
        baseline_id: Optional[str] = None,
    ) -> List[BaselineEntry]:
        if not self.enabled or not self.db_path.is_file():
            return []
        query = "SELECT * FROM baselines WHERE schema_version = ?"
        params: List = [self.schema_version]
        if baseline_id is not None:
            query += " AND baseline_id = ?"
            params.append(baseline_id)
        with self._lock:
            rows = self._connect().execute(query, params).fetchall()
        return [
            BaselineEntry(
                dataset_id=r["dataset_id"],
                baseline_id=r["baseline_id"],
                fingerprint=r["fingerprint"],
                profile_key=r["profile_key"],
                accuracy=float(r["accuracy"]),
                schema_version=r["schema_version"],
                computed_at=r["computed_at"],
            )
            for r in rows
        ]

    def to_dataframe(self, baseline_id: Optional[str] = None) -> pd.DataFrame:
        entries = self.list_entries(baseline_id=baseline_id)
        if not entries:
            return pd.DataFrame(
                columns=[
                    "dataset_id",
                    "baseline_id",
                    "fingerprint",
                    "profile_key",
                    "accuracy",
                    "schema_version",
                    "computed_at",
                ]
            )
        return pd.DataFrame([e.__dict__ for e in entries])

    def export_table(self, path: Path) -> Path:
        """Exporta a tabela para Parquet (se pyarrow existir) ou CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            out = path.with_suffix(".csv")
            df.to_csv(out, index=False)
            return out

    def summary(self) -> str:
        if not self.enabled:
            return "baseline_store=off"
        n = len(self.list_entries())
        total = self.hits + self.misses
        hit_pct = f" ({100*self.hits/total:.0f}% hits)" if total else ""
        return f"baseline_store entries={n} hits={self.hits} misses={self.misses}{hit_pct}"


def _compute_one(
    X: np.ndarray,
    y: np.ndarray,
    dataset_id: str,
    baseline_id: str,
    profile: UtilityProfile,
    store: BaselineStore,
    delta: float,
) -> Tuple[str, str, float, bool]:
    """Retorna (dataset_id, baseline_id, accuracy, was_cached)."""
    y_enc = LabelEncoder().fit_transform(y)
    fp = _data_fingerprint(X, y_enc)
    cached = store.get(dataset_id, baseline_id, fp)
    if cached is not None:
        return dataset_id, baseline_id, cached, True

    store._inc_misses()
    ev = DPUtilityEvaluator(delta=delta, profile=profile, cache=None)
    score = ev._cv_score(X, y_enc, profile)
    store.set(dataset_id, baseline_id, fp, profile, score)
    return dataset_id, baseline_id, score, False


def precompute_baselines(
    datasets: Sequence[DatasetTuple],
    baseline_ids: Optional[Sequence[str]] = None,
    registry: BaselineRegistry = DEFAULT_BASELINE_REGISTRY,
    store: Optional[BaselineStore] = None,
    delta: float = DELTA_DEFAULT,
    parallel: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Calcula e persiste baselines ausentes para cada (dataset, baseline_id).

    Entradas já presentes com fingerprint compatível são ignoradas.
    """
    store = store or BaselineStore()
    ids = list(baseline_ids or registry.ids())

    tasks: List[Tuple] = []
    for X, y, name in datasets:
        for bid in ids:
            profile = registry.get(bid)
            y_enc = LabelEncoder().fit_transform(y)
            fp = _data_fingerprint(X, y_enc)
            if store.has(name, bid, fp):
                continue
            tasks.append((X, y, name, bid, profile))

    if not tasks:
        _log.info("[baselines] Nada a calcular (%d algoritmo(s), store cheio).", len(ids))
        return store.to_dataframe()

    _log.info(
        "[baselines] Calculando %d par(es) (%d datasets × %d algoritmos)...",
        len(tasks), len(datasets), len(ids),
    )

    def _run(task):
        X, y, name, bid, profile = task
        return _compute_one(X, y, name, bid, profile, store, delta)

    if parallel and len(tasks) > 1:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run)(t) for t in tqdm(tasks, desc="Baselines")
        )
    else:
        results = [_run(t) for t in tqdm(tasks, desc="Baselines")]

    computed = sum(1 for *_, cached in results if not cached)
    _log.info("[baselines] Novos: %d | %s", computed, store.summary())
    return store.to_dataframe()
