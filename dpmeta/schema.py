"""Public-schema feature encoding. All data-dependent scaling is forbidden."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p
import json

import numpy as np
import pandas as pd

MISSING = "__PUBLIC_MISSING__"
PUBLIC_FEATURE_NAMES = (
    "log_n", "log_original_features", "log_encoded_features", "log_classes",
    "log_nominal_features", "log_numeric_features", "log_nominal_levels",
    "log_max_nominal_cardinality", "log_mean_nominal_cardinality",
    "numeric_fraction", "nominal_fraction", "binary_nominal_fraction",
    "log_classes_per_encoded_feature", "log_epsilon", "log_n_epsilon_per_feature",
    "numeric_bounds_fraction", "log_mean_numeric_bound_width",
)


@dataclass(frozen=True)
class PublicSchema:
    feature_names: tuple[str, ...]
    feature_kinds: tuple[str, ...]
    nominal_domains: tuple[tuple[str, ...], ...]
    classes: tuple[str, ...]
    numeric_bounds: tuple[tuple[float, float] | None, ...] | None = None

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        if n == 0 or len(self.feature_kinds) != n or len(self.nominal_domains) != n:
            raise ValueError("schema de features inconsistente")
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("o schema público requer ao menos duas classes distintas")
        if any(kind not in {"numeric", "nominal"} for kind in self.feature_kinds):
            raise ValueError("apenas features numéricas e nominais são suportadas")
        if self.numeric_bounds is not None and len(self.numeric_bounds) != n:
            raise ValueError("limites numéricos precisam alinhar com as features")
        for kind, domain in zip(self.feature_kinds, self.nominal_domains):
            if kind == "nominal" and (not domain or len(set(domain)) != len(domain)):
                raise ValueError("domínio nominal público precisa ser completo e sem duplicatas")
            if kind == "numeric" and domain:
                raise ValueError("feature numérica não pode declarar domínio nominal")
        if self.numeric_bounds is not None:
            for kind, bounds in zip(self.feature_kinds, self.numeric_bounds):
                if kind == "nominal" and bounds is not None:
                    raise ValueError("feature nominal não pode declarar limites numéricos")
                if kind == "numeric":
                    if bounds is None or len(bounds) != 2:
                        raise ValueError("cada feature numérica precisa de limites públicos [min, max]")
                    lower, upper = map(float, bounds)
                    if (not np.isfinite(lower) or not np.isfinite(upper)
                            or not np.isfinite(upper - lower) or lower >= upper):
                        raise ValueError("limites numéricos precisam ser finitos e crescentes")

    @property
    def encoded_dim(self) -> int:
        return sum(1 if kind == "numeric" else len(domain) + 1
                   for kind, domain in zip(self.feature_kinds, self.nominal_domains))

    @property
    def nominal_count(self) -> int:
        return sum(kind == "nominal" for kind in self.feature_kinds)

    @classmethod
    def from_manifest_json(cls, raw: str | dict) -> "PublicSchema":
        data = json.loads(raw) if isinstance(raw, str) else raw
        return cls(
            feature_names=tuple(data["feature_names"]),
            feature_kinds=tuple(data["feature_kinds"]),
            nominal_domains=tuple(tuple(domain) for domain in data["nominal_domains"]),
            classes=tuple(data["classes"]),
            numeric_bounds=(
                tuple(None if bounds is None else tuple(map(float, bounds))
                      for bounds in data["numeric_bounds"])
                if data.get("numeric_bounds") is not None else None
            ),
        )

    def encode(self, frame, *, numeric_transform: str = "tanh") -> np.ndarray:
        if numeric_transform not in {"tanh", "asinh10", "public_bounds"}:
            raise ValueError("numeric_transform deve ser 'tanh', 'asinh10' ou 'public_bounds'")
        if numeric_transform == "public_bounds" and self.numeric_bounds is None:
            raise ValueError("public_bounds requer limites explícitos no schema público")
        if list(frame.columns) != list(self.feature_names):
            raise ValueError("nomes/ordem das colunas diferem do schema público")
        columns: list[np.ndarray] = []
        bounds_by_name = self.numeric_bounds or (None,) * len(self.feature_names)
        for name, kind, domain, bounds in zip(
                self.feature_names, self.feature_kinds, self.nominal_domains, bounds_by_name):
            values = frame[name]
            if kind == "numeric":
                numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                invalid = ~np.isfinite(numeric)
                numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
                if numeric_transform == "tanh":
                    encoded_numeric = np.tanh(numeric)
                elif numeric_transform == "asinh10":
                    # Fixed, data-independent compression with a public [-1, 1] bound.
                    # Unlike tanh(x), asinh(x)/10 retains more ordering for moderate
                    # magnitudes while still bounding arbitrarily large inputs.
                    encoded_numeric = np.clip(np.arcsinh(numeric) / 10.0, -1.0, 1.0)
                else:
                    lower, upper = bounds
                    encoded_numeric = np.clip(
                        2.0 * (numeric - lower) / (upper - lower) - 1.0, -1.0, 1.0,
                    )
                encoded_numeric[invalid] = 0.0
                columns.append(encoded_numeric.reshape(-1, 1))
                continue
            exact = {value: index for index, value in enumerate(domain)}
            folded = {}
            for index, value in enumerate(domain):
                folded.setdefault(value.casefold(), []).append(index)
            missing_index = len(domain)
            indices = []
            for value in values.to_numpy(dtype=object):
                if pd.isna(value):
                    indices.append(missing_index)
                    continue
                key = str(value)
                if key in exact:
                    indices.append(exact[key])
                elif len(folded.get(key.casefold(), [])) == 1:
                    indices.append(folded[key.casefold()][0])
                else:
                    raise ValueError(f"valor nominal fora do schema público em {name}: {key!r}")
            one_hot = np.zeros((len(indices), len(domain) + 1), dtype=float)
            one_hot[np.arange(len(indices)), np.asarray(indices)] = 1.0
            columns.append(one_hot)
        encoded = np.concatenate(columns, axis=1)
        if encoded.size and (not np.isfinite(encoded).all() or encoded.min() < -1 or encoded.max() > 1):
            raise AssertionError("encoding violou o domínio público [-1,1]")
        return encoded

    def class_indices(self, values) -> np.ndarray:
        mapping = {value: i for i, value in enumerate(self.classes)}
        folded = {}
        for index, value in enumerate(self.classes):
            folded.setdefault(value.casefold(), []).append(index)
        indices = []
        for value in values:
            key = str(value)
            if key in mapping:
                indices.append(mapping[key])
            elif len(folded.get(key.casefold(), [])) == 1:
                indices.append(folded[key.casefold()][0])
            else:
                raise ValueError(f"rótulo fora do conjunto público de classes: {key!r}")
        return np.asarray(indices, dtype=int)


def public_features(n: int, schema: PublicSchema, epsilon: float) -> np.ndarray:
    """Only public schema descriptors and epsilon; deliberately no data values."""
    if n < 1 or epsilon <= 0:
        raise ValueError("n e epsilon precisam ser positivos")
    p = schema.encoded_dim
    nominal_domains = [len(domain) for kind, domain in
                       zip(schema.feature_kinds, schema.nominal_domains) if kind == "nominal"]
    nominal_count = len(nominal_domains)
    numeric_count = len(schema.feature_names) - nominal_count
    if schema.numeric_bounds is None:
        bounded_numeric = []
    else:
        bounded_numeric = [upper - lower
                           for kind, bounds in zip(schema.feature_kinds, schema.numeric_bounds)
                           if kind == "numeric" and bounds is not None
                           for lower, upper in [bounds]]
    bounds_fraction = len(bounded_numeric) / numeric_count if numeric_count else 0.0
    mean_log_bound_width = (float(np.mean(np.log1p(bounded_numeric)))
                            if bounded_numeric else 0.0)
    maximum_nominal_cardinality = max(nominal_domains, default=0)
    mean_nominal_cardinality = float(np.mean(nominal_domains)) if nominal_domains else 0.0
    return np.asarray([
        log1p(n),
        log1p(len(schema.feature_names)),
        log1p(p),
        log1p(len(schema.classes)),
        log1p(nominal_count),
        log1p(numeric_count),
        log1p(sum(nominal_domains)),
        log1p(maximum_nominal_cardinality),
        log1p(mean_nominal_cardinality),
        numeric_count / len(schema.feature_names),
        nominal_count / len(schema.feature_names),
        (sum(cardinality <= 2 for cardinality in nominal_domains) / nominal_count
         if nominal_count else 0.0),
        log1p(len(schema.classes) / p),
        log1p(epsilon),
        log1p(n * epsilon / p),
        bounds_fraction,
        mean_log_bound_width,
    ], dtype=float)
