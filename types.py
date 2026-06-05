"""Tipos compartilhados do framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class Dataset:
    """
    Wrapper para um dataset tabular de classificação.

    Suporta unpacking posicional (``X, y, name = ds``) para retrocompatibilidade
    com o código existente que itera sobre listas de DatasetTuple.
    """

    X: np.ndarray
    y: np.ndarray
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.X, np.ndarray):
            raise TypeError(f"X deve ser np.ndarray, não {type(self.X).__name__}")
        if not isinstance(self.y, np.ndarray):
            raise TypeError(f"y deve ser np.ndarray, não {type(self.y).__name__}")
        if self.X.ndim != 2:
            raise ValueError(f"X deve ter 2 dimensões, tem {self.X.ndim}")
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X e y têm tamanhos diferentes: {len(self.X)} vs {len(self.y)}"
            )
        if len(self.X) == 0:
            raise ValueError("Dataset vazio (X sem linhas)")
        if not self.name:
            raise ValueError("name não pode ser vazio")

    # ── retrocompatibilidade: ``X, y, name = dataset`` ────────────────────────
    def __iter__(self) -> Iterator:
        yield self.X
        yield self.y
        yield self.name

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int):
        return (self.X, self.y, self.name)[idx]

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y)))

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, "
            f"n_samples={self.n_samples}, n_features={self.n_features}, "
            f"n_classes={self.n_classes})"
        )


# Retrocompatibilidade: código antigo que usa DatasetTuple continua funcionando
DatasetTuple = Tuple[np.ndarray, np.ndarray, str]
