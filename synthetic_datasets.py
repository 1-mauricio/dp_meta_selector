"""Geração de datasets sintéticos para balanceamento do meta-dataset.

Este módulo gera datasets sintéticos que favorecem cada família de mecanismos DP,
garantindo representatividade no treino do meta-modelo.
"""

import logging
from typing import List, Optional

import numpy as np
from sklearn.datasets import make_classification

from .types import Dataset

_log = logging.getLogger(__name__)


def generate_continuous_dataset(
    n_samples: int = 500,
    n_features: int = 20,
    n_classes: int = 3,
    seed: int = 42,
    name_suffix: str = "",
) -> Dataset:
    """Gera dataset contínuo que favorece Laplace/Gaussian.
    
    Características:
    - Features contínuas com alta cardinalidade
    - Distribuição aproximadamente normal
    - Boa separabilidade entre classes
    """
    rng = np.random.RandomState(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        n_clusters_per_class=2,
        flip_y=0.05,
        random_state=seed,
    )
    # Adiciona ruído contínuo para garantir alta cardinalidade
    X = X + rng.normal(0, 0.1, X.shape)
    name = f"synthetic:continuous{name_suffix}"
    return Dataset(X=X.astype(float), y=y.astype(int), name=name)


def generate_discrete_dataset(
    n_samples: int = 500,
    n_features: int = 15,
    n_classes: int = 3,
    max_value: int = 20,
    seed: int = 42,
    name_suffix: str = "",
) -> Dataset:
    """Gera dataset discreto/inteiro que favorece Geometric.
    
    Características:
    - Todas as features são inteiras
    - Range pequeno por coluna (0 a max_value)
    - Simula dados de contagem
    """
    rng = np.random.RandomState(seed)
    
    # Gera features inteiras com distribuições variadas
    X = np.zeros((n_samples, n_features), dtype=float)
    for j in range(n_features):
        dist_type = j % 3
        if dist_type == 0:
            # Poisson-like (contagens)
            lam = rng.uniform(2, max_value // 2)
            X[:, j] = np.clip(rng.poisson(lam, n_samples), 0, max_value)
        elif dist_type == 1:
            # Uniforme discreta
            X[:, j] = rng.randint(0, max_value + 1, n_samples)
        else:
            # Binomial
            n_trials = max_value
            p = rng.uniform(0.2, 0.8)
            X[:, j] = rng.binomial(n_trials, p, n_samples)
    
    # Gera labels correlacionados com algumas features
    score = X[:, :3].sum(axis=1) + rng.normal(0, 1, n_samples)
    y = np.digitize(score, np.percentile(score, np.linspace(0, 100, n_classes + 1)[1:-1]))
    y = np.clip(y, 0, n_classes - 1)
    
    name = f"synthetic:discrete{name_suffix}"
    return Dataset(X=X.astype(float), y=y.astype(int), name=name)


def generate_categorical_dataset(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 4,
    max_categories: int = 5,
    seed: int = 42,
    name_suffix: str = "",
) -> Dataset:
    """Gera dataset categórico que favorece Exponential.
    
    Características:
    - Todas as features têm baixa cardinalidade (≤ max_categories)
    - Simula dados one-hot ou ordinal-encoded
    - Classes relativamente balanceadas
    """
    rng = np.random.RandomState(seed)
    
    X = np.zeros((n_samples, n_features), dtype=float)
    for j in range(n_features):
        n_cats = rng.randint(2, max_categories + 1)
        X[:, j] = rng.randint(0, n_cats, n_samples)
    
    # Labels baseados em combinação de features categóricas
    score = np.zeros(n_samples)
    for j in range(min(3, n_features)):
        score += X[:, j] * (j + 1)
    score += rng.normal(0, 0.5, n_samples)
    y = np.digitize(score, np.percentile(score, np.linspace(0, 100, n_classes + 1)[1:-1]))
    y = np.clip(y, 0, n_classes - 1)
    
    name = f"synthetic:categorical{name_suffix}"
    return Dataset(X=X.astype(float), y=y.astype(int), name=name)


def generate_high_dim_dataset(
    n_samples: int = 300,
    n_features: int = 100,
    n_classes: int = 3,
    seed: int = 42,
    name_suffix: str = "",
) -> Dataset:
    """Gera dataset de alta dimensionalidade que favorece GaussianAnalytic.
    
    Características:
    - Muitas features (≥ 50)
    - Variância distribuída entre muitas componentes (PCA spread alto)
    - Baixa sensibilidade normalizada por feature
    """
    rng = np.random.RandomState(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 3,
        n_redundant=n_features // 3,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.02,
        random_state=seed,
    )
    # Escala features para ter range similar (baixa sensibilidade)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    X = X * 0.5  # Reduz range
    
    name = f"synthetic:high_dim{name_suffix}"
    return Dataset(X=X.astype(float), y=y.astype(int), name=name)


def generate_binary_dataset(
    n_samples: int = 500,
    n_features: int = 20,
    n_classes: int = 2,
    binary_ratio: float = 0.7,
    seed: int = 42,
    name_suffix: str = "",
) -> Dataset:
    """Gera dataset com muitas features binárias.
    
    Características:
    - Maioria das features são binárias (0/1)
    - Algumas features contínuas
    - Útil para testar robustez em dados mistos
    """
    rng = np.random.RandomState(seed)
    n_binary = int(n_features * binary_ratio)
    n_continuous = n_features - n_binary
    
    # Features binárias
    X_bin = rng.randint(0, 2, (n_samples, n_binary)).astype(float)
    
    # Features contínuas
    X_cont = rng.randn(n_samples, n_continuous)
    
    X = np.hstack([X_bin, X_cont])
    
    # Labels
    score = X_bin[:, :3].sum(axis=1) + X_cont[:, 0] if n_continuous > 0 else X_bin[:, :3].sum(axis=1)
    score += rng.normal(0, 0.5, n_samples)
    y = (score > np.median(score)).astype(int)
    
    name = f"synthetic:binary{name_suffix}"
    return Dataset(X=X.astype(float), y=y.astype(int), name=name)


def generate_mixed_dataset(
    n_samples: int = 500,
    n_features: int = 20,
    n_classes: int = 3,
    seed: int = 42,
    name_suffix: str = "",
) -> Dataset:
    """Gera dataset misto com features de diferentes tipos.
    
    Características:
    - Mix de features contínuas, discretas e binárias
    - Útil para testar generalização do meta-modelo
    """
    rng = np.random.RandomState(seed)
    n_cont = n_features // 3
    n_disc = n_features // 3
    n_bin = n_features - n_cont - n_disc
    
    X_cont = rng.randn(n_samples, n_cont) * 2
    X_disc = rng.randint(0, 10, (n_samples, n_disc)).astype(float)
    X_bin = rng.randint(0, 2, (n_samples, n_bin)).astype(float)
    
    X = np.hstack([X_cont, X_disc, X_bin])
    
    # Shuffle columns
    perm = rng.permutation(n_features)
    X = X[:, perm]
    
    # Labels baseados em todas as partes
    score = X_cont.mean(axis=1) + X_disc.mean(axis=1) + X_bin.sum(axis=1)
    score += rng.normal(0, 1, n_samples)
    y = np.digitize(score, np.percentile(score, np.linspace(0, 100, n_classes + 1)[1:-1]))
    y = np.clip(y, 0, n_classes - 1)
    
    name = f"synthetic:mixed{name_suffix}"
    return Dataset(X=X.astype(float), y=y.astype(int), name=name)


def generate_synthetic_training_datasets(
    n_per_type: int = 10,
    base_seed: int = 42,
) -> List[Dataset]:
    """Gera conjunto completo de datasets sintéticos para treino.
    
    Parameters
    ----------
    n_per_type:
        Número de datasets por tipo/família.
    base_seed:
        Seed base para reprodutibilidade.
    
    Returns
    -------
    Lista de datasets sintéticos cobrindo todas as famílias.
    """
    datasets: List[Dataset] = []
    
    # Configurações variadas para cada tipo
    configs = [
        # (n_samples, n_features, n_classes)
        (300, 10, 2),
        (500, 15, 3),
        (800, 20, 4),
        (400, 25, 3),
        (600, 12, 2),
    ]
    
    _log.info("[synthetic] Gerando %d datasets por tipo (5 tipos)...", n_per_type)
    
    for i in range(n_per_type):
        seed = base_seed + i * 100
        cfg = configs[i % len(configs)]
        n_samples, n_features, n_classes = cfg
        suffix = f"_{i}"
        
        # Contínuos (Laplace, Gaussian)
        datasets.append(generate_continuous_dataset(
            n_samples=n_samples, n_features=n_features,
            n_classes=n_classes, seed=seed, name_suffix=suffix
        ))
        
        # Discretos (Geometric)
        datasets.append(generate_discrete_dataset(
            n_samples=n_samples, n_features=n_features,
            n_classes=n_classes, seed=seed + 1, name_suffix=suffix
        ))
        
        # Categóricos (Exponential)
        datasets.append(generate_categorical_dataset(
            n_samples=n_samples, n_features=min(n_features, 12),
            n_classes=n_classes, seed=seed + 2, name_suffix=suffix
        ))
        
        # Alta dimensionalidade (GaussianAnalytic)
        datasets.append(generate_high_dim_dataset(
            n_samples=min(n_samples, 400), n_features=max(n_features * 3, 60),
            n_classes=n_classes, seed=seed + 3, name_suffix=suffix
        ))
        
        # Mistos (teste de robustez)
        if i < n_per_type // 2:
            datasets.append(generate_mixed_dataset(
                n_samples=n_samples, n_features=n_features,
                n_classes=n_classes, seed=seed + 4, name_suffix=suffix
            ))
    
    _log.info("[synthetic] Gerados %d datasets sintéticos.", len(datasets))
    return datasets


def augment_training_datasets(
    real_datasets: List[Dataset],
    synthetic_ratio: float = 0.3,
    min_synthetic: int = 20,
    seed: int = 42,
) -> List[Dataset]:
    """Aumenta datasets reais com sintéticos para melhor cobertura.
    
    Parameters
    ----------
    real_datasets:
        Lista de datasets reais (ex: OpenML).
    synthetic_ratio:
        Proporção de sintéticos em relação aos reais.
    min_synthetic:
        Número mínimo de sintéticos a gerar.
    seed:
        Seed para reprodutibilidade.
    
    Returns
    -------
    Lista combinada de datasets reais + sintéticos.
    """
    n_synthetic = max(min_synthetic, int(len(real_datasets) * synthetic_ratio))
    n_per_type = max(2, n_synthetic // 5)
    
    synthetic = generate_synthetic_training_datasets(n_per_type=n_per_type, base_seed=seed)
    
    combined = list(real_datasets) + synthetic
    _log.info(
        "[augment] Combinados: %d reais + %d sintéticos = %d total",
        len(real_datasets), len(synthetic), len(combined)
    )
    return combined
