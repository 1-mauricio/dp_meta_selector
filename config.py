"""Configurações globais centralizadas do dp_meta_selector."""

from pathlib import Path

# ── Versão ────────────────────────────────────────────────────────────────────
FRAMEWORK_VERSION = "5"

# ── Cache / persistência ──────────────────────────────────────────────────────
DEFAULT_CACHE_DIR: Path = Path(".dp_meta_cache")
DEFAULT_MODEL_PATH: Path = Path("dp_meta_selector.joblib")

# ── Privacidade diferencial ───────────────────────────────────────────────────
DELTA_DEFAULT: float = 1e-5
TARGET_NOISE_RATIO: float = 0.30   # σ_ruído / σ_sinal alvo

# ── Datasets ──────────────────────────────────────────────────────────────────
OPENML_TRAINING_TARGET: int = 500   # número-alvo de datasets de treino
OPENML_CC18_SUITE_ID: int = 99
MAX_ROWS_PER_DATASET: int = 3000    # subsampling máximo por dataset

# ── Avaliação de utilidade ────────────────────────────────────────────────────
BASELINE_SCHEMA_VERSION: str = "1"
FINGERPRINT_SAMPLE_SIZE: int = 512  # linhas amostradas no fingerprint

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT: str = "[%(levelname)s] %(name)s — %(message)s"
