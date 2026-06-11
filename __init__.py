"""
dp_meta_selector — seleção automática de mecanismos DP via meta-aprendizagem.

Dependências:
  pip install diffprivlib scikit-learn pandas numpy scipy tqdm openml joblib
"""

import warnings

# EN4: suprimir apenas warnings conhecidos e inócuos — não silenciar tudo
warnings.filterwarnings(
    "ignore",
    message=".*ConvergenceWarning.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*n_splits.*",  # StratifiedKFold com poucas amostras
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*UndefinedMetricWarning.*",
    category=UserWarning,
)
# diffprivlib emite FutureWarning sobre parâmetros; suprime para não poluir saída acadêmica
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="diffprivlib",
)
# sklearn ConvergenceWarning em LogisticRegression com poucos dados
try:
    from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    # CalibratedClassifierCV cv='prefit' FutureWarning em sklearn >= 1.6
    warnings.filterwarnings(
        "ignore",
        message=".*cv='prefit'.*",
        category=FutureWarning,
    )
except ImportError:
    pass

from .applicator import DPApplicator
from .baseline_store import (
    BASELINE_SCHEMA_VERSION,
    DEFAULT_BASELINE_DB,
    DEFAULT_BASELINE_REGISTRY,
    BaselineEntry,
    BaselineRegistry,
    BaselineStore,
    precompute_baselines,
)
from .calibration import (
    DELTA_DEFAULT,
    FAMILY_EPSILON,
    TARGET_NOISE_RATIO,
    calibrate_epsilon,
)
from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_PATH,
    FRAMEWORK_VERSION,
    MAX_ROWS_PER_DATASET,
    OPENML_TRAINING_TARGET,
)
from .types import Dataset, DatasetTuple
from .datasets import (
    OPENML_TRAINING_SPECS,
    OPENML_TRAINING_SPECS_CORE,
    build_openml_training_specs,
    load_openml_dataset,
    load_openml_training_datasets,
    split_meta_datasets,
)
from .diagnostics import (
    ablation_study,
    compute_calibration_data,
    compute_confusion_matrix,
    compute_family_f1_scores,
    dataset_level_kfold_cv,
    print_calibration_report,
    print_confusion_matrix,
    print_family_f1_report,
    run_full_diagnostics,
)
from .evaluator import FrameworkEvaluator
from .main import cli, main, run_precompute_baselines
from .mechanisms import (
    DP_MECHANISMS,
    DPMechanism,
    FAMILY_OF,
    MECHANISM_NAMES,
    SCREENING_MECHANISMS,
)
from .meta_dataset import MetaDatasetBuilder
from .meta_features import MetaFeatureExtractor
from .meta_learner import MetaLearner
from .selector import DPMechanismSelector
from .utility import (
    DPUtilityEvaluator,
    EVAL_FAST_PROFILE,
    EVAL_FULL_PROFILE,
    META_ALIGNED_PROFILE,
    META_FAST_PROFILE,
    UtilityProfile,
    UtilityResultCache,
)

__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "DEFAULT_BASELINE_DB",
    "DEFAULT_BASELINE_REGISTRY",
    "BaselineEntry",
    "BaselineRegistry",
    "BaselineStore",
    "Dataset",
    "DatasetTuple",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MODEL_PATH",
    "DELTA_DEFAULT",
    "DPApplicator",
    "DPMechanism",
    "DPMechanismSelector",
    "DP_MECHANISMS",
    "DPUtilityEvaluator",
    "EVAL_FAST_PROFILE",
    "EVAL_FULL_PROFILE",
    "FAMILY_EPSILON",
    "FAMILY_OF",
    "FRAMEWORK_VERSION",
    "FrameworkEvaluator",
    "MAX_ROWS_PER_DATASET",
    "MECHANISM_NAMES",
    "META_ALIGNED_PROFILE",
    "META_FAST_PROFILE",
    "MetaDatasetBuilder",
    "MetaFeatureExtractor",
    "MetaLearner",
    "OPENML_TRAINING_SPECS",
    "OPENML_TRAINING_SPECS_CORE",
    "OPENML_TRAINING_TARGET",
    "SCREENING_MECHANISMS",
    "TARGET_NOISE_RATIO",
    "UtilityProfile",
    "UtilityResultCache",
    "ablation_study",
    "build_openml_training_specs",
    "calibrate_epsilon",
    "cli",
    "compute_calibration_data",
    "compute_confusion_matrix",
    "compute_family_f1_scores",
    "dataset_level_kfold_cv",
    "precompute_baselines",
    "print_calibration_report",
    "print_confusion_matrix",
    "print_family_f1_report",
    "run_full_diagnostics",
    "run_precompute_baselines",
    "load_openml_dataset",
    "load_openml_training_datasets",
    "main",
    "split_meta_datasets",
]

__version__ = "5.0.0"
