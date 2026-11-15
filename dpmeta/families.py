"""Conservative grouping for known related OpenML dataset families."""

from .external_confirmation import confirmation_family
from .openml_benchmark import canonical_family


def _related_family(name: str) -> str:
    """Group aliases for repeated or derived catalog datasets."""
    normalized = canonical_family(name)
    aliases = {
        "australian": "australian_credit_approval",
        "credit_approval": "australian_credit_approval",
        "credit": "credit_data_variants",
        "credit_g": "credit_data_variants",
        "credit_g_copy": "credit_data_variants",
        "dataset_credit_g": "credit_data_variants",
        "boston_corrected": "boston",
        "credit_g_reproduced_1": "credit_data_variants",
        "credit_approval_reproduced": "australian_credit_approval",
    }
    if normalized.startswith("fri_c"):
        return "friedman_synthetic"
    for prefix, family in (
        ("monks_problems_", "monks_problems"),
        ("analcatdata_apnea", "analcatdata_apnea"),
        ("chscase_census", "chscase_census"),
        ("jedit_", "jedit"),
        ("rmftsa_", "rmftsa"),
        ("arsenic_female_", "arsenic_female"),
        ("arsenic_male_", "arsenic_male"),
        ("accidents_prediction", "accidents_prediction"),
        ("pc1_", "nasa_software_defects"),
        ("pc3_", "nasa_software_defects"),
        ("pc4_", "nasa_software_defects"),
        ("kc1_", "nasa_software_defects"),
        ("kc2_", "nasa_software_defects"),
        ("bwin_", "bwin"),
        ("doa_bwin", "bwin"),
    ):
        if normalized.startswith(prefix):
            return family
    if normalized in {"heart_c", "heart_h", "heart_statlog", "cleve", "heart_disease"}:
        return "heart_disease"
    return aliases.get(normalized, normalized)


def conservative_family(name: str) -> str:
    """Unify aliases documented by the benchmark runners."""
    canonical = canonical_family(name)
    confirmed = confirmation_family(name)
    related = _related_family(name)
    return confirmed if confirmed != canonical else related
