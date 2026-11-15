"""Replace-one privacy sensitivities and sequential privacy ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import log, sqrt


@dataclass(frozen=True)
class Sensitivity:
    l1: float
    l2: float
    linf: float
    max_changed_coordinates: int | None = None


def sufficient_statistic_sensitivities(p: int) -> dict[str, Sensitivity]:
    """Global sensitivities for joint class/feature vectors under replace-one.

    Each row contributes to one public class block. Feature coordinates are
    bounded to [-1, 1], labels are included in adjacency, and all classes are
    released together. ``p`` is the public encoded feature count.
    """
    if p < 1:
        raise ValueError("p precisa ser positivo")
    return {
        "counts": Sensitivity(2.0, sqrt(2.0), 1.0, 2),
        "sums": Sensitivity(2.0 * p, 2.0 * sqrt(p), 2.0, 2 * p),
        "seconds": Sensitivity(2.0 * p, sqrt(2.0 * p), 1.0, 2 * p),
    }


@dataclass(frozen=True)
class Release:
    name: str
    mechanism: str
    epsilon: float
    delta: float
    sensitivity: Sensitivity
    noise_scale: float
    rho: float = 0.0


@dataclass
class PrivacyLedger:
    epsilon_limit: float
    delta_limit: float
    accounting: str = "basic"
    releases: list[Release] = field(default_factory=list)

    def record(self, release: Release) -> None:
        if self.accounting == "basic":
            if release.epsilon <= 0 or release.delta < 0 or release.rho != 0:
                raise ValueError("liberação básica requer epsilon > 0, delta >= 0 e rho=0")
        elif self.accounting == "zcdp":
            if (self.delta_limit <= 0 or release.epsilon != 0 or release.delta != 0
                    or release.rho <= 0):
                raise ValueError("liberação zCDP requer rho > 0 e delta total positivo")
        else:
            raise ValueError(f"accounting desconhecido: {self.accounting}")
        self.releases.append(release)
        self.verify()

    @property
    def epsilon_used(self) -> float:
        if self.accounting == "zcdp":
            rho = self.rho_used
            if rho == 0:
                return 0.0
            return rho + 2.0 * sqrt(rho * log(1.0 / self.delta_limit))
        return sum(item.epsilon for item in self.releases)

    @property
    def delta_used(self) -> float:
        if self.accounting == "zcdp":
            return self.delta_limit if self.rho_used > 0 else 0.0
        return sum(item.delta for item in self.releases)

    @property
    def rho_used(self) -> float:
        return sum(item.rho for item in self.releases)

    def verify(self, tolerance: float = 1e-10) -> "PrivacyLedger":
        if self.accounting == "zcdp":
            if self.delta_limit <= 0 or self.delta_limit >= 1:
                raise ValueError("zCDP requer delta total em (0,1)")
        if self.epsilon_used > self.epsilon_limit + tolerance:
            raise ValueError(f"composição excede epsilon: {self.epsilon_used} > {self.epsilon_limit}")
        if self.delta_used > self.delta_limit + tolerance:
            raise ValueError(f"composição excede delta: {self.delta_used} > {self.delta_limit}")
        return self

    def as_dict(self) -> dict:
        return {
            "accounting": self.accounting,
            "epsilon_limit": self.epsilon_limit,
            "epsilon_used": self.epsilon_used,
            "delta_limit": self.delta_limit,
            "delta_used": self.delta_used,
            "rho_used": self.rho_used,
            "releases": [
                {**asdict(release), "sensitivity": asdict(release.sensitivity)}
                for release in self.releases
            ],
        }


@dataclass
class ParallelPrivacyLedger:
    """Basic parallel composition for releases on disjoint record partitions.

    This ledger is valid only when every record is assigned to exactly one
    branch by a partition independent of record values. Branches may themselves
    have already-composed ledgers; the caller passes their final (epsilon, delta).
    """

    epsilon_limit: float
    delta_limit: float
    releases: list[Release] = field(default_factory=list)

    def record(self, release: Release) -> None:
        if (release.epsilon <= 0 or release.delta < 0 or release.delta >= 1
                or release.rho != 0):
            raise ValueError("parallel branch precisa de epsilon > 0, delta em [0,1) e rho=0")
        self.releases.append(release)
        self.verify()

    @property
    def epsilon_used(self) -> float:
        return max((item.epsilon for item in self.releases), default=0.0)

    @property
    def delta_used(self) -> float:
        return max((item.delta for item in self.releases), default=0.0)

    def verify(self, tolerance: float = 1e-10) -> "ParallelPrivacyLedger":
        if (self.epsilon_limit <= 0 or not 0 <= self.delta_limit < 1
                or not self.releases):
            raise ValueError("limites positivos e ao menos um ramo são necessários")
        if self.epsilon_used > self.epsilon_limit + tolerance:
            raise ValueError(
                f"ramo paralelo excede epsilon: {self.epsilon_used} > {self.epsilon_limit}"
            )
        if self.delta_used > self.delta_limit + tolerance:
            raise ValueError(
                f"ramo paralelo excede delta: {self.delta_used} > {self.delta_limit}"
            )
        return self

    def as_dict(self) -> dict:
        return {
            "accounting": "parallel_disjoint_record_partitions",
            "assumption": (
                "cada registro pertence a exatamente um ramo, e a partição é independente "
                "dos valores dos registros"
            ),
            "epsilon_limit": self.epsilon_limit,
            "epsilon_used": self.epsilon_used,
            "delta_limit": self.delta_limit,
            "delta_used": self.delta_used,
            "branches": [
                {**asdict(release), "sensitivity": asdict(release.sensitivity)}
                for release in self.releases
            ],
        }
