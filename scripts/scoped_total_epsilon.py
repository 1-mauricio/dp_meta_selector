"""Compare a DP-rich selector with a fixed pipeline at equal total epsilon.

The selector run uses ``epsilon_model`` plus a separate metadata release.  A
fair pilot therefore compares its output at ``e`` with a fixed-pipeline run at
``e + epsilon_meta``.  This script only combines already-computed pilot runs;
it does not refit models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def run(selector_predictions: Path, fixed_predictions: Path,
        metadata_epsilon: float = 1.0) -> dict:
    selector = pd.read_csv(selector_predictions)
    fixed = pd.read_csv(fixed_predictions)
    selector = selector[selector.feature_mode == "dp_research_rich"]
    fixed = fixed[fixed.feature_mode == "dp_research_rich"]
    rows = []
    for epsilon in sorted(selector.epsilon.unique()):
        total = float(epsilon + metadata_epsilon)
        left = selector[selector.epsilon == epsilon]
        right = fixed[fixed.epsilon == total]
        joined = left[["dataset_id", "family_id", "selector_regret"]].merge(
            right[["dataset_id", "family_id", "best_fixed_regret"]],
            on=["dataset_id", "family_id"], how="inner",
        )
        family = joined.groupby("family_id").mean(numeric_only=True)
        rows.append({
            "model_epsilon": float(epsilon),
            "metadata_epsilon": float(metadata_epsilon),
            "total_epsilon": total,
            "n_families": int(len(family)),
            "selector_regret": float(family.selector_regret.mean()),
            "fixed_total_regret": float(family.best_fixed_regret.mean()),
            "fixed_minus_selector": float(
                (family.best_fixed_regret - family.selector_regret).mean()
            ),
        })
    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--fixed-total", type=Path, required=True)
    parser.add_argument("--metadata-epsilon", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.selector, args.fixed_total, args.metadata_epsilon)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
