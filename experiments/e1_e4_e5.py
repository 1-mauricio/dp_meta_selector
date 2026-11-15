"""Scoped, reproducible E1/E4/E5 experiments; no changes to dpmeta modules."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import types

os.environ.setdefault("OPENML_CACHE_DIR", str(Path(".cache/openml").resolve()))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from dpmeta import classifiers as c, model as g
from dpmeta.openml_benchmark import _get_openml_data, _openml, _seed_for, stratified_holdout_indices
from dpmeta.portfolio_benchmark import _ledger_fields

SEED = 20261110
EPS = (0.1, 1.0, 4.0)
OUT = Path("results/experiments")
PIPES = ("categorical_naive_bayes__laplace_histograms", "softmax__dp_sgd_clip2_reg003", "gnb__gaussian_zcdp_equal")


class ZeroNoise:
    def normal(self, loc=0., scale=1., size=None):
        return np.full(size, loc)

    laplace = normal


def with_globals(function, **overrides):
    """Reuse the exact existing function body with isolated global bindings."""
    wrapped = types.FunctionType(function.__code__, {**function.__globals__, **overrides},
                                 function.__name__, function.__defaults__, function.__closure__)
    wrapped.__kwdefaults__ = function.__kwdefaults__
    return wrapped


zero_sgd = with_globals(c.fit_private_sgd_softmax, privacy_rng=lambda seed: ZeroNoise())
zero_nb = with_globals(c.fit_private_categorical_nb, privacy_rng=lambda seed: ZeroNoise())
# GNB draws noise in a nested helper; replace only that helper's RNG binding.
zero_stats = with_globals(g.privatize_sufficient_statistics, privacy_rng=lambda seed: ZeroNoise())
zero_gnb = with_globals(g.fit_private, privatize_sufficient_statistics=zero_stats)


def nb_stats(x, y, k, schema, bins):
    categories, cards = c.categorical_nb_features(x, schema, numeric_bins=bins)
    counts, features = c._categorical_nb_counts(categories, y, k, cards)
    return c.CategoricalNBStatistics(categories, cards, counts, features)


def select_sample():
    eligible = pd.read_csv("data/openml_manifest_984_completed.csv", keep_default_na=False)
    eligible = eligible[(eligible.n_instances <= 5000) & eligible.semantic_family_id.ne("")].copy()
    cuts = eligible.n_instances.quantile([1/3, 2/3]).to_numpy()
    eligible["n_tercile"] = pd.cut(eligible.n_instances, [-np.inf, *cuts, np.inf],
                                   labels=["T1", "T2", "T3"]).astype(str)
    eligible["feature_group"] = np.where(eligible.n_nominal_features.eq(eligible.n_features),
                                          "all_nominal", "has_numeric")
    # Random representative per semantic family before balanced allocation.
    pool = eligible.sample(frac=1, random_state=SEED).drop_duplicates("semantic_family_id")
    cells = list(pool.groupby(["n_tercile", "feature_group"], sort=True))
    chosen = []
    for index, ((tercile, kind), block) in enumerate(cells):
        quota = 7 if index < 4 else 6
        for order, (_, row) in enumerate(block.head(quota).iterrows()):
            chosen.append({**row.to_dict(), "within_stratum_order": order})
    sample = pd.DataFrame(chosen).sort_values(["within_stratum_order", "n_tercile", "feature_group"])
    assert len(sample) == 40 and sample.semantic_family_id.nunique() == 40
    sample.to_csv(OUT / "sample_E1_E4_E5.csv", index=False)
    pool.to_csv(OUT / "eligible_families_E1_E4_E5.csv", index=False)
    return sample, cuts, len(eligible), len(pool)


class Writer:
    def __init__(self, filename, *, append=False):
        self.path = OUT / filename
        if self.path.exists() and not append:
            raise FileExistsError(self.path)
        self.file = self.path.open("a" if append else "w", newline="")
        self.writer = None
        if append and self.path.stat().st_size:
            with self.path.open(newline="") as existing:
                fields = next(csv.reader(existing))
            self.writer = csv.DictWriter(self.file, fieldnames=fields)

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()


class Experiment:
    def __init__(self, *, append=False):
        self.runs = {e: Writer(f"{e}_runs.csv", append=append) for e in ("E1", "E4", "E5")}
        self.ledgers = Writer("budget_ledgers_E1_E4_E5.csv", append=append)
        self.audits = Writer("privacy_audit_E1_E4_E5.csv", append=append)
        self.seen_ledgers = set()
        if append:
            self.seen_ledgers = set(pd.read_csv(self.ledgers.path).ledger_id)
        self.reference = {}

    def fit(self, pipe, x, y, k, schema, eps, seed, stats, bins=16, reg=.03, zero=False):
        if pipe == PIPES[0]:
            model, _, ledger = (zero_nb if zero else c.fit_private_categorical_nb)(
                x, y, k, schema, epsilon=eps, seed=seed, statistics=stats[bins])
        elif pipe == PIPES[1]:
            model, ledger = (zero_sgd if zero else c.fit_private_sgd_softmax)(
                x, y, k, epsilon=eps, delta=g.DELTA_TOTAL, seed=seed,
                n_steps=100, gradient_clip=2., regularization=reg)
        else:
            model, ledger = (zero_gnb if zero else g.fit_private)(
                x, y, k, epsilon=eps, delta=g.DELTA_TOTAL, mechanism="gaussian_zcdp",
                allocation="equal", seed=seed, stats=stats["gnb"])
        return model, ledger

    def record(self, experiment, row, pipe, eps, rep, split_seed, dp_seed, model, ledger,
               xt, yt, schema, *, mode="private", bins=16, prior="current", reg=.03,
               clean_ba=np.nan, fit_seconds=0.):
        if pipe == PIPES[0]:
            categories, _ = c.categorical_nb_features(xt, schema, numeric_bins=bins)
            if prior == "uniform":
                model = c.CategoricalNaiveBayesModel(np.ones_like(model.class_counts),
                                                     model.feature_counts, model.feature_cardinalities)
            prediction = model.predict(categories)
        else:
            prediction = g.predict(xt, model) if pipe == PIPES[2] else model.predict(xt)
        fields = _ledger_fields(ledger)
        ledger_json = fields.pop("ledger_json")
        ledger_id = hashlib.sha256(ledger_json.encode()).hexdigest()
        if ledger_id not in self.seen_ledgers:
            self.ledgers.add(dict(ledger_id=ledger_id, ledger_json=ledger_json))
            self.seen_ledgers.add(ledger_id)
        key = (int(row.dataset_id), split_seed, eps, pipe)
        identical = None
        is_private = mode in ("private", "split", "noise")
        if is_private:
            ledger.verify()
            if key not in self.reference:
                assert (bins == 16 and reg == .03), "baseline must run first"
                self.reference[key] = ledger_json
            identical = ledger_json == self.reference[key]
            assert identical, "variant changed privacy accounting"
            assert abs(fields["epsilon_used"] - eps) < 1e-9
            assert abs(fields["delta_used"] - (0. if pipe == PIPES[0] else 1e-6)) < 1e-15
        result = dict(dataset_id=int(row.dataset_id), semantic_family_id=row.semantic_family_id,
                      n_instances=int(row.n_instances), n_tercile=row.n_tercile,
                      feature_group=row.feature_group, pipeline=pipe, epsilon=eps,
                      repetition=rep, mode=mode, bins=bins, prior=prior, regularization=reg,
                      split_seed=split_seed, dp_seed=dp_seed, test_n=len(yt), encoded_dim=schema.encoded_dim,
                      private_ba=g.balanced_accuracy(yt, prediction), clean_ba=clean_ba,
                      fit_seconds=fit_seconds, ledger_id=ledger_id,
                      ledger_identical_to_baseline=identical,
                      is_dp_release=is_private, **fields)
        self.runs[experiment].add(result)
        self.audits.add(dict(experiment=experiment, dataset_id=int(row.dataset_id),
                             pipeline=pipe, epsilon=eps, repetition=rep, mode=mode,
                             bins=bins, prior=prior, regularization=reg, ledger_id=ledger_id,
                             verified_private=is_private, identical_to_baseline=identical,
                             epsilon_used=fields["epsilon_used"], delta_used=fields["delta_used"],
                             noise_enabled=is_private))

    def dataset(self, openml, row):
        started = time.monotonic()
        source = row.copy()
        # The manifest pins absent parquet files in this copy. Use the loader's
        # existing download branch, preserving all schema and row validations.
        if source.local_data_path and not Path(source.local_data_path).exists():
            source.local_data_path = ""
        x, y, schema = _get_openml_data(openml, source, numeric_transform="tanh")
        k = len(schema.classes)
        clean = {}
        fixed = {}
        for split in range(10):
            split_seed = _seed_for(SEED, int(row.dataset_id), split, "portfolio_split")
            tr, te = stratified_holdout_indices(y, split_seed)
            xx, yy, xt, yt = x[tr], y[tr], x[te], y[te]
            stats = {b: nb_stats(xx, yy, k, schema, b) for b in (16, 4, 8)}
            stats["gnb"] = g.sufficient_statistics(xx, yy, k)
            if split == 0:
                clean_nb, _ = c.fit_clean_categorical_nb(xx, yy, k, schema, statistics=stats[16])
                clean[PIPES[0]] = g.balanced_accuracy(yt, clean_nb.predict(c.categorical_nb_features(xt, schema)[0]))
                clean[PIPES[2]] = g.balanced_accuracy(yt, g.predict(xt, g.model_from_statistics(stats["gnb"])))
                clean[PIPES[1]] = g.balanced_accuracy(yt, c.fit_clean_softmax(
                    xx, yy, k, regularization=.03, optimizer_tolerance=1e-5,
                    record_norm_bound=np.sqrt(len(schema.feature_names) + 1)).model.predict(xt))
            for eps in EPS:
                for pipe in PIPES:
                    seed0 = _seed_for(SEED, int(row.dataset_id), eps, pipe, "dp", 0)
                    t = time.monotonic()
                    model, ledger = self.fit(pipe, xx, yy, k, schema, eps, seed0, stats)
                    elapsed = time.monotonic() - t
                    args = (row, pipe, eps)
                    self.record("E1", *args, split, split_seed, seed0, model, ledger, xt, yt, schema,
                                mode="split", clean_ba=clean.get(pipe, np.nan), fit_seconds=elapsed)
                    # 'split' and 'noise' are private; record uses a mode flag below.
                    if split == 0:
                        fixed[(pipe, eps)] = (model, ledger)
                        self.record("E1", *args, 0, split_seed, seed0, model, ledger, xt, yt, schema,
                                    mode="noise", clean_ba=clean[pipe])
                        for rep in range(1, 10):
                            dp_seed = _seed_for(SEED, int(row.dataset_id), eps, pipe, "dp", rep)
                            t = time.monotonic()
                            m, l = self.fit(pipe, xx, yy, k, schema, eps, dp_seed, stats)
                            self.record("E1", *args, rep, split_seed, dp_seed, m, l, xt, yt, schema,
                                        mode="noise", clean_ba=clean[pipe], fit_seconds=time.monotonic()-t)
                        m, l = self.fit(pipe, xx, yy, k, schema, eps, seed0, stats, zero=True)
                        self.record("E1", *args, 0, split_seed, seed0, m, l, xt, yt, schema,
                                    mode="zero_noise_NON_DP", clean_ba=clean[pipe])
                    if split < 3 and pipe in PIPES[:2]:
                        dp_seed = _seed_for(SEED, int(row.dataset_id), eps, pipe, "dp", split)
                        if split == 0:
                            base_model, base_ledger = model, ledger
                        else:
                            base_model, base_ledger = self.fit(pipe, xx, yy, k, schema, eps, dp_seed, stats)
                        if pipe == PIPES[0]:
                            for bins in (16, 4, 8):
                                m, l = ((base_model, base_ledger) if bins == 16 else
                                        self.fit(pipe, xx, yy, k, schema, eps, dp_seed, stats, bins=bins))
                                for prior in ("current", "uniform"):
                                    self.record("E4", *args, split, split_seed, dp_seed, m, l, xt, yt, schema,
                                                bins=bins, prior=prior)
                        else:
                            for reg in (.03, .01, .003, .001):
                                m, l = ((base_model, base_ledger) if reg == .03 else
                                        self.fit(pipe, xx, yy, k, schema, eps, dp_seed, stats, reg=reg))
                                self.record("E5", *args, split, split_seed, dp_seed, m, l, xt, yt, schema, reg=reg)
        return dict(dataset_id=int(row.dataset_id), seconds=time.monotonic()-started,
                    encoded_dim=schema.encoded_dim, completed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=55)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    sample, cuts, eligible_n, families_n = select_sample()
    exp = Experiment()
    openml = _openml(Path(".cache/openml"))
    times = Writer("timing_E1_E4_E5.csv")
    errors = Writer("errors_E1_E4_E5.csv")
    metadata = dict(seed=SEED, tercile_boundaries=cuts.tolist(), eligible_records=eligible_n,
                    eligible_families=families_n, planned_n=40, completed_ids=[], failed_ids=[],
                    time_limit_minutes=args.minutes)
    for index, (_, row) in enumerate(sample.iterrows()):
        elapsed = time.monotonic()-start
        if elapsed > args.minutes*60 - 60:
            break
        # A hard per-dataset cap preserves the total budget; incomplete datasets
        # are excluded from every summary rather than silently dropping runs.
        remaining = args.minutes*60 - elapsed
        def alarm(signum, frame):
            raise TimeoutError("dataset/runtime limit")
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(int(min(remaining, 240)))
        print(f"START {index+1}/40 id={row.dataset_id} n={row.n_instances} d={row.n_encoded_features} elapsed={elapsed:.1f}s", flush=True)
        try:
            with threadpool_limits(limits=1):
                timing = exp.dataset(openml, row)
            times.add(timing)
            metadata["completed_ids"].append(int(row.dataset_id))
            print(f"DONE {row.dataset_id} {timing['seconds']:.1f}s", flush=True)
        except Exception as error:
            metadata["failed_ids"].append(int(row.dataset_id))
            errors.add(dict(dataset_id=int(row.dataset_id), error=repr(error), traceback=traceback.format_exc()))
            print(f"FAIL {row.dataset_id}: {error!r}", flush=True)
        finally:
            signal.alarm(0)
            metadata["runtime_seconds"] = time.monotonic()-start
            (OUT / "metadata_E1_E4_E5.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
