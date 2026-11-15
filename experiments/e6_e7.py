"""Scoped E6/E7; native E1/E4/E5 loader, RNG pairing, wrappers and accountant.

Run with PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python.
All new data, including fallback OpenML cache, stay in results/experiments/e6_e7.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENML_CACHE_DIR', str(__import__('pathlib').Path('results/experiments/e6_e7/openml_cache').resolve()))
import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.e1_e4_e5 import (c, g, zero_sgd, _get_openml_data, _openml,
                            _seed_for, stratified_holdout_indices, SEED as PAIR_SEED, PIPES)
from dpmeta.privacy import PrivacyLedger, Release, Sensitivity
from dpmeta.mechanisms import zcdp_rho_for_epsilon

OUT = Path('results/experiments/e6_e7')
SEED = 20261111
LAMBDAS = (.03, .01, .003, .001, .0003)
EPS6 = (.1, .5, 1., 4., 16.)
EPS7 = (.1, 1., 4.)
STEPS = (100, 200, 400)
CLIPS = (.5, 1., 2., 4.)
BIN_EDGES = np.array([-np.inf, 2., 3., 4., 5., np.inf])


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False))


def initialize_protection():
    """Record preexisting files before a fresh reproduction; never replace it."""
    if not (OUT/'protected_initial.json').exists():
        entries = {}
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if not (Path(root)/d).as_posix().startswith(str(OUT))]
            for name in files:
                path = Path(root)/name
                if path.is_symlink():
                    continue
                st = path.stat()
                record = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
                if (path.suffix in ('.py','.toml','.md','.csv','.json') and
                    not str(path).startswith(('.venv/','.cache/'))) or str(path).startswith('.git/'):
                    record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                entries[str(path)] = record
        save_json('protected_initial.json', entries)
    if not (OUT/'run_metadata.json').exists():
        from datetime import datetime, timezone
        save_json('run_metadata.json', dict(request_start_utc=datetime.now(timezone.utc).isoformat(),
                  run_start_epoch=time.time(),workers=8,full_grid_planned=True,python='.venv/bin/python',
                  environment=dict(PYTHONPATH='.',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')))


def sample():
    base = pd.read_csv('results/experiments/effective_sample_E1_E4_E5.csv', keep_default_na=False)
    assert len(base) == 40
    base['sample_group'] = 'original40'
    m = pd.read_csv('data/openml_manifest_984_completed.csv', keep_default_na=False)
    e = m[m.n_instances.between(5001, 50000) & m.semantic_family_id.ne('') &
          ~m.semantic_family_id.isin(base.semantic_family_id)].copy()
    e['feature_group'] = np.where(e.n_nominal_features.eq(e.n_features), 'all_nominal', 'has_numeric')
    # Public alias absent from manifest normalization; same chess endgame family.
    aliases = {'king rook vs king': 'kropt'}
    e['sampling_family_id'] = e.semantic_family_id.replace(aliases)
    pool = e.sample(frac=1, random_state=SEED).drop_duplicates('sampling_family_id')
    pool.to_csv(OUT / 'large_reserve_order.csv', index=False)
    nominal_quota = min(7, int(pool.feature_group.eq('all_nominal').sum()))
    numeric_quota = 15 - nominal_quota
    large = pd.concat([pool[pool.feature_group.eq(kind)].head(q) for kind, q in
                       [('all_nominal', nominal_quota), ('has_numeric', numeric_quota)]], ignore_index=True)
    assert len(large) == 15
    large['sample_group'] = 'large15'
    large['n_tercile'] = 'LARGE'
    full = pd.concat([base, large], ignore_index=True)
    assert full.semantic_family_id.nunique() == 55
    assert full.semantic_family_id.replace(aliases).nunique() == 55
    cuts = full.n_instances.quantile([1/3, 2/3]).to_numpy()
    full['combined_n_tercile'] = pd.cut(full.n_instances, [-np.inf, *cuts, np.inf],
                                       labels=['C1','C2','C3']).astype(str)
    full.to_csv(OUT / 'sample_E6_E7.csv', index=False)
    save_json('sample_metadata.json', dict(seed=SEED, pairing_seed=PAIR_SEED,
              combined_tercile_cuts=cuts.tolist(), eligible_large_records=len(e),
              eligible_large_families=len(pool), public_family_aliases=aliases,
              large_quotas=dict(all_nominal=nominal_quota, has_numeric=numeric_quota)))
    return full


def load(row):
    source = row.copy()
    pinned = str(source.local_data_path)
    candidates = [Path(pinned), Path(f'.cache/openml/org/openml/www/datasets/{row.dataset_id}/dataset_{row.dataset_id}.pq')]
    source.local_data_path = next((str(p) for p in candidates if p.is_file()), '')
    # Read existing parquets directly; downloads, metadata and logs use a NEW cache.
    openml = None if source.local_data_path else _openml(OUT / 'openml_cache')
    x, y, schema = _get_openml_data(openml, source, numeric_transform='tanh')
    return x, y, schema, source.local_data_path or 'new_openml_cache'


def ledger_from_dict(d):
    ledger = PrivacyLedger(d['epsilon_limit'], d['delta_limit'], d['accounting'])
    ledger.releases = [Release(**{**r, 'sensitivity': Sensitivity(**r['sensitivity'])}) for r in d['releases']]
    return ledger.verify()


def audit(ledger, eps, steps, clip, n, k, dim):
    ledger.verify()
    rho = zcdp_rho_for_epsilon(eps, g.DELTA_TOTAL)
    assert ledger.accounting == 'zcdp' and len(ledger.releases) == steps
    assert abs(ledger.epsilon_used - eps) < 1e-9
    assert abs(ledger.delta_used - g.DELTA_TOTAL) < 1e-15
    assert np.isclose(ledger.rho_used, rho, rtol=1e-12, atol=1e-16)
    sens = 2 * clip / n
    sigma = sens / np.sqrt(2 * rho / steps)
    for r in ledger.releases:
        assert r.mechanism == 'gaussian_zcdp_full_batch_gradient'
        assert np.isclose(r.rho, rho/steps, rtol=1e-12)
        assert r.sensitivity.l2 == sens
        assert np.isclose(r.noise_scale, sigma, rtol=1e-12)
    fields = __import__('dpmeta.portfolio_benchmark', fromlist=['_ledger_fields'])._ledger_fields(ledger)
    js = fields.pop('ledger_json')
    return dict(**fields, rho_used=ledger.rho_used, rho_per_step=rho/steps,
                sensitivity_l2=sens, sigma=sigma, ledger_verified=True,
                accounting='zcdp', accounting_method_changed=False,
                release_schedule_changed=(steps != 100 or clip != 2.),
                total_budget_identical=True, ledger_id=hashlib.sha256(js.encode()).hexdigest()), js


class Worker:
    def __init__(self, row, stage):
        self.row, self.stage = row, stage
        self.rows, self.ledgers, self.cache = [], {}, {}
        self.x, self.y, self.schema, self.source = load(row)
        self.k = len(self.schema.classes)
        self.splits = {}
        for rep in range(3):
            split_seed = _seed_for(PAIR_SEED, int(row.dataset_id), rep, 'portfolio_split')
            tr, te = stratified_holdout_indices(self.y, split_seed)
            self.splits[rep] = (self.x[tr], self.y[tr], self.x[te], self.y[te], split_seed)
        # Resume only exact parameter/seed matches from this task's new files.
        prior = OUT / 'parts' / f'{stage}_{int(row.dataset_id)}.csv'
        self.clean_cache = {}
        if stage == 'E7' and prior.exists():
            old = pd.read_csv(prior)
            self.ledgers.update(json.loads(prior.with_name(prior.stem+'_ledgers.json').read_text()))
            for r in old.to_dict('records'):
                rep = int(r['repetition'])
                assert int(r['split_seed']) == self.splits[rep][4]
                assert int(r['n_train']) == len(self.splits[rep][1])
                zero = r['mode'] == 'zero_noise_NON_DP'
                key = (r['epsilon'],rep,r['regularization'],int(r['steps']),r['clip'],zero)
                ledger_from_dict(json.loads(self.ledgers[r['ledger_id']])).verify()
                self.cache[key] = r
                if zero and np.isfinite(r.get('clean_ba', np.nan)):
                    self.clean_cache[r['regularization']] = (r['clean_ba'],r['clean_fit_seconds'])

    def run(self, eps, rep, reg, steps=100, clip=2., policy='grid', zero=False):
        xx, yy, xt, yt, split_seed = self.splits[rep]
        dp_seed = _seed_for(PAIR_SEED, int(self.row.dataset_id), eps, PIPES[1], 'dp', rep)
        key = (eps, rep, reg, steps, clip, zero)
        if key in self.cache:
            result = {**self.cache[key], 'policy': policy, 'fit_seconds': 0., 'source': 'within_stage_reuse'}
        else:
            t = time.monotonic()
            model, ledger = (zero_sgd if zero else c.fit_private_sgd_softmax)(
                xx, yy, self.k, epsilon=eps, delta=g.DELTA_TOTAL, seed=dp_seed,
                n_steps=steps, gradient_clip=clip, regularization=reg)
            elapsed = time.monotonic()-t
            fields, js = audit(ledger, eps, steps, clip, len(yy), self.k, self.schema.encoded_dim)
            self.ledgers[fields['ledger_id']] = js
            if zero:
                fields['ledger_verified'] = False  # calibration ledger, NOT a private run
            result = dict(dataset_id=int(self.row.dataset_id), epsilon=eps, repetition=rep,
                          regularization=reg, steps=steps, clip=clip, policy=policy,
                          mode='zero_noise_NON_DP' if zero else 'private', is_dp_release=not zero,
                          split_seed=split_seed, dp_seed=dp_seed, n_train=len(yy), test_n=len(yt),
                          encoded_dim=self.schema.encoded_dim,
                          ba=g.balanced_accuracy(yt, model.predict(xt)), fit_seconds=elapsed,
                          source='native_fit', **fields)
            self.cache[key] = result
        self.rows.append(result)
        return result

    def persist(self):
        stem = f'{self.stage}_{int(self.row.dataset_id)}'
        pd.DataFrame(self.rows).to_csv(OUT / 'parts' / f'{stem}.csv', index=False)
        (OUT / 'parts' / f'{stem}_ledgers.json').write_text(json.dumps(self.ledgers))


def run_dataset(payload):
    row, stage, predictions, configs = payload
    started = time.monotonic()
    def timeout(*args):
        raise TimeoutError('per-dataset cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    w = None
    try:
        with threadpool_limits(limits=1):
            w = Worker(pd.Series(row), stage)
            if stage == 'E6':
                for eps in EPS6:
                    for rep in range(3):
                        for reg in LAMBDAS:
                            w.run(eps, rep, reg)
            else:
                clean = w.clean_cache
                xx, yy, xt, yt, _ = w.splits[0]
                for eps in EPS7:
                    reg_rule = float(predictions[(int(w.row.dataset_id), eps)])
                    for policy, reg in [('fixed003', .03), ('rule_lodo', reg_rule)]:
                        if reg not in clean:
                            t = time.monotonic()
                            fit = c.fit_clean_softmax(xx, yy, w.k, regularization=reg,
                                  optimizer_tolerance=1e-5,
                                  record_norm_bound=np.sqrt(len(w.schema.feature_names)+1))
                            clean[reg] = (g.balanced_accuracy(yt, fit.model.predict(xt)), time.monotonic()-t)
                        for steps, clip in configs:
                            for rep in range(3):
                                w.run(eps, rep, reg, steps, clip, policy)
                            z = w.run(eps, 0, reg, steps, clip, policy, zero=True)
                            z['clean_ba'], z['clean_fit_seconds'] = clean[reg]
                            z['clean_minus_zero'] = z['clean_ba'] - z['ba']
            w.persist()
        return dict(dataset_id=int(row['dataset_id']), stage=stage, complete=True,
                    seconds=time.monotonic()-started, rows=len(w.rows), data_source=w.source)
    except Exception as exc:
        if w is not None and w.rows:
            w.persist()
        return dict(dataset_id=int(row['dataset_id']), stage=stage, complete=False,
                    seconds=time.monotonic()-started, error=repr(exc), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)


def bin_index(n, eps):
    return np.digitize(np.log10(np.asarray(n)*np.asarray(eps)), BIN_EDGES[1:-1], right=False)


def fit_rule(frame):
    """Fixed public bins, train-dataset equal weights, conservative tie breaking."""
    ds = frame.groupby(['dataset_id','epsilon','regularization'], as_index=False).ba.mean()
    n = frame[['dataset_id','n_instances']].drop_duplicates().set_index('dataset_id').n_instances
    ds['bin'] = bin_index(ds.dataset_id.map(n), ds.epsilon)
    def best(block):
        # Give each training dataset equal weight inside the bin.
        scores = block.groupby(['dataset_id','regularization']).ba.mean().groupby('regularization').mean()
        top = scores.max()
        return float(max(scores.index[np.isclose(scores, top, atol=1e-12, rtol=0)]))
    fallback = best(ds)
    values, support = [], []
    for b in range(5):
        block = ds[ds.bin.eq(b)]
        count = block.dataset_id.nunique()
        support.append(int(count))
        values.append(best(block) if count >= 5 else fallback)
    return dict(bin_edges_log10_n_epsilon=['-inf',2,3,4,5,'inf'], lambdas=values,
                unique_training_datasets_per_bin=support, fallback_lambda=fallback,
                min_datasets=5, tie_break='largest_lambda', equal_weight='dataset_within_bin')


def build_rules(sample_frame):
    f = pd.read_csv(OUT/'E6_runs.csv').merge(sample_frame[['dataset_id','n_instances']], on='dataset_id')
    rows, folds = [], {}
    for dataset_id, block in f.groupby('dataset_id'):
        train = f[f.dataset_id.ne(dataset_id)]
        assert dataset_id not in train.dataset_id.unique()
        rule = fit_rule(train)
        folds[int(dataset_id)] = rule
        n = int(block.n_instances.iloc[0])
        for eps in EPS6:
            rows.append(dict(dataset_id=int(dataset_id), epsilon=eps, n_instances=n,
                             selected_lambda=rule['lambdas'][int(bin_index(n,eps))],
                             training_datasets=train.dataset_id.nunique()))
    pd.DataFrame(rows).to_csv(OUT/'E6_lodo_predictions.csv', index=False)
    save_json('E6_lodo_rules.json', folds)
    save_json('E6_final_public_rule.json', fit_rule(f))
    return {(r['dataset_id'],r['epsilon']):r['selected_lambda'] for r in rows}


def stage_run(stage, sample_frame, workers, predictions=None, configs=None):
    payloads = [(row.to_dict(),stage,predictions,configs) for _,row in sample_frame.iterrows()]
    # Expensive datasets first, using public size/dimension only.
    payloads.sort(key=lambda p: float(p[0]['n_instances'])*float(p[0]['n_encoded_features']), reverse=True)
    timing = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_dataset, p) for p in payloads]
        for future in cf.as_completed(futures):
            result = future.result()
            timing.append(result)
            save_json(f'{stage}_timing.json', timing)
            print(json.dumps({k:v for k,v in result.items() if k!='traceback'}), flush=True)
    complete = [r['dataset_id'] for r in timing if r['complete']]
    frames = [pd.read_csv(OUT/'parts'/f'{stage}_{i}.csv') for i in complete]
    pd.concat(frames,ignore_index=True).to_csv(OUT/f'{stage}_runs.csv',index=False)
    return complete


def main():
    p=argparse.ArgumentParser(); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--stage',choices=['sample','pilot','E6','E7'],required=True)
    p.add_argument('--configs',default='full'); args=p.parse_args()
    (OUT/'parts').mkdir(parents=True,exist_ok=True)
    if args.stage=='sample': initialize_protection()
    s=sample() if args.stage=='sample' else pd.read_csv(OUT/'sample_E6_E7.csv',keep_default_na=False)
    if args.stage=='sample': return
    if args.stage=='pilot':
        rows=[]
        for _,row in s[s.sample_group.eq('large15')].iterrows():
            t=time.monotonic()
            try:
                with threadpool_limits(limits=1):
                    w=Worker(row,'pilot'); r=w.run(1.,0,.03)
                rows.append(dict(dataset_id=int(row.dataset_id),load_and_fit_seconds=time.monotonic()-t,
                                 fit_seconds=r['fit_seconds'],data_source=w.source,complete=True))
            except Exception as e:
                rows.append(dict(dataset_id=int(row.dataset_id),complete=False,error=repr(e)))
            print(json.dumps(rows[-1]),flush=True)
            save_json('pilot.json',rows)
        return
    if args.stage=='E6':
        complete=stage_run('E6',s,args.workers)
        assert len(complete)==55, 'Resolve failures before fitting LODO rule'
        build_rules(s)
    else:
        predictions=build_rules(s)
        configs=[(t,c) for t in STEPS for c in CLIPS] if args.configs=='full' else json.loads(args.configs)
        save_json('E7_grid.json',dict(configurations=configs,full_grid=args.configs=='full'))
        stage_run('E7',s,args.workers,predictions,configs)


if __name__=='__main__': main()
