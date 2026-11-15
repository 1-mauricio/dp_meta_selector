"""Resumable, disk-guarded extension of the original 984-dataset benchmark.

Run alignment first, then run, then summarize. Dataset JSON commits are atomic;
resume skips only validated complete commits and never substitutes datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback

# Set before importing numerical libraries or starting spawn workers.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('OPENML_CACHE_DIR', str(Path('.cache/openml').resolve()))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from dpmeta.openml_benchmark import _load_manifest, _openml, _seed_for
from dpmeta.portfolio_benchmark import (NB8U_PIPELINE, PIPELINES, run_dataset,
                                       complete_dataset_ids, validate_condition_grid)
from dpmeta.provenance import (file_sha256, numerical_implementation_sha256,
                               runtime_fingerprint, save_source_snapshot)

OUT = Path('results/portfolio-984-nb8u')
ORIGINAL = Path('results/portfolio-984-current-slim')
MANIFEST = Path('data/openml_manifest_984_completed.csv')
CACHE = Path('.cache/openml')
KEYS = ['dataset_id', 'epsilon', 'repetition']
MIN_FREE = 5 * 1024**3


class DiskLimit(RuntimeError):
    pass


def disk_guard():
    free = shutil.disk_usage(CACHE).free
    if free < MIN_FREE:
        raise DiskLimit(f'free disk {free} bytes < 5 GiB; stopping')
    return free


def atomic_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def source_row(row):
    row = row.copy()
    local = str(row.get('local_data_path', ''))
    if local and not Path(local).is_file():
        row.local_data_path = ''  # Existing download branch; all validations remain.
    return row


def rejected_dataset_ids():
    """Never resume or aggregate input sources rejected by alignment validation."""
    path = OUT / 'validation_rejections.json'
    return set(map(int, json.loads(path.read_text()))) if path.exists() else set()


def inputs():
    config = json.loads((ORIGINAL / 'run_config.json').read_text())
    manifest = _load_manifest(MANIFEST)
    if file_sha256(MANIFEST) != config['manifest_sha256']:
        raise ValueError('manifest hash differs from original run')
    old = pd.read_csv(ORIGINAL / 'conditions.csv', float_precision='round_trip')
    validate_condition_grid(old, manifest, epsilons=tuple(config['epsilons']),
                            repeats=config['repeats'], pipelines=tuple(config['pipelines']))
    if set(old.dataset_id) != set(manifest.dataset_id):
        raise ValueError('manifest/original IDs differ')
    return config, manifest, old


def alignment():
    config, manifest, old = inputs()
    # Deterministic cached representatives: five with numeric features and five
    # all nominal, then fill any shortfall; prefer smaller datasets.
    cached = manifest[manifest.dataset_id.map(
        lambda d: (CACHE / f'org/openml/www/datasets/{d}/dataset.arff').is_file()
        or (CACHE / f'org/openml/www/datasets/{d}/dataset.pq').is_file()
        or (CACHE / f'org/openml/www/datasets/{d}').is_dir())].copy()
    cached = cached[cached.dataset_id.ne(47266)].sort_values(['n_instances', 'dataset_id'])
    cached['has_numeric'] = cached.n_features.gt(cached.n_nominal_features)
    selected = pd.concat([cached[cached.has_numeric].head(5), cached[~cached.has_numeric].head(5)])
    if len(selected) < 10:
        selected = pd.concat([selected, manifest]).drop_duplicates('dataset_id').head(10)
    selected.to_csv(OUT / 'alignment_manifest.csv', index=False)
    om = _openml(CACHE)
    checks, details = [], []
    with threadpool_limits(limits=1):
        for _, row in selected.iterrows():
            disk_guard()
            started = time.monotonic()
            generated = pd.DataFrame(run_dataset(
                om, source_row(row), tuple(config['epsilons']), config['repeats'], config['seed'],
                pipelines=(PIPELINES[10],), include_research_features=False,
                numeric_transform=config['numeric_transform']))
            ref = old[(old.dataset_id == row.dataset_id) & (old.pipeline == PIPELINES[10])]
            paired = generated.merge(ref, on=KEYS + ['pipeline'], suffixes=('_rerun', '_original'),
                                     validate='one_to_one')
            paired['abs_error'] = abs(paired.private_ba_rerun - paired.private_ba_original)
            paired['clean_abs_error'] = abs(paired.clean_ba_rerun - paired.clean_ba_original)
            details.append(paired[KEYS + ['pipeline','private_ba_rerun','private_ba_original',
                                        'abs_error','clean_abs_error']])
            check = dict(dataset_id=int(row.dataset_id), has_numeric=bool(row.has_numeric),
                         conditions=len(paired), max_abs_error=float(paired.abs_error.max()),
                         clean_max_abs_error=float(paired.clean_abs_error.max()),
                         seconds=time.monotonic()-started)
            checks.append(check)
            print('ALIGN', check, flush=True)
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(OUT / 'alignment_conditions.csv', index=False)
    passed = len(checks) == 10 and len(detail) == 240 and detail.abs_error.max() <= 1e-12
    result = dict(passed=bool(passed), datasets=checks, conditions=len(detail),
                  max_abs_error=float(detail.abs_error.max()),
                  seed=config['seed'], split_namespace='portfolio_split',
                  noise_namespace='categorical_nb_laplace',
                  implementation_sha256=numerical_implementation_sha256(), runtime=runtime_fingerprint())
    atomic_json(OUT / 'alignment.json', result)
    if not passed:
        raise RuntimeError('alignment failed; full run prohibited')


def worker(queue, stop, config):
    om = _openml(CACHE)
    with threadpool_limits(limits=1):
        while not stop.is_set():
            task = queue.get()
            if task is None:
                return
            raw, reference = task
            row = pd.Series(raw)
            dataset_id = int(row.dataset_id)
            started = time.monotonic()
            try:
                free_before = disk_guard()
                generated = run_dataset(
                    om, source_row(row), tuple(config['epsilons']), config['repeats'], config['seed'],
                    pipelines=(NB8U_PIPELINE,), include_research_features=False,
                    numeric_transform=config['numeric_transform'],
                    best_clean_by_condition=reference)
                # Ledger equality is checked against the full original NB ledger
                # schema, determined by p+1 queries (independent of data/bins/prior).
                from dpmeta.privacy import PrivacyLedger, Release, Sensitivity
                for r in generated:
                    ledger = json.loads(r['ledger_json'])
                    expected = PrivacyLedger(r['epsilon'], 0., accounting='basic')
                    queries = r['original_feature_count'] + 1
                    for i in range(queries):
                        expected.record(Release(
                            name='class_prior_histogram' if i == 0 else f'feature_{i-1}_histogram',
                            mechanism='laplace', epsilon=r['epsilon']/queries, delta=0.,
                            sensitivity=Sensitivity(2., np.sqrt(2.), 1.),
                            noise_scale=2. / (r['epsilon']/queries)))
                    expected.verify()
                    if ledger != expected.as_dict():
                        raise AssertionError('NB ledger differs from original accounting')
                    r['split_seed'] = _seed_for(config['seed'], dataset_id, r['repetition'], 'portfolio_split')
                    r['dp_seed'] = _seed_for(config['seed'], dataset_id, r['repetition'],
                                             r['epsilon'], 'categorical_nb_laplace')
                    r['ledger_identical_to_current_nb'] = True
                frame = pd.DataFrame(generated)
                validate_condition_grid(frame, pd.DataFrame([raw]), epsilons=tuple(config['epsilons']),
                                        repeats=config['repeats'], pipelines=(NB8U_PIPELINE,))
                free_after = disk_guard()
                atomic_json(OUT / 'datasets' / f'{dataset_id}.json', dict(
                    dataset_id=dataset_id, seconds=time.monotonic()-started,
                    free_before=free_before, free_after=free_after, rows=generated))
                print(f'DONE {dataset_id} {time.monotonic()-started:.2f}s free={free_after/1024**3:.2f}GiB', flush=True)
            except Exception as exc:
                failure = dict(dataset_id=dataset_id, dataset_name=str(row.dataset_name),
                               error_type=type(exc).__name__, error=str(exc),
                               traceback=traceback.format_exc(), seconds=time.monotonic()-started)
                atomic_json(OUT / 'failure_details' / f'{dataset_id}.json', failure)
                print(f'FAIL {dataset_id}: {exc!r}', flush=True)
                if isinstance(exc, DiskLimit):
                    stop.set()
                    return


def run(workers, retry_failures=False):
    config, manifest, old = inputs()
    check = json.loads((OUT / 'alignment.json').read_text())
    if not check['passed'] or check['implementation_sha256'] != numerical_implementation_sha256():
        raise RuntimeError('a passing alignment for this implementation is required')
    disk_guard()
    for name in ('datasets', 'failure_details'):
        (OUT / name).mkdir(exist_ok=True)
    expected = dict(original_run_config=config, manifest_sha256=file_sha256(MANIFEST),
                    original_conditions_sha256=file_sha256(ORIGINAL / 'conditions.csv'),
                    implementation_sha256=numerical_implementation_sha256(),
                    executor_sha256=file_sha256(Path(__file__)),
                    runtime=runtime_fingerprint(), pipelines=[NB8U_PIPELINE], workers=workers,
                    numeric_bins=8, uniform_prior=True, prior_query_retained=True,
                    metadata_release=False, cache=str(CACHE), min_free_bytes=MIN_FREE,
                    clean_reference='original best_clean_portfolio_ba of matching condition')
    config_path = OUT / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != expected:
        raise ValueError('resume configuration changed')
    atomic_json(config_path, expected)
    save_source_snapshot(OUT, extra_files=('scripts/parallel_nb8u.py', 'tests/test_nb8u.py'))
    completed = set()
    for path in sorted((OUT / 'datasets').glob('*.json')):
        payload = json.loads(path.read_text())
        if complete_dataset_ids(pd.DataFrame(payload['rows']), epsilons=tuple(config['epsilons']),
                                repeats=config['repeats'], pipelines=(NB8U_PIPELINE,)) == {payload['dataset_id']}:
            completed.add(payload['dataset_id'])
        else:
            raise ValueError(f'invalid resume commit {path}')
    failed = {int(f.stem) for f in (OUT / 'failure_details').glob('*.json')}
    pending = manifest[~manifest.dataset_id.isin(completed | rejected_dataset_ids() | (set() if retry_failures else failed))]
    # NB work scales with original feature count, rows, and class/cardinality tables.
    pending = pending.assign(work=pending.n_instances * pending.n_features).sort_values('work', ascending=False)
    print(f'RUN completed={len(completed)} failed={len(failed-completed)} pending={len(pending)} workers={workers}', flush=True)
    ctx = mp.get_context('spawn')
    queue, stop = ctx.Queue(), ctx.Event()
    reference = old.groupby(KEYS).best_clean_portfolio_ba.first()
    for _, row in pending.iterrows():
        baseline = {(float(e), int(r)): float(b) for (e, r), b in reference.loc[int(row.dataset_id)].items()}
        queue.put((row.drop('work').to_dict(), baseline))
    for _ in range(workers):
        queue.put(None)
    processes = [ctx.Process(target=worker, args=(queue, stop, config)) for _ in range(workers)]
    wall = time.time()
    attempt = dict(started_unix=wall, pending=len(pending), completed_before=len(completed),
                   workers=workers, free_before=shutil.disk_usage(CACHE).free)
    attempts_path = OUT / 'attempts.json'
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    attempts.append(attempt)
    atomic_json(attempts_path, attempts)
    try:
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            if proc.pid is not None:
                proc.join()
        raise
    finally:
        # Pending queued tasks need not be flushed after an early disk stop.
        queue.cancel_join_thread()
        queue.close()
        attempt.update(seconds=time.time()-wall, ended_unix=time.time(),
                       disk_stopped=stop.is_set(), free_after=shutil.disk_usage(CACHE).free,
                       worker_exit_codes=[p.exitcode for p in processes])
        atomic_json(attempts_path, attempts)
    aggregate()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError('worker failure; committed datasets are resumable')


def aggregate():
    config, manifest, old = inputs()
    payloads = [json.loads(path.read_text()) for path in sorted((OUT/'datasets').glob('*.json'))
                if int(path.stem) not in rejected_dataset_ids()]
    if not payloads:
        raise RuntimeError('no completed datasets')
    full = pd.DataFrame([r for payload in payloads for r in payload['rows']])
    validate_condition_grid(full, manifest, epsilons=tuple(config['epsilons']), repeats=config['repeats'],
                            pipelines=(NB8U_PIPELINE,))
    new = full[list(old.columns)].sort_values(KEYS + ['pipeline'])
    new.to_csv(OUT / 'conditions_new_option.csv', index=False)
    combined = pd.concat([old, new], ignore_index=True)
    counts = combined.groupby(KEYS).pipeline.agg(['size','nunique'])
    good = counts[(counts['size'] == 13) & (counts['nunique'] == 13)].index
    combined = combined.set_index(KEYS).loc[good].reset_index()[list(old.columns)]
    validate_condition_grid(combined, manifest, epsilons=tuple(config['epsilons']), repeats=config['repeats'],
                            pipelines=(*PIPELINES, NB8U_PIPELINE))
    combined.sort_values(KEYS+['pipeline']).to_csv(OUT/'conditions_13.csv', index=False)
    ledgers = {}
    audit = []
    for r in full.to_dict('records'):
        identifier = hashlib.sha256(r['ledger_json'].encode()).hexdigest()
        ledgers[identifier] = r['ledger_json']
        audit.append({k:r[k] for k in KEYS + ['pipeline','split_seed','dp_seed','train_n','test_n',
                                              'encoded_dim','epsilon_used','delta_used',
                                              'ledger_identical_to_current_nb']} | dict(
            ledger_id=identifier, numeric_bins=8, uniform_prior=True, prior_query_retained=True))
    pd.DataFrame([dict(ledger_id=k,ledger_json=v) for k,v in sorted(ledgers.items())]).to_csv(OUT/'ledgers.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'privacy_audit.csv',index=False)
    pd.DataFrame([{k:v for k,v in p.items() if k != 'rows'} for p in payloads]).to_csv(OUT/'timing.csv',index=False)
    failures = [json.loads(p.read_text()) for p in sorted((OUT/'failure_details').glob('*.json'))
                if int(p.stem) not in set(full.dataset_id)]
    pd.DataFrame(failures, columns=['dataset_id','dataset_name','error_type','error','traceback','seconds']).to_csv(OUT/'failures.csv',index=False)
    atomic_json(OUT/'coverage.json',dict(planned=len(manifest),completed=full.dataset_id.nunique(),
                failed=len(failures), unattempted_ids=sorted(set(manifest.dataset_id)-set(full.dataset_id)-
                {f['dataset_id'] for f in failures}), new_rows=len(new), combined_rows=len(combined)))
    print('AGGREGATED',len(payloads),'datasets',len(new),'new rows',len(combined),'combined',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['alignment','run','aggregate'])
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--retry-failures',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    if args.stage=='alignment': alignment()
    elif args.stage=='aggregate': aggregate()
    else: run(args.workers,args.retry_failures)

if __name__=='__main__':
    main()
