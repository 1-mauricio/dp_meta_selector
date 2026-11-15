"""Replace unavailable planned datasets by the next fixed-seed same-stratum reserve."""
import json
import signal
import time
import traceback
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.e1_e4_e5 import OUT, Experiment, Writer, _openml
from pathlib import Path


def main():
    meta = json.loads((OUT/"metadata_E1_E4_E5.json").read_text())
    assert not meta.get("replacement_attempts"), "replacement stage already attempted"
    sample = pd.read_csv(OUT/"sample_E1_E4_E5.csv",keep_default_na=False)
    pool = pd.read_csv(OUT/"eligible_families_E1_E4_E5.csv",keep_default_na=False)
    actual = sample[sample.dataset_id.isin(meta["completed_ids"])].copy()
    used = set(sample.semantic_family_id)
    exp = Experiment(append=True)
    times = Writer("timing_E1_E4_E5.csv",append=True)
    errors = Writer("errors_E1_E4_E5.csv",append=True)
    openml = _openml(Path(".cache/openml"))
    started = time.monotonic()
    original_runtime = meta["runtime_seconds"]
    meta["replacement_attempts"] = []
    for _, failed in sample[~sample.dataset_id.isin(meta["completed_ids"])].iterrows():
        candidates = pool[(pool.n_tercile==failed.n_tercile) & (pool.feature_group==failed.feature_group)]
        for _, row in candidates.iterrows():
            if row.semantic_family_id in used:
                continue
            if original_runtime + time.monotonic()-started > 49*60:
                break
            used.add(row.semantic_family_id)
            print(f"REPLACE {failed.dataset_id} -> {row.dataset_id} ({row.n_tercile}/{row.feature_group})",flush=True)
            def alarm(signum,frame):
                raise TimeoutError("replacement dataset limit")
            signal.signal(signal.SIGALRM,alarm)
            signal.alarm(240)
            succeeded = False
            try:
                with threadpool_limits(limits=1):
                    result = exp.dataset(openml,row)
                times.add(result)
                meta["completed_ids"].append(int(row.dataset_id))
                replacement = row.to_dict()
                replacement["within_stratum_order"] = failed.within_stratum_order
                actual = pd.concat([actual,pd.DataFrame([replacement])],ignore_index=True)
                succeeded = True
                print(f"DONE replacement {row.dataset_id} {result['seconds']:.1f}s",flush=True)
            except Exception as error:
                meta["failed_ids"].append(int(row.dataset_id))
                errors.add(dict(dataset_id=int(row.dataset_id),error=repr(error),traceback=traceback.format_exc()))
                print(f"FAIL replacement {row.dataset_id}: {error!r}",flush=True)
            finally:
                signal.alarm(0)
                meta["replacement_attempts"].append(dict(original_id=int(failed.dataset_id),
                    replacement_id=int(row.dataset_id),succeeded=succeeded,n_tercile=row.n_tercile,
                    feature_group=row.feature_group))
                meta["runtime_seconds"] = original_runtime+time.monotonic()-started
                (OUT/"metadata_E1_E4_E5.json").write_text(json.dumps(meta,indent=2))
            if succeeded:
                break
    assert actual.semantic_family_id.nunique()==len(actual)
    actual.to_csv(OUT/"effective_sample_E1_E4_E5.csv",index=False)
    print(json.dumps(meta,indent=2))


if __name__=="__main__":
    main()
