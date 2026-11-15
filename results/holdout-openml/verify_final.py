"""Read-only verification of completed protocol artifacts; writes its own audit."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image
from scripts.confirmatory import OUT, MANIFEST, EXPECTED, P15, KEYS
from dpmeta.provenance import file_sha256

def main():
    checks={}
    frozen=json.loads((OUT/'frozen.json').read_text())
    for p,h in frozen['hashes'].items():assert file_sha256(Path(p))==h,p
    checks['protocol_models_development_and_dpmeta_hashes_unchanged']=True
    assert file_sha256(Path('protocols/CONFIRMATORY_PROTOCOL_FINAL.md'))==EXPECTED
    selected=json.loads(MANIFEST.with_suffix('.selection.json').read_text())
    assert file_sha256(MANIFEST)==selected['manifest_sha256']
    for entry in selected['exclusions']:
        assert file_sha256(Path(entry['path']))==entry['sha256']
    checks['all_21_exclusion_sources_unchanged']=True
    conf=pd.read_csv(OUT/'confirmatory/conditions.csv',float_precision='round_trip')
    old=pd.read_csv(OUT/'exploratory_41/conditions_15.csv',float_precision='round_trip')
    assert len(conf)==2*15*8*3 and len(old)==41*15*8*3
    assert len(conf.drop_duplicates(KEYS))==len(conf)
    assert len(old.drop_duplicates(KEYS))==len(old)
    for frame in (conf,old):
        assert set(frame.pipeline)==set(P15)
        assert frame.groupby('dataset_id').size().eq(360).all()
        assert np.isfinite(frame.private_ba).all()
    checks['complete_720_confirmatory_and_14760_exploratory_grids']=True
    prior_path=Path('results/dissertation-portfolio13/holdout/conditions_13.csv')
    # Digest logged before starting any exploratory fits (execution.log).
    assert file_sha256(prior_path)=='571631972451d0117372dd02d20ff4e1a247a78c9c636be865f9b1a6f24ff037'
    prior=pd.read_csv(prior_path,float_precision='round_trip')
    retained=old[old.pipeline.isin(prior.pipeline.unique())][prior.columns]
    pd.testing.assert_frame_equal(prior.sort_values(KEYS).reset_index(drop=True),
        retained.sort_values(KEYS).reset_index(drop=True),check_exact=True,check_dtype=False)
    checks['all_12792_old_13_option_rows_preserved_exactly']=True
    audits=[];times=[]
    for relative,expected in [('confirmatory',720),('exploratory_41/extension',1968)]:
        folder=OUT/relative
        audit=pd.read_csv(folder/'privacy_audit.csv')
        assert len(audit)==expected and audit.ledger_audit_passed.all()
        ledgers=pd.read_csv(folder/'ledgers.csv').set_index('ledger_id').ledger_json.to_dict()
        for key,payload in ledgers.items():
            assert hashlib.sha256(payload.encode()).hexdigest()==key
        assert set(audit.ledger_id)<=set(ledgers)
        commits=[json.loads(p.read_text()) for p in (folder/'datasets').glob('*.json')]
        assert all(c['signature']['executor_sha256']==file_sha256(Path('scripts/confirmatory.py')) for c in commits)
        if relative=='exploratory_41/extension':
            private=OUT/'exploratory_41/private_extension'
            extra=[json.loads(p.read_text()) for p in (private/'datasets').glob('*.json')]
            assert all(c['signature']['executor_sha256']==file_sha256(OUT/'private_extension.py') for c in extra)
            align=json.loads((private/'alignment.json').read_text())
            assert align['passed'] and align['datasets']==3 and align['rows']==54
            assert len(pd.read_csv(private/'alignment_privacy_audit.csv'))==54
            for commit in extra:
                assert all(r['clean_metrics_evaluated'] is False for r in commit['rows'])
                assert all('clean_ba' not in r for r in commit['rows'])
            commits+=extra
            checks['exact_alignment_54_additional_audits_and_no_fabricated_clean_metrics']=True
            assert not list((private/'failure_details').glob('*.json'))
        assert len({c['dataset_id'] for c in commits})==(2 if relative=='confirmatory' else 41)
        assert not list((folder/'failure_details').glob('*.json'))
        audits.append(len(audit));times.append([datetime.fromisoformat(c['completed_at']) for c in commits])
    checks['all_2688_new_fit_ledger_audits_and_payload_hashes']=sum(audits)
    checks['no_model_run_failures']=True
    c=json.loads((OUT/'confirmatory/analysis.json').read_text())
    e=json.loads((OUT/'exploratory_41/analysis.json').read_text())
    assert not c['exploratory'] and e['exploratory']
    assert c['success']==(c['primary']['ci_low']>0)
    assert len(c['secondaries'])==len(e['secondaries'])==4
    assert c['bootstrap_seed']==e['bootstrap_seed']==20261111
    assert c['bootstrap_draws']==e['bootstrap_draws']==20000
    assert datetime.fromisoformat(frozen['completed_at'])<datetime.fromisoformat(selected['completed_at'])<min(times[0])
    assert max(times[0])<datetime.fromisoformat(c['completed_at'])<min(times[1])
    assert max(times[1])<datetime.fromisoformat(e['completed_at'])
    checks['ordered_freeze_selection_confirmatory_analysis_exploratory']=True
    figures=Path('figures/dissertation');details={}
    for stem in ['fig14_holdout_gains_vs_b8u','fig13_holdout_cd','gains_by_epsilon']:
        pdf=figures/f'{stem}.pdf';png=figures/f'{stem}.png'
        assert pdf.read_bytes().startswith(b'%PDF-')
        with Image.open(png) as im:
            im.verify()
        with Image.open(png) as im:details[stem]=dict(png_dimensions=list(im.size),pdf_bytes=pdf.stat().st_size)
        assert datetime.fromtimestamp(pdf.stat().st_mtime,timezone.utc)>datetime.fromisoformat(e['completed_at'])
    checks['all_six_pdf_png_artifacts_valid']=details
    assert (OUT/'SUMMARY.md').is_file()
    (OUT/'verification_final.json').write_text(json.dumps(dict(completed_at=datetime.now(timezone.utc).isoformat(),checks=checks),indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
