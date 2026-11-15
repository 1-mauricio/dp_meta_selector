"""Final figure QA: accurate lambda label and an unclipped large CD bar.

Consumes completed analysis outputs. No fitting, recommendations, or tests are
changed. Keep the executing script immutable so its resume signatures remain valid.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dpmeta import portfolio_benchmark as pb
from dpmeta.figures import _cd_diagram, _save
from scripts.confirmatory import OUT, MANIFEST, P15, B8, NB, log

def main():
    assert (OUT/'exploratory_41/analysis.json').exists()
    folder=Path('figures/dissertation')
    table=pd.read_csv(OUT/'figure_forest_data.csv').sort_values('gain')
    labels={p:pb.PIPELINE_DISPLAY.get(p,p) for p in P15}
    labels.update({B8:'NB b8u',NB:'NB antigo',pb.SGD_NEW_PIPELINES[0]:'DP-SGD λ regra',
        pb.SGD_NEW_PIPELINES[1]:'DP-SGD λ=0,0003','gbm13_1':'Seletor GBM-15',
        'gbm13':'Seletor GBM-13','rf12':'Seletor RF-12'})
    fig,ax=plt.subplots(figsize=(12,9));y=np.arange(len(table))
    ax.hlines(y,100*table.ci_low,100*table.ci_high,color='#555555')
    ax.scatter(100*table.gain,y,c=['#c04532' if c=='gbm13_1' else '#256582' for c in table.method],zorder=3)
    ax.set_yticks(y,[labels[c] for c in table.method]);ax.axvline(0,color='gray',lw=1)
    ax.set_xlabel('Ganho de acurácia balanceada frente a b8u (p.p.; IC95% por família)')
    ax.set_title('Holdout final: 2 famílias novas');fig.tight_layout()
    _save(fig,folder,'fig14_holdout_gains_vs_b8u')
    pred=pd.read_csv(OUT/'confirmatory/predictions.csv')
    cond=pd.read_csv(OUT/'confirmatory/conditions.csv')
    meta=pd.read_csv(MANIFEST).set_index('dataset_id')
    scores=cond.groupby(['dataset_id','epsilon','pipeline']).private_ba.mean().unstack('pipeline')
    for name in ('gbm13_1','gbm13','rf12'):
        scores[name]=pred.set_index(['dataset_id','epsilon'])[name+'_ba']
    scores=scores.reset_index();scores['family_id']=scores.dataset_id.map(meta.semantic_family_id)
    columns=[*P15,'gbm13_1','gbm13','rf12']
    family=scores.groupby(['family_id','dataset_id'])[columns].mean().groupby('family_id').mean()
    ranks=family.rank(axis=1,ascending=False).mean().rename(index=labels)
    record=json.loads((OUT/'critical_difference.json').read_text())
    assert np.allclose(sorted(ranks),sorted(record['ranks'].values()),rtol=0,atol=0)
    fig,ax=plt.subplots(figsize=(16,10));_cd_diagram(ax,ranks,record['cd'])
    # With N=2, CD can exceed the full observed rank range. The original helper
    # limits its axis to observed ranks and clips the CD bar; preserve its scale.
    lo,hi=ax.get_xlim();ax.set_xlim(lo,max(hi,lo+.02+record['cd']+.4))
    ax.set_title(f"Holdout final: Nemenyi α=0,05; 2 famílias; Friedman p={record['friedman_p']:.3g}\nDescritivo: amostra muito pequena")
    _save(fig,folder,'fig13_holdout_cd')
    record['ranks']=ranks.to_dict()
    (OUT/'critical_difference.json').write_text(json.dumps(record,indent=2))
    log('QA das figuras: legenda de lam0003 corrigida para λ=0,0003; eixo ampliado para mostrar a barra CD inteira. Números, rankings e inferência preservados; script finalize_figures.py arquivado.')

if __name__=='__main__':main()
