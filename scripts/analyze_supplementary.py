"""Corpus, appendix, label-noise oracle and holdout diagnostic tables for Phase A."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from dpmeta import dissertation_selector as sel
from dpmeta import portfolio_benchmark as pb
from dpmeta.openml_benchmark import _load_manifest
from scripts.dissertation_analysis import dump,mean_ci,fam,paired_distribution

OUT=Path('results/dissertation');OUT.mkdir(parents=True,exist_ok=True)
c=pd.read_csv('results/portfolio-984-sgd/conditions_15.csv');a,m,pipes=sel.prepare(c,_load_manifest('data/openml_manifest_984_completed.csv'))
cols=['n_instances','n_features','n_encoded_features','n_nominal_features','n_classes']
rows=[]
for col in cols:
    v=m[col];rows.append(dict(variable=col,min=v.min(),q1=v.quantile(.25),median=v.median(),q3=v.quantile(.75),max=v.max(),mean=v.mean()))
pd.DataFrame(rows).to_csv(OUT/'appendix_public_dimensions.csv',index=False)
m.selection_source.value_counts().rename('datasets').reset_index().assign(percent=lambda x:x.datasets/len(m)*100).to_csv(OUT/'appendix_origins.csv',index=False)
bands=[]
for col,bins,labels in [('n_instances',[0,100,1000,10000,np.inf],['<100','100–999','1000–9999','≥10000']),
         ('n_encoded_features',[0,11,51,201,np.inf],['≤10','11–50','51–200','>200']),
         ('n_classes',[2,3,6,11,np.inf],['2','3–5','6–10','>10']),
         ('nominal_fraction',[0,1e-12,.5,1,1.01],['none','<.5','≥.5 but <1','all'])]:
    v=m.n_nominal_features/m.n_features if col=='nominal_fraction' else m[col]
    ct=pd.cut(v,bins,labels=labels,right=False).value_counts().reindex(labels)
    bands += [dict(variable=col,band=str(b),datasets=int(n),percent=n/len(m)*100) for b,n in ct.items()]
pd.DataFrame(bands).to_csv(OUT/'appendix_bands.csv',index=False)
counts=m.family_id.value_counts();counts.rename('datasets').to_csv(OUT/'appendix_semantic_families.csv')
full=_load_manifest('data/openml_manifest_984_completed.csv');validated=_load_manifest('data/openml_manifest_1000_validated.csv')
validated[~validated.dataset_id.isin(full.dataset_id)].to_csv(OUT/'appendix_cost_exclusions.csv',index=False)
full[~full.dataset_id.isin(m.dataset_id)].to_csv(OUT/'appendix_validation_exclusions.csv',index=False)
missing_semantic=m.semantic_family_id.astype(str).str.strip().str.lower().isin(['','nan','none','<na>'])
dump(OUT/'corpus_summary.json',dict(datasets=len(m),canonical_families=m.dataset_id.nunique(),
     semantic_lodo_families=m.family_id.nunique(),semantic_declared_families=m.loc[~missing_semantic,'semantic_family_id'].nunique(),
     missing_semantic_ids=m[missing_semantic].dataset_id.tolist(),
     semantic_missing_unique_families=m.loc[~missing_semantic,'semantic_family_id'].nunique()+missing_semantic.sum(),
     multi_dataset_families=int(counts.gt(1).sum()),datasets_with_relatives=int(counts[counts.gt(1)].sum()),
     independent_datasets=int(counts.eq(1).sum()),validation_failures=sorted(set(full.dataset_id)-set(m.dataset_id)),
     cost_exclusions=16))
# Cross-repetition oracle, including paired evaluation against the same evaluation repetitions.
w=c.pivot(index=['dataset_id','epsilon','repetition'],columns='pipeline',values='private_ba')[list(pipes)]
reps=sorted(c.repetition.unique());outs=[]
for held in reps:
    for training in [(held,),tuple(r for r in reps if r!=held)]:
        evaluation=tuple(r for r in reps if r not in training)
        train=w.loc[w.index.get_level_values('repetition').isin(training)].groupby(['dataset_id','epsilon']).mean()
        test=w.loc[w.index.get_level_values('repetition').isin(evaluation)].groupby(['dataset_id','epsilon']).mean()
        choice=train.to_numpy().argmax(axis=1)
        selected=test.to_numpy()[np.arange(len(test)),choice]
        out=test.reset_index()[['dataset_id','epsilon']].assign(
            gain=selected-test[sel.NB8U].to_numpy(),old_nb_gain=selected-test[sel.NB].to_numpy(),
            scheme='train1_eval2' if len(training)==1 else 'train2_eval1',split=held)
        outs.append(out)
mean=w.groupby(['dataset_id','epsilon']).mean()
outs.append(mean.reset_index()[['dataset_id','epsilon']].assign(gain=mean.max(axis=1).to_numpy()-mean[sel.NB8U].to_numpy(),
            old_nb_gain=mean.max(axis=1).to_numpy()-mean[sel.NB].to_numpy(),scheme='oracle_mean3_optimistic',split=-1))
frame=pd.concat(outs).merge(m[['dataset_id','family_id']],on='dataset_id');frame.to_csv(OUT/'noisy_oracle_predictions.csv',index=False)
rows=[]
for scheme,g in frame.groupby('scheme'):
    for col in ['gain','old_nb_gain']:
        rows.append(dict(scheme=scheme,reference='b8u' if col=='gain' else 'old_nb',estimand='equal_family',**mean_ci(fam(g,col))))
        rows.append(dict(scheme=scheme,reference='b8u' if col=='gain' else 'old_nb',estimand='equal_dataset',**mean_ci(g.groupby('dataset_id')[col].mean())))
pd.DataFrame(rows).to_csv(OUT/'noisy_oracle.csv',index=False)
