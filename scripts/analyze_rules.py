"""Threshold rules with leave-one-semantic-family-out tuning (exploratory)."""
import argparse,itertools,json
from pathlib import Path
import numpy as np,pandas as pd
from dpmeta import dissertation_selector as sel
from dpmeta import portfolio_benchmark as pb
from dpmeta.openml_benchmark import _load_manifest
from scripts.dissertation_analysis import paired_distribution,dump,fam


def matrix_inputs(a,m,pipes):
    wide=a.pivot(index=['dataset_id','epsilon'],columns='pipeline',values='portfolio_regret')[list(pipes)]
    ids=wide.index.get_level_values('dataset_id');eps=wide.index.get_level_values('epsilon').to_numpy()
    meta=m.set_index('dataset_id').loc[ids]
    ratios={'nε/(d+1)':np.log10(meta.n_instances.to_numpy()*eps/(meta.n_features.to_numpy()+1)),
            'nε/p':np.log10(meta.n_instances.to_numpy()*eps/meta.n_encoded_features.to_numpy())}
    nominal=(meta.n_nominal_features==meta.n_features).to_numpy()
    return wide,meta,ratios,nominal


def choices(config,pipes,ratios,nominal):
    kind,x,t1,t2,low,mid=config;col={p:i for i,p in enumerate(pipes)}
    ch=np.where(ratios[x]<t1,col[low],col[sel.NB8U]) if t2 is None else np.where(
        ratios[x]<t1,col[low],np.where(ratios[x]<t2,col[mid],col[sel.NB8U]))
    if '+nom' in kind:ch=np.where(nominal,col[sel.NB8U],ch)
    return ch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions',type=Path,default=Path('results/portfolio-984-sgd/conditions_15.csv'))
    parser.add_argument('--manifest',type=Path,default=Path('data/openml_manifest_984_completed.csv'))
    parser.add_argument('--output',type=Path,default=Path('results/dissertation/rules'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    a,m,pipes=sel.prepare(pd.read_csv(args.conditions),_load_manifest(args.manifest))
    wide,meta,ratios,nominal=matrix_inputs(a,m,pipes);R=wide.to_numpy();fams=sorted(meta.family_id.unique())
    fi=pd.Categorical(meta.family_id,categories=fams).codes
    # Equal family, then dataset, then ε; equivalent to mean condition within each full-grid family.
    counts=np.bincount(fi);configs=[]
    for x in ratios:
        for t in np.round(np.arange(-1,4.01,.1),2):
            for low in pipes:
                if low==sel.NB8U:continue
                for kind in ['R1','R1+nom']:configs.append((kind,x,float(t),None,low,None))
    mids=[pb.PIPELINES[i] for i in (8,5,7,6,0)]
    for x in ratios:
        for t1,t2 in itertools.combinations(np.round(np.arange(-1,4.01,.25),2),2):
            for low,mid in itertools.permutations(mids,2):configs.append(('R2+nom',x,float(t1),float(t2),low,mid))
    M=np.empty((len(configs),len(fams)))
    for i,c in enumerate(configs):
        ch=choices(c,pipes,ratios,nominal);reg=R[np.arange(len(R)),ch]
        M[i]=np.bincount(fi,reg)/counts
    base=sel.from_lodo('results/experiments/lodo_gbm15/lodo_picks.csv',a).set_index(['dataset_id','epsilon'])
    rows=[];frozen={};predrows=[]
    for kind in ('R1','R1+nom','R2+nom'):
        inds=np.flatnonzero([c[0]==kind for c in configs]);cost=M[inds]
        selected=inds[((cost.sum(axis=1)[:,None]-cost)/(len(fams)-1)).argmin(axis=0)]
        pred=np.empty(len(R));selpipe=np.empty(len(R),dtype=object)
        for familyidx,configidx in enumerate(selected):
            mask=fi==familyidx;ch=choices(configs[configidx],pipes,ratios,nominal)
            pred[mask]=R[np.flatnonzero(mask),ch[mask]]
            selpipe[mask]=np.asarray(pipes)[ch[mask]]
        frame=wide.reset_index()[['dataset_id','epsilon']].assign(family_id=meta.family_id.to_numpy(),regret=pred,pipeline=selpipe)
        frame['selector_regret']=base.selector_regret.reindex(wide.index).to_numpy()
        frame['fixed_regret']=wide[sel.NB8U].to_numpy()
        stats=paired_distribution(fam(frame,'regret'),fam(frame,'fixed_regret'))
        advantage=paired_distribution(fam(frame,'regret'),fam(frame,'selector_regret'))
        row=dict(rule=kind,regret=stats['selector']['mean'],median=stats['selector']['median'],
                  q90=stats['selector']['q90'],cvar90=stats['selector']['cvar90'],
                  **{f'gain_{k}':v for k,v in stats['gain']['mean'].items()},
                  **{f'vs_selector_{k}':v for k,v in advantage['gain']['mean'].items()})
        rows.append(row);predrows.append(frame.assign(rule=kind))
        bestidx=inds[cost.mean(axis=1).argmin()]
        frozen[kind]=dict(config=configs[bestidx],share_lodo_matching_full=float((selected==bestidx).mean()),
                          most_common_lodo=pd.Series([str(configs[i]) for i in selected]).value_counts().to_dict())
    pd.DataFrame(rows).to_csv(args.output/'summary.csv',index=False)
    pd.concat(predrows).to_csv(args.output/'lodo_predictions.csv',index=False)
    dump(args.output/'frozen_rules.json',frozen)
    hold=Path('results/dissertation/holdout/conditions_13.csv')
    if hold.exists():
        ha,hm,_=sel.prepare(pd.read_csv(hold),_load_manifest('data/openml_confirmatory_manifest_41_combined.csv'))
        hw,meta,ratios,nominal=matrix_inputs(ha,hm,pipes);R=hw.to_numpy();out={}
        hp=pd.read_csv('results/dissertation/holdout/predictions.csv').set_index(['dataset_id','epsilon'])
        for name,f in frozen.items():
            ch=choices(f['config'],pipes,ratios,nominal)
            frame=hw.reset_index()[['dataset_id','epsilon']].assign(family_id=meta.family_id.to_numpy(),regret=R[np.arange(len(R)),ch],
                selector_regret=hp.selector_regret.reindex(hw.index).to_numpy(),fixed_regret=hw[sel.NB8U].to_numpy(),
                h27_regret=hw[pb.H27_PIPELINE].to_numpy())
            out[name]={ref:paired_distribution(fam(frame,'regret'),fam(frame,ref)) for ref in ['selector_regret','fixed_regret','h27_regret']}
            inside=frame.dataset_id.isin(hm[hm.n_instances<=m.n_instances.max()].dataset_id)
            for tag,mask in [('inside',inside),('outside',~inside)]:
                g=frame[mask];out[name][tag]=paired_distribution(fam(g,'regret'),fam(g,'fixed_regret'))
        dump(args.output/'holdout_exploratory.json',out)

if __name__=='__main__':main()
