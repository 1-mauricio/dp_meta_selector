"""GBM-13 dissertation analyses; all writes remain in explicitly supplied output directories."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon, binomtest
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from dpmeta import dissertation_selector as sel
from dpmeta import portfolio_benchmark as pb
from dpmeta.openml_benchmark import _load_manifest
from dpmeta.provenance import file_sha256


def dump(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:float(x))+'\n')


def mean_ci(v, seed=sel.SEED):
    m,l,h=pb._bootstrap_ci(np.asarray(v),seed)
    return dict(mean=m,ci_low=l,ci_high=h)


def fam(frame,col):
    frame=frame.assign(family_id=frame.family_id.fillna('').astype(str).replace('nan',''))
    return frame.groupby(['family_id','dataset_id'])[col].mean().groupby('family_id').mean()


def paired_distribution(a,b,seed=sel.SEED):
    """Paired family resampling; tails use ceil(10% N), as the holdout analysis."""
    from dpmeta.external_confirmation import _metric_summary
    if isinstance(a,pd.Series) and isinstance(b,pd.Series):
        if set(a.index) != set(b.index):raise ValueError('Paired statistics require identical family IDs')
        b=b.reindex(a.index)
    a,b=np.asarray(a),np.asarray(b)
    if a.shape != b.shape:raise ValueError('Paired statistics require identical family coverage')
    mean,low,high=pb._bootstrap_ci(b-a,seed)
    tails=pb._bootstrap_tail_gains(a,b,seed,draws=20000)
    return {'selector':_metric_summary(a),'reference':_metric_summary(b),
            'gain':{'mean':dict(mean=mean,ci_low=low,ci_high=high),
                    **{key:dict(mean=v['gain'],ci_low=v['ci95'][0],ci_high=v['ci95'][1])
                       for key,v in tails.items()}},
            'sign_flip_p':pb._sign_flip_p(b-a,seed),
            'wilcoxon_p':float(wilcoxon(b-a).pvalue) if np.any(b!=a) else 1.,
            'wins_ties_losses_zero':[int(((b-a)>1e-9).sum()),int((abs(b-a)<=1e-9).sum()),
                                     int(((b-a)<-1e-9).sum())]}


def base(conditions,manifest,output,lodo):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(conditions);a,m,pipes=sel.prepare(c,_load_manifest(manifest))
    pred=sel.from_lodo(lodo,a)
    worst=a.groupby(['family_id','pipeline']).portfolio_regret.mean().groupby('pipeline').mean().idxmax()
    worstreg=a[a.pipeline==worst].set_index(['dataset_id','epsilon']).portfolio_regret
    pred['worst_fixed_regret']=[worstreg.loc[(d,e)] for d,e in zip(pred.dataset_id,pred.epsilon)]
    pred.to_csv(output/'outer_predictions.csv',index=False)
    m.to_csv(output/'development_manifest.csv',index=False)
    dump(output/'run_config.json',dict(selector='gbm_residual_fallback',tau=.02,
         source_manifest_sha256=file_sha256(Path(manifest)),source_conditions_sha256=file_sha256(Path(conditions)),
         source_split_seed=20260917,repeats=3,seed=sel.SEED,pipelines=pipes,
         estimand='peso igual por familia semantica; dataset; epsilon',
         semantic_missing_policy="astype(str), reproduz LODO existente; NaN forma grupo compartilhado"))
    rows=[]
    for p,g in a.groupby('pipeline'):
        rows.append(dict(pipeline=p,**{f'{col}_{k}':v for col in ['private_ba','clean_ba','portfolio_regret']
                      for k,v in mean_ci(fam(g,col)).items()}))
    pd.DataFrame(rows).to_csv(output/'option_performance.csv',index=False)
    summary={}
    for col in ['best_fixed_regret','h27_fixed_regret','gnb_fixed_regret','oracle_regret']:
        summary[col]=paired_distribution(fam(pred,'selector_regret'),fam(pred,col))
    sf=fam(pred,'selector_regret');bf=fam(pred,'best_fixed_regret')
    pd.DataFrame({'selector':sf,'best_fixed':bf,'gain':bf-sf}).to_csv(output/'family_scores.csv')
    summary.update(datasets=len(m),families=m.family_id.nunique(),options=len(pipes),
          conditions=a.groupby(['dataset_id','epsilon']).ngroups,evaluations=len(c),
          best_fixed=sel.best_fixed(a),worst_fixed=worst,
          best_fixed_ba=mean_ci(fam(a[a.pipeline==sel.best_fixed(a)],'private_ba')),
          selector_ba=mean_ci(fam(pred,'selector_ba')),
          selection_shares=pred.selector_pipeline.value_counts(normalize=True).to_dict(),
          top1_strict=float(pred.selector_top1.mean()),fixed_top1_strict=float(pred.fixed_top1.mean()))
    gain=bf-sf
    ds=pred.groupby('dataset_id')[['selector_regret','best_fixed_regret','selector_top1','fixed_top1']].mean()
    ds_gain=ds.best_fixed_regret-ds.selector_regret
    summary['practical']=dict(rank_biserial=pb._rank_biserial(gain.to_numpy()),
        median_dataset_gain=float(ds_gain.median()),dataset_wins=int((ds_gain>1e-9).sum()),
        dataset_ties=int((abs(ds_gain)<=1e-9).sum()),dataset_losses=int((ds_gain<-1e-9).sum()),
        constant_best_fixed_datasets=int(pred.assign(_fixed=pred.selector_pipeline==pred.best_fixed_pipeline).groupby('dataset_id')._fixed.all().sum()),
        mcnemar_dataset=pb._mc_nemar(ds.selector_top1.to_numpy()>.5,ds.fixed_top1.to_numpy()>.5),
        conditions_worse_over_1pp=float(((pred.selector_regret-pred.best_fixed_regret)>.01).mean()),
        cvar90_condition_loss=float(np.sort((pred.selector_regret-pred.best_fixed_regret).to_numpy())[-max(1,len(pred)//10):].mean()),
        top1_argmax=float(pred.selector_top1_argmax.mean()),top1_tie_inclusive=float(pred.selector_top1_tie_inclusive.mean()))
    summary['references']={col:dict(mean=float(fam(pred,col).mean())) for col in
          ['random_pipeline_regret','worst_pipeline_regret','worst_fixed_regret','oracle_regret']}
    eps=[]
    for e,g in pred.groupby('epsilon'):
        row=dict(epsilon=e,datasets=g.dataset_id.nunique(),families=g.family_id.nunique(),
                 top1=float(g.selector_top1.mean()),fixed_top1=float(g.fixed_top1.mean()))
        for col in ['selector_regret','best_fixed_regret','h27_fixed_regret','oracle_regret','selector_ba','worst_fixed_regret','random_pipeline_regret','worst_pipeline_regret']:
            row.update({f'{col}_{k}':v for k,v in mean_ci(fam(g,col)).items()})
        row.update({f'gain_{k}':v for k,v in mean_ci(fam(g,'best_fixed_regret')-fam(g,'selector_regret')).items()})
        eps.append(row)
    pd.DataFrame(eps).to_csv(output/'regret_by_epsilon.csv',index=False)
    wide=a.pivot(index=['dataset_id','epsilon'],columns='pipeline',values='private_ba')
    w=wide.sub(wide.max(axis=1),axis=0).abs().lt(1e-12); shares=w.div(w.sum(axis=1),axis=0)
    shares.groupby('epsilon').mean().T.assign(Total=shares.mean()).to_csv(output/'win_share_by_epsilon.csv')
    n_winners=w.groupby('dataset_id').any().sum(axis=1)
    common=(w.sum(axis=1).eq(1).groupby('dataset_id').all() & w.groupby('dataset_id').sum().max(axis=1).eq(8))
    r=a.pivot_table(index='dataset_id',columns='pipeline',values='portfolio_regret')
    chi,p=friedmanchisquare(*[r[col] for col in r])
    # Both descriptive dataset and inferential family ranks are reported explicitly.
    rf=a.groupby(['family_id','pipeline']).portfolio_regret.mean().unstack()
    chif,pf=friedmanchisquare(*[rf[col] for col in rf]);n,k=rf.shape
    cd=studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n))
    ranked=rf.rank(axis=1).mean()
    pd.DataFrame({'mean_rank':ranked}).to_csv(output/'portfolio_ranks_family.csv')
    npair=[]
    for i,p1 in enumerate(pipes):
        for p2 in pipes[i+1:]:
            diff=abs(ranked[p1]-ranked[p2]);q=diff/np.sqrt(k*(k+1)/(6*n))*np.sqrt(2)
            npair.append(dict(option1=p1,option2=p2,rank_difference=diff,p_nemenyi=studentized_range.sf(q,k,np.inf)))
    pd.DataFrame(npair).to_csv(output/'portfolio_nemenyi.csv',index=False)
    spread=wide.max(axis=1)-wide.min(axis=1); sortedba=np.sort(wide,axis=1)
    clean=a.groupby('dataset_id').best_clean_portfolio_ba_gbm13.mean();chance=1/m.set_index('dataset_id').n_classes
    near=clean-chance<=.02
    summary['h1']=dict(tied_conditions=int(w.sum(axis=1).gt(1).sum()),
          tied_fraction=float(w.sum(axis=1).gt(1).mean()),all_tied=int(w.all(axis=1).sum()),
          constant_unique_datasets=int(common.sum()),median_distinct_winners=float(n_winners.median()),
          friedman_dataset_chi2=chi,friedman_dataset_p=p,kendall_W_dataset=chi/(len(r)*(k-1)),
          friedman_family_chi2=chif,friedman_family_p=pf,kendall_W_family=chif/(n*(k-1)),nemenyi_cd=cd,
          spread_median=float(spread.median()),spread_p90=float(spread.quantile(.9)),
          spread_gt_10pp=float(spread.gt(.1).mean()),top_two_gap_median=float(np.median(sortedba[:,-1]-sortedba[:,-2])),
          near_chance_clean_datasets=int(near.sum()),near_chance_clean_fraction=float(near.mean()),
          near_chance_private_fraction=float((wide.max(axis=1)-chance.reindex(wide.index.get_level_values(0)).to_numpy()<=.02).mean()))
    ds=a.groupby(['dataset_id','pipeline']).agg(private_ba=('private_ba','mean'),clean_ba=('clean_ba','mean')).reset_index()
    ds=ds.pivot(index='dataset_id',columns='pipeline',values='private_ba')
    head=(wide.max(axis=1)-wide[sel.best_fixed(a)]).groupby('dataset_id').mean()
    bywin=shares.groupby('dataset_id').sum()
    tally=bywin.stack().rename('wins').reset_index().merge(a.groupby(['dataset_id','pipeline'],as_index=False).private_ba.mean())
    winner=tally.sort_values(['wins','private_ba'],ascending=False).groupby('dataset_id').pipeline.first()
    appendix=m.set_index('dataset_id').join(pd.DataFrame(dict(winner=winner,distinct_winners=n_winners,
             constant_unique=common,headroom=head,best_private=ds.max(axis=1),
             best_clean=clean,best_clean_reference_12=a.groupby('dataset_id').best_clean_portfolio_ba.mean(),best_fixed_private=ds[sel.best_fixed(a)])))
    appendix.to_csv(output/'appendix_datasets.csv')
    winner.value_counts().rename('datasets').to_csv(output/'appendix_modal_winners.csv')
    n_winners.value_counts().sort_index().rename('datasets').to_csv(output/'appendix_distinct_winners.csv')
    summary['headroom']=dict(mean=float(head.mean()),median=float(head.median()),below_005=int(head.lt(.005).sum()),
                             above_10=int(head.gt(.1).sum()),captured_family=float((bf-sf).mean()/
                                   (bf-fam(pred,'oracle_regret')).mean()))
    summary['repetition_sd']=c.groupby(['dataset_id','epsilon','pipeline']).private_ba.std().agg(['mean','median']).to_dict()
    summary['without_epsilon16']=mean_ci(fam(pred[pred.epsilon!=16],'best_fixed_regret')-
                                        fam(pred[pred.epsilon!=16],'selector_regret'))
    summary['size_groups']={}
    for name,mask in [('n_ge200',m.n_instances>=200),('n_lt200',m.n_instances<200),('n_lt100',m.n_instances<100),
                      ('informative',~m.dataset_id.map(near)),('near_chance',m.dataset_id.map(near))]:
        block=pred[pred.dataset_id.isin(m[mask].dataset_id)]
        summary['size_groups'][name]=dict(datasets=block.dataset_id.nunique(),
                  **mean_ci(fam(block,'best_fixed_regret')-fam(block,'selector_regret')))
    dump(output/'summary.json',summary)
    return a,m,pred


def publication(conditions,manifest,output,figures,lodo):
    import matplotlib.pyplot as plt
    from dpmeta.figures import _save,_cd_diagram
    from scripts import analyze_dissertation_extras as x
    from scripts import regen_tied_winner_figures as tw
    output,figures=Path(output),Path(figures);figures.mkdir(parents=True,exist_ok=True)
    a,m,pred=base(conditions,manifest,output,lodo)
    pipes=tuple(sorted(a.pipeline.unique()));x.PIPES=pipes;x.EXPANDED=True
    tw.x.PIPES=pipes;tw.x.EXPANDED=True
    tw.PT[pb.NB8U_PIPELINE]='Laplace · NB b8u'
    tw.fig_winners_pt(a,figures)
    # LODO curves: semantic-family averages; strict unique-winner top-1 is explicit.
    rows=[]
    worst=a.groupby(['family_id','pipeline']).portfolio_regret.mean().groupby('pipeline').mean().idxmax()
    labels={'selector_regret':'GBM residual + fallback (LODO)','best_fixed_regret':'b8u congelada',
         'h27_fixed_regret':'H27','gnb_fixed_regret':'GNB zCDP','oracle_regret':'Oráculo',
         'random_pipeline_regret':'Aleatória','worst_pipeline_regret':'Pior por condição','worst_fixed_regret':'Pior fixa'}
    for e,g in pred.groupby('epsilon'):
        rows.append(dict(epsilon=e,**{col:fam(g,col).mean() for col in labels},
            selector_top1=g.selector_top1.mean(),fixed_top1=g.fixed_top1.mean()))
    t=pd.DataFrame(rows).set_index('epsilon');fig,ax=plt.subplots(1,2,figsize=(14,5))
    for col,label in labels.items():ax[0].plot(range(len(t)),t[col]*100,marker='o',label=label)
    ax[1].plot(range(len(t)),t.selector_top1*100,marker='o',label='Seletor GBM-15')
    ax[1].plot(range(len(t)),t.fixed_top1*100,marker='s',label='b8u congelada')
    for i in range(2):ax[i].set_xticks(range(len(t)),[str(e) for e in t.index]);ax[i].set_xlabel('ε');ax[i].legend(fontsize=8)
    ax[0].set_ylabel('Arrependimento (p.p.; peso por família)');ax[1].set_ylabel('Top-1 estrito (% de condições)')
    _save(fig,figures,'fig2_regret_and_top1_curves_12opcoes')
    # Figures 3 and 4 follow the original five-fold scope, grouped semantically.
    fp=output/'extras/fivefold_predictions.csv'
    if not fp.exists():
        with threadpool_limits(limits=1):s,five=sel.grouped(pd.read_csv(conditions),m)
        five.to_csv(output/'fivefold_predictions.csv',index=False)
    else:five=pd.read_csv(fp)
    schema_path=output/'modes/outer_predictions.csv'
    schema_parts=sorted((output/'modes').glob('public_schema_only_fold*.csv'))
    if schema_parts:schema=pd.concat([pd.read_csv(p) for p in schema_parts])
    elif schema_path.exists():schema=pd.read_csv(schema_path).query("feature_mode=='public_schema_only'")
    else:
        with threadpool_limits(limits=1):_,schema=sel.grouped(pd.read_csv(conditions),m,mode='public_schema_only')
    wide=pd.DataFrame({'GBM residual + fallback':fam(five,'selector_regret'),
          'Só schema (GBM-15)':fam(schema,'selector_regret'),'b8u congelada':fam(five,'best_fixed_regret'),
          'H27':fam(five,'h27_fixed_regret'),'GNB zCDP':fam(five,'gnb_fixed_regret')})
    n,k=wide.shape;chi,p=friedmanchisquare(*[wide[c] for c in wide]);ranks=wide.rank(axis=1).mean()
    cd=studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n))
    fig,ax=plt.subplots(figsize=(12,4));_cd_diagram(ax,ranks,cd)
    ax.set_title(f'5 partições por família semântica; Nemenyi α=0,05; N={n}; Friedman p={p:.2g}')
    _save(fig,figures,'fig3_critical_difference_nemenyi_12opcoes')
    dump(output/'methods_nemenyi.json',dict(n=n,cd=cd,friedman_chi2=chi,friedman_p=p,ranks=ranks.to_dict()))
    wide.to_csv(output/'methods_fivefold_family.csv')
    gain=(wide['b8u congelada']-wide['GBM residual + fallback'])*100
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].hist(gain,bins=50,color='#2a78d6');axes[0].axvline(0,color='k',ls='--')
    axes[0].set(xlabel='Ganho vs b8u (p.p.)',ylabel='Famílias semânticas',title='5 partições: redução pareada do arrependimento')
    for col in wide:
        vals=np.sort(wide[col])*100;axes[1].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=col)
    axes[1].set(xlabel='Arrependimento médio por família (p.p.)',ylabel='Distribuição acumulada');axes[1].legend(fontsize=8)
    _save(fig,figures,'fig4_regret_distribution_and_headroom_12opcoes')


def holdout(dev_conditions,conditions,dev_manifest,manifest,output,figures):
    from scripts import analyze_holdout as h
    output,figures=Path(output),Path(figures);output.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    da,dm,pipes=sel.prepare(pd.read_csv(dev_conditions),_load_manifest(dev_manifest))
    ta,tm,_=sel.prepare(pd.read_csv(conditions),_load_manifest(manifest));ta=sel.ordered(ta,pipes)
    with threadpool_limits(limits=1):
        model=sel.fit(da,dm,pipes);chosen=sel.pick(ta,model.predict(sel.feature_matrix(ta,tm,pipes)))
    import joblib
    joblib.dump(model,output/'frozen_gbm13_selector.joblib')
    dump(output/'frozen_selector_config.json',dict(selector='GBM residual + fallback',seed=sel.SEED,
        tau=sel.TAU,hyperparameters=model.estimator.get_params(),pipelines=pipes,
        best_fixed=model.fixed,dev_conditions_sha256=file_sha256(dev_conditions),
        weighting='sum of row weights = number of semantic training families',
        status='EXPLORATÓRIO: escolhido depois de abrir o holdout'))
    pred=sel.predictions(ta,chosen,da)
    pred.to_csv(output/'predictions.csv',index=False)
    # Preserve executable RF-12 holdout methods as historical frozen comparators.
    old=pd.read_csv('results/openml-holdout-41/method_regret.csv')
    old=old[~old.method.isin(['random','oracle','fixed_nb'])].copy()
    old.loc[old.method=='selector','method']='selector_rf12'
    old['family_id']=old.dataset_id.map(tm.set_index('dataset_id').family_id)
    long=[old]
    for method,p in [('fixed_b8u',sel.NB8U),('fixed_old_nb',sel.NB)]:
        long.append(ta[ta.pipeline==p][['dataset_id','epsilon','family_id','portfolio_regret']].rename(
            columns={'portfolio_regret':'regret'}).assign(method=method))
    long.append(pred[['dataset_id','epsilon','family_id','selector_regret']].rename(
        columns={'selector_regret':'regret'}).assign(method='selector'))
    for method,op in [('oracle','min'),('random','mean')]:
        long.append(ta.groupby(['family_id','dataset_id','epsilon']).portfolio_regret.agg(op).rename('regret').reset_index().assign(method=method))
    long=pd.concat(long,ignore_index=True);long.to_csv(output/'method_regret.csv',index=False)
    wide=h.per_family(long)
    h.LABELS.update(selector='GBM residual + fallback GBM-13 (EXPLORATÓRIO)',selector_rf12='RF-12 congelada',
                    fixed_b8u='Melhor fixa dev (NB b8u)',fixed_old_nb='NB antigo')
    # Same bootstrap, sign-flip, Holm and tail definitions as analyze_holdout.
    extra=[]
    for reference in ['fixed_h27','fixed_b8u','fixed_old_nb','selector_rf12']:
        for method in wide:
            if method==reference or method in h.DESCRIPTIVE:continue
            diff=(wide[reference]-wide[method]).to_numpy()
            from dpmeta.external_confirmation import _metric_summary
            summary=_metric_summary(wide[method].to_numpy())
            extra.append(dict(method=method,label=h.LABELS[method],reference=reference,
                 regret=summary['mean'],median=summary['median'],q90=summary['q90'],cvar90=summary['cvar90'],
                 gain=diff.mean(),ci_low=mean_ci(diff)['ci_low'],ci_high=mean_ci(diff)['ci_high'],
                 sign_flip_p=pb._sign_flip_p(diff,sel.SEED)))
    table=pd.DataFrame(extra)
    # All GBM-13 comparisons are exploratory; no exempt primary contrast.
    holm=pb._holm_adjust({str(i):p for i,p in enumerate(table.sign_flip_p)})
    table['p_holm']=[holm[str(i)] for i in range(len(table))]
    table.to_csv(output/'comparisons.csv',index=False)
    strata={}
    for name,mask in [('inside_dev_range',tm.n_instances.between(dm.n_instances.min(),dm.n_instances.max())),
                      ('outside_dev_range',~tm.n_instances.between(dm.n_instances.min(),dm.n_instances.max()))]:
        ids=tm.loc[mask,'dataset_id'];block=h.per_family(long,ids)
        entries={ref:paired_distribution(block.selector,block[ref]) for ref in ['fixed_b8u','fixed_h27','fixed_old_nb','selector_rf12']}
        strata[name]=dict(datasets=len(ids),families=len(block),comparisons=entries)
    stats={ref:paired_distribution(wide.selector,wide[ref]) for ref in ['fixed_b8u','fixed_h27','fixed_old_nb','selector_rf12']}
    # Explicit exploratory status appears on the figures themselves.
    import matplotlib.pyplot as plt
    from dpmeta.figures import _save,_cd_diagram
    data=wide.drop(columns=[c for c in h.DESCRIPTIVE if c in wide]);n,k=data.shape
    cd=studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n))
    chi,p=friedmanchisquare(*[data[c] for c in data]);ranks=data.rank(axis=1).mean()
    fig,ax=plt.subplots(figsize=(13,6));_cd_diagram(ax,ranks.rename(index=h.LABELS),cd)
    ax.set_title(f'Holdout GBM-13 EXPLORATÓRIO — N={n} famílias; Nemenyi α=0,05; Friedman p={p:.2g}')
    _save(fig,figures,'fig13_holdout_cd')
    t=table[(table.reference=='fixed_h27')].sort_values('gain');fig,ax=plt.subplots(figsize=(11,7))
    y=np.arange(len(t));ax.errorbar(t.gain*100,y,xerr=np.vstack([(t.gain-t.ci_low)*100,(t.ci_high-t.gain)*100]),fmt='o')
    ax.set_yticks(y,t.label);ax.axvline(0,color='k',ls='--');ax.set_xlabel('Ganho vs H27 (p.p.; IC95% por família)')
    ax.set_title('Holdout GBM-13 EXPLORATÓRIO — mudanças escolhidas após abertura')
    _save(fig,figures,'fig14_holdout_gains_vs_h27')
    dump(output/'summary.json',dict(status='EXPLORATÓRIO: mudanças GBM-13 escolhidas após abrir holdout',
        completed_datasets=len(tm),completed_families=tm.family_id.nunique(),development_best_fixed=sel.best_fixed(da),
        comparisons=stats,strata=strata,nemenyi=dict(cd=cd,ranks=ranks.to_dict(),friedman_chi2=chi,friedman_p=p),
        historical_comparators='Resultados congelados RF-12 do holdout; buscas continuam com 12 opções',
        conditions_sha256=file_sha256(Path(conditions)),dev_conditions_sha256=file_sha256(dev_conditions)))


def size_analysis(predictions,manifest,output,conditions):
    from scripts.analyze_size_sensitivity import SIZE_BINS,SIZE_LABELS
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    m=sel.semantic_manifest(_load_manifest(manifest));p=pd.read_csv(predictions)
    ids=set(pd.read_csv(conditions,usecols=['dataset_id']).dataset_id)
    if set(p.dataset_id) != ids:raise ValueError('Prediction/condition dataset IDs differ')
    if 'feature_mode' in p:p=p[p.feature_mode=='public_plus_dp']
    p=p.merge(m[['dataset_id','n_instances','n_classes']],on='dataset_id',validate='many_to_one')
    p['family_id']=p.family_id.astype(str)
    ds=p.groupby(['family_id','dataset_id','n_instances'],as_index=False).agg(
        selector_regret=('selector_regret','mean'),fixed_regret=('best_fixed_regret','mean'))
    ds['gain']=ds.fixed_regret-ds.selector_regret
    ds['size_band']=pd.cut(ds.n_instances,bins=SIZE_BINS,labels=SIZE_LABELS,right=False)
    rows=[]
    for band,g in ds.groupby('size_band',observed=True):
        rows.append(dict(size_band=str(band),n_datasets=g.dataset_id.nunique(),families=g.family_id.nunique(),
            **mean_ci(g.groupby('family_id').gain.mean()),mean_selector_regret=fam(g,'selector_regret').mean(),
            mean_fixed_regret=fam(g,'fixed_regret').mean()))
    pd.DataFrame(rows).to_csv(output/'sensitivity_by_size.csv',index=False)
    ds.to_csv(output/'sensitivity_per_dataset.csv',index=False)
    # Splits and classes are unchanged: retain aligned original coverage for retained datasets.
    coverage=pd.read_csv('results/portfolio-984-shared-selector/small_dataset_coverage.csv')
    coverage=coverage[coverage.dataset_id.isin(ds.dataset_id)]
    coverage.to_csv(output/'small_dataset_coverage.csv',index=False)
    dump(output/'coverage_provenance.json',dict(source='results/portfolio-984-shared-selector/small_dataset_coverage.csv',
         reason='mesmos datasets, schema, seeds e splits; descritores independem do seletor',
         source_sha256=file_sha256(Path('results/portfolio-984-shared-selector/small_dataset_coverage.csv')),
         incomplete_test_classes=int((coverage.test_class_coverage<1).sum()),splits=len(coverage)))


def descriptors(output):
    from dpmeta.meta_features import PRIVATE_META_COLUMNS,DP_META_COLUMNS
    target=Path(output).parent/'training_descriptors.csv'
    cols=['dataset_id',*PRIVATE_META_COLUMNS,*DP_META_COLUMNS]
    if target.exists():return pd.read_csv(target)
    # Metadata release is repeated across options, but the canonical selector uses the
    # first aggregated row of each dataset. Read only that vector, without rerunning DP.
    parts=[];seen=set()
    for chunk in pd.read_csv('results/portfolio-984-current/conditions.csv',usecols=cols,chunksize=20000):
        first=chunk[~chunk.dataset_id.isin(seen)].drop_duplicates('dataset_id')
        parts.append(first);seen.update(first.dataset_id)
    result=pd.concat(parts,ignore_index=True);result.to_csv(target,index=False)
    return result


def modes_analysis(conditions,manifest,output,folds=5,repeats=5):
    from scripts.analyze_selector_modes import pairwise_modes,equal_total_budget,metadata_cost
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    a,m,pipes=sel.prepare(pd.read_csv(conditions),_load_manifest(manifest))
    a=a.merge(descriptors(output),on='dataset_id',validate='many_to_one');a=sel.ordered(a,pipes)
    modes=('public_schema_only','public_plus_dp','dp_research_rich','research_privileged','research_rich')
    ids=m.sort_values('dataset_id');splits=list(GroupKFold(folds).split(ids,groups=ids.family_id))
    allpred=[];groups_rows=[];tables=[]
    rng=np.random.default_rng(sel.SEED)
    for mode in modes:
        print('MODE',mode,flush=True)
        names=sel.feature_names(pipes,mode);x=sel.feature_matrix(a,m,pipes,mode)
        groupcols={}
        for i,(_,g) in enumerate(names):
            if g!='option':groupcols.setdefault(g,[]).append(i)
        groupcols['all_calibration']=[i for g,cc in groupcols.items() if g.startswith('calib_') for i in cc]
        groupcols['all_meta_features']=[i for i,(_,g) in enumerate(names) if g!='option']
        changes={g:[] for g,cc in groupcols.items() if cc};predparts=[]
        nb=a[a.pipeline==sel.NB].set_index(['dataset_id','epsilon']).private_ba
        y=a.private_ba.to_numpy()-nb.loc[list(zip(a.dataset_id,a.epsilon))].to_numpy()
        for fold,(tr,te) in enumerate(splits):
            mask=a.dataset_id.isin(ids.iloc[tr].dataset_id).to_numpy();train=a[mask];test=a[~mask]
            checkpoint=output/f'{mode}_fold{fold}.csv'
            with threadpool_limits(limits=1):
                model=sel.FallbackPredictor(sel.gbm().fit(x[mask],y[mask],sample_weight=sel.weights(train)),
                                            pipes,sel.NB8U)
                testx=x[~mask];chosen=sel.pick(test,model.predict(testx))
                pred=sel.predictions(test,chosen,train,fold).assign(feature_mode=mode)
                predparts.append(pred);pred.to_csv(checkpoint,index=False)
                baseline=fam(chosen,'portfolio_regret')
                for g,columns in groupcols.items():
                    if not columns:continue
                    delta=[]
                    for _ in range(repeats):
                        perm=testx.copy()
                        for e in test.epsilon.unique():
                            rows=np.flatnonzero(test.epsilon.to_numpy()==e)
                            block=perm[rows].reshape(-1,len(pipes),perm.shape[1])
                            block[:,:,columns]=block[rng.permutation(len(block))][:,:,columns]
                            perm[rows]=block.reshape(len(rows),-1)
                        chosen_perm=sel.pick(test,model.predict(perm))
                        delta.append(fam(chosen_perm,'portfolio_regret')-baseline)
                    changes[g].append(sum(delta)/repeats)
            print(mode,'fold',fold,flush=True)
        mp=pd.concat(predparts);allpred.append(mp)
        z=paired_distribution(fam(mp,'selector_regret'),fam(mp,'best_fixed_regret'))
        s=z['selector'];gain=z['gain']['mean']
        tables.append(dict(mode=mode,metadata_epsilon=metadata_cost(mode,1.),dp_guarantee=mode not in
              ('research_privileged','research_rich'),regret_mean=s['mean'],regret_median=s['median'],
              regret_q90=s['q90'],regret_cvar90=s['cvar90'],gain_vs_best_fixed=gain['mean'],
              gain_ci95_low=gain['ci_low'],gain_ci95_high=gain['ci_high']))
        for g,parts in changes.items():
            vals=pd.concat(parts)
            groups_rows.append(dict(mode=mode,group=g,n_features=len(groupcols[g]),
                                    regret_increase=vals.mean(),ci95_low=mean_ci(vals)['ci_low'],
                                    ci95_high=mean_ci(vals)['ci_high'],impurity_share=np.nan))
        pd.DataFrame(tables).to_csv(output/'modes.csv',index=False)
        pd.DataFrame(groups_rows).to_csv(output/'importance_groups.csv',index=False)
    pred=pd.concat(allpred);pred.to_csv(output/'outer_predictions.csv',index=False)
    pairwise_modes(pred,modes,sel.SEED).to_csv(output/'pairwise.csv',index=False)
    equal_total_budget(pred,modes,1.,sel.SEED).to_csv(output/'equal_total.csv',index=False)
    dump(output/'summary.json',dict(selector='GBM residual + fallback',families=m.family_id.nunique(),
         folds=folds,permutation_repeats=repeats,impurity='não definida para HistGradientBoostingRegressor',
         metadata_epsilon=1.,source_descriptors='results/portfolio-984-current/conditions.csv',
         new_option_features='calibração própria; one-hot de 15 opções; fallback b8u; tau=.02',modes=tables))
