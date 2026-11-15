"""Reconcile Phase-C artifacts, regenerate both holdout panels, and numerical audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,ast,re,shutil
import numpy as np,pandas as pd
from scipy.stats import studentized_range,friedmanchisquare
from dpmeta import dissertation_selector as sel
from scripts.dissertation_analysis import fam,mean_ci,paired_distribution,dump
from dpmeta.openml_benchmark import _load_manifest
from dpmeta.figures import _save,_cd_diagram
from scripts import analyze_dissertation_extras as x
import matplotlib.pyplot as plt
OUT=Path('results/dissertation');FIG=Path('figures/dissertation');HOLD=Path('results/holdout-openml')

def cleanfamily(frame):
    frame['family_id']=frame.family_id.fillna('').astype(str).replace('nan','')
    return frame

def reconcile():
    c=pd.read_csv('results/portfolio-984-sgd/conditions_15.csv');a,m,pipes=sel.prepare(c,_load_manifest('data/openml_manifest_984_completed.csv'))
    score=a.set_index(['dataset_id','epsilon','pipeline']).portfolio_regret
    all_scores=a.pivot(index=['dataset_id','epsilon'],columns='pipeline',values='private_ba')
    unique=all_scores.sub(all_scores.max(axis=1),axis=0).abs().lt(1e-12).sum(axis=1).eq(1)
    for path in [OUT/'modes/outer_predictions.csv',OUT/'extras/fivefold_predictions.csv',*sorted((OUT/'modes').glob('*_fold*.csv'))]:
        p=cleanfamily(pd.read_csv(path));p['best_fixed_pipeline']=sel.NB8U
        p['best_fixed_regret']=[score.loc[(d,e,sel.NB8U)] for d,e in zip(p.dataset_id,p.epsilon)]
        p['fixed_top1']=[float(unique.loc[(d,e)] and all_scores.loc[(d,e),sel.NB8U]>=all_scores.loc[(d,e)].max()-1e-12) for d,e in zip(p.dataset_id,p.epsilon)]
        p.to_csv(path,index=False)
    p=pd.read_csv(OUT/'modes/outer_predictions.csv');p=cleanfamily(p);rows=[]
    from scripts.analyze_selector_modes import metadata_cost,pairwise_modes,equal_total_budget
    for mode,g in p.groupby('feature_mode'):
        z=paired_distribution(fam(g,'selector_regret'),fam(g,'best_fixed_regret'));s=z['selector'];ci=z['gain']['mean']
        rows.append(dict(mode=mode,metadata_epsilon=metadata_cost(mode,1.),regret_mean=s['mean'],regret_median=s['median'],regret_q90=s['q90'],regret_cvar90=s['cvar90'],gain_vs_best_fixed=ci['mean'],gain_ci95_low=ci['ci_low'],gain_ci95_high=ci['ci_high']))
    pd.DataFrame(rows).to_csv(OUT/'modes/modes.csv',index=False)
    mode_summary=json.loads((OUT/'modes/summary.json').read_text());mode_summary.update(modes=rows,tau=.02,fallback=sel.NB8U,new_option_features='own calibration + 15 one-hot');dump(OUT/'modes/summary.json',mode_summary)
    modes=tuple(p.feature_mode.unique());pairwise_modes(p,modes,sel.SEED).to_csv(OUT/'modes/pairwise.csv',index=False);equal_total_budget(p,modes,1.,sel.SEED).to_csv(OUT/'modes/equal_total.csv',index=False)
    # All meta-models use family -> dataset -> epsilon weighting, including missing group.
    metas=cleanfamily(pd.read_csv(OUT/'extras/metamodel_predictions.csv'));metas['metamodel']=metas.metamodel.str.replace('(GBM-13)','(GBM-15)',regex=False);metas.to_csv(OUT/'extras/metamodel_predictions.csv',index=False);baseline=fam(a[a.pipeline==sel.NB8U],'portfolio_regret');rows=[]
    for name,g in metas.groupby('metamodel'):
        z=fam(g,'portfolio_regret');rows.append(dict(metamodel=name,regret=z.mean(),top1=fam(g,'top1').mean(),**mean_ci(baseline-z)))
    rows.append(dict(metamodel='b8u congelada',regret=baseline.mean(),top1=np.nan,mean=0.,ci_low=0.,ci_high=0.))
    pd.DataFrame(rows).to_csv(OUT/'extras/metamodels.csv',index=False)
    wide=metas.groupby(['family_id','metamodel']).portfolio_regret.mean().unstack();wide['b8u congelada']=baseline
    assert not wide.isna().any().any()
    x.EXPANDED=True;x.PIPES=pipes;cd=x.fig_metamodels(wide,FIG)
    curve=pd.read_csv(OUT/'extras/topk.csv');x.fig_topk(curve,float(curve.loc[curve.k.eq(len(pipes)),'selector'].iloc[0]),FIG)
    x.fig_importance(pd.read_csv(OUT/'extras/shapley_primitives.csv'),pd.read_csv(OUT/'extras/importance_columns.csv'),FIG)
    extra=json.loads((OUT/'extras/summary.json').read_text());extra['metamodel_cd']=cd;dump(OUT/'extras/summary.json',extra)
    from itertools import combinations
    rows=[]
    for left,right in combinations(wide.columns,2):
        g=wide[right]-wide[left];rows.append(dict(method_a=left,method_b=right,**mean_ci(g),wilcoxon_p=sel.pb._safe_wilcoxon(g.to_numpy())))
    tab=pd.DataFrame(rows);ph=sel.pb._holm_adjust({str(i):p for i,p in enumerate(tab.wilcoxon_p)});tab['p_holm']=[ph[str(i)] for i in range(len(tab))];tab.to_csv(OUT/'extras/metamodel_pairwise.csv',index=False)
    p=cleanfamily(pd.read_csv(OUT/'outer_predictions.csv'))
    empirical=sel.empirical_best(a)
    for name,pipe in [('empirical_fixed',empirical),('old_nb_fixed',sel.NB)]:
        p[name+'_regret']=[score.loc[(d,e,pipe)] for d,e in zip(p.dataset_id,p.epsilon)]
    p.to_csv(OUT/'outer_predictions.csv',index=False)
    summary=json.loads((OUT/'summary.json').read_text())
    summary['empirical_best_fixed_equal_family']=empirical
    summary['empirical_fixed_regret']=paired_distribution(fam(p,'selector_regret'),fam(p,'empirical_fixed_regret'))
    summary['old_nb_fixed_regret']=paired_distribution(fam(p,'selector_regret'),fam(p,'old_nb_fixed_regret'))
    dump(OUT/'summary.json',summary)
    config=json.loads((OUT/'run_config.json').read_text());config.update(tau=.02,fallback=sel.NB8U,reference='b8u frozen; empirical best reported separately',features='own calibration + 15 option one-hot; width 99');dump(OUT/'run_config.json',config)
    rows=[]
    for name,col in [('b8u congelada','best_fixed_regret'),('Constante (melhor por família)','empirical_fixed_regret'),('H27','h27_fixed_regret'),('GNB zCDP','gnb_fixed_regret'),('Seletor GBM-15 LODO','selector_regret'),('Oráculo','oracle_regret'),('Aleatória','random_pipeline_regret'),('Pior fixa','worst_fixed_regret')]:
        z=paired_distribution(fam(p,col),fam(p,'best_fixed_regret'));rows.append(dict(method=name,validation='LODO',regret=z['selector']['mean'],median=z['selector']['median'],q90=z['selector']['q90'],cvar90=z['selector']['cvar90'],gain=z['gain']['mean']['mean'],ci_low=z['gain']['mean']['ci_low'],ci_high=z['gain']['mean']['ci_high']))
    for name,path in [('Seletor GBM-15 5 partições','extras/fivefold_predictions.csv'),('Seletor GBM-15 por epsilon','strategies/per_epsilon_predictions.csv')]:
        f=cleanfamily(pd.read_csv(OUT/path));z=paired_distribution(fam(f,'selector_regret'),fam(f,'best_fixed_regret'));rows.append(dict(method=name,validation='5 folds',regret=z['selector']['mean'],median=z['selector']['median'],q90=z['selector']['q90'],cvar90=z['selector']['cvar90'],gain=z['gain']['mean']['mean'],ci_low=z['gain']['mean']['ci_low'],ci_high=z['gain']['mean']['ci_high']))
    pd.DataFrame(rows).to_csv(OUT/'method_summary.csv',index=False)
    # Independent reproduction of compare_vs_b8u logic, plus explicit missing-group sensitivity.
    rows=[]
    for name in ['lodo_13','lodo_13b','lodo_14','lodo_15_preds','lodo_gbm15']:
        path=Path('results/experiments')/name/'lodo_picks.csv'
        if not path.exists():continue
        picks=pd.read_csv(path);picks=picks[(picks.variant=='GBM residual')&(picks.fallback=='sim')].copy()
        lookup=a.set_index(['dataset_id','epsilon','pipeline']).private_ba
        picks['gain']=[lookup.loc[(d,e,q)]-lookup.loc[(d,e,sel.NB8U)] for d,e,q in zip(picks.dataset_id,picks.epsilon,picks.pipeline)]
        picks['oldgain']=[lookup.loc[(d,e,q)]-lookup.loc[(d,e,sel.NB)] for d,e,q in zip(picks.dataset_id,picks.epsilon,picks.pipeline)]
        for convention in ['published_drop_missing','all_667_groups']:
            g=picks.copy()
            if convention=='all_667_groups':g.fam=g.fam.fillna('')
            by=g.groupby('fam')[['gain','oldgain']].mean();ci=mean_ci(by.gain)
            rows.append(dict(portfolio=name,convention=convention,families=len(by),vs_b8u=ci['mean'],ci_low=ci['ci_low'],ci_high=ci['ci_high'],vs_old_nb=by.oldgain.mean(),loss_gt_1pp=(g.gain<-.01).mean(),cvar90_loss=-np.sort(g.gain)[:len(g)//10].mean()))
    pd.DataFrame(rows).to_csv(OUT/'lodo_portfolio_comparison.csv',index=False)
    shares=a.pivot(index=['dataset_id','epsilon'],columns='pipeline',values='private_ba');wins=shares.sub(shares.max(axis=1),axis=0).abs().lt(1e-12);rows=[]
    for method,col in [('final','selector_pipeline'),('b8u','best_fixed_pipeline')]:
        pp=p.set_index(['dataset_id','epsilon']).reindex(shares.index)[col];unique=wins.sum(axis=1).eq(1)
        hit=np.array([wins.loc[idx,q] for idx,q in pp.items()])
        rows.append(dict(run=method,true_unique_winner=(hit&unique).mean(),tie_inclusive=hit.mean(),conditions=len(wins)))
    pd.DataFrame(rows).to_csv(OUT/'top1_definition_audit.csv',index=False)
    # Unavailable extended transformations: retained as explicitly historical evidence.
    dump(OUT/'unavailable_analyses.json',{'transformations':'15-option alternative-transform scores absent; preserve GBM-13 13-option/199-dataset diagnostic as history only','holdout_clean':'new DP-SGD clean auxiliary fits absent for exploratory 41-family extension; no clean scores imputed','old_private_search':'historical confirmatory RF-12 search baselines remain 12-option runs; not rebranded final','saturation':'descriptors unchanged, GBM-13 historical diagnostic not reinterpreted as 15-option result'})

def holdout_figures():
    labels={'gbm15_ba':'Seletor GBM-15','gbm13_ba':'Seletor GBM-13','rf12_ba':'RF-12','b8u_ba':'NB b8u','old_nb_ba':'NB antigo','h27_ba':'H27'}
    cohorts=[('confirmatory','2 famílias novas: protocolo congelado; poder insuficiente'),('exploratory_41','41 famílias antigas: EXPLORATÓRIO')]
    fig,axes=plt.subplots(2,1,figsize=(13,9))
    cds=[]
    for ax,(folder,title) in zip(axes,cohorts):
        pred=pd.read_csv(HOLD/folder/'predictions.csv');f=pred.groupby('family_id')[list(labels)].mean();r=f.rank(axis=1,ascending=False).mean().rename(index=labels);n,k=f.shape
        cd=studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n));chi,p=friedmanchisquare(*[f[c] for c in f]);_cd_diagram(ax,r,cd)
        lo,hi=ax.get_xlim();ax.set_xlim(lo,max(hi,lo+.02+cd+.4));ax.set_title(title+'\nPostos descritivos; métodos executáveis; Nemenyi α=0,05',fontsize=11)
        cds.append(dict(cohort=folder,families=n,cd=cd,ranks=r.to_dict(),friedman_p=p))
    fig.tight_layout();_save(fig,FIG,'fig13_holdout_cd');dump(OUT/'holdout_cd.json',cds)
    fig,axes=plt.subplots(1,2,figsize=(13,5.8),sharex=True)
    rows=[]
    names={'b8u':'b8u (primária)','H27':'H27','old_NB':'NB antigo','gbm13_selector':'Seletor GBM-13','rf12_selector':'RF-12'}
    for ax,(folder,title) in zip(axes,cohorts):
        analysis=json.loads((HOLD/folder/'analysis.json').read_text());entries=[analysis['primary'],*analysis['secondaries']]
        for y,r in enumerate(entries):
            ax.hlines(y,r['ci_low']*100,r['ci_high']*100,color='#256582');ax.plot(r['gain']*100,y,'o',color='#c04532' if y==0 else '#256582');rows.append(dict(cohort=folder,**r))
        ax.set_yticks(range(len(entries)),[names[r['reference']] for r in entries]);ax.axvline(0,color='gray',ls='--');ax.set_title(title,fontsize=10);ax.set_xlabel('Ganho final vs referência (p.p.; IC95%)');ax.invert_yaxis()
    fig.tight_layout();_save(fig,FIG,'fig14_holdout_gains_vs_h27');pd.DataFrame(rows).to_csv(OUT/'holdout_comparisons.csv',index=False)
    # Keep previous final filename as a consistent alias, in addition to original 14 filenames.
    for ext in ['pdf','png']:shutil.copyfile(FIG/f'fig14_holdout_gains_vs_h27.{ext}',FIG/f'fig14_holdout_gains_vs_b8u.{ext}')
    dump(OUT/'holdout_summary.json',{folder:json.loads((HOLD/folder/'analysis.json').read_text()) for folder,_ in cohorts})

def mdtable(df):
    df=df.drop(columns=['public_schema_json','local_data_path','dataset_sha256'],errors='ignore')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(str(x).replace('|','/').replace('\n',' ') for x in r)+' |' for r in df.itertuples(index=False,name=None)])

def flatten(obj,prefix=''):
    if isinstance(obj,dict):return [(k,v) for key,val in obj.items() for k,v in flatten(val,f'{prefix}.{key}' if prefix else key)]
    if isinstance(obj,list):return [(k,v) for i,val in enumerate(obj) for k,v in flatten(val,f'{prefix}[{i}]')]
    return [(prefix,obj)]

def numbers():
    old=Path('results/dissertation-portfolio13');lines=['# Números da dissertação GBM-13 → GBM-15 — Fase C','',
    'Regerado a partir de conditions_15 e picks LODO GBM-15. BA/regret em escala 0–1; pp = 100 × diferença. Referência limpa original preservada. Os CSVs integrais conservam schema JSON e caminhos de dados; essas colunas não numéricas são omitidas nas transcrições Markdown para legibilidade. Comparador e fallback congelados: b8u; τ=0,02. A configuração e a escolha de tau são exploratórias, posteriores ao protocolo confirmatório inicial, no desenvolvimento. O protocolo confirmatório final foi congelado antes da seleção de novas famílias.','',
    'A disponibilidade de somente duas famílias novas impede confirmação com poder: critério primário não atingido. O holdout antigo de 41 famílias é exclusivamente exploratório. Os ICs são pontuais, não ajustam a seleção adaptativa nem a sobreposição de treinos LODO.','',
    '## Auditoria do denominador semântico','',
    'A lógica compare_vs_b8u.py usa groupby(fam), que descarta os dois IDs sem família na leitura padrão. São 667 grupos nos ajustes, mas 666 na média publicada. Esta regeneração inclui o grupo compartilhado em todas as médias principais: ganho 0,0142630529 [0,0112394134; 0,0174540945] (20.000 bootstrap). A fonte inicial arredondava 0,0142 [0,0113; 0,0174]; fallback_grid registra IC [0,0113; 0,0176] (outra reamostragem). A diferença de estimando é explicitada, sem inventar identidade numérica.','',
    '## Valores centrais: GBM-13 → GBM-15','']
    rows=[]
    for rel in ['summary.json','corpus_summary.json','extras/summary.json','modes/summary.json','strategies/summary.json']:
        if not (OUT/rel).exists():continue
        before=dict(flatten(json.loads((old/rel).read_text()))) if (old/rel).exists() else {}
        for key,val in flatten(json.loads((OUT/rel).read_text())):
            if isinstance(val,(int,float)) and not isinstance(val,bool):rows.append(dict(metrica=rel+'#'+key,GBM-13=before.get(key,'não se aplica'),gbm13_1=val,fonte=rel))
    cross=pd.DataFrame(rows);cross.to_csv(OUT/'numeric_crosswalk.csv',index=False);lines += [mdtable(cross),'','## Tabelas: valores antigos e novos, linha a linha','']
    tree=ast.parse(Path('scripts/write_numbers_portfolio13.py').read_text());tablemap={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='tablemap' for t in node.targets):tablemap=ast.literal_eval(node.value)
    tablemap['tab:corpus']=(['appendix_public_dimensions.csv','corpus_summary.json'],'Tabela própria do artigo: perfil do mesmo corpus; portfólio 15, avaliações 350.280.')
    tablemap['tab:comparacao-portfolios']=(['lodo_portfolio_comparison.csv'],'Nova tabela com referência b8u comum e auditoria do grupo sem nome.')
    for label in ['tab:frameworks','tab:mecanismos-revisao','tab:classificadores-revisao']:
        tablemap[label]=([], 'Tabela conceitual preservada; a linha deste trabalho em frameworks passa de 13 para 15 opções.')
    tablemap['tab:holdout']=(['holdout_comparisons.csv','holdout_summary.json'],'2 novas famílias e 41 antigas em blocos distintos. Holm somente nos quatro secundários, conforme protocolo confirmatório final.')
    coverage=[]
    for label,(sources,note) in tablemap.items():
        lines += ['### '+label,'',note,'']
        oldfolder='artigo' if label=='tab:corpus' else 'latex-dissertacao'
        for tex in sorted((OUT/'source_portfolio13'/oldfolder).rglob('*.tex')):
            for match in re.finditer(r'\\begin\{(?:table|longtable)\}.*?\\end\{(?:table|longtable)\}',tex.read_text(),re.S):
                if '\\label{'+label+'}' in match.group():lines += ['Valores GBM-13 impressos:','','```tex',match.group(),'```','']
        for tex in sorted(Path('../mestrado-papers/latex-dissertacao').rglob('*.tex')):
            for match in re.finditer(r'\\begin\{(?:table|longtable)\}.*?\\end\{(?:table|longtable)\}',tex.read_text(),re.S):
                if '\\label{'+label+'}' in match.group():lines += ['Valores final impressos:','','```tex',match.group(),'```','']
        statuses=[]
        for rel in sources:
            new=OUT/rel;previous=old/rel
            if previous.exists():
                lines += ['Valores GBM-13 — `'+rel+'`:','',mdtable(pd.read_csv(previous)) if previous.suffix=='.csv' else '```json\n'+previous.read_text()+'\n```','']
            if new.exists():
                lines += ['Valores final — `'+rel+'`:','',mdtable(pd.read_csv(new)) if new.suffix=='.csv' else '```json\n'+new.read_text()+'\n```',''];statuses.append('recalculado')
            else:lines += ['`'+rel+'`: não atualizado; diagnóstico histórico GBM-13, sem imputação.',''];statuses.append('histórico/não aplicável')
        if label=='tab:opcoes':lines+=['13 → 15: +softmax__dp_sgd_clip2_lamrule e +softmax__dp_sgd_clip2_lam0003; 100 passos, clip=2, mesma contabilidade H27. Regra pública nε: <100→0,03; [100,1000)→0,01; ≥1000→0,0003.','']
        coverage.append(dict(table=label,sources=';'.join(sources),status=';'.join(statuses)))
    lines += ['## Análises adicionais auditáveis','']
    for p in sorted(OUT.rglob('*.csv')):
        if any(s in p.parts for s in ['source_portfolio13','cache','learning_curve']) or p.name in ['numeric_crosswalk.csv']:continue
        # Fold/individual prediction matrices are linked, not expanded redundantly.
        if any(s in p.name for s in ['predictions','fold','per_dataset']):continue
        lines += ['### '+str(p.relative_to(OUT)),'',mdtable(pd.read_csv(p)),'']
    for p in [OUT/'unavailable_analyses.json',OUT/'holdout_summary.json']:
        lines += ['### '+p.name,'','```json',p.read_text(),'```','']
    def expanded(path,seen=None,root=None):
        root=path.parent if root is None else root
        seen=set() if seen is None else seen
        if path in seen or not path.exists():return ''
        seen.add(path);source=path.read_text()
        def include(match):
            target=path.parent/match.group(1)
            if not target.suffix:target=target.with_suffix('.tex')
            if not target.exists():
                target=root/match.group(1)
                if not target.suffix:target=target.with_suffix('.tex')
            return expanded(target,seen,root) if target.suffix=='.tex' and target.exists() else match.group()
        return re.sub(r'\\(?:input|include)\{([^}]+)\}',include,source)
    inventory=[]
    for folder,targets in [('latex-dissertacao',['main.tex']),('artigo',['main.tex']),('latex-problema-pesquisa',['main.tex','main-resultados.tex'])]:
        for target in targets:
            current=Path('../mestrado-papers')/folder/target
            if not current.exists():continue
            for version,path in [('gbm13',OUT/'source_portfolio13'/folder/target),('final',current)]:
                source=expanded(path)
                labels=re.findall(r'\\label\{(tab:[^}]+)\}',source)
                for label in labels:inventory.append(dict(document=folder,target=target,version=version,table=label))
    pd.DataFrame(inventory).to_csv(OUT/'per_document_table_inventory.csv',index=False)
    lines += ['## Cobertura por documento','','Inventário de tabelas alcançáveis por cada main; inclui tabelas conceituais e cronograma preservados. Tabelas experimentais são regeneradas; as duas exceções históricas são identificadas nos captions.','',mdtable(pd.DataFrame(inventory)),'']
    input_sources=['results/portfolio-984-sgd/conditions_15.csv','results/experiments/lodo_gbm15/lodo_picks.csv','results/experiments/lodo_15_preds/fallback_grid.csv','results/experiments/e6_e7/E6_final_public_rule.json','protocols/CONFIRMATORY_PROTOCOL_FINAL.md','protocols/CONFIRMATORY_PROTOCOL_FINAL.sha256','results/holdout-openml/confirmatory/analysis.json','results/holdout-openml/exploratory_41/analysis.json','data/openml_manifest_984_completed.csv']
    dump(OUT/'input_fingerprints.json',{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in input_sources})
    lines += ['## Fontes e integridade','','```json',(OUT/'input_fingerprints.json').read_text(),'```','']
    (OUT/'NUMBERS.md').write_text('\n'.join(lines));pd.DataFrame(coverage).to_csv(OUT/'table_coverage.csv',index=False)
    files=[p for p in OUT.rglob('*') if p.is_file() and 'cache' not in p.parts and p.name!='artifact_manifest.json']
    dump(OUT/'artifact_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

if __name__=='__main__':
    reconcile();holdout_figures()
    import subprocess,sys
    subprocess.run([sys.executable,'scripts/winner_share_ci.py'],check=True)
    numbers()
