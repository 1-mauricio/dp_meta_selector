"""Dataset-paired bootstrap, integrity checks and Portuguese E6/E7 report."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.e6_e7 import (OUT, LAMBDAS, EPS6, EPS7, STEPS, CLIPS, audit,
                          ledger_from_dict, BIN_EDGES, fit_rule, bin_index)
from experiments.summarize_e1_e4_e5 import bootstrap, table

META = ['dataset_id','n_instances','n_tercile','combined_n_tercile','feature_group','sample_group']


def paired_summary(frame, label, variants=()):
    # First reps and eps within a dataset, then resample whole datasets.
    reports=[]
    scopes=[('geral',frame), *[(f'epsilon={e:g}',frame[frame.epsilon.eq(e)]) for e in sorted(frame.epsilon.unique())],
            *[(f'tercil={t}',frame[frame.combined_n_tercile.eq(t)]) for t in ('C1','C2','C3')],
            *[(f'faixa={t}',frame[frame.n_tercile.eq(t)]) for t in ('T1','T2','T3','LARGE')],
            *[(f'tipo={t}',frame[frame.feature_group.eq(t)]) for t in ('all_nominal','has_numeric')],
            *[(f'{t}/epsilon={e:g}',frame[frame.combined_n_tercile.eq(t)&frame.epsilon.eq(e)])
              for t in ('C1','C2','C3') for e in sorted(frame.epsilon.unique())],
            *[(f'LARGE/epsilon={e:g}',frame[frame.sample_group.eq('large15')&frame.epsilon.eq(e)])
              for e in sorted(frame.epsilon.unique())]]
    for scope,selected in scopes:
        groups=selected.groupby(list(variants),sort=True) if variants else [((),selected)]
        for key,block in groups:
            key=key if isinstance(key,tuple) else (key,)
            ds=block.groupby('dataset_id',as_index=False).agg(
                ba=('ba','mean'),baseline_ba=('baseline_ba','mean'),delta=('delta','mean'))
            if not len(ds): continue
            avg,lo,hi=bootstrap(ds.delta,(label,scope,key))
            reports.append(dict(scope=scope,**dict(zip(variants,key)),n_datasets=len(ds),
                                ba=ds.ba.mean(),baseline_ba=ds.baseline_ba.mean(),
                                delta=avg,ci95_low=lo,ci95_high=hi,
                                worse_gt_1pp=int(ds.delta.lt(-.01).sum()),
                                share_worse_gt_1pp=ds.delta.lt(-.01).mean()))
    result=pd.DataFrame(reports)
    result.to_csv(OUT/f'{label}_bootstrap_summary.csv',index=False)
    return result


def cell(r):
    return f'{r.delta*100:+.2f} [{r.ci95_low*100:+.2f}; {r.ci95_high*100:+.2f}]'


def fmt_summary(f, scopes):
    return table(['Grupo','N','BA regra (%)','Δ BA [IC95%] (pp)','Piora >1 pp'],
                 [(r.scope,r.n_datasets,f'{100*r.ba:.2f}',cell(r),
                   f'{r.worse_gt_1pp}/{r.n_datasets} ({100*r.share_worse_gt_1pp:.1f}%)')
                  for r in f[f.scope.isin(scopes)].itertuples()])


def verify_files():
    protected=json.loads((OUT/'protected_initial.json').read_text())
    changed=[]
    for name,initial in protected.items():
        p=Path(name)
        if not p.exists():
            changed.append(dict(path=name,reason='missing')); continue
        st=p.stat()
        if st.st_size!=initial['size'] or st.st_mtime_ns!=initial['mtime_ns']:
            changed.append(dict(path=name,reason='size_or_mtime'))
        elif 'sha256' in initial and hashlib.sha256(p.read_bytes()).hexdigest()!=initial['sha256']:
            changed.append(dict(path=name,reason='hash'))
    new=[]
    import os
    for root,ds,fs in os.walk('.'):
        for name in fs:
            p=Path(root)/name
            if str(p) not in protected and not p.is_symlink(): new.append(str(p))
    outside=[p for p in new if not p.startswith(('experiments/','results/experiments/e6_e7/'))]
    result=dict(existing_files_checked=len(protected),changed_existing_files=changed,
                new_files_outside_authorized_paths=outside,new_files_count=len(new),passed=not changed and not outside)
    (OUT/'file_integrity.json').write_text(json.dumps(result,indent=2))
    assert result['passed'],result
    return result


def main():
    sample=pd.read_csv(OUT/'sample_E6_E7.csv',keep_default_na=False)
    e6=pd.read_csv(OUT/'E6_runs.csv'); e7=pd.read_csv(OUT/'E7_runs.csv')
    assert len(sample)==55 and sample.semantic_family_id.nunique()==55
    assert len(e6)==55*5*5*3
    assert not e6.duplicated(['dataset_id','epsilon','repetition','regularization']).any()
    assert e6.groupby('dataset_id').size().eq(75).all()
    assert len(e7)==55*3*2*12*4
    assert e7.groupby('dataset_id').size().eq(288).all()
    assert not e7.duplicated(['dataset_id','epsilon','repetition','policy','steps','clip','mode']).any()
    assert e6.ba.between(0,1).all() and e7.ba.between(0,1).all()
    ledger_map={}
    for p in (OUT/'parts').glob('*_ledgers.json'): ledger_map.update(json.loads(p.read_text()))
    reconstructed={}
    for lid,js in ledger_map.items():
        assert hashlib.sha256(js.encode()).hexdigest()==lid
        reconstructed[lid]=ledger_from_dict(json.loads(js))
    private=pd.concat([e6,e7[e7.is_dp_release]],ignore_index=True)
    all_executed_parts=pd.concat([pd.read_csv(p) for p in (OUT/'parts').glob('E[67]_*.csv')],ignore_index=True)
    all_executed_private=all_executed_parts[all_executed_parts.is_dp_release]
    # Every private row resolves to a verified full ledger. Audit calibration formula per unique ledger.
    assert private.ledger_verified.all() and private.total_budget_identical.all()
    assert not private.accounting_method_changed.any()
    assert np.allclose(private.epsilon,private.epsilon_used,atol=1e-9,rtol=0)
    assert all_executed_private.ledger_verified.all()
    assert np.allclose(all_executed_private.epsilon,all_executed_private.epsilon_used,atol=1e-9,rtol=0)
    for lid,block in all_executed_private.groupby('ledger_id'):
        r=block.iloc[0]
        audit(reconstructed[lid],r.epsilon,int(r.steps),r["clip"],int(r.n_train),2,int(r.encoded_dim))
    # Lambda has no accounting effect on the E6 grid.
    assert e6.groupby(['dataset_id','epsilon','repetition']).ledger_id.nunique().eq(1).all()
    keys=['dataset_id','epsilon','repetition']
    prediction=pd.read_csv(OUT/'E6_lodo_predictions.csv')
    assert prediction.training_datasets.eq(54).all()
    rules=json.loads((OUT/'E6_lodo_rules.json').read_text())
    training=e6.merge(sample[['dataset_id','n_instances']],on='dataset_id')
    for did,rule in rules.items():
        recalculated=fit_rule(training[training.dataset_id.ne(int(did))])
        assert rule==recalculated
    selected=e6.merge(prediction[keys[:2]+['selected_lambda']],on=keys[:2],validate='many_to_one')
    selected=selected[selected.regularization.eq(selected.selected_lambda)].copy()
    base=e6[e6.regularization.eq(.03)][keys+['ba']].rename(columns={'ba':'baseline_ba'})
    pair6=selected.merge(base,on=keys,validate='one_to_one').merge(sample[META],on='dataset_id')
    pair6['delta']=pair6.ba-pair6.baseline_ba
    pair6.to_csv(OUT/'E6_lodo_paired_runs.csv',index=False)
    pair6.groupby(['dataset_id','epsilon'],as_index=False).agg(
        ba=('ba','mean'),baseline_ba=('baseline_ba','mean'),delta=('delta','mean')).to_csv(OUT/'E6_lodo_dataset_deltas.csv',index=False)
    s6=paired_summary(pair6,'E6_lodo')
    grid6=e6.merge(base,on=keys,validate='many_to_one').merge(sample[META],on='dataset_id')
    grid6['delta']=grid6.ba-grid6.baseline_ba
    g6=paired_summary(grid6,'E6_lambda_grid',['regularization'])
    priv7=e7[e7.is_dp_release].copy()
    h27=priv7[priv7.policy.eq('fixed003')&priv7.steps.eq(100)&priv7["clip"].eq(2)]
    exact=h27.merge(base,on=keys,validate='one_to_one')
    assert np.array_equal(exact.ba,exact.baseline_ba)
    rule7=priv7[priv7.policy.eq('rule_lodo')&priv7.steps.eq(100)&priv7["clip"].eq(2)]
    exact=rule7.merge(pair6[keys+['ba']].rename(columns={'ba':'e6_ba'}),on=keys,validate='one_to_one')
    assert np.array_equal(exact.ba,exact.e6_ba)
    pair7=priv7.merge(h27[keys+['ba']].rename(columns={'ba':'baseline_ba'}),on=keys,validate='many_to_one')
    pair7['delta']=pair7.ba-pair7.baseline_ba
    pair7=pair7.merge(sample[META],on='dataset_id')
    pair7.to_csv(OUT/'E7_paired_runs_vs_H27.csv',index=False)
    s7=paired_summary(pair7,'E7_vs_H27',['policy','steps','clip'])
    own=priv7[priv7.steps.eq(100)&priv7["clip"].eq(2)][keys+['policy','ba']].rename(columns={'ba':'matched_baseline_ba'})
    matched=pair7.merge(own,on=keys+['policy'],validate='many_to_one')
    matched['baseline_ba']=matched.matched_baseline_ba
    matched['delta']=matched.ba-matched.baseline_ba
    m7=paired_summary(matched,'E7_vs_same_lambda_100_clip2',['policy','steps','clip'])
    zero=e7[~e7.is_dp_release].copy()
    assert zero.groupby(['dataset_id','regularization','steps','clip']).ba.nunique().eq(1).all()
    # Gap uses ONLY the fixed split, with the corresponding lambda clean solution.
    zkeys=['dataset_id','epsilon','policy','steps','clip']
    private0=priv7[priv7.repetition.eq(0)][zkeys+['ba']].rename(columns={'ba':'private_ba'})
    gaps=zero.merge(private0,on=zkeys,validate='one_to_one').merge(sample[META],on='dataset_id')
    gaps['noise_gap']=gaps.ba-gaps.private_ba
    gaps.to_csv(OUT/'E7_fixed_split_gaps.csv',index=False)
    # Reuse paired inference on clean - zero; positive is a remaining gap.
    gapframe=gaps.copy();gapframe['baseline_ba']=gapframe.ba;gapframe['ba']=gapframe.clean_ba
    gapframe['delta']=gapframe.clean_ba-gapframe.baseline_ba
    zs=paired_summary(gapframe,'E7_clean_minus_zero',['policy','steps','clip'])
    winners=[]
    for scope in ['geral',*[f'epsilon={e:g}' for e in EPS7]]:
        for policy in ['fixed003','rule_lodo','both']:
            block=s7[s7.scope.eq(scope)]
            if policy!='both':block=block[block.policy.eq(policy)]
            winner=block.sort_values(['ba','steps','clip'],ascending=[False,True,True]).iloc[0]
            winners.append(dict(selection_scope=scope,policy_pool=policy,**winner.to_dict()))
    winners=pd.DataFrame(winners); winners.to_csv(OUT/'E7_exploratory_best.csv',index=False)
    # Original E5 baseline reuse is checked, though every E6 run was freshly fitted.
    old=pd.read_csv('results/experiments/E5_runs.csv')
    old=old[old.dataset_id.isin(sample[sample.sample_group.eq('original40')].dataset_id)]
    comp=e6.merge(old[keys+['regularization','private_ba']],on=keys+['regularization'],validate='one_to_one')
    old_diff=float(np.max(np.abs(comp.ba-comp.private_ba)))
    assert old_diff<1e-12
    integrity=verify_files()
    meta=json.loads((OUT/'run_metadata.json').read_text())
    runtime=(time.time()-pd.Timestamp(meta['request_start_utc']).timestamp())/60
    audit_result=dict(passed=True,private_rows=len(private),zero_noise_non_dp_rows=len(zero),
                      all_executed_private_rows_including_discarded=len(all_executed_private),
                      discarded_private_rows=len(all_executed_private)-len(private),
                      unique_full_ledgers=len(ledger_map),epsilon_max_abs_error=float(abs(private.epsilon-private.epsilon_used).max()),
                      release_schedule_changed_rows=int(private.release_schedule_changed.sum()),
                      accounting_method_changed_rows=0,all_private_ledger_verify=True,
                      E6_lambda_exact_ledger_identity=True,E7_E6_baseline_exact=True,
                      original_E5_matched_rows=len(comp),original_E5_max_abs_ba_difference=old_diff,
                      lodo_rules_refit_verified=55,existing_files_checked=integrity['existing_files_checked'])
    (OUT/'final_verification.json').write_text(json.dumps(audit_result,indent=2))
    meta.update(runtime_minutes_including_preparation=runtime,full_grid_completed=True,completed_datasets=55,
                execution_deviation=dict(requested_max_fit_processes=8,observed_peak_fit_processes=9,
                    auxiliary_E6_overlap_seconds=59.70724220806733,
                    reason='public alias correction run overlapped the already active eight-process E7 pool',
                    finalization_fit_processes=1))
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    final_rule=json.loads((OUT/'E6_final_public_rule.json').read_text())
    smeta=json.loads((OUT/'sample_metadata.json').read_text())
    best=winners[winners.policy_pool.eq('both')]
    globalbest=best[best.selection_scope.eq('geral')].iloc[0]
    decomposition=[]
    for scope in ['geral',*[f'epsilon={e:g}' for e in EPS7]]:
        epsilon=None if scope=='geral' else float(scope.split('=')[1])
        for label,policy,steps,clip in [('H27 atual','fixed003',100,2.),
                                      ('Configuração recomendada',globalbest.policy,int(globalbest.steps),float(globalbest["clip"]))]:
            block=gaps[gaps.policy.eq(policy)&gaps.steps.eq(steps)&gaps["clip"].eq(clip)]
            if epsilon is not None: block=block[block.epsilon.eq(epsilon)]
            ds=block.groupby('dataset_id')[['clean_minus_zero','noise_gap','clean_ba','ba','private_ba']].mean()
            a,lo,hi=bootstrap(ds.clean_minus_zero,('decomposition',scope,label,'optimization'))
            b,blo,bhi=bootstrap(ds.noise_gap,('decomposition',scope,label,'noise'))
            decomposition.append(dict(scope=scope,configuration=label,clean_ba=ds.clean_ba.mean(),
                                      zero_ba=ds.ba.mean(),private_ba=ds.private_ba.mean(),
                                      clean_minus_zero=a,opt_low=lo,opt_high=hi,
                                      zero_minus_private=b,noise_low=blo,noise_high=bhi))
    decomposition=pd.DataFrame(decomposition)
    decomposition.to_csv(OUT/'E7_current_recommended_fixed_split_decomposition.csv',index=False)
    # Recommend one public configuration, selected exploratorily over the E7 grid.
    recommendation=dict(lambda_policy=globalbest.policy,public_rule=final_rule if globalbest.policy=='rule_lodo' else {'lambda':.03},
                        steps=int(globalbest.steps),clip=float(globalbest["clip"]),delta=1e-6,
                        full_batch=True,total_zcdp_budget='rho(epsilon, delta); rho_step=rho_total/steps',
                        status='exploratory_for_future_full_benchmark',private_n_assumption='n is public')
    (OUT/'recommended_configuration.json').write_text(json.dumps(recommendation,indent=2))
    r6=s6[s6.scope.eq('geral')].iloc[0]
    lines=['# E6 e E7 — regularização pública e gap de otimização', '',
      f'Concluídos **55/55 datasets** (40 originais + 15 grandes), com as duas grades completas, em **{runtime:.2f} minutos** incluindo preparação, piloto, execução e análise. Sem redução de grade. E6 contém {len(e6):,} ajustes privados; E7 contém {len(priv7):,} registros privados e {len(zero):,} condições sem ruído, estas explicitamente **não DP**. Ajustes com λ idêntico entre políticas são reutilizados dentro do dataset; contagens são de condições registradas, não de computações distintas.', '',
      f'E6 LODO: Δ BA global **{cell(r6)} pp**; {r6.worse_gt_1pp}/{r6.n_datasets} datasets pioram mais de 1 pp (média dos cinco ε e três repetições). Melhor configuração observada em E7: **{globalbest.policy}, {int(globalbest.steps)} passos, clip={globalbest["clip"]:g}**, Δ **{cell(globalbest)} pp** versus H27 atual. A seleção de configuração em E7 é exploratória.', '',
      '## Amostra e execução', '',
      'Base imutável: `results/experiments/effective_sample_E1_E4_E5.csv`. Extras: manifesto `data/openml_manifest_984_completed.csv`, 5000<n≤50000, família preenchida, excluídas as famílias dos 40 originais. Embaralhamento com semente 20261111, primeiro representante por família, 6 famílias `all_nominal` e 9 `has_numeric` (inclui mistos). O alias público `king rook vs king`→`kropt` foi unificado, retendo o primeiro representante na ordem sorteada (ID 184) e substituindo ID 46173 pelo próximo numérico (ID 46468). Os 6 nominais esgotam as famílias nominais elegíveis após a união; o estrato numérico tinha 106 famílias. A correção usou somente informações públicas, antes de avaliar configurações finais. A composição é equilibrada por desenho e não proporcional ao manifesto. As famílias seguem o manifesto, com essa correção explícita de alias; não foi feita uma revisão semântica exaustiva dos 984 registros.', '',
      'Tercis empíricos calculados publicamente sobre os 55 n. C1 contém n=15–662 (18 datasets), C2 n=683–3200 (19), C3 n=3511–45000 (18). C3 inclui os 15 grandes e datasets originais maiores. Também preservamos T1/T2/T3 originais e reportamos LARGE separadamente. Os cortes numéricos exatos estão em `sample_metadata.json`. A análise usa peso igual por dataset, por ε e por repetição.', '',
      table(['Grupo','Tipo','Datasets','n mínimo–máximo'],[(a,b,len(d),f'{d.n_instances.min()}–{d.n_instances.max()}') for (a,b),d in sample.groupby(['sample_group','feature_group'])]), '',
      'Os 40 IDs originais são preservados integralmente no CSV da amostra. Extras efetivamente usados:', '',
      table(['ID OpenML','Dataset','Família','n','Tipo'],[(int(r.dataset_id),r.dataset_name,r.semantic_family_id,int(r.n_instances),r.feature_group)
            for r in sample[sample.sample_group.eq('large15')].itertuples()]), '',
      'Loader, encoding público `tanh`, validações de schema/linhas/classes, holdout estratificado 75/25, tratamento de singleton, DP-SGD e wrapper `zero_sgd` são importados de `experiments/e1_e4_e5.py`. Cada repetição usa o mesmo split e a mesma semente DP de E5 (semente base 20261110), pareados entre λ, passos e clip. A semente 20261111 é usada na seleção dos extras. O bootstrap usa a função anterior: 20.000 reamostragens de datasets e semente estável; primeiro promedia repetições/ε dentro do dataset. CIs percentis de 95%, em pp. Pool principal de 8 processos, `.venv/bin/python`, `PYTHONPATH=.`, `OMP_NUM_THREADS=1` e uma thread numérica por processo; bytecode desativado. Nenhum download foi necessário.', '',
      '**Desvio do limite de processos:** o ajuste auxiliar E6 do dataset substituto foi iniciado enquanto o pool E7 de oito processos já estava ativo, produzindo pico de **9 processos de ajuste por cerca de 60 s**, acima do máximo solicitado de 8. A finalização E7 do substituto usou um único processo, após terminar o pool. O desvio de execução não mudou sementes, dados ou contabilidade e está registrado em `run_metadata.json`; a grade estatística permaneceu integral.', '',
      f'Os {len(comp):,} resultados compartilhados com E5 foram reproduzidos com diferença máxima de BA {old_diff:g}. As referências de E7 a 100 passos/clip 2 reproduzem exatamente E6 tanto para λ fixo quanto para a regra LODO.', '',
      '## E6 — regra pública λ(n, ε)', '',
      'Grade: λ∈{0,03; 0,01; 0,003; 0,001; 0,0003}, ε∈{0,1; 0,5; 1; 4; 16}, 100 passos, clip=2, três repetições pareadas. A regra tem cinco faixas fixadas antes da análise: n·ε<100, [100,1000), [1000,10000), [10000,100000), ≥100000. Em cada faixa, escolhe o λ com maior BA de treinamento média, com peso igual por dataset; empates favorecem λ maior. Faixas com menos de cinco datasets usam o melhor λ global de treinamento. Sem otimizar cortes ou escolher entre modelos de regra após ver os resultados.', '',
      'LODO: cada uma das 55 regras é ajustada nos outros 54 datasets, removendo todas as repetições e todos os ε do dataset avaliado. E7 usa essas mesmas previsões LODO. Os ICs condicionam às previsões LODO obtidas; o bootstrap não reajusta a regra em cada sorteio, nem resolve a dependência causada pela sobreposição dos treinamentos.', '',
      fmt_summary(s6,['geral',*[f'epsilon={e:g}' for e in EPS6],*[f'tercil={t}' for t in ('C1','C2','C3')],'faixa=LARGE']), '',
      'Resultados por ε e tercil (incluindo os grandes no C3):', '',
      fmt_summary(s6,[f'{t}/epsilon={e:g}' for t in ('C1','C2','C3') for e in EPS6]), '',
      'Regra final ajustada nos 55 datasets, para uso como hiperparâmetro público em benchmark futuro:', '',
      table(['n·ε (n total público do dataset)','λ final','Datasets de treinamento na faixa'],
            [(label,lam,support) for label,lam,support in zip(['<100','[100,1000)','[1000,10000)','[10000,100000)','≥100000'],final_rule['lambdas'],final_rule['unique_training_datasets_per_bin'])]), '',
      'A regra usa n total do dataset para preservar a definição do experimento; a escala do mecanismo usa n_train. Em uma aplicação com dados privados, n deve ser público ou contabilizado separadamente. A regra final, treinada com os 55 datasets, não tem avaliação independente aqui; o resultado apresentado é LODO. Não se usa BA, classe ou metafeature privada do novo dataset para escolher λ.', '',
      'Deltas globais das constantes da grade (comparações descritivas):', '',
      table(['λ','Δ BA [IC95%] (pp)'],[(r.regularization,cell(r)) for r in g6[g6.scope.eq('geral')].itertuples()]), '',
      '## E7 — passos, clipping e ruído', '',
      'Grade completa: passos∈{100,200,400} × clip∈{0,5;1;2;4}, ε∈{0,1;1;4}, três repetições privadas, com λ=0,03 e com a regra E6 LODO. Referência principal: H27 atual (λ=0,03, 100 passos, clip 2). A coluna Δ mesma λ isola a alteração de passos/clip, usando como referência 100 passos/clip 2 sob a mesma política de λ. Ruído zero e clean usam somente o split fixo 0, sem replicar uma medição determinística como se fossem três observações independentes.', '']
    for policy in ['fixed003','rule_lodo']:
        rows=[]
        for r in s7[s7.scope.eq('geral')&s7.policy.eq(policy)].itertuples():
            mr=m7[m7.scope.eq('geral')&m7.policy.eq(policy)&m7.steps.eq(r.steps)&m7["clip"].eq(r.clip)].iloc[0]
            zr=zs[zs.scope.eq('geral')&zs.policy.eq(policy)&zs.steps.eq(r.steps)&zs["clip"].eq(r.clip)].iloc[0]
            per=[]
            for eps in EPS7:
                er=s7[s7.scope.eq(f'epsilon={eps:g}')&s7.policy.eq(policy)&s7.steps.eq(r.steps)&s7["clip"].eq(r.clip)].iloc[0]
                per.append(cell(er))
            rows.append((r.steps,r.clip,cell(r),cell(mr),*per,cell(zr)))
        lines.extend([f'### Política {policy}', '',table(['Passos','Clip','Δ vs H27 geral','Δ mesma λ geral','Δ ε=0,1','Δ ε=1','Δ ε=4','Clean−zero geral'],rows),''])
    lines.extend(['Interpretação: com a regra LODO e clip 2, passar de 100 para 400 passos reduz clean−zero de 3,34 para 0,04 pp, enquanto o ganho privado contra H27 cai de +1,23 para +0,37 pp. A melhoria de otimização sem ruído é compensada pela perturbação DP sob orçamento total fixo. Com λ=0,03/clip 2, o gap cai de 4,02 para 1,37 pp, mas a BA privada cai 1,62 pp [−2,84; −0,32] frente a 100 passos. Não há suporte global nesta amostra para aumentar os passos da configuração recomendada.', '', 'Melhores configurações observadas (maior BA média; desempate por menos passos e menor clip):','',
        table(['Grupo','Política','Passos','Clip','BA (%)','Δ vs H27 [IC95%] (pp)'],
              [(r.selection_scope,r.policy,int(r.steps),r.clip,f'{100*r.ba:.2f}',cell(r)) for r in best.itertuples()]), '',
        'Decomposição no mesmo split fixo (clean−privado = clean−zero + zero−privado). Cada termo usa seu próprio IC bootstrap por dataset; os limites dos ICs não devem ser somados:', '',
        table(['Grupo','Configuração','BA clean / zero / privada (%)','Clean−zero [IC95%] pp','Zero−privado [IC95%] pp'],
              [(r.scope,r.configuration,f'{100*r.clean_ba:.2f} / {100*r.zero_ba:.2f} / {100*r.private_ba:.2f}',
                f'{100*r.clean_minus_zero:+.2f} [{100*r.opt_low:+.2f}; {100*r.opt_high:+.2f}]',
                f'{100*r.zero_minus_private:+.2f} [{100*r.noise_low:+.2f}; {100*r.noise_high:+.2f}]')
               for r in decomposition.itertuples()]), '',
        'Clean é `fit_clean_softmax` existente, λ correspondente, tolerância 1e−5 e limite público de norma sqrt(p_original+1), minimizando a perda regularizada. Não é o máximo de BA de teste. O gap clean−zero positivo mede perda de BA associada conjuntamente a clipping e otimização finita; pode ser negativo. Não identifica isoladamente o efeito de clipping. O arquivo `E7_fixed_split_gaps.csv` contém clean−zero e zero−privado no mesmo split, permitindo decompor clean−privado. Os ajustes clean e sem ruído são diagnósticos não DP; ledgers associados aos sem ruído mostram somente a calibração contrafactual.', '',
        '## Auditoria de privacidade e integridade', '',
        f'Todos os **{len(private):,} registros privados** passaram `ledger.verify()` e auditoria de igualdade do ε total solicitado; erro absoluto máximo **{audit_result["epsilon_max_abs_error"]:.3g}**. δ=10⁻⁶ em todos os ajustes. Foram persistidos **{len(ledger_map):,} ledgers completos**, ligados por SHA-256. As {len(zero):,} condições zero-noise estão marcadas `is_dp_release=False` e `ledger_verified=False` para não atribuir garantia DP ao diagnóstico.', '',
        f'A família substituída permaneceu nos payloads da execução provisória já iniciada. Seus 75 registros privados E6, 216 privados E7 e 72 diagnósticos zero-noise estão preservados em `parts/`, excluídos de todas as médias e CIs finais. A auditoria adicional também verificou esses ajustes: **{len(all_executed_private):,} registros privados incluindo os descartados**. `provisional_before_alias_correction/` preserva a consolidação E7 anterior; `alias_correction_verification.json` comprova que as 162 escolhas de λ E7 dos 54 datasets mantidos não mudaram, permitindo reutilização exata.', '',
        'Contabilidade nativa preservada: ρ_total = (sqrt(log(1/δ)+ε)−sqrt(log(1/δ)))²; ρ_step=ρ_total/T, sensibilidade L2=2·clip/n_train, σ=(2·clip/n_train)/sqrt(2ρ_step). Logo aumentar T aumenta σ por passo como sqrt(T), dividindo o mesmo orçamento, sem amplificação por subsampling. λ não muda o ledger. Passos ou clip diferentes **mudam o ledger completo** (número de liberações, sensibilidade e/ou σ); isto é marcado por `release_schedule_changed=True`. Não houve mudança de método contábil ou orçamento total (`accounting_method_changed=False`, `total_budget_identical=True`).', '',
        'O ε refere-se a cada treinamento individual. A execução/divulgação conjunta de todas as variantes, repetições ou a seleção baseada em dados privados requer composição adicional; este estudo usa datasets públicos. BA de teste, holdout estratificado por rótulo, ajuste da regra, zero-noise e clean não constituem novas liberações DP certificadas. A recomendação pode ser congelada e aplicada a um novo dataset com n/ε públicos, sem tuning privado adicional.', '',
        f'Integridade: **{integrity["existing_files_checked"]:,} arquivos preexistentes** verificados por tamanho/mtime, com SHA-256 adicional para fontes/resultados textuais e `.git/`: nenhum foi alterado. Não houve escrita em `dpmeta/`, arquivos anteriores, `.git/` ou no projeto anterior. Código novo somente em `experiments/`; saídas somente em `results/experiments/e6_e7/`. A configuração de cache de fallback também aponta para esta pasta. Nenhum comando que altere o estado git foi executado.', '',
        '## Recomendação concreta e limitações', '',
        f'Para uma opção futura no benchmark completo: **DP-SGD full batch, {int(globalbest.steps)} passos, clip={globalbest["clip"]:g}, '+('λ pela regra pública final de E6 acima' if globalbest.policy=='rule_lodo' else 'λ=0,03 constante')+'**, δ=10⁻⁶, mesmo ρ_total por ε, aceleração/tamanho de passo nativos. Parâmetros exatos em `recommended_configuration.json`. A regra final E6 está preservada em `E6_final_public_rule.json`.', '',
        'Estudo exploratório, 55 datasets e somente três repetições; estratificação equilibra tipos e acrescenta deliberadamente datasets grandes. Muitos testes e escolha do melhor na mesma amostra geram otimismo; os ICs são pontuais, sem correção por múltiplas comparações, e não são ICs pós-seleção. E6 tem exclusão por dataset no ajuste da regra, mas E7 seleciona passos/clip usando todos os datasets; validar essa configuração congelada no benchmark completo ou em famílias novas antes de adotá-la. A amostra não permite atribuir causalmente efeitos a n, separado de dimensão, classes ou família. O clean fornece ótimo da perda regularizada aproximado, sem promessa de maior BA.', '',
        'A constante λ=0,0003 teve ganho descritivo E6 de +2,04 pp [+0,88; +3,41], comparado a +1,83 pp da regra LODO. Este estudo não demonstra superioridade da forma dependente de n·ε sobre essa constante; a escolha da constante pela grade usa a própria amostra. Incluí-la como controle congelado no benchmark futuro ajudará a verificar se a complexidade da regra é necessária.', '',
        'A extrapolação para n>50000 ou ε fora de [0,1;16] não foi validada. E7 examina somente ε∈{0,1;1;4}; a melhor configuração nesse subconjunto não garante o mesmo comportamento em ε=0,5 ou 16. Os ganhos dos datasets grandes não justificam alterar o mecanismo de privacidade ou tratar n privado como público.', '',
        '## Artefatos e reprodução', '',
        '`sample_E6_E7.csv`: amostra completa; `large_reserve_order.csv`: sorteio elegível; `E6_runs.csv`/`E7_runs.csv`: condições individuais com sementes, tempo e auditoria; `parts/*_ledgers.json`: ledgers completos; `E6_lodo_predictions.csv` e `E6_lodo_rules.json`: previsões/regras por fold; `*_bootstrap_summary.csv`: tabelas completas por ε, tercil, tamanho e tipo; `E7_fixed_split_gaps.csv`: decomposição no split fixo; `E7_exploratory_best.csv`: vencedores globais e por política; `final_verification.json` e `file_integrity.json`: verificações.', '',
        'Reprodução, em uma pasta de resultados vazia para preservar a regra de não sobrescrever arquivos existentes:', '',
        '```sh\nPYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python experiments/e6_e7.py --stage sample\nPYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python experiments/e6_e7.py --stage E6 --workers 8\nPYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python experiments/e6_e7.py --stage E7 --workers 8\nPYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python experiments/summarize_e6_e7.py\n```', ''])
    (OUT/'SUMMARY_E6_E7.md').write_text('\n'.join(lines))
    print(json.dumps(dict(E6_overall=r6.to_dict(),E7_best=best.to_dict('records'),runtime_minutes=runtime),indent=2))


if __name__=='__main__': main()
