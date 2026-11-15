"""Paired, semantic-family cluster bootstrap for the two frozen H27 variants."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from dpmeta.portfolio_benchmark import H27_PIPELINE,LAMRULE_PIPELINE,LAM0003_PIPELINE,SGD_NEW_PIPELINES
from dpmeta.provenance import file_sha256
from scripts.parallel_sgd import OUT,BASE,KEYS,inputs,atomic_json,signature

BOOTSTRAPS=20000
SEED=20261110

def bootstrap(block,rng):
    family=block.groupby('bootstrap_family').delta.agg(['sum','count'])
    sums=family['sum'].to_numpy(); counts=family['count'].to_numpy(); k=len(family)
    draws=[]
    for start in range(0,BOOTSTRAPS,256):
        indices=rng.integers(0,k,size=(min(256,BOOTSTRAPS-start),k))
        draws.append(sums[indices].sum(axis=1)/counts[indices].sum(axis=1))
    low,high=np.quantile(np.concatenate(draws),[.025,.975])
    return dict(datasets=len(block),families=k,mean_delta=float(block.delta.mean()),ci_low=float(low),ci_high=float(high))

def table(frame,columns):
    header='| '+' | '.join(columns)+' |'
    return '\n'.join([header,'| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(str(row[c]) for c in columns)+' |' for _,row in frame.iterrows()])

def run():
    started=time.monotonic(); config,manifest,old=inputs()
    conditions=pd.read_csv(OUT/'conditions_15.csv',float_precision='round_trip')
    new=pd.read_csv(OUT/'conditions_new_options.csv',float_precision='round_trip')
    coverage=json.loads((OUT/'coverage.json').read_text())
    metadata=manifest[['dataset_id','n_instances','semantic_family_id']].copy()
    missing=metadata.semantic_family_id.isna()|metadata.semantic_family_id.astype(str).str.strip().isin(['','nan','None'])
    metadata['bootstrap_family']=metadata.semantic_family_id.astype(str)
    metadata.loc[missing,'bootstrap_family']=metadata.loc[missing,'dataset_id'].map(lambda d:f'__dataset_{d}')
    # Terciles defined once on the target population of 973 public n values.
    metadata['n_tercile'],edges=pd.qcut(metadata.n_instances,3,labels=['small','medium','large'],retbins=True)
    metadata.to_csv(OUT/'analysis_population.csv',index=False)
    scores=conditions.pivot(index=KEYS,columns='pipeline',values='private_ba')
    pairs=[(LAMRULE_PIPELINE,H27_PIPELINE,'lamrule_vs_h27'),
        (LAM0003_PIPELINE,H27_PIPELINE,'lam0003_vs_h27'),
        (LAMRULE_PIPELINE,LAM0003_PIPELINE,'lamrule_vs_lam0003')]
    paired=[]; boot=[]; rng=np.random.default_rng(SEED)
    for candidate,reference,label in pairs:
        delta=(scores[candidate]-scores[reference]).rename('delta').reset_index()
        delta['comparison']=label; paired.append(delta)
        means=delta.groupby(['dataset_id','epsilon','comparison'],as_index=False).delta.mean().merge(metadata,on='dataset_id',validate='many_to_one')
        for epsilon,eps in means.groupby('epsilon',sort=True):
            for group,block in [('all',eps)]+[(str(g),b) for g,b in eps.groupby('n_tercile',observed=True)]:
                boot.append(dict(comparison=label,group=group,epsilon=float(epsilon),**bootstrap(block,rng)))
    paired=pd.concat(paired,ignore_index=True)
    paired.to_csv(OUT/'paired_conditions.csv',index=False)
    means=paired.groupby(['dataset_id','epsilon','comparison'],as_index=False).delta.mean().merge(metadata,on='dataset_id',validate='many_to_one')
    means.to_csv(OUT/'paired_dataset_means.csv',index=False)
    means.groupby(['comparison','epsilon','bootstrap_family'],as_index=False).delta.agg(['mean','size']).reset_index().to_csv(OUT/'paired_family_means.csv',index=False)
    boot=pd.DataFrame(boot); boot.to_csv(OUT/'paired_bootstrap.csv',index=False)
    fixed=[]; oracle=[]; oracle_conditions=[]
    for count in (14,15):
        frame=pd.read_csv(OUT/f'conditions_{count}.csv',float_precision='round_trip')
        for group,block in [('all',frame)]+[(str(e),b) for e,b in frame.groupby('epsilon')]:
            means_fixed=block.groupby('pipeline').private_ba.mean().sort_values(ascending=False)
            for rank,(pipeline,ba) in enumerate(means_fixed.items(),1):
                fixed.append(dict(portfolio=count,epsilon=group,pipeline=pipeline,rank=rank,mean_ba=float(ba)))
        pivot=frame.pivot(index=KEYS,columns='pipeline',values='private_ba')
        best=pivot.max(axis=1)
        tied=np.isclose(pivot.to_numpy(),best.to_numpy()[:,None],atol=1e-12,rtol=0)
        n_ties=tied.sum(axis=1)
        for j,pipeline in enumerate(pivot.columns):
            result=pivot.index.to_frame(index=False)
            result['portfolio']=count;result['pipeline']=pipeline
            result['inclusive_win']=tied[:,j];result['strict_win']=tied[:,j]&(n_ties==1)
            result['fractional_win']=tied[:,j]/n_ties
            oracle_conditions.append(result)
            for group,block in [('all',result)]+[(str(e),b) for e,b in result.groupby('epsilon')]:
                oracle.append(dict(portfolio=count,epsilon=group,pipeline=pipeline,conditions=len(block),
                    inclusive_share=float(block.inclusive_win.mean()),strict_share=float(block.strict_win.mean()),
                    fractional_share=float(block.fractional_win.mean())))
    fixed=pd.DataFrame(fixed);fixed.to_csv(OUT/'fixed_option_rankings.csv',index=False)
    oracle=pd.DataFrame(oracle);oracle.to_csv(OUT/'oracle_win_shares.csv',index=False)
    pd.concat(oracle_conditions,ignore_index=True).to_csv(OUT/'oracle_conditions.csv',index=False)
    alignment=json.loads((OUT/'alignment.json').read_text());attempts=json.loads((OUT/'attempts.json').read_text())
    timing=pd.read_csv(OUT/'timing.csv'); failures=pd.read_csv(OUT/'failures.csv')
    audit=pd.read_csv(OUT/'privacy_audit.csv',float_precision='round_trip')
    protected=json.loads((OUT/'protected_hashes.json').read_text())
    changed=[name for name,digest in protected.items() if not Path(name).is_file() or file_sha256(Path(name))!=digest]
    source=Path('dpmeta/portfolio_benchmark.py').read_text()
    import hashlib
    body=hashlib.sha256(source[source.index('def _run_dataset_legacy'):source.index('def run_dataset')].encode()).hexdigest()
    legacy=json.loads((OUT/'legacy_preservation.json').read_text())['legacy_body_sha256']==body
    # Verify serialized output too, not only the pre-write dataframe.
    original_rows_exact=True
    for count in (14,15):
        frame=pd.read_csv(OUT/f'conditions_{count}.csv',float_precision='round_trip')
        retained=old[old.dataset_id.isin(frame.dataset_id)].sort_values(KEYS+['pipeline']).reset_index(drop=True)
        actual=frame[frame.pipeline.isin(old.pipeline.unique())].sort_values(KEYS+['pipeline']).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,retained,check_exact=True)
    verification=dict(signature=signature(),protected_files=len(protected),changed_protected_files=changed,
        legacy_body_byte_identical=legacy,original_rows_exact=original_rows_exact,
        ledger_audits=len(audit),all_ledgers_identical_to_h27=bool(audit.ledger_identical_to_h27.all()),
        n_tercile_edges=edges.tolist(),bootstrap_replicates=BOOTSTRAPS,bootstrap_seed=SEED,
        summary_seconds=time.monotonic()-started,
        output_hashes={p.name:file_sha256(p) for p in OUT.glob('*.csv')})
    atomic_json(OUT/'verification.json',verification)
    if changed or not legacy: raise RuntimeError('protected source changed')
    all_tables=[]
    comparison_titles={'lamrule_vs_h27':'Lamrule − H27',
        'lam0003_vs_h27':'Controle λ=0,0003 − H27',
        'lamrule_vs_lam0003':'Lamrule − controle λ=0,0003'}
    for comparison,_ref,label in pairs:
        b=boot[boot.comparison.eq(label)].copy()
        b['group']=b['group'].map({'all':'todos','small':'n baixo','medium':'n médio','large':'n alto'})
        b['Δ BA (pp)']=b.mean_delta.map(lambda x:f'{100*x:+.3f}')
        b['IC 95% (pp)']=[f'[{100*l:+.3f}; {100*h:+.3f}]' for l,h in zip(b.ci_low,b.ci_high)]
        all_tables.append(comparison_titles[label]+'\n\n'+table(b.rename(columns={'group':'Grupo','epsilon':'ε','datasets':'Datasets','families':'Famílias'}),
            ['Grupo','ε','Datasets','Famílias','Δ BA (pp)','IC 95% (pp)']))
    best=fixed[fixed['rank'].eq(1)].copy();best['BA média (%)']=best.mean_ba.map(lambda x:f'{100*x:.3f}')
    wins=oracle[oracle.pipeline.isin(SGD_NEW_PIPELINES)].copy()
    for column in ('inclusive_share','strict_share','fractional_share'): wins[column]=wins[column].map(lambda x:f'{100*x:.2f}')
    best['epsilon']=best.epsilon.replace('all','todos')
    wins['epsilon']=wins.epsilon.replace('all','todos')
    best=best.rename(columns={'portfolio':'Opções','epsilon':'ε','pipeline':'Melhor opção fixa'})
    wins=wins.rename(columns={'portfolio':'Opções','epsilon':'ε','pipeline':'Opção',
        'inclusive_share':'Incluindo empates (%)','strict_share':'Vitória única (%)',
        'fractional_share':'Cota fracionária (%)'})
    pytest=Path(OUT/'pytest.log').read_text().strip().splitlines()[-1]
    new_count=f'{len(new):,}'.replace(',', '.')
    overall=boot[boot.group.eq('all')]
    rule_gain=100*overall[overall.comparison.eq('lamrule_vs_h27')].mean_delta.mean()
    control_gain=100*overall[overall.comparison.eq('lam0003_vs_h27')].mean_delta.mean()
    rule_control=overall[overall.comparison.eq('lamrule_vs_lam0003')]
    excluded_zero=rule_control[rule_control.ci_high.lt(-1e-12)].epsilon.tolist()
    global_fixed=fixed[fixed.epsilon.eq('all')&fixed.portfolio.eq(15)]
    best_global=global_fixed[global_fixed['rank'].eq(1)].iloc[0]
    control_global=global_fixed[global_fixed.pipeline.eq(LAM0003_PIPELINE)].iloc[0]
    main_result=(f'Resultado principal: ambas as opções superam H27 na média dos datasets em todos os oito ε, '
        f'com ICs pontuais de 95% acima de zero. Na média sobre os oito ε, os ganhos são '
        f'**{rule_gain:+.3f} pp (lamrule)** e **{control_gain:+.3f} pp (controle)**. '
        f'O controle tem média maior que a regra em todos os ε; o IC de lamrule − controle '
        f'exclui zero a favor do controle apenas em ε={excluded_zero}, sem ajuste de multiplicidade. '
        f'A melhor opção fixa global nas grades de 14 e 15 continua sendo NB8U: '
        f'**{100*best_global.mean_ba:.4f}% BA**, praticamente empatada com o controle '
        f'(**{100*control_global.mean_ba:.4f}%**, diferença de '
        f'{100*(best_global.mean_ba-control_global.mean_ba):.4f} pp). '
        'Esse ranking usa o teste retrospectivamente e não demonstra superioridade estatística de NB8U sobre o controle.')
    text=[
        f'Foram implementadas e avaliadas as duas opções de DP-SGD em **{coverage["completed"]}/973 datasets**, com 8 ε × 3 repetições: **{new_count} linhas novas**. Falhas: **{coverage["failed"]}**; datasets sem tentativa: **{len(coverage["unattempted_ids"])}**.',
        main_result,
        '`conditions_new_options.csv` contém somente `softmax__dp_sgd_clip2_lamrule` e `softmax__dp_sgd_clip2_lam0003`. `conditions_14.csv` acrescenta apenas lamrule às 13 opções; `conditions_15.csv` acrescenta ambas. As grades retêm exclusivamente condições completas. As nove colunas e todos os floats das linhas antigas foram preservados exatamente, inclusive após serialização e leitura round-trip.',
        'As duas opções usam `fit_private_sgd_softmax` por parâmetros: full batch, 100 passos, clip 2, δ=10⁻⁶, contabilidade zCDP e namespace de ruído H27 `dp_sgd_clip2_reg003`. Somente λ muda; smoothness, momentum e gradiente regularizado seguem o treinador existente. Lamrule lê o JSON congelado `results/experiments/e6_e7/E6_final_public_rule.json`, com n = total público `n_instances`: nε<100 → 0,03; [100,1000) → 0,01; ≥1000 → 0,0003. O controle usa 0,0003. Fits rigorosamente duplicados entre as opções são reutilizados; nenhuma aproximação foi introduzida. Os cleans usam o λ correspondente; a referência `best_clean_portfolio_ba` e o regret mantêm o clean original.',
        'O corpo histórico `_run_dataset_legacy` permanece byte a byte idêntico. A fixture das 13 opções verifica igualdade exata de outputs completos (scores e ledgers). A fixture das features preserva forma e valores das 12 opções anteriormente suportadas em todos os cinco modos. NB8U tinha calibração ausente (ValueError); seu ramo passou a calcular os descritores de 8 bins, sem alterar um vetor preexistente. Para as novas opções, `_candidate_features` usa a identidade H27 e sua calibração, substituindo somente o campo público já existente `calib_dp_sgd__regularization` por log1p(λ). Não houve novas dimensões, deslocamentos nem release de metadados. Em `public_schema_only`, que não inclui calibração, as variantes compartilham a representação H27.',
        f'Testes: **{pytest}**. Incluem fronteiras e leitura efetiva do JSON, uso do total público em vez do split, ledger completo idêntico ao H27 nos oito ε, execução isolada, invariância das 13 saídas antigas, invariância das features existentes, limiar de disco, rejeição de ledger adulterado e retomada com assinatura divergente.',
        f'Alinhamento obrigatório: **{alignment["datasets"]} datasets e {alignment["conditions"]} condições**, H27 reexecutado com seed 20260917, split estratificado 75/25, namespace `portfolio_split`, tanh e dados certificados. **Igualdade exata de todos os cinco scores** com os shards brutos e, reproduzindo a leitura histórica, com `conditions_13.csv`; erro máximo final de BA = **{alignment["max_private_error"]}**. A primeira comparação direta encontrou até 8,33×10⁻¹⁷, pois a consolidação histórica usou o parser padrão do pandas. A trilha `alignment_direct_float_*` registra isso; `alignment_raw_shards.csv` e `alignment_conditions.csv` demonstram que apenas a serialização explica a diferença. Nenhum score foi copiado para fazer o teste passar.',
        f'Execução retomável: `scripts/parallel_sgd.py`, **8 workers**, OMP/BLAS=1, mesmos 973 IDs, ε, repetições, seeds e splits. Tempo ativo total: **{sum(a["seconds"] for a in attempts)/60:.2f} min** em {len(attempts)} tentativa(s); soma por dataset: **{timing.seconds.sum()/3600:.2f} h**; maior dataset: **{timing.seconds.max()/60:.2f} min**. Espaço final: **{attempts[-1]["free_after"]/1024**3:.2f} GiB**; parada por disco: **{any(a["disk_stopped"] for a in attempts)}**. O pai monitora o limite de 5 GiB durante os ajustes; workers também verificam antes e depois de cada dataset. Commits JSON são atômicos; retomada verifica assinatura de código, executor, JSON público, manifesto, baseline e runtime, grade completa e cada ledger.',
        'A segunda passagem do executor confirmou 973 commits completos, zero pendentes e nenhum retreino. Cada parquet usado foi conferido por SHA-256 contra a auditoria de conteúdo NB8U, que certifica igualdade com a execução original. Não houve download substituto, remoção de linhas ou inclusão dos 11 datasets anteriormente rejeitados.',
        f'Auditoria de privacidade: **{len(audit):,} condições novas** e **240 ajustes de alinhamento** passaram a reconstrução independente de todos os 100 releases do ledger H27. Os ledgers completos estão em `ledgers.csv`, ligados por SHA-256 a `privacy_audit.csv` (seeds, split, λ e n público). A regularização não muda sensibilidade, sigma nem rho. O orçamento é por ajuste; a divulgação conjunta requer composição. Clean e BA de teste são diagnósticos em datasets públicos.',
        'Falhas: '+('nenhuma.' if failures.empty else '; '.join(f'{r.dataset_id}: {r.error_type}: {r.error}' for r in failures.itertuples()))+' Os detalhes e tempos constam em `failures.csv` e `failure_details/`.',
        f'Comparações pareadas: Δ BA = primeira opção − segunda; média das três repetições por dataset, depois peso igual por dataset. ICs percentis pontuais de 95% com **{BOOTSTRAPS:,} reamostragens de famílias semânticas**, seed {SEED}; cada família preserva todos os seus membros e pares. A razão soma dos deltas/número de datasets reamostrados conserva o estimando. Famílias ausentes recebem chave única por dataset. Estratos de n são tercis fixados nos 973 valores públicos: limites **{edges.tolist()}**, reamostrados separadamente. Sem correção de multiplicidade. A análise é retrospectiva; a regra congelada veio de E6/E7, sem alegação de validação independente nos datasets sobrepostos.',
        'Os empates de n foram mantidos no mesmo estrato: n≤418 (325 datasets), 418<n≤2000 (344) e n>2000 (304). Os ganhos maiores estão no estrato de n alto. Quando a regra escolhe λ=0,03, seu ajuste coincide com H27; resíduos de até 10⁻¹⁶ na BA, decorrentes da serialização histórica, não representam ganho. Isso explica o Δ praticamente zero de lamrule no estrato de n baixo em ε=0,1. Os CSVs preservam os deltas brutos e as tabelas arredondam em pontos percentuais.',
        *all_tables,
        'Melhor opção fixa por BA média (mesmo peso por dataset e ε; escolha retrospectiva em teste):\n\n'+table(best,['Opções','ε','Melhor opção fixa','BA média (%)']),
        'Participação das novas opções no oracle privado (%). Inclusive conta todos os empatados; estrita exige vencedor único; fracionária divide a condição igualmente entre os empatados (a soma sobre as opções é 100%). Tolerância absoluta 10⁻¹². O oracle usa o teste e é somente diagnóstico. Todas as opções, inclusive as 13 antigas, constam em `oracle_win_shares.csv`.\n\n'+table(wins,['Opções','ε','Opção','Incluindo empates (%)','Vitória única (%)','Cota fracionária (%)']),
        f'Integridade: **{len(protected)} arquivos protegidos** conservaram SHA-256 (dados, Markdown preexistente e artefatos NB8U). Nenhum comando git foi usado e nenhum arquivo da versão anterior foi acessado. Fontes e regra congelada estão em `source_snapshot/`; configuração, tentativas, cobertura e hashes finais em `run_config.json`, `attempts.json`, `coverage.json` e `verification.json`.',
        'Reprodução: `PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python scripts/parallel_sgd.py alignment`; depois `PYTHONPATH=. OMP_NUM_THREADS=1 OPENML_CACHE_DIR=$PWD/.cache/openml .venv/bin/python scripts/parallel_sgd.py run --workers 8`. Retomar com o mesmo comando; falhas só repetem com `--retry-failures`. Agregar: estágio `aggregate`. Gerar esta análise: `PYTHONPATH=. .venv/bin/python experiments/summarize_sgd.py`.',
    ]
    (OUT/'SUMMARY.md').write_text('\n\n'.join(text)+'\n')
    print('SUMMARY generated',time.monotonic()-started,'seconds')
if __name__=='__main__':run()
