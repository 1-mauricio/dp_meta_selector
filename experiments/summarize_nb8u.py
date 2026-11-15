"""Paired dataset means, semantic-family cluster bootstrap, and oracle audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from dpmeta.portfolio_benchmark import NB8U_PIPELINE, PIPELINES, validate_condition_grid
from dpmeta.openml_benchmark import _load_manifest
from dpmeta.provenance import file_sha256, numerical_implementation_sha256, save_source_snapshot

OUT = Path('results/portfolio-984-nb8u')
ROOT = Path.cwd()
N_BOOT = 20_000
BOOT_SEED = 20261110
KEYS = ['dataset_id', 'epsilon', 'repetition']


def cluster_bootstrap(block, rng):
    # Resample families with replacement and retain every member dataset and its
    # paired repetition mean. Ratio preserves equal dataset weighting, while
    # accounting for within-family dependence.
    groups = block.groupby('bootstrap_family').delta.agg(['sum', 'count'])
    sums, counts = groups['sum'].to_numpy(), groups['count'].to_numpy()
    estimates = np.empty(N_BOOT)
    for first in range(0, N_BOOT, 256):
        size = min(256, N_BOOT-first)
        index = rng.integers(0, len(groups), size=(size,len(groups)))
        estimates[first:first+size] = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    lower, upper = np.quantile(estimates, [.025,.975])
    return float(block.delta.mean()), float(lower), float(upper), len(groups)


def main():
    config = json.loads((OUT/'run_config.json').read_text())['original_run_config']
    coverage = json.loads((OUT/'coverage.json').read_text())
    alignment = json.loads((OUT/'alignment.json').read_text())
    manifest = _load_manifest(Path('data/openml_manifest_984_completed.csv'))
    new = pd.read_csv(OUT/'conditions_new_option.csv',float_precision='round_trip')
    combined = pd.read_csv(OUT/'conditions_13.csv',float_precision='round_trip')
    old = pd.read_csv('results/portfolio-984-current-slim/conditions.csv',float_precision='round_trip')
    validate_condition_grid(new,manifest,epsilons=tuple(config['epsilons']),repeats=config['repeats'],
                            pipelines=(NB8U_PIPELINE,))
    validate_condition_grid(combined,manifest,epsilons=tuple(config['epsilons']),repeats=config['repeats'],
                            pipelines=(*PIPELINES,NB8U_PIPELINE))
    # Exact round-trip equality of all retained original values, no rewriting of
    # their clean reference or recomputing the original portfolio's regret.
    original_retained = old[old.dataset_id.isin(new.dataset_id)].sort_values(KEYS+['pipeline']).reset_index(drop=True)
    combined_original = combined[combined.pipeline.ne(NB8U_PIPELINE)].sort_values(KEYS+['pipeline']).reset_index(drop=True)
    pd.testing.assert_frame_equal(original_retained,combined_original,check_exact=True)
    ref = old[old.pipeline.eq(PIPELINES[10])]
    pairs = new.merge(ref,on=KEYS,suffixes=('_new','_current'),validate='one_to_one')
    assert np.array_equal(pairs.best_clean_portfolio_ba_new,pairs.best_clean_portfolio_ba_current)
    assert np.allclose(pairs.portfolio_regret_new,
                       pairs.best_clean_portfolio_ba_current-pairs.private_ba_new,atol=1e-15,rtol=0)
    pairs['delta'] = pairs.private_ba_new - pairs.private_ba_current
    metadata = manifest[['dataset_id','semantic_family_id','n_features','n_nominal_features']].copy()
    metadata['has_numeric'] = metadata.n_features.gt(metadata.n_nominal_features)
    metadata['bootstrap_family'] = metadata.semantic_family_id
    missing = metadata.bootstrap_family.eq('')
    metadata.loc[missing,'bootstrap_family'] = metadata.loc[missing,'dataset_id'].map(lambda d:f'__missing_dataset_{d}')
    means = pairs.groupby(['dataset_id','epsilon'],as_index=False).agg(
        delta=('delta','mean'),ba_new=('private_ba_new','mean'),ba_current=('private_ba_current','mean'),
        repetitions=('repetition','nunique')).merge(metadata,on='dataset_id',validate='many_to_one')
    assert means.repetitions.eq(3).all()
    pairs.to_csv(OUT/'paired_conditions.csv',index=False)
    means.to_csv(OUT/'paired_dataset_means.csv',index=False)
    means.groupby(['bootstrap_family','epsilon'],as_index=False).agg(
        delta=('delta','mean'),datasets=('dataset_id','nunique')).to_csv(OUT/'paired_family_means.csv',index=False)
    rng=np.random.default_rng(BOOT_SEED)
    stats=[]
    for group, selected in [('all',means),('has_numeric',means[means.has_numeric]),('all_nominal',means[~means.has_numeric])]:
        for epsilon, block in selected.groupby('epsilon',sort=True):
            mean, lower, upper, families = cluster_bootstrap(block,rng)
            stats.append(dict(group=group,epsilon=float(epsilon),datasets=len(block),families=families,
                              mean_ba_new=float(block.ba_new.mean()),mean_ba_current=float(block.ba_current.mean()),
                              mean_delta=mean,ci95_low=lower,ci95_high=upper,bootstrap_replicates=N_BOOT,
                              bootstrap_seed=BOOT_SEED))
    stats=pd.DataFrame(stats)
    stats.to_csv(OUT/'paired_bootstrap.csv',index=False)
    best = old.groupby(KEYS).private_ba.max().rename('best_original_private_ba')
    oracle = new.merge(best,on=KEYS,validate='one_to_one')
    oracle['strict_win'] = oracle.private_ba.gt(oracle.best_original_private_ba+1e-12)
    oracle['tied_win'] = abs(oracle.private_ba-oracle.best_original_private_ba).le(1e-12)
    oracle['oracle_win_including_ties'] = oracle.strict_win | oracle.tied_win
    oracle['oracle_gain'] = np.maximum(oracle.private_ba-oracle.best_original_private_ba,0.)
    oracle.to_csv(OUT/'oracle_conditions.csv',index=False)
    wins=oracle.groupby('epsilon',as_index=False).agg(
        conditions=('dataset_id','size'),strict_wins=('strict_win','sum'),tied_wins=('tied_win','sum'),
        oracle_wins_including_ties=('oracle_win_including_ties','sum'),mean_oracle_gain=('oracle_gain','mean'))
    wins['strict_win_pct'] = 100*wins.strict_wins/wins.conditions
    wins['oracle_win_including_ties_pct'] = 100*wins.oracle_wins_including_ties/wins.conditions
    wins.to_csv(OUT/'oracle_summary.csv',index=False)
    content_reference=pd.read_csv('data/openml_manifest_984_content_audit.csv')
    content_rows=[]
    completed_ids=set(new.dataset_id)
    for r in content_reference.itertuples():
        original_path=Path(r.local_data_path)
        cache=Path(f'.cache/openml/org/openml/www/datasets/{r.dataset_id}')
        actual=(original_path if original_path.is_file() else
                cache/f'dataset_{r.dataset_id}.pq' if (cache/f'dataset_{r.dataset_id}.pq').is_file()
                else cache/'dataset.arff')
        exists=actual.is_file()
        parquet=exists and actual.suffix in ('.pq','.parquet')
        digest=file_sha256(actual) if exists else None
        content_rows.append(dict(dataset_id=int(r.dataset_id),completed=int(r.dataset_id) in completed_ids,
            original_path=str(original_path),actual_path=str(actual),exists=exists,
            format='parquet' if parquet else 'arff' if exists else 'missing',
            original_sha256=r.sha256,actual_sha256=digest,byte_comparison_available=bool(parquet),
            byte_identical=(digest==r.sha256) if parquet else None))
    content=pd.DataFrame(content_rows)
    content.to_csv(OUT/'data_content_audit.csv',index=False)
    comparable=content[content.completed & content.byte_comparison_available]
    if not comparable.byte_identical.all():
        raise AssertionError('downloaded parquet differs from original byte-level audit')
    protected=json.loads((OUT/'protected_hashes.json').read_text())
    changed = [p for p,h in protected.items() if not Path(p).is_file() or file_sha256(Path(p))!=h]
    if changed: raise AssertionError(f'protected files changed: {changed}')
    tests=(OUT/'pytest.log').read_text()
    test_result=re.search(r'\d+ passed[^\n]*',tests).group(0)
    attempts=json.loads((OUT/'attempts.json').read_text())
    runtime=attempts[-1]['ended_unix']-attempts[0]['started_unix']
    runner_seconds=sum(a.get('seconds',0.) for a in attempts)
    timing=pd.read_csv(OUT/'timing.csv')
    failures=pd.read_csv(OUT/'failures.csv')
    audit=pd.read_csv(OUT/'privacy_audit.csv')
    assert len(audit)==len(new) and audit.ledger_identical_to_current_nb.all()
    assert audit.prior_query_retained.all() and audit.numeric_bins.eq(8).all()
    assert audit.delta_used.eq(0).all()
    assert np.allclose(audit.epsilon_used,audit.epsilon,atol=1e-10,rtol=0)
    files=['conditions_new_option.csv','conditions_13.csv','ledgers.csv','privacy_audit.csv',
           'paired_bootstrap.csv','oracle_summary.csv','failures.csv','alignment_conditions.csv','alignment_arff_350.csv','data_content_audit.csv','validation_rejections.json','alignment_guard_amendment.json']
    verification=dict(coverage=coverage,alignment_passed=alignment['passed'],
                      original_rows_exactly_preserved=True,
                      protected_files_checked=len(protected),protected_files_changed=changed,
                      private_ledgers_verified=len(audit),test_result=test_result,
                      supplementary_alignment_350_passed=False,
                      rejected_dataset_ids=[350],all_retained_inputs_byte_identical=True,
                      implementation_sha256=numerical_implementation_sha256(),
                      files_sha256={name:file_sha256(OUT/name) for name in files})
    save_source_snapshot(OUT, extra_files=('scripts/parallel_nb8u.py', 'experiments/summarize_nb8u.py',
                        'tests/test_nb8u.py', 'tests/fixtures/portfolio12_golden.json',
                        'experiments/fetch_har_blocks.py', 'tests/test_nb8u_runner.py'))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    incomplete = coverage['unattempted_ids']
    text = [
        f"A opção `{NB8U_PIPELINE}` foi implementada e avaliada no manifesto completo de **984 datasets**. Não houve substituições. "
        f"Concluíram **{coverage['completed']}/984**; **{coverage['failed']}** falharam; **{len(incomplete)}** ficaram sem tentativa. "
        f"Foram gravadas **{len(new):,} condições novas** e **{len(combined):,} linhas** na grade com 13 opções. ".replace(',', '.'),
        "`conditions_13.csv` contém apenas condições com uma linha de cada uma das 13 opções. "
        "Todos os datasets concluídos têm 8 ε × 3 repetições. `conditions_new_option.csv` mantém exatamente as nove colunas do arquivo slim, com BA em escala 0–1.",
        "Em `dpmeta/classifiers.py`, `numeric_bins` passou a ser parâmetro das estatísticas e ajustes NB; `uniform_prior` troca as contagens de classe por uns somente no modelo de decisão, exatamente como E4. "
        "A consulta de prior, seu sorteio e os histogramas condicionais são preservados. O novo clean também usa 8 bins e prior uniforme. "
        "Em `dpmeta/portfolio_benchmark.py`, `PIPELINE_REGISTRY` contém as 13 opções; `PIPELINES` conserva o padrão histórico de 12 e sua ordem. "
        "`run_dataset(..., pipelines=(... ,))` e `--pipelines` selecionam opções, com caminho dedicado para NB. "
        "A opção nova usa a mesma namespace de ruído do NB atual. A execução padrão continua pelo corpo histórico sem alterar as sementes e resultados.",
        f"Testes: **{test_result}** (`pytest.log`). `tests/test_nb8u.py` cobre ledger idêntico nos oito ε, retenção da consulta de prior, "
        "efeito do prior uniforme com histogramas condicionais iguais, equivalência ao wrapper E4, fronteiras dos 8 bins, "
        "execução isolada da opção nova e referência clean original. `pyproject.toml` limita a descoberta à suíte canônica `tests/`, evitando coletar cópias de testes dentro dos snapshots de resultados. A fixture `tests/fixtures/portfolio12_golden.json`, "
        "capturada antes das alterações, confirma igualdade exata de 48 saídas das 12 opções, incluindo ledgers, em dois ε e duas repetições. `legacy_preservation.json` confirma também que o corpo histórico de `run_dataset` foi preservado byte a byte (normalizando apenas o nome da função).",
        f"Alinhamento antes do benchmark: **10 datasets, 240 condições**, cinco com atributos numéricos e cinco somente nominais. "
        f"O NB atual foi reexecutado por `run_dataset`; erro máximo de BA privada **{alignment['max_abs_error']:.3g}**, abaixo de 10⁻¹². "
        "A configuração original foi preservada: seed **20260917**, transformação `tanh`, split estratificado 75/25 "
        "com namespace `portfolio_split`, e ruído com namespace `categorical_nb_laplace`. Não foi necessário corrigir sementes ou splits. "
        "Os detalhes estão em `alignment.json`, `alignment_manifest.csv` e `alignment_conditions.csv`. A semente exploratória 20261110 do E4 não foi usada no benchmark. Parear sementes não torna o ruído coordenada a coordenada idêntico quando mudar bins altera as dimensões dos histogramas.",
        f"Execução: `scripts/parallel_nb8u.py`, **8 workers**, uma thread em OMP/BLAS e cache `.cache/openml`. "
        f"Janela de execução do benchmark, incluindo retomadas e recuperação de download: **{runtime/60:.2f} min** em **{len(attempts)} tentativa(s)** (runners ativos por {runner_seconds/60:.2f} min); soma dos tempos de datasets concluídos: {timing.seconds.sum()/60:.2f} min; "
        f"maior duração concluída: {timing.seconds.max():.2f} s. Downloads estão incluídos. "
        f"Espaço livre final: **{attempts[-1]['free_after']/1024**3:.2f} GiB**; "
        f"parada por espaço: **{'sim' if any(a.get('disk_stopped') for a in attempts) else 'não'}**. "
        "O runner verifica o limite de 5 GiB antes e depois de cada dataset. Commits JSON por dataset são atômicos; "
        "a retomada valida a grade e os hashes de código/configuração, reaproveita somente datasets completos e mantém falhas registradas.",
        "Os parquets locais indicados pelo manifesto não estavam disponíveis; nesses casos foi acionado o ramo de download do loader original. "
        "As validações de atributos, ordem, classes, rótulos ausentes e número público de linhas permaneceram ativas. "
        "Nenhum dataset foi substituído ou teve linhas removidas para completar a grade.",
        f"Falhas (**{len(failures)}**): " + (', '.join(str(int(d)) for d in failures.dataset_id) if len(failures) else 'nenhuma') + '. '
        + "`failures.csv` e `failure_details/` registram tipo, mensagem, traceback e duração por dataset."
    ]
    first_log = (OUT/'run.log').read_text()
    first_failed = set(map(int, re.findall(r'^FAIL (\d+):', first_log, flags=re.M)))
    recovered = sorted(first_failed & set(new.dataset_id))
    text.append(f'Falhas da primeira passagem recuperadas na retomada: **{len(recovered)}**. Os logs de cada tentativa foram preservados; nenhuma recuperação substituiu um dataset por outro.')
    if len(failures):
        text.append('Tipos de falha: ' + '; '.join(f'{k}: {v}' for k,v in failures.error_type.value_counts().items()) + '.')
    if (OUT/'har_download.json').exists():
        har=json.loads((OUT/'har_download.json').read_text())
        text.append(f'O download MinIO lento de HAR (1478) foi interrompido após salvar os demais 973 resultados. `experiments/fetch_har_blocks.py` recuperou exatamente o mesmo parquet oficial por HTTP Range com oito conexões, em {har["seconds"]:.2f} s; o arquivo completo passou MD5 contra o ETag oficial, SHA-256 idêntico ao registrado na auditoria de conteúdo original, leitura parquet e validação de dimensão. O runner foi retomado com o cache completo e todas as validações originais; não houve substituição de dataset. Os detalhes estão em `har_download.json`, `har_blocks.log` e nos logs de todas as tentativas.')
    text.append(f'Auditoria adicional de conteúdo: todos os **{len(comparable)} datasets retidos** têm parquets com SHA-256 exatamente igual ao da execução original. `data_content_audit.csv` registra os caminhos e hashes. A validação suplementar do dataset 350 por ARFF falhou: erro máximo de BA privada 0,06194987001614971 em 24 condições (`alignment_arff_350.csv`). O parquet fixado não estava na cópia, e seu endpoint oficial retornou 404. Esse dataset foi excluído integralmente, sem substituição; o resultado bruto foi preservado em `rejected_datasets/350.json`. A equivalência dos dados/decodificação ARFF não pôde ser certificada sem o parquet fixado. As sementes e regras de split eram as mesmas; não se tentou ajustar resultados para fazê-los coincidir.')
    text.append('`validation_rejections.json` e a proteção em `scripts/parallel_nb8u.py` impedem que o 350 volte à grade por agregação ou retomada, inclusive com `--retry-failures`. `alignment_guard_amendment.json`, `run_config_before_alignment_guard.json` e `source_before_alignment_guard/` preservam a revisão anterior do executor; somente essa proteção de entrada mudou, e o hash da implementação numérica permaneceu igual. A retomada final confirmou zero datasets pendentes, sem treinar novamente os resultados válidos. `tests/test_nb8u_runner.py` verifica a exclusão mesmo se uma grade bruta completa for recolocada entre os commits.')

    if incomplete: text.append('Datasets sem tentativa: '+', '.join(map(str,incomplete))+'.')
    text.extend([
        "O regret da opção nova usa **o `best_clean_portfolio_ba` original da mesma combinação dataset/ε/repetição**, "
        "sem atualizar essa referência com o clean da opção nova. Assim, `portfolio_regret = best_clean_original − private_ba`; "
        "o penalty é `clean_ba_new − private_ba_new`. As linhas originais retidas foram verificadas por igualdade exata após leitura com round-trip de floats.",
        "Comparação pareada: primeiro se calcula Δ BA = nova − NB atual em cada condição; depois se tira a média das três repetições por dataset. "
        "As médias abaixo dão peso igual a cada dataset concluído. Os ICs percentis de 95% usam **20.000 reamostragens de famílias semânticas**, "
        "seed 20261110, mantendo juntos os membros de cada família e os pares. A razão entre soma de deltas e número de datasets reamostrados "
        "preserva o estimando com peso igual por dataset. Os estratos `has_numeric` (inclui mistos) e `all_nominal` são reamostrados separadamente. "
        "Famílias semânticas ausentes recebem uma chave exclusiva por dataset; os deltas e identificadores ficam em `paired_dataset_means.csv` e `paired_family_means.csv`. "
        "Os intervalos são pontuais, sem correção por múltiplas comparações. Valores da tabela estão em **pontos percentuais**.",
        '| Grupo | ε | Datasets | Famílias | Δ BA (pp) | IC 95% (pp) |',
        '| --- | ---: | ---: | ---: | ---: | --- |'
    ])
    for r in stats.itertuples():
        text.append(f'| {r.group} | {r.epsilon:g} | {r.datasets} | {r.families} | {100*r.mean_delta:+.3f} | [{100*r.ci95_low:+.3f}; {100*r.ci95_high:+.3f}] |')
    total=len(oracle); strict=int(oracle.strict_win.sum()); tied=int(oracle.tied_win.sum())
    text.extend(['', f"A opção nova seria vencedora do oracle privado em **{strict+tied}/{total} condições ({100*(strict+tied)/total:.2f}%)**, "
                 f"incluindo empates: **{strict} vitórias estritas ({100*strict/total:.2f}%)** e **{tied} empates ({100*tied/total:.2f}%)**. "
                 "Vitória estrita exige superar todas as 12 BAs privadas originais por mais de 10⁻¹²; empate usa tolerância de 10⁻¹². "
                 "O denominador inclui apenas condições com as 13 opções. Oracle é diagnóstico em teste, não um seletor validado.",
                 '| ε | Condições | Vitórias estritas | Empates | Vence incl. empates (%) | Ganho médio do oracle (pp) |',
                 '| ---: | ---: | ---: | ---: | ---: | ---: |'])
    for r in wins.itertuples():
        text.append(f'| {r.epsilon:g} | {r.conditions} | {r.strict_wins} | {r.tied_wins} | {r.oracle_win_including_ties_pct:.2f} | {100*r.mean_oracle_gain:.3f} |')
    text.extend(['', f"Privacidade: **{len(audit):,} ajustes** passaram a auditoria de ledger completo idêntico ao NB atual. ".replace(',','.')
                 + "Cada ajuste mantém p+1 histogramas Laplace, ε/(p+1) por consulta, sensibilidade L1=2, escala 2(p+1)/ε e δ=0. "
                 "`ledgers.csv` armazena os ledgers completos por SHA-256 e `privacy_audit.csv` vincula cada condição ao ledger, sementes e tamanho do split. "
                 "Não foi emitido release extra de metafeatures nesta extensão. O ε refere-se a cada ajuste; a divulgação conjunta de ajustes exige composição adicional. "
                 "Clean e BA de teste são diagnósticos em dados públicos.",
                 f"Integridade: **{len(protected)} arquivos protegidos** (Markdown preexistente, dados e slim original) mantiveram seus hashes. "
                 "Nenhum comando git foi usado e nenhum arquivo da versão anterior foi escrito. `source_snapshot/`, `run_config.json`, "
                 "`attempts.json`, `coverage.json` e `verification.json` registram fontes, versões, hashes, tentativas, cobertura e verificações finais. "
                 "As falhas de validação delimitam a população efetivamente comparada; as médias não imputam resultados aos datasets ausentes.",
                 "Reprodução/retomada no diretório do repo: `PYTHONPATH=. OMP_NUM_THREADS=1 OPENML_CACHE_DIR=$PWD/.cache/openml "
                 ".venv/bin/python scripts/parallel_nb8u.py run --workers 8`. A etapa `alignment` deve preceder a primeira execução. "
                 "Falhas só são tentadas novamente com `--retry-failures`; não há substituição. Para regenerar os CSVs: etapa `aggregate`; "
                 "para análise e este único Markdown novo: `PYTHONPATH=. .venv/bin/python experiments/summarize_nb8u.py`."])
    report = '\n\n'.join(text)+'\n'
    report = re.sub(r'(?<=\|)\n\n(?=\|)', '\n', report)
    report = re.sub(r'\n{3,}', '\n\n', report)
    (OUT/'SUMMARY.md').write_text(report)
    print(json.dumps(verification,indent=2))

if __name__=='__main__': main()
