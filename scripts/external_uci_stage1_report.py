"""Portuguese STAGE 1 status and integrity report, without outcome analysis."""
from datetime import datetime,timezone
from pathlib import Path
import json
import re
import shutil
import pandas as pd
from dpmeta.provenance import file_sha256,save_source_snapshot
from scripts import external_uci_stage1 as stage


def main():
    stage.verify();out=stage.OUT
    selection=json.loads(stage.MAN.with_suffix('.selection.json').read_text())
    assert file_sha256(stage.MAN)==selection['manifest_sha256']
    assert file_sha256(out/'candidate_order.csv')==selection['candidate_order_sha256']
    assert file_sha256(out/'selection_decisions.csv')==selection['decisions_sha256']
    man=pd.read_csv(stage.MAN);dec=pd.read_csv(out/'selection_decisions.csv')
    coverage=json.loads((out/'confirmatory/coverage.json').read_text())
    audit=pd.read_csv(out/'confirmatory/privacy_audit.csv')
    rec=pd.read_csv(out/'recommendations.csv')
    assert len(audit)==len(man)*15*8*3==coverage['ledger_audits']==1800
    assert audit.ledger_audit_passed.all()
    assert not audit.duplicated(stage.frozen.KEYS).any()
    assert len(rec)==len(man)*8*3 and rec.public_features_only.all()
    assert not rec.duplicated(['dataset_id','epsilon','selector']).any()
    assert all(r.split_seed==stage._seed_for(stage.frozen.RUN_SEED,int(r.dataset_id),int(r.repetition),'portfolio_split') for r in audit.itertuples())
    assert man.semantic_family_id.nunique()==len(man)
    assert man.n_instances.between(200,50000).all() and man.n_classes.between(2,30).all() and man.n_encoded_features.le(512).all()
    # Stage 1 never emits analysis artifacts. Do not read/aggregate BA columns.
    assert not any(out.rglob('analysis.json')) and not any(out.rglob('family_gains.csv'))
    failures=list((out/'confirmatory/failure_details').glob('*.json'));assert not failures
    resume=(out/'resume_verification.log').read_text();assert '5 completos; 0 pendentes' in resume
    tests=(out/'pytest_final.log').read_text();assert '111 passed' in tests
    now=datetime.now(timezone.utc)
    initial=re.search(r'2026-\d\d-\d\dT\S+', (out/'LOG.md').read_text()).group(0)
    elapsed=(now-datetime.fromisoformat(initial)).total_seconds()
    model_log=(out/'model_execution.log').read_text()
    dates=[datetime.fromisoformat(x) for x in re.findall(r'2026-\d\d-\d\dT\S+',model_log)]
    model_seconds=(dates[-1]-dates[0]).total_seconds()
    sums=[json.loads(p.read_text())['seconds'] for p in (out/'confirmatory/datasets').glob('*.json')]
    snapshot=save_source_snapshot(out,extra_files=('dpmeta/uci_external.py','scripts/external_uci_catalog.py',
        'scripts/external_uci_stage1.py','scripts/external_uci_stage1_report.py',
        'scripts/confirmatory.py','scripts/parallel_nb8u.py','scripts/parallel_sgd.py',
        'tests/test_uci_external.py','tests/test_uci_external_archives.py'))
    verification=dict(at=now.isoformat(),amendment_sha256=file_sha256(stage.AMENDMENT),manifest_sha256=file_sha256(stage.MAN),
        frozen_models_and_dependencies_verified=True,protected_tree_files_verified=len(json.loads((out/'protected_sha256.json').read_text())),
        complete=True,conditions=1800,ledger_audits=1800,recommendations=120,public_features_only=True,
        resume_completed=5,resume_pending=0,tests_passed=111,primary_analysis_computed=False,
        elapsed_seconds=elapsed,model_wall_seconds=model_seconds,worker_seconds=sum(sums),
        stage2_required=len(man)<30,disk_free_bytes=shutil.disk_usage(out).free)
    stage.atomic_json(out/'verification_final.json',verification)
    reasons={
        'excluded_name_or_family':'Nome normalizado ou família canônica/semântica já existente',
        'excluded_uncertain_same_base':'Possível mesma base de origem (exclusão conservadora)',
        'excluded_rows':'Número público de linhas fora de 200–50.000',
        'excluded_declared_target':'Ausência de alvo único declarado pela fonte',
        'excluded_feature_lower_bound':'Número mínimo de atributos codificados acima de 512',
        'excluded_declared_origin':'Origem declarada já presente',
        'excluded_selected_family':'Outra representante da mesma família já selecionada',
        'validation_loading_failure':'Falha de validação/carregamento',
        'accepted':'Família aceita'}
    failure_translations={
        'encoded_features_above_512':'Mais de 512 atributos codificados.',
        'unsupported_declared_type':'Tipo declarado Date não suportado pelo schema existente.',
        'source_row_count_disagrees_with_metadata':'Quantidade de linhas do arquivo diverge do metadado público.',
        'invalid_feature_names':'Nomes de atributos/alvo declarados não formam schema válido.',
        'no_unique_archive_table_matching_declared_schema':'Arquivo não contém uma única tabela identificável que corresponda exatamente ao schema declarado.'}
    lines=['# STAGE 1 — UCI: concluído',f'Atualizado em **{now.isoformat()} (UTC)**.',
        '**5 famílias UCI elegíveis obtidas**: Land Mines; National Poll on Healthy Aging (NPHA); Rice (Cammeo and Osmancik); Turkish Music Emotion; Raisin.',
        'A API pública UCI forneceu 690 entradas, com metadados obtidos sem falhas: 526 declaram classificação e 164 não declaram. Os 526 candidatos foram permutados com semente **20261112**; toda a ordem foi percorrida. O catálogo OpenML foi atualizado para **25.245 IDs** nos estados ativo, desativado e em preparação. Todos os manifestos de `data/`, famílias de `dpmeta.families`, nomes normalizados e origens públicas foram considerados.',
        '| Decisão / motivo | Quantidade |\n|---|---:|']
    lines[-1]+='\n'+'\n'.join(f'| {reasons[k]} | {v} |' for k,v in sorted(dec.status.value_counts().items()))
    lines+=['| UCI | Dataset | Linhas | Atributos codificados | Classes |\n|---:|---|---:|---:|---:|'+''.join(f'\n| {r.uci_id} | {r.dataset_name} | {r.n_instances} | {r.n_encoded_features} | {r.n_classes} |' for r in man.itertuples()),
        '**Falhas de validação/carregamento: 11.** Nenhuma substituição usou resultados de modelos. Não foram escolhidos alvos, variantes ou arquivos ambíguos; não houve limpeza adicional de valores.',
        '| UCI | Dataset | Falha |\n|---:|---|---|'+''.join(f'\n| {r.uci_id} | {r["name"]} | {next((pt for key,pt in failure_translations.items() if key in r.reason),r.reason)} |' for _,r in dec[dec.status.eq('validation_loading_failure')].iterrows()),
        '**Execução: 1.800/1.800 ajustes privados concluídos**, correspondendo a 5 datasets × 15 opções × 8 ε × 3 repetições. **1.800 auditorias de ledger aprovadas; zero falhas de execução.** Split estratificado 75/25, semente base 20260918, namespaces e transformações idênticos à execução confirmatória interna; 8 workers, `OMP_NUM_THREADS=1`, parada abaixo de 5 GiB.',
        '**120 recomendações** dos três seletores congelados, calculadas somente a partir de features públicas. Hashes dos modelos e dependências congeladas verificados; nenhum retreino. Manifesto e decisões finais foram gravados antes das recomendações e dos ajustes.',
        f'Tempo total decorrido: **{elapsed/60:.2f} minutos** ({elapsed:.2f} s), incluindo aquisição, revisão, testes e execução. Execução dos modelos, em tempo de parede: **{model_seconds:.2f} s**; soma dos tempos dos workers: {sum(sums):.2f} s. Espaço livre final: {verification["disk_free_bytes"]/1024**3:.2f} GiB.',
        '**Verificação:** 111 testes passaram (14 avisos de dependências); retomada validada com 5 datasets completos e 0 pendentes, sem refazer ajustes. Hashes dos protocolos e de todos os arquivos previamente existentes em `results/holdout-openml/` conferem. Nenhum comando git foi executado. `ucimlrepo` não foi necessário; acesso direto via `requests`. Versões adicionais: `openpyxl==3.1.5`, `xlrd==2.0.2`, registradas em `adapter_dependencies.json`.',
        '**Etapa 2 (Kaggle) necessária: SIM**, pois **5 < 30**. Kaggle não foi iniciado.',
        '**Análise primária NÃO calculada.** Não foram calculados ganhos entre seletores, bootstrap, caudas ou testes de hipótese. A execução encerra nesta etapa.',
        'Arquivos: `data/external_manifest_uci.csv` e `.selection.json`; `candidate_order.csv`; `selection_decisions.csv`; `origin_review.json`; `recommendations.csv`; `confirmatory/conditions.csv`, `privacy_audit.csv`, `ledgers.csv` e commits por dataset; `verification_final.json`; `LOG.md` com timestamps UTC.',
        'Ressalva operacional: o inventário inicial usou `find ..` com filtro de caminhos RF-12; esse filtro não impede a travessia de diretórios, portanto pode ter enumerado seus nomes. Nenhum conteúdo de arquivo da versão anterior foi aberto ou escrito. As operações seguintes ficaram restritas à versão final e às dependências autorizadas. Essa ressalva não afetou a seleção nem os modelos.',
        'Para retomar: `PYTHONPATH=. .venv/bin/python scripts/external_uci_stage1.py run`. Para regenerar este status: `PYTHONPATH=. .venv/bin/python scripts/external_uci_stage1_report.py`.']
    (out/'STAGE1_STATUS.md').write_text('\n\n'.join(lines)+'\n')
    stage.log('Verificação final: 111 testes passaram, retomada 5 completos/0 pendentes, 1.800 ledgers, 120 recomendações públicas, hashes protegidos OK. STAGE1_STATUS.md escrito em português; etapa 2 necessária; nenhuma análise primária.')
    stage.log('Ressalva operacional registrada: inventário inicial find .. com filtro RF-12 não poda diretórios; pode ter enumerado nomes, sem leitura de conteúdo/escrita de arquivos RF-12. Escopo corrigido nas operações seguintes.')
    paths=[p for p in out.rglob('*') if p.is_file() and p.name not in {'ARTIFACTS_SHA256SUMS','LOG.md','final_verification.log'} and '__pycache__' not in p.parts]
    paths += [stage.MAN,stage.MAN.with_suffix('.selection.json')]
    (out/'ARTIFACTS_SHA256SUMS').write_text(''.join(f'{file_sha256(p)}  {p}\n' for p in sorted(paths)))
    stage.log(f'Inventário SHA256 de {len(paths)} artefatos salvo (LOG.md e saída corrente final_verification.log excluídos). STAGE 1 encerrado.')

if __name__=='__main__':main()
