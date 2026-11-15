"""Finalize Portuguese reporting from verified, completed results only."""
from pathlib import Path
import json
import re
import shutil
import pandas as pd
from scripts.confirmatory import OUT, MANIFEST, EXPECTED, now, log

def number(value,digits=3,signed=False):
    return format(float(value),('+' if signed else '')+f'.{digits}f').replace('.',',')

def effect(row):
    return f"{number(100*row['gain'],signed=True)} p.p.; IC95% [{number(100*row['ci_low'])}; {number(100*row['ci_high'])}] p.p."

def main():
    audit=json.loads((OUT/'verification_final.json').read_text())
    assert audit['checks']['no_model_run_failures']
    c=json.loads((OUT/'confirmatory/analysis.json').read_text())
    e=json.loads((OUT/'exploratory_41/analysis.json').read_text())
    s=json.loads(MANIFEST.with_suffix('.selection.json').read_text())
    m=pd.read_csv(MANIFEST)
    old=pd.read_csv('data/openml_confirmatory_manifest_41_combined.csv')
    decisions=pd.read_csv(OUT/'selection_decisions.csv')
    starts={};ends={}
    for line in (OUT/'LOG.md').read_text().splitlines():
        match=re.match(r'^- (\S+) — Etapa ([1-7]) (INÍCIO|CONCLUÍDA)',line)
        if match:
            stamp,stage,kind=match.groups()
            if kind=='INÍCIO':starts.setdefault(stage,stamp)
            else:ends[stage]=stamp
        if 'Etapa 7: redação' in line:starts.setdefault('7',line.split()[1])
    ends['7']=now()
    status='ATINGIDO' if c['success'] else 'NÃO ATINGIDO'
    lines=['# Confirmação final — famílias novas','',
        f"**Critério H2a final: {status}.** Ganho sobre b8u: **{effect(c['primary'])}**.",
        'O critério congelado exige limite inferior do IC95% > 0. Foram usadas somente duas famílias elegíveis; o resultado não confirma transferência e tem poder/generalização muito limitados.','',
        f'Protocolo SHA256: `{EXPECTED}`. O [protocolo original](../../protocols/CONFIRMATORY_PROTOCOL_FINAL.md) foi preservado. Nenhum comando git foi executado; o checkout anterior não foi acessado.','',
        'Cronologia em UTC (11/11/2026; registros completos no [LOG.md](LOG.md)):','',
        '| Etapa | Início UTC | Fim UTC |','|---|---|---|']
    names={'1':'Congelamento dos seletores','2':'Seleção/validação do holdout','3':'15 opções no novo holdout',
        '4':'Análise confirmatória','5':'Extensão/análise exploratória antiga','6':'Figuras PDF+PNG','7':'Resumo e QA final'}
    for stage in map(str,range(1,8)):
        lines.append(f"| {stage}. {names[stage]} | {starts.get(stage,'—')} | {ends.get(stage,'—')} |")
    lines += ['', 'Os três seletores foram salvos com hashes antes da seleção: GBM-15 (15 opções, fallback b8u, τ=0,02), GBM-13 (13 opções, τ=0,01), ambos em 973 datasets; RF-12 em 984 datasets/12 opções. Features, alvos, pesos e hiperparâmetros seguem as rotinas especificadas. Modelos e hashes de treinamento: [frozen.json](frozen.json), [SHA256SUMS](models/SHA256SUMS).','',
        f"Amostra nova: **{len(m)} famílias/datasets**, n de **{int(m.n_instances.min()):,} a {int(m.n_instances.max()):,}** (separador de milhar abaixo normalizado). O alvo de 40 não foi alcançado: foram esgotados os {len(decisions)} candidatos da ordem sorteada no catálogo ativo; o protocolo autoriza usar todas as elegíveis.",
        '', '| ID OpenML | Dataset/família | n | Classes | Atributos codificados |', '|---:|---|---:|---:|---:|']
    for r in m.itertuples():
        lines.append(f'| {r.dataset_id} | `{r.dataset_name}` | {r.n_instances} | {r.n_classes} | {r.n_encoded_features} |')
    lines += ['',f"Elegibilidade: 200–50.000 linhas, 2–30 classes, ≤512 atributos codificados. Disjunção por ID, família canônica e semântica contra os **{len(s['exclusions'])} CSVs** de data/, listados no LOG e em [exclusion_manifests.json](exclusion_manifests.json). Manifesto/seleção foram salvos antes das avaliações.",
        f"Falhas/substituições: **{len(s['validation_failures'])} rejeições de validação**, substituídas somente pelo próximo candidato na permutação da semente 20261111; {len(decisions)-len(s['validation_failures'])-len(m)} exclusões por família/origem. **Sem erros de dados/ajustes** e nenhuma substituição após medir desempenho. A extensão antiga teve interrupção operacional e retomada documentadas. Detalhes em [selection_decisions.csv](selection_decisions.csv).",
        'A revisão de procedência resolveu pais públicos declarados por TabClustPFN e excluiu os 18 derivados de famílias anteriores, preservando a mesma ordem, antes de ajustar modelos no holdout. O registro inicial da validação foi arquivado.','',
        'Execução nova: **720 saídas e 720 auditorias de ledger aprovadas** (2 × 15 opções × 8 ε × 3 repetições), tanh, 8 workers, OMP_NUM_THREADS=1, commits atômicos retomáveis e guarda de disco de 5 GiB. A semente base 20260918 e os namespaces foram herdados de external_confirmation/benchmark do holdout; seleção e bootstrap usam 20261111. As recomendações dos três seletores usam somente features públicas e foram reproduzidas independentemente.','',
        'Inferência primária: média sobre ε/repetições por dataset e peso igual por família; bootstrap percentil de famílias, 20.000 reamostragens, semente 20261111.','']
    t=c['tails']
    lines += [f"Caudas (restrição, sem teste): perda >1 p.p. em **{number(100*t['fraction_loss_gt_1pp'],2)}%** das condições, IC95% [{number(100*t['fraction_ci95'][0],2)}; {number(100*t['fraction_ci95'][1],2)}]%. **CVaR90 da perda: {number(100*t['cvar90_loss'])} p.p.**, IC95% [{number(100*t['cvar90_ci95'][0])}; {number(100*t['cvar90_ci95'][1])}] p.p. Perda assinada = BA(b8u) − BA(seletor); CVaR segue a média das maiores floor(10% × condições) perdas, como no desenvolvimento.",
        '', 'Secundários: sign-flip pareado bilateral por família, 50.000 sorteios; Holm aplicado exatamente aos quatro testes. ICs abaixo são marginais.','',
        '| Referência | Ganho e IC95% | p Holm |','|---|---|---:|']
    labels={'H27':'H27','old_NB':'NB antigo','gbm13_selector':'Seletor GBM-13','rf12_selector':'Seletor RF-12'}
    for r in c['secondaries']:
        lines.append(f"| {labels[r['reference']]} | {effect(r)} | {number(r['p_holm'],5)} |")
    t=e['tails']
    lines += ['', '**EXPLORATÓRIO — holdout antigo já aberto:**',
        f"{e['datasets']} datasets/{e['families']} famílias; n de {int(old.n_instances.min())} a {int(old.n_instances.max())}. Foram acrescentadas as duas opções DP-SGD: **1.968 saídas novas, todas auditadas**, mais **54 auditorias de alinhamento exato**; b8u da fase A foi reutilizada com suas 984 auditorias aprovadas. As 12.792 linhas anteriores de 13 opções foram preservadas exatamente. Após alinhamento exato em três datasets, os pendentes foram executados com o mesmo kernel privado, omitindo somente ajustes limpos auxiliares. Esses campos ficam explicitamente ausentes/nulos; nenhuma métrica limpa foi fabricada.",
        f"Ganho final vs b8u: **{effect(e['primary'])}**. Perda >1 p.p.: {number(100*t['fraction_loss_gt_1pp'],2)}%, IC95% [{number(100*t['fraction_ci95'][0],2)}; {number(100*t['fraction_ci95'][1],2)}]%; CVaR90 {number(100*t['cvar90_loss'])} p.p., IC95% [{number(100*t['cvar90_ci95'][0])}; {number(100*t['cvar90_ci95'][1])}] p.p. Os quatro secundários/Holm estão em [exploratory_41/analysis.json](exploratory_41/analysis.json). Esses números não constituem confirmação independente.",
        'Estratos por ε, faixa de n (≤1000; 1001–5000; 5001–10000; 10001–50000; >50000) e atributos numeric/nominal/mixed: [holdout antigo](exploratory_41/exploratory_strata.csv) e [holdout novo](exploratory_new_holdout/strata.csv), ambos exclusivamente exploratórios.','',
        'Figuras do holdout novo: [forest PDF](../../figures/dissertation/fig14_holdout_gains_vs_b8u.pdf) / [PNG](../../figures/dissertation/fig14_holdout_gains_vs_b8u.png); [CD PDF](../../figures/dissertation/fig13_holdout_cd.pdf) / [PNG](../../figures/dissertation/fig13_holdout_cd.png); [ganhos por ε PDF](../../figures/dissertation/gains_by_epsilon.pdf) / [PNG](../../figures/dissertation/gains_by_epsilon.png). CD é descritivo, com apenas duas famílias.','',
        '**Desvios operacionais:** os estratos exploratórios do novo holdout foram gravados ao fim da etapa 4, antes da etapa 5; a extensão antiga foi interrompida e retomada sem ajustes limpos auxiliares, após alinhamento privado exato. A [emenda datada](AMENDMENT_EXECUTION.md) mantém tudo que depende dessas mudanças como exploratório. A confirmação nova, os modelos e o código existente de dpmeta foram preservados. Não houve retuning, alteração do protocolo ou substituição por desempenho. Catálogo ativo, semente base herdada, teste bilateral e convenção CVaR foram documentados antes dos resultados.',
        'Limitações prévias: a configuração foi escolhida no desenvolvimento e com o holdout antigo já aberto; o protocolo é interno/versionado, não um pré-registro público. A escassez de famílias novas torna a interpretação particularmente limitada.','',
        'Validação: **102 testes passaram** (14 avisos), verificação independente das recomendações/primária, hashes preservados, disjunção, grades completas, preservação dos resultados antigos e seis arquivos gráficos válidos: [verification_final.json](verification_final.json), [verification_confirmatory.json](verification_confirmatory.json), [pytest_final.log](pytest_final.log).']
    # Use Portuguese thousands separators in the single prose range.
    text='\n'.join(lines).replace('4,800 a 8,993','4.800 a 8.993').replace(' (separador de milhar abaixo normalizado)','')+'\n'
    separate=OUT/'exploratory_new_holdout';separate.mkdir(exist_ok=True)
    shutil.copyfile(OUT/'confirmatory/exploratory_strata.csv',separate/'strata.csv')
    (OUT/'SUMMARY.md').write_text(text)
    log('Etapa 7 FINALIZADA: SUMMARY.md revisado, cronologia e emenda incorporadas; verificação final aprovada.')

if __name__=='__main__':main()
