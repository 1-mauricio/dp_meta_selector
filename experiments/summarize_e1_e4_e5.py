"""Dataset-paired inference and Portuguese report for E1/E4/E5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/experiments")
SEED = 20261110
NAMES = {"categorical_naive_bayes__laplace_histograms": "NB categórico",
         "softmax__dp_sgd_clip2_reg003": "Softmax DP-SGD",
         "gnb__gaussian_zcdp_equal": "GNB zCDP"}


def bootstrap(values, key):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all() and len(values)
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{key}".encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(20000, len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(means, [.025, .975]).tolist()


def paired_summary(frame, variant_columns, baseline_filter, filename):
    keys = ["dataset_id", "epsilon", "repetition"]
    base = frame.loc[baseline_filter, keys+["private_ba"]].rename(columns={"private_ba":"baseline_ba"})
    assert not base.duplicated(keys).any()
    paired = frame.merge(base, on=keys, validate="many_to_one")
    paired["delta_ba"] = paired.private_ba - paired.baseline_ba
    paired.to_csv(OUT / f"{filename}_paired_runs.csv", index=False)
    columns = ["dataset_id", "epsilon", "n_tercile", "feature_group", *variant_columns]
    dataset = paired.groupby(columns, as_index=False).agg(
        private_ba=("private_ba","mean"), baseline_ba=("baseline_ba","mean"),
        delta_ba=("delta_ba","mean"), repetitions=("repetition","nunique"))
    assert dataset.repetitions.eq(3).all()
    dataset.to_csv(OUT / f"{filename}_dataset_deltas.csv", index=False)
    summaries = []
    for scope, selected in [("all", dataset),
                            ("has_numeric", dataset[dataset.feature_group.eq("has_numeric")]),
                            ("all_nominal", dataset[dataset.feature_group.eq("all_nominal")]),
                            *[(t,dataset[dataset.n_tercile.eq(t)]) for t in ("T1","T2","T3")]]:
        for key, block in selected.groupby(["epsilon", *variant_columns], sort=True):
            avg, lo, hi = bootstrap(block.delta_ba, (filename,scope,key))
            summaries.append(dict(scope=scope, **dict(zip(["epsilon", *variant_columns],key)),
                                  n_datasets=len(block), mean_ba=block.private_ba.mean(),
                                  baseline_ba=block.baseline_ba.mean(), delta_ba=avg,
                                  ci95_low=lo, ci95_high=hi))
    result = pd.DataFrame(summaries)
    result.to_csv(OUT / f"{filename}_bootstrap_summary.csv", index=False)
    return result, dataset


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str,headers)) + " |",
                       "| " + " | ".join(["---"]*len(headers)) + " |",
                       *["| " + " | ".join(map(str,row)) + " |" for row in rows]])


def delta_cell(row):
    return f"{100*row.delta_ba:+.2f} [{100*row.ci95_low:+.2f}; {100*row.ci95_high:+.2f}]"


def main():
    meta = json.loads((OUT/"metadata_E1_E4_E5.json").read_text())
    sample_path = OUT/"effective_sample_E1_E4_E5.csv"
    sample = pd.read_csv(sample_path if sample_path.exists() else OUT/"sample_E1_E4_E5.csv")
    complete = sample[sample.dataset_id.isin(meta["completed_ids"])]
    n = len(complete)
    assert n > 0
    frames = {}
    for experiment, expected in (("E1",189),("E4",54),("E5",36)):
        raw = pd.read_csv(OUT/f"{experiment}_runs.csv")
        frame = raw[raw.dataset_id.isin(meta["completed_ids"])].copy()
        assert frame.groupby("dataset_id").size().eq(expected).all()
        assert frame.private_ba.between(0,1).all()
        assert not frame.duplicated(["dataset_id","pipeline","epsilon","repetition","mode","bins","prior","regularization"]).any()
        assert len(frame) == n*expected
        frames[experiment] = frame
    e1 = frames["E1"]
    components = []
    for (dataset_id, pipe, eps), block in e1.groupby(["dataset_id","pipeline","epsilon"]):
        noise = block[block["mode"].eq("noise")]
        split = block[block["mode"].eq("split")]
        zero = block[block["mode"].eq("zero_noise_NON_DP")].iloc[0]
        assert len(noise) == len(split) == 10
        assert noise.split_seed.nunique() == 1 and noise.dp_seed.nunique() == 10
        assert split.split_seed.nunique() == 10 and split.dp_seed.nunique() == 1
        assert noise.loc[noise.repetition.eq(0),"private_ba"].iloc[0] == split.loc[split.repetition.eq(0),"private_ba"].iloc[0]
        vn, vs = noise.private_ba.var(ddof=1), split.private_ba.var(ddof=1)
        components.append(dict(dataset_id=dataset_id,pipeline=pipe,epsilon=eps,
                               variance_dp=vn,variance_split=vs,
                               share_dp=vn/(vn+vs) if vn+vs else np.nan,
                               zero_total_variance=vn+vs==0,
                               noise_mean_ba=noise.private_ba.mean(),split_mean_ba=split.private_ba.mean(),
                               zero_noise_ba=zero.private_ba,clean_ba=zero.clean_ba,
                               optimization_clipping_gap=zero.clean_ba-zero.private_ba))
    comp = pd.DataFrame(components)
    # Noise-off predictions must be invariant to epsilon and match analytical NB fits.
    assert comp.groupby(["dataset_id","pipeline"]).zero_noise_ba.nunique().eq(1).all()
    assert comp.loc[comp.pipeline.ne("softmax__dp_sgd_clip2_reg003"),"optimization_clipping_gap"].eq(0).all()
    comp.to_csv(OUT/"E1_variance_components_by_dataset.csv",index=False)
    aggregate = comp.groupby(["pipeline","epsilon"],as_index=False).agg(
        variance_dp=("variance_dp","mean"),variance_split=("variance_split","mean"),
        mean_dataset_share_dp=("share_dp","mean"),zero_variance_datasets=("zero_total_variance","sum"),
        private_ba=("noise_mean_ba","mean"),split_ba=("split_mean_ba","mean"),
        zero_noise_ba=("zero_noise_ba","mean"),clean_ba=("clean_ba","mean"),
        optimization_clipping_gap=("optimization_clipping_gap","mean"))
    aggregate["share_dp"] = aggregate.variance_dp / (aggregate.variance_dp+aggregate.variance_split)
    aggregate["share_split"] = 1-aggregate.share_dp
    aggregate.to_csv(OUT/"E1_variance_summary.csv",index=False)
    e4 = frames["E4"]
    summary4, data4 = paired_summary(e4,["bins","prior"],e4.bins.eq(16)&e4.prior.eq("current"),"E4")
    e5 = frames["E5"]
    summary5, data5 = paired_summary(e5,["regularization"],e5.regularization.eq(.03),"E5")
    # Check actual-data invariance of bin choices on nominal-only datasets.
    nom = e4[e4.feature_group.eq("all_nominal")]
    assert nom.groupby(["dataset_id","epsilon","repetition","prior"]).private_ba.nunique().eq(1).all()

    audit = pd.read_csv(OUT/"privacy_audit_E1_E4_E5.csv")
    all_private = audit[audit.noise_enabled.eq(True)]
    assert all_private.verified_private.eq(True).all()
    assert all_private.identical_to_baseline.eq(True).all()
    protected = json.loads((OUT/"protected_hashes_E1_E4_E5.json").read_text())
    changed = [p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed, changed
    verification = dict(completed_datasets=n, verified_private_rows=len(all_private),
                        non_dp_zero_noise_rows=int(audit.noise_enabled.eq(False).sum()),
                        identical_ledgers=True, nominal_bins_invariant=True,
                        complete_grids=True, protected_files_verified=len(protected),
                        protected_files_changed=changed)
    (OUT/"final_verification_E1_E4_E5.json").write_text(json.dumps(verification,indent=2))
    completed_runs = sum(len(frame) for frame in frames.values())
    timings = pd.read_csv(OUT/"timing_E1_E4_E5.csv")
    strata = complete.groupby(["n_tercile","feature_group"]).agg(n=("dataset_id","size"),
                    n_min=("n_instances","min"),n_max=("n_instances","max")).reset_index()
    strata_table = table(["Tercil", "Tipo", "Datasets", "n mínimo–máximo"],
                         [(r.n_tercile,r.feature_group,r.n,f"{r.n_min}–{r.n_max}") for r in strata.itertuples()])
    e1_table = table(["Pipeline","ε","Var. DP","Var. split","DP / split (%)","BA privada (%)","BA zero (%)","BA clean (%)","Gap clean−zero (pp)"],
                    [(NAMES[r.pipeline],f"{r.epsilon:g}",f"{r.variance_dp:.5f}",f"{r.variance_split:.5f}",
                      f"{100*r.share_dp:.1f} / {100*r.share_split:.1f}",f"{100*r.private_ba:.2f}",
                      f"{100*r.zero_noise_ba:.2f}",f"{100*r.clean_ba:.2f}",f"{100*r.optimization_clipping_gap:+.2f}")
                     for r in aggregate.itertuples()])
    def variant_table(summary, scopes, variantcols, filters=lambda x:x):
        rows=[]
        for scope in scopes:
            subset=filters(summary[summary.scope.eq(scope)])
            for key,block in subset.groupby(variantcols,sort=True):
                key = key if isinstance(key,tuple) else (key,)
                cells=[delta_cell(block[block.epsilon.eq(eps)].iloc[0]) for eps in (.1,1.,4.)]
                rows.append((scope,*key,*cells))
        return table(["Grupo",*variantcols,"ε=0,1","ε=1","ε=4"],rows)
    e4_table = variant_table(summary4,["all"],["bins","prior"])
    e4_numeric = variant_table(summary4,["has_numeric"],["bins","prior"])
    e5_table = variant_table(summary5,["all","T1","T2","T3"],["regularization"],
                              lambda d:d[d.regularization.ne(.03)])
    baseline = table(["Experimento","Grupo","ε=0,1","ε=1","ε=4"],
                     [(ex,scope,*[f"{100*d[(d.scope==scope)&(d.epsilon==eps)].baseline_ba.iloc[0]:.2f}" for eps in (.1,1.,4.)])
                      for ex,d,scopes in [("E4",summary4,["all","has_numeric"]),("E5",summary5,["all","T1","T2","T3"])]
                      for scope in scopes])
    failed = ", ".join(map(str,meta["failed_ids"])) or "nenhum"
    successful_replacements = [r for r in meta.get("replacement_attempts",[]) if r["succeeded"]]
    replacements_text = ("Substituições por indisponibilidade: " + "; ".join(
        f"{r['original_id']} → {r['replacement_id']} ({r['n_tercile']}/{r['feature_group']})"
        for r in successful_replacements) + ". Foi usado o próximo representante na ordem aleatória já fixada, no mesmo estrato e sem repetir família."
        if successful_replacements else "Não houve substituições.")
    reduction = "Sem redução da amostra." if n==40 else f"Amostra reduzida de 40 para {n}; IDs com falha/limite: {failed}. Só grades completas entram nas análises."
    best4 = summary4[(summary4.scope=="all") & ~((summary4.bins==16)&(summary4.prior=="current"))]
    best5 = summary5[(summary5.scope=="all") & summary5.regularization.ne(.03)]
    positive4 = best4[best4.ci95_low>0]
    positive5 = best5[best5.ci95_low>0]
    low_eps = aggregate[aggregate.epsilon.eq(.1)]
    high_eps = aggregate[aggregate.epsilon.eq(4.)]
    nb1 = summary4[(summary4.scope=="all") & (summary4.bins==8) & (summary4.prior=="uniform") & (summary4.epsilon==1.)].iloc[0]
    nb4 = summary4[(summary4.scope=="all") & (summary4.bins==4) & (summary4.prior=="uniform") & (summary4.epsilon==4.)].iloc[0]
    sgd4 = summary5[(summary5.scope=="all") & (summary5.regularization==.001) & (summary5.epsilon==4.)].iloc[0]
    sgd_large = summary5[(summary5.scope=="T3") & (summary5.regularization==.001) & (summary5.epsilon==4.)].iloc[0]
    gap = aggregate[aggregate.pipeline.eq("softmax__dp_sgd_clip2_reg003")].optimization_clipping_gap.iloc[0]
    findings = (f"Em E1, a participação descritiva do ruído DP varia de {100*low_eps.share_dp.min():.1f}% a {100*low_eps.share_dp.max():.1f}% em ε=0,1, "
        f"e de {100*high_eps.share_dp.min():.1f}% a {100*high_eps.share_dp.max():.1f}% em ε=4. "
        f"O softmax apresenta gap médio clean−zero de {100*gap:.2f} pp. "
        f"Em E4, 8 bins/prior uniforme em ε=1 obtêm {delta_cell(nb1)} pp, e 4 bins/prior uniforme em ε=4 obtêm {delta_cell(nb4)} pp. "
        f"Em E5, λ=0,001 em ε=4 obtém {delta_cell(sgd4)} pp no conjunto completo e {delta_cell(sgd_large)} pp em T3. "
        "Os ICs dessas comparações são pontuais e os achados são exploratórios.")
    count_runs = f"{completed_runs:,}".replace(",", ".")
    count_private = f"{len(all_private):,}".replace(",", ".")
    report = f"""# Experimentos pequenos E1, E4 e E5

Foram concluídos **{n}/40 datasets**, com **{count_runs} registros de condições** e **{meta['runtime_seconds']/60:.2f} minutos** de execução, incluindo downloads. {reduction} A semente foi **20261110**. Os resultados usam a balanced accuracy de `dpmeta.model.balanced_accuracy`, expressa em %; deltas e gaps são pontos percentuais (pp).

{findings}

## Amostra e protocolo

Manifesto: `data/openml_manifest_984_completed.csv`; filtro `n_instances <= 5000` e família semântica preenchida ({meta['eligible_records']} registros, {meta['eligible_families']} famílias). Um registro elegível sem `semantic_family_id` foi excluído. Primeiro foi sorteado um representante por família; depois foram escolhidos 6–7 representantes por estrato. Tercis públicos do conjunto elegível: **T1 ≤250; T2 251–1000; T3 1001–5000**. `has_numeric` inclui datasets mistos. A alocação equilibrada dá peso igual aos dois tipos, e não estima a média proporcional do manifesto. {replacements_text}

{strata_table}

Loader: `_get_openml_data`, o mesmo de `portfolio_benchmark.run_dataset`, cache `.cache/openml`. Como os parquets fixados no manifesto não estavam nesta cópia, o wrapper acionou o ramo de download do loader sem alterar o manifesto. Foram mantidas as validações de schema, classes e número de linhas, o encoding público com `tanh` e o holdout estratificado 75/25, incluindo a regra existente para classes singleton. Nenhuma metafeature privada adicional foi liberada.

E1 usa 10 sementes DP no split 0; dez splits com a mesma semente DP; e ruído desligado no split 0. O split 0/semente DP 0 é compartilhado pelos dois ensaios. E4 e E5 usam três splits e três sementes DP pareadas entre variantes; para cada repetição, apenas a configuração muda. `OMP_NUM_THREADS=1`, `.venv/bin/python`, `PYTHONPATH=.` e limite de uma thread nas bibliotecas numéricas. O tempo máximo de planejamento foi 50 minutos e o limite preventivo por dataset, 240 s.

## E1 — variância do ruído e dos splits

Cada variância é amostral (`ddof=1`) em escala BA 0–1, calculada dentro de cada dataset e depois promediada com peso igual. As participações são **média(Var DP) / [média(Var DP) + média(Var split)]** e seu complemento. São participações descritivas das duas perturbações controladas, não uma ANOVA identificada: o ensaio de splits contém interação com o ruído fixo, e não mede a variância de uma pipeline sem ruído.

{e1_table}

`BA privada` é a média das dez sementes no split fixo. `Gap clean−zero` positivo significa perda com clipping/otimização finita; negativo significa que o ajuste de 100 passos obteve BA de teste maior que o clean. Para softmax, clean é o ajuste existente de `fit_clean_softmax`, λ=0,03, tolerância 1e−5, com o limite público de norma do benchmark; zero mantém exatamente 100 passos, clip=2, regularização, aceleração e tamanho de passo do DP-SGD. O gap combina clipping e otimização finita, sem separá-los. O clean minimiza a perda regularizada, não maximiza BA de teste. Em NB e GNB, clean usa os estimadores analíticos existentes: o gap é zero em todos os datasets.

O ruído foi desligado por cópia isolada do objeto de função com um gerador que retorna zeros, preservando o corpo da função original e sem modificar módulos. As condições sem ruído **não são DP**; seus ledgers registram apenas a calibração de referência e não uma garantia de privacidade efetiva.

## E4 — NB categórico: bins e prior

Referência: **16 bins + prior current**. Cada célula é **Δ BA média [IC bootstrap 95%]**, em pp. Primeiro se calcula o delta pareado em cada repetição, depois a média das três repetições por dataset; o bootstrap reamostra datasets (20.000 sorteios, semente fixa), mantendo os pares. A família semântica coincide com a unidade de reamostragem porque não se repete na amostra. ICs percentis e pontuais, sem correção por múltiplas comparações.

{e4_table}

Como bins afeta apenas atributos numéricos, segue também o subconjunto `has_numeric`:

{e4_numeric}

Em todos os datasets `all_nominal`, alterar somente bins produziu exatamente as mesmas BAs em cada repetição, como esperado. `uniform` substitui apenas o prior de classe na decisão por probabilidades iguais; conserva os histogramas condicionais e a consulta de prior original, mesmo que seu resultado não seja usado. Assim o ε total é idêntico. A função existente já aceita estatísticas com cardinalidades diferentes; não foi preciso copiar o mecanismo Laplace. As sementes são pareadas entre bins, mas mudar a dimensão dos histogramas muda a sequência de sorteios alocada aos atributos subsequentes.

## E5 — DP-SGD: regularização

Referência: **λ=0,03**, clip=2 e 100 passos em todas as variantes. Mesma apresentação de delta e IC de E4, com reamostragem de datasets dentro de cada grupo. O algoritmo existente ajusta passo/aceleração em função de λ; foram preservadas essas regras, o clipping e o número de passos.

{e5_table}

Para contextualizar os deltas, BA média (%) das referências de E4/E5, nas três repetições:

{baseline}

## Contabilidade de privacidade e verificação

Foram auditados **{count_private} registros privados** (incluindo condições parciais, se existirem): todos passaram `ledger.verify()`, usam o ε solicitado e têm **ledger completo idêntico ao da configuração atual**, no mesmo dataset/split/ε. Há **{verification['non_dp_zero_noise_rows']} registros diagnósticos sem ruído**, explicitamente marcados como não DP.

NB: composição básica de `p+1` histogramas, sensibilidade L1=2, ε/(p+1) por consulta, escala Laplace 2(p+1)/ε e δ=0, independente de bins/prior. Softmax: 100 liberações Gaussianas zCDP, sensibilidade L2=4/n_train e ρ_total/100 por passo, δ=10⁻⁶; λ não altera essas quantidades. GNB: três blocos de estatísticas, alocação igual de ρ, δ=10⁻⁶. Os ledgers são armazenados integralmente e vinculados aos registros por SHA-256.

O orçamento vale por ajuste/liberação individual. Executar ou divulgar todos os ajustes conjuntamente requer composição adicional; as repetições não compartilham um único ε global. BA em teste, a seleção estratificada usando rótulos, referências clean e condições sem ruído são diagnósticos de benchmark em dados públicos; os valores de BA não passaram por um mecanismo de avaliação DP.

Verificações adicionais: equivalência exata do wrapper NB com a configuração nativa; ledgers iguais para todos os bins/λ em dados sintéticos e reais; GNB/NB sem ruído iguais ao clean; softmax sem ruído independente da semente e de ε; grades completas, ausência de duplicatas e BA finita. **{len(protected)} arquivos existentes** de `dpmeta/`, `scripts/`, `data/` e Markdown da raiz mantiveram seus hashes. Nenhum comando git foi usado. O código novo está apenas em `experiments/`; os resultados estão em `results/experiments/`, além do cache OpenML solicitado.

## Leitura dos resultados e limitações

O limite inferior do IC ficou acima de zero em {len(positive4)} comparações de E4 e {len(positive5)} de E5 na amostra completa, contando separadamente os ε. Essas contagens são exploratórias e não selecionam uma configuração universal. A amostra pequena, os 10 sorteios de E1 e as três repetições E4/E5 geram incerteza; os ICs representam variação entre datasets, e não toda a incerteza Monte Carlo. Resultados por tercil têm amostras menores. Famílias são as declaradas no manifesto, sem uma auditoria semântica adicional. O split fixo de E1 pode ser atípico. Clipping e convergência finita não são identificados separadamente, e diferenças de BA não medem distância ao ótimo da função objetivo.

Downloads e codificação pública estão incluídos no runtime; o dataset mais demorado concluído levou {timings.seconds.max():.1f} s. Datasets que excedem o limite ou falham são excluídos integralmente das tabelas e ficam identificados em `errors_E1_E4_E5.csv` e no metadata. Falhas nesta execução: **{failed}**. O dataset 47266 retornou atributos diferentes do schema fixado; a validação original do loader foi respeitada. O tempo de execução mede os runners e não inclui a elaboração inicial dos scripts ou a redação deste resumo.

## Arquivos reproduzíveis

- `experiments/e1_e4_e5.py`: amostragem, downloads, execuções e auditoria.
- `experiments/verify_e1_e4_e5.py`: verificações dos wrappers e hashes protegidos.
- `experiments/replace_failed_e1_e4_e5.py`: reposição determinística no mesmo estrato quando o download não passa na validação original.
- `experiments/summarize_e1_e4_e5.py`: deltas pareados, bootstrap e este relatório.
- `results/experiments/sample_E1_E4_E5.csv` e `eligible_families_E1_E4_E5.csv`: amostra planejada e representantes elegíveis.
- `effective_sample_E1_E4_E5.csv`: amostra final, incluindo eventuais substituições.
- `E1_runs.csv`, `E4_runs.csv`, `E5_runs.csv`: resultados por condição, com sementes, BA e referência ao ledger.
- `E1_variance_components_by_dataset.csv`, `E1_variance_summary.csv`: decomposição descritiva.
- `E4/E5_paired_runs.csv`, `E4/E5_dataset_deltas.csv`, `E4/E5_bootstrap_summary.csv`: diferenças e inferência (nomes com prefixos E4 ou E5).
- `budget_ledgers_E1_E4_E5.csv`, `privacy_audit_E1_E4_E5.csv`: ledgers completos e auditoria por condição.
- `metadata_E1_E4_E5.json`, `timing_E1_E4_E5.csv`, `E1_E4_E5.log`, `errors_E1_E4_E5.csv`: execução, runtime e falhas.
- `wrapper_verification_E1_E4_E5.json`, `final_verification_E1_E4_E5.json`, `protected_hashes_E1_E4_E5.json`: verificações.

Reprodução em um diretório de saída sem arquivos E1/E4/E5 anteriores (o runner recusa sobrescrever os CSVs de execuções): `PYTHONPATH=. OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 OPENML_CACHE_DIR=$PWD/.cache/openml .venv/bin/python experiments/e1_e4_e5.py --minutes 50`; depois execute `replace_failed_e1_e4_e5.py`, `verify_e1_e4_e5.py` e `summarize_e1_e4_e5.py` com o mesmo ambiente. Nenhum arquivo Markdown existente foi alterado; este resumo novo é a exceção explicitamente solicitada.
"""
    (OUT/"SUMMARY_E1_E4_E5.md").write_text(report)
    print(json.dumps(verification,indent=2))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
