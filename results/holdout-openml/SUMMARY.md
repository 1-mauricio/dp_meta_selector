# Confirmação final — famílias novas

**Critério H2a final: NÃO ATINGIDO.** Ganho sobre b8u: **+1,080 p.p.; IC95% [-0,278; 2,438] p.p.**.
O critério congelado exige limite inferior do IC95% > 0. Foram usadas somente duas famílias elegíveis; o resultado não confirma transferência e tem poder/generalização muito limitados.

Protocolo SHA256: `66933d771548a8048b38c5952dbc4a9e1675c31694421c0b39d5063af5dfc183`. O [protocolo original](../../protocols/CONFIRMATORY_PROTOCOL_FINAL.md) foi preservado. Nenhum comando git foi executado; o checkout anterior não foi acessado.

Cronologia em UTC (11/11/2026; registros completos no [LOG.md](LOG.md)):

| Etapa | Início UTC | Fim UTC |
|---|---|---|
| 1. Congelamento dos seletores | 2026-11-11T21:02:41.409497+00:00 | 2026-11-11T21:03:58.622550+00:00 |
| 2. Seleção/validação do holdout | 2026-11-11T21:04:09.722701+00:00 | 2026-11-11T21:22:54.771447+00:00 |
| 3. 15 opções no novo holdout | 2026-11-11T21:23:02.290306+00:00 | 2026-11-11T21:23:26.589743+00:00 |
| 4. Análise confirmatória | 2026-11-11T21:23:26.589921+00:00 | 2026-11-11T21:23:26.618189+00:00 |
| 5. Extensão/análise exploratória antiga | 2026-11-11T21:23:26.618317+00:00 | 2026-11-11T22:37:29.928154+00:00 |
| 6. Figuras PDF+PNG | 2026-11-11T22:38:59.769485+00:00 | 2026-11-11T22:39:10.625517+00:00 |
| 7. Resumo e QA final | 2026-11-11T22:39:11.294294+00:00 | 2026-11-11T22:41:24.164674+00:00 |

Os três seletores foram salvos com hashes antes da seleção: GBM-15 (15 opções, fallback b8u, τ=0,02), GBM-13 (13 opções, τ=0,01), ambos em 973 datasets; RF-12 em 984 datasets/12 opções. Features, alvos, pesos e hiperparâmetros seguem as rotinas especificadas. Modelos e hashes de treinamento: [frozen.json](frozen.json), [SHA256SUMS](models/SHA256SUMS).

Amostra nova: **2 famílias/datasets**, n de **4.800 a 8.993**. O alvo de 40 não foi alcançado: foram esgotados os 869 candidatos da ordem sorteada no catálogo ativo; o protocolo autoriza usar todas as elegíveis.

| ID OpenML | Dataset/família | n | Classes | Atributos codificados |
|---:|---|---:|---:|---:|
| 47588 | `marketing-income` | 8993 | 9 | 15 |
| 48184 | `NovAsia_Property_Listing_Dedup_Synthetic` | 4800 | 2 | 9 |

Elegibilidade: 200–50.000 linhas, 2–30 classes, ≤512 atributos codificados. Disjunção por ID, família canônica e semântica contra os **21 CSVs** de data/, listados no LOG e em [exclusion_manifests.json](exclusion_manifests.json). Manifesto/seleção foram salvos antes das avaliações.
Falhas/substituições: **296 rejeições de validação**, substituídas somente pelo próximo candidato na permutação da semente 20261111; 571 exclusões por família/origem. **Sem erros de dados/ajustes** e nenhuma substituição após medir desempenho. A extensão antiga teve interrupção operacional e retomada documentadas. Detalhes em [selection_decisions.csv](selection_decisions.csv).
A revisão de procedência resolveu pais públicos declarados por TabClustPFN e excluiu os 18 derivados de famílias anteriores, preservando a mesma ordem, antes de ajustar modelos no holdout. O registro inicial da validação foi arquivado.

Execução nova: **720 saídas e 720 auditorias de ledger aprovadas** (2 × 15 opções × 8 ε × 3 repetições), tanh, 8 workers, OMP_NUM_THREADS=1, commits atômicos retomáveis e guarda de disco de 5 GiB. A semente base 20260918 e os namespaces foram herdados de external_confirmation/benchmark do holdout; seleção e bootstrap usam 20261111. As recomendações dos três seletores usam somente features públicas e foram reproduzidas independentemente.

Inferência primária: média sobre ε/repetições por dataset e peso igual por família; bootstrap percentil de famílias, 20.000 reamostragens, semente 20261111.

Caudas (restrição, sem teste): perda >1 p.p. em **6,25%** das condições, IC95% [0,00; 12,50]%. **CVaR90 da perda: 2,694 p.p.**, IC95% [0,000; 2,694] p.p. Perda assinada = BA(b8u) − BA(seletor); CVaR segue a média das maiores floor(10% × condições) perdas, como no desenvolvimento.

Secundários: sign-flip pareado bilateral por família, 50.000 sorteios; Holm aplicado exatamente aos quatro testes. ICs abaixo são marginais.

| Referência | Ganho e IC95% | p Holm |
|---|---|---:|
| H27 | +1,990 p.p.; IC95% [0,611; 3,370] p.p. | 1,00000 |
| NB antigo | -2,336 p.p.; IC95% [-4,703; 0,031] p.p. | 1,00000 |
| Seletor GBM-13 | +1,080 p.p.; IC95% [-0,278; 2,438] p.p. | 1,00000 |
| Seletor RF-12 | -1,933 p.p.; IC95% [-3,897; 0,031] p.p. | 1,00000 |

**EXPLORATÓRIO — holdout antigo já aberto:**
41 datasets/41 famílias; n de 65 a 1000000. Foram acrescentadas as duas opções DP-SGD: **1.968 saídas novas, todas auditadas**, mais **54 auditorias de alinhamento exato**; b8u da fase A foi reutilizada com suas 984 auditorias aprovadas. As 12.792 linhas anteriores de 13 opções foram preservadas exatamente. Após alinhamento exato em três datasets, os pendentes foram executados com o mesmo kernel privado, omitindo somente ajustes limpos auxiliares. Esses campos ficam explicitamente ausentes/nulos; nenhuma métrica limpa foi fabricada.
Ganho final vs b8u: **+0,677 p.p.; IC95% [0,020; 1,618] p.p.**. Perda >1 p.p.: 0,30%, IC95% [0,00; 0,91]%; CVaR90 0,079 p.p., IC95% [0,000; 0,227] p.p. Os quatro secundários/Holm estão em [exploratory_41/analysis.json](exploratory_41/analysis.json). Esses números não constituem confirmação independente.
Estratos por ε, faixa de n (≤1000; 1001–5000; 5001–10000; 10001–50000; >50000) e atributos numeric/nominal/mixed: [holdout antigo](exploratory_41/exploratory_strata.csv) e [holdout novo](exploratory_new_holdout/strata.csv), ambos exclusivamente exploratórios.

Figuras do holdout novo: [forest PDF](../../figures/dissertation/fig14_holdout_gains_vs_b8u.pdf) / [PNG](../../figures/dissertation/fig14_holdout_gains_vs_b8u.png); [CD PDF](../../figures/dissertation/fig13_holdout_cd.pdf) / [PNG](../../figures/dissertation/fig13_holdout_cd.png); [ganhos por ε PDF](../../figures/dissertation/gains_by_epsilon.pdf) / [PNG](../../figures/dissertation/gains_by_epsilon.png). CD é descritivo, com apenas duas famílias.

**Desvios operacionais:** os estratos exploratórios do novo holdout foram gravados ao fim da etapa 4, antes da etapa 5; a extensão antiga foi interrompida e retomada sem ajustes limpos auxiliares, após alinhamento privado exato. A [emenda datada](AMENDMENT_EXECUTION.md) mantém tudo que depende dessas mudanças como exploratório. A confirmação nova, os modelos e o código existente de dpmeta foram preservados. Não houve retuning, alteração do protocolo ou substituição por desempenho. Catálogo ativo, semente base herdada, teste bilateral e convenção CVaR foram documentados antes dos resultados.
Limitações prévias: a configuração foi escolhida no desenvolvimento e com o holdout antigo já aberto; o protocolo é interno/versionado, não um pré-registro público. A escassez de famílias novas torna a interpretação particularmente limitada.

Validação: **102 testes passaram** (14 avisos), verificação independente das recomendações/primária, hashes preservados, disjunção, grades completas, preservação dos resultados antigos e seis arquivos gráficos válidos: [verification_final.json](verification_final.json), [verification_confirmatory.json](verification_confirmatory.json), [pytest_final.log](pytest_final.log).
