# Confirmação externa final

Atualizado em 2026-11-12T15:37:14.587962+00:00 (UTC).

**Resultado primário: 0.283 p.p.; IC95% [-0.026; 0.719] frente a b8u. Critério pré-especificado: limite inferior do IC95% > 0. Resultado: CRITÉRIO NÃO ATINGIDO.**

Análise com peso igual por família, 31 famílias novas (2 OpenML + 5 UCI + 24 Kaggle); média sobre 8 ε e 3 repetições. Bootstrap percentil por família: 20.000 reamostragens, semente 20261111.

## Linha do tempo e fontes

- Protocolo base congelado em 2026-11-11; emenda externa congelada em 2026-11-12 antes da aquisição UCI/Kaggle. SHA256 conferidos antes e depois desta continuação.

- Etapa UCI concluída em 2026-11-12T14:40:42Z: 5 famílias, 1.800 ajustes e auditorias. Artefatos UCI e holdout OpenML preservados.

- Kaggle: ordem dos primeiros 600 candidatos públicos por votos congelada em 2026-11-12T14:50:42.560922+00:00; ambas tags `tabular,classification`, filtro CSV, 30 páginas de 20. Permutação 20261112. Consulta exata e hashes em `kaggle/query.json`.

- Manifesto Kaggle final salvo em 2026-11-12T15:29:33.185330+00:00, antes dos modelos. Recomendações somente de features públicas; hashes dos três modelos congelados e dependências conferidos. Sem retreino.

- Ajustes Kaggle concluídos em 2026-11-12T15:32:09Z: 8.640/8.640; retomada verificada em 15:32:45Z com 24 datasets completos e 0 pendentes, sem refazer ajustes. O texto STAGE1 no log é um rótulo herdado do executor UCI reutilizado; as saídas estão isoladas em kaggle/.

- Análise iniciada somente depois das grades completas: 11,160 ajustes privados e auditorias (OpenML + UCI + Kaggle); 15 opções × 8 ε × 3 repetições. 8 workers, OMP_NUM_THREADS=1; split estratificado 75/25, semente base 20260918 e tanh/encoding originais. Guarda de disco: 5 GiB; commits atômicos e retomada por assinatura.

## Seleção e exclusões

Disjunção por nome normalizado, família canônica e semântica, metadados/origem declarada, todos os manifestos em data/, catálogo OpenML atualizado de 25.245 IDs e catálogo/lista UCI. Cópias e origens incertas excluídas conservadoramente. Alvo único exige evidência textual na descrição da fonte; todos os demais atributos são mantidos, com tipo inferido pelo dtype. Apenas linhas sem alvo são removidas.

| Decisão Kaggle | Quantidade |
|---|---:|
| excluded_uncertain_same_base | 218 |
| excluded_declared_target | 199 |
| excluded_rows | 36 |
| excluded_name_or_family | 34 |
| accepted | 24 |
| excluded_declared_origin | 24 |
| validation_loading_failure | 23 |
| excluded_origin_family | 21 |
| excluded_declared_repository_origin | 9 |
| excluded_ambiguous_source_table | 6 |
| excluded_feature_lower_bound | 3 |
| excluded_classes | 3 |

Foram percorridos 600/600 candidatos na ordem congelada; 24 famílias Kaggle aceitas. Total externo: 29/40. Ordem esgotada; usadas todas as elegíveis.

Decisões e evidências por candidato: `kaggle/selection_decisions.csv` e `kaggle/review.json`. Substituições somente por validação/carregamento, sem acesso a resultados. Múltiplos CSVs/variantes sem tabela única inequívoca foram rejeitados; não foram concatenados ou amostrados.

## Amostra

| Fonte | Dataset | n | d codificado | Classes |
|---|---|---:|---:|---:|
| OpenML | [marketing-income](https://www.openml.org/d/47588) | 8993 | 15 | 9 |
| OpenML | [NovAsia_Property_Listing_Dedup_Synthetic](https://www.openml.org/d/48184) | 4800 | 9 | 2 |
| UCI | [Land Mines](https://archive.ics.uci.edu/dataset/763/land+mines-1) | 338 | 3 | 5 |
| UCI | [National Poll on Healthy Aging (NPHA)](https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha)) | 714 | 63 | 3 |
| UCI | [Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) | 3810 | 7 | 2 |
| UCI | [Turkish Music Emotion](https://archive.ics.uci.edu/dataset/862/turkish+music+emotion) | 400 | 50 | 4 |
| UCI | [Raisin](https://archive.ics.uci.edu/dataset/850/raisin) | 900 | 7 | 2 |
| Kaggle | [LidarDataFrames](https://www.kaggle.com/datasets/tareqalhmiedat/lidardataframes) | 411 | 361 | 4 |
| Kaggle | [Klasifikasi Tingkat Kemiskinan Seseorang](https://www.kaggle.com/datasets/neevvv/tingkat-kemiskinan-di-negri-hytamputyh) | 5000 | 57 | 6 |
| Kaggle | [🧠 Alzheimer's Disease Dataset 🧠](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) | 2149 | 35 | 2 |
| Kaggle | [HR-dataset/classif](https://www.kaggle.com/datasets/avikumart/hrdatasetclassif) | 8995 | 52 | 2 |
| Kaggle | [🏠 Predict Smart Home Device Efficiency Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-smart-home-device-efficiency-dataset) | 5403 | 12 | 2 |
| Kaggle | [Cyber-threat-detection](https://www.kaggle.com/datasets/prashantshukla91/cyber-threat-detection) | 4000 | 348 | 5 |
| Kaggle | [🏭 Predicting Manufacturing Defects Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset) | 3240 | 16 | 2 |
| Kaggle | [Mobile Usage and Addiction Habits, 2026](https://www.kaggle.com/datasets/abirambharathisa/mobile-usage-and-addiction-habits-2026) | 3000 | 12 | 2 |
| Kaggle | [Human Stress Detection](https://www.kaggle.com/datasets/laavanya/stress-level-detection) | 2001 | 3 | 3 |
| Kaggle | [676 cases of pharyngitis in children](https://www.kaggle.com/datasets/yoshifumimiya/pharyngitis) | 676 | 19 | 2 |
| Kaggle | [📈 Predict Conversion in Digital Marketing Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset) | 8000 | 32 | 2 |
| Kaggle | [Human Stress Detection in and through Sleep ](https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep) | 630 | 8 | 5 |
| Kaggle | [Lone dataset](https://www.kaggle.com/datasets/abbas829/lone-dataset) | 1000 | 5 | 2 |
| Kaggle | [DnD Stats](https://www.kaggle.com/datasets/andrewabeles/dnd-stats) | 10000 | 9 | 9 |
| Kaggle | [car-price-classification-ready-data](https://www.kaggle.com/datasets/tatheerabbas/car-price-classification-ready-data) | 300 | 346 | 4 |
| Kaggle | [🐾 Predict Pet Adoption Status Dataset 🐾](https://www.kaggle.com/datasets/rabieelkharoua/predict-pet-adoption-status-dataset) | 2007 | 31 | 2 |
| Kaggle | [🎓 Predict Online Course Engagement Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-course-engagement-dataset) | 9000 | 13 | 2 |
| Kaggle | [🎮 Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset) | 40034 | 26 | 3 |
| Kaggle | [🛍️ Predict Customer Purchase Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset) | 1500 | 8 | 2 |
| Kaggle | [🦸 Fictional Character Battle Outcome Prediction](https://www.kaggle.com/datasets/rabieelkharoua/fictional-character-battle-outcome-prediction) | 2351 | 25 | 2 |
| Kaggle | [Student Placement Preparation Dataset](https://www.kaggle.com/datasets/chanchalsh05/student-placement-preparation-dataset) | 1000 | 10 | 2 |
| Kaggle | [💼 Predicting Hiring Decisions in Recruitment Data](https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data) | 1500 | 10 | 2 |
| Kaggle | [Classification of human poses - keypoints](https://www.kaggle.com/datasets/jorgemora/classification-of-human-poses-keypoints) | 1086 | 16 | 3 |
| Kaggle | [Dataset of Formalin Tofu Aroma](https://www.kaggle.com/datasets/yudhaislamisulistya/dataset-aroma-tahu-berfomalin) | 45000 | 8 | 2 |

## Externas somente

UCI + Kaggle (29 famílias): 0.228 p.p.; IC95% [-0.052; 0.671]; critério não atingido. A análise primária é a combinada, incluindo as duas famílias OpenML novas.

## Secundários

Testes bilaterais sign-flip pareados por família, rotina congelada (50.000 sorteios ou enumeração exata para amostras pequenas), Holm exatamente sobre quatro comparações.

| Referência | Ganho de BA e IC95% | p Holm | Rejeita a 5% |
|---|---|---:|---|
| H27 | 3.306 p.p.; IC95% [1.820; 4.986] | 0.000319994 | sim |
| old_NB | 0.358 p.p.; IC95% [-1.413; 1.989] | 1 | não |
| gbm13_selector | 0.232 p.p.; IC95% [-0.019; 0.589] | 0.554869 | não |
| rf12_selector | 0.311 p.p.; IC95% [-1.438; 1.922] | 1 | não |

## Caudas

Combinada: perda >1 p.p. em 2.42% das condições dataset–ε (IC95% [0.40; 5.24]%). CVaR90 da perda assinada: 1.220 p.p. (IC95% [0.175; 2.720]). Perda = BA b8u − BA seletor; média das maiores max(1, floor(0,1 × N)) perdas. IC por bootstrap de famílias; restrição descritiva, sem teste adicional.

Caudas e quatro secundários das externas somente: `analysis/externals_only/analysis.json`.

## Onde ganha — descritivo

Mesmos cortes, lógica, peso familiar, 5.000 reamostragens e semente 20261112 de `experiments/where_it_wins.py`: ε; n (≤500, 501–2k, 2k–10k, >10k); n×ε/(d+1) (<10, 10–100, 100–1k, >1k); atributos; classes; proximidade ao acaso (BA média do melhor portfólio limpo − 1/k ≤0,02). A tabela externa aqui usa todas as novas famílias combinadas. Comparação com `results/development-strata/gain_by_stratum.csv`; desenvolvimento é LODO e externo é transferência dos modelos finais congelados.

| Estrato | Nível | Desenvolvimento (p.p.) | Novas combinadas (p.p.) | Famílias novas |
|---|---|---:|---:|---:|
| acaso | informativo | 1.65 | 0.34 | 26 |
| acaso | perto do acaso | 0.02 | 0.01 | 5 |
| classes | 2 | 1.06 | 0.34 | 18 |
| classes | 3–5 | 1.48 | 0.01 | 10 |
| classes | >5 | 2.68 | 0.81 | 3 |
| eps | 0.1 | 0.56 | 0.07 | 31 |
| eps | 0.25 | 0.93 | 0.39 | 31 |
| eps | 0.5 | 1.46 | 0.50 | 31 |
| eps | 1.0 | 1.56 | 0.68 | 31 |
| eps | 16.0 | 1.48 | 0.12 | 31 |
| eps | 2.0 | 1.81 | 0.33 | 31 |
| eps | 4.0 | 1.82 | 0.35 | 31 |
| eps | 8.0 | 1.78 | -0.19 | 31 |
| n | 2k–10k | 3.13 | 0.50 | 16 |
| n | 501–2k | 1.70 | -0.09 | 9 |
| n | >10k | 0.71 | 0.00 | 2 |
| n | ≤500 | 0.73 | 0.40 | 4 |
| ratio | 100–1k | 1.13 | 0.49 | 29 |
| ratio | 10–100 | 2.47 | 0.33 | 29 |
| ratio | <10 | 1.04 | 0.11 | 14 |
| ratio | >1k | 0.03 | -0.03 | 24 |
| tipo | mistos | 0.83 | 0.62 | 13 |
| tipo | só nominais | 0.42 | 0.00 | 1 |
| tipo | só numéricos | 2.06 | 0.04 | 17 |

O padrão foi atenuado em relação ao desenvolvimento. Em atributos só numéricos, o ganho passou de 2,06 para 0,04 p.p.; em mistos, de 0,83 para 0,62 p.p. Para n entre 2k e 10k, caiu de 3,13 para 0,50 p.p. A maior estimativa por número de classes aparece em >5 classes (0,81 p.p.), sustentada por apenas 3 famílias. Entre ε, a maior estimativa externa foi em ε=1 (0,68 p.p.); em ε=8 foi −0,19 p.p. Perto do acaso, o ganho permaneceu próximo de zero (0,01 p.p.). Essas estimativas não demonstram diferenças confirmatórias entre estratos.

Diferenças entre estratos e entre desenvolvimento/externo são descritivas; nenhum ajuste de seletor, corte ou hipótese foi feito após observar os resultados.

## Figuras

Em `figures/external/`, PNG e PDF: forest de ganhos do seletor vs cinco referências (combinada e externas somente), ganho por ε e estratos com desenvolvimento e novas famílias lado a lado.

## Verificação, desvios e limitações

Testes: `116 passed, 14 warnings in 17.77s`. Protocolos, modelos, arquivos preexistentes de data/, results/holdout-openml/ e saídas da etapa UCI conferidos por SHA256. Nenhum comando git executado. Credenciais usadas somente em memória em HTTP Basic; nenhum kaggle.json criado.

As primeiras tentativas HTTP retornaram 404; a consulta completa com grupo público e cabeçalhos do cliente oficial funcionou. Depois houve HTTP429: a aquisição foi pausada, as falhas temporárias foram repetidas e não serviram como exclusões. A consulta fixa, o limite e a permutação foram preservados. Falhas não temporárias após tentativas são registradas como carregamento; inventários de arquivos foram paginados integralmente. Evidências preservadas sem credenciais.

Correspondências entre declaração do alvo e cabeçalhos públicos foram conferidas antes dos modelos (inclusive o identificador ambiental id de LidarDataFrames e a abreviação sl de SaYoPillow). A revisão preliminar foi preservada em kaggle/before_label_mapping_review.json; só o manifesto final foi usado nos ajustes. Não houve escolha de alvo orientada por BA, limpeza ou imputação adicional às regras do loader, engenharia de atributos ou concatenação de tabelas. IDs/textos foram mantidos. Arquivos compactados acima de 1 GiB mantêm o limite de recursos do adaptador UCI e podem falhar no carregamento.

Ressalva operacional: o inventário inicial desta continuação repetiu `find .. -name AGENTS.md`; pode ter enumerado nomes de diretórios do checkout anterior. Nenhum arquivo dele foi aberto ou escrito. Operações posteriores ficaram restritas à versão final e dependências públicas.

Nenhum parâmetro, fallback, portfólio, alvo ou critério confirmatório foi alterado. A emenda não fixa convenção de p dos secundários ou CVaR; foram mantidas as implementações da confirmação base. IDs Kaggle: 2000000 + ID público Kaggle, documentados antes dos modelos.

Limitações: amostra condicionada às primeiras 600 bases por votos e às exclusões conservadoras; poder dependente do número final de famílias; possível parentesco não declarado; qualidade e dados sintéticos distintos entre repositórios; schema/domínios/classes extraídos de bases públicas. O desenvolvimento e o holdout antigo já eram conhecidos ao congelar final. Protocolo interno, sem pré-registro público. Bootstrap condicionado às famílias observadas e estratos pequenos ou ausentes não sustentam generalização ampla.
