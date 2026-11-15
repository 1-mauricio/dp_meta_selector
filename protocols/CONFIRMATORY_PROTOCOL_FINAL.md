# Protocolo confirmatório final — famílias novas

**Estado:** congelado em 2026-11-11, ANTES de selecionar, baixar ou avaliar
qualquer dataset do holdout final. Nenhuma decisão abaixo pode ser alterada
depois de abrir os dados. Desvios devem ser registrados como emenda datada,
e tudo que depender deles é exploratório.

## Configuração congelada (GBM-15)

- **Portfólio (15 opções):** as 12 opções do protocolo confirmatório inicial +
  `categorical_naive_bayes__laplace_histograms_b8u` +
  `softmax__dp_sgd_clip2_lamrule` + `softmax__dp_sgd_clip2_lam0003`, como
  implementadas em `dpmeta/` neste repositório (testes: 102 passando).
- **Seletor:** `HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
  max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
  random_state=20261110)`; alvo = BA privada média (3 rep.) da opção menos a do
  NB categórico antigo; features `public_plus_dp` de cada opção
  (`dpmeta.portfolio_benchmark._candidate_features`) + one-hot das 15 opções;
  peso igual por família semântica, depois por dataset e opção — exatamente
  `experiments/lodo_gbm_residual.py`.
- **Fallback:** recomenda `categorical_naive_bayes__laplace_histograms_b8u`
  salvo quando a vantagem prevista da melhor opção sobre ela for > τ = 0,02.
- **Treino:** uma única vez, em todo o desenvolvimento
  `results/portfolio-984-sgd/conditions_15.csv` (973 datasets). O modelo
  treinado e o hash do arquivo de condições devem ser salvos ANTES da seleção do
  holdout.

## Holdout final

- Fonte: catálogo OpenML, semente de seleção `20261111`.
- Elegibilidade: 200–50.000 instâncias (dentro da faixa do desenvolvimento),
  2–30 classes, no máximo 512 atributos codificados, schema público válido
  pelos validadores existentes.
- Disjunção obrigatória, por ID OpenML, família canônica e família semântica,
  contra: todos os manifestos de `data/` (desenvolvimento 984, triagens
  250/500/1000, holdouts 30/40/41 e adendos).
- Uma família por dataset (primeiro representante na ordem sorteada).
- Alvo: **40 famílias**; se a disponibilidade for menor, usar todas as elegíveis
  e registrar. Falhas de validação: substituir pelo próximo na ordem sorteada,
  no mesmo procedimento, sem olhar resultados.
- Execução: 15 opções, 8 ε {0,1; 0,25; 0,5; 1; 2; 4; 8; 16}, 3 repetições,
  transformação `tanh`, mesmas regras de split/sementes do benchmark.

## Análise pré-especificada

- **Desfecho primário:** ganho de BA privada da recomendação final sobre a
  opção fixa `..._b8u` (melhor fixa do desenvolvimento). Média sobre ε e
  repetições por dataset; depois média com peso igual por família.
- **Inferência:** IC95% bootstrap percentil por família (20.000 reamostragens,
  semente 20261111). **Critério de sucesso (H2a final):** limite inferior do
  IC95% > 0.
- **Caudas (restrição, não teste):** fração de condições dataset–ε com perda
  > 1 p.p. frente à b8u e CVaR90 da perda; reportar com IC.
- **Secundários (Holm sobre os quatro):** ganho vs H27 (`softmax__dp_sgd_clip2_reg003`),
  vs NB antigo, vs seletor GBM-13 (13 opções, fallback b8u, τ=0,01) e vs seletor
  RF-12 congelado — todos treinados só no desenvolvimento.
- **Exploratório (não confirmatório):** os mesmos números no holdout antigo de
  41 famílias (já aberto); efeitos por ε, por faixa de n e por tipo de atributo.

## Limitações conhecidas antes da execução

- A configuração final foi escolhida com o corpus de desenvolvimento e com o
  holdout de 41 famílias já aberto; este holdout novo é o primeiro teste
  independente dela.
- Poder estatístico desconhecido: o efeito no desenvolvimento é ~0,014 com
  ~667 grupos; com ~40 famílias o IC tende a ser largo. Um resultado nulo não
  refuta o efeito do desenvolvimento.
- Protocolo interno versionado, não pré-registro público.
