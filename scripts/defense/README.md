# Análises posteriores ao protocolo (Seção 3.8.4 da dissertação)

Exploratórias, feitas depois de aberto o holdout. Rodar da raiz do repositório,
com `PYTHONPATH=.`; as saídas ficam em `results/defense-analyses/`.

| Script | Análise | Entrada |
|---|---|---|
| `q2_noise_floor.py` | piso de ruído do rótulo, confiabilidade, W sem bases no acaso | `results/portfolio-984-current-slim/conditions.csv` |
| `q1_make_manifests.py results/defense-analyses 200` | amostra de 200 conjuntos e manifesto com limites min–max | manifesto 984 |
| `parallel_h27_round.py --numeric-transform {tanh,asinh10,public_bounds}` | reexecução do portfólio (saídas `q1_<transformação>/`) | manifestos da amostra |
| `q1_compare.py tanh asinh10 public_bounds` | comparação entre transformações | `q1_*/conditions.csv` |
| `q3_diagnose.py` | softmax limpa com λ=0,03 e 0,001 no holdout | `q3_holdout.csv` (BA por base e opção no holdout) e cache OpenML |
| `q4_rule.py` | regra de limiar sobre nε/(d+1) por LODO | condições e LODO do seletor |
| `q4_rule_holdout.py` | regra congelada (limiar 2,2 e exceção dos só nominais) nas 41 famílias do holdout | `results/openml-holdout-41/` |
| `q7_semantic_bootstrap.py` | ganho do LODO com peso igual e bootstrap por família semântica | LODO do seletor e manifesto |
| `q6_learning_curve.py` | curva de aprendizado e partições por família semântica | condições e manifesto |

Os números de saturação da `tanh` citados na dissertação (amostra de 80 conjuntos e
correlação nos 588 só numéricos) não têm script neste repositório.
