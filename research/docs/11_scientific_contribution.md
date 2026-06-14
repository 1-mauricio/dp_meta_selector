# 11 — Contribuição Científica Principal

> Documento de suporte à dissertação. Sintetiza o achado científico central do DP-Meta-Selector e sua fundamentação teórica e empírica.

---

## 1. Problema Científico Endereçado

Sistemas de AutoML tradicionais selecionam algoritmos de aprendizado de máquina com base em **meta-features estáticas** dos dados (número de instâncias, colunas, tipo de atributo, correlações). Esta abordagem é suficiente para selecionar modelos de ML convencional, mas **falha sistematicamente** na seleção de mecanismos de Privacidade Diferencial (DP).

A hipótese central deste trabalho é:

> **O comportamento de mecanismos DP depende intrinsecamente da geometria interna dos dados e do contexto operacional do usuário — não apenas das suas características superficiais.**

Três pilares matemáticos fundamentam essa hipótese:

### 1.1 Sensibilidade Global e Caudas Longas

O ruído injetado pelo Mecanismo de Laplace e pelo Mecanismo Gaussiano é diretamente proporcional à **sensibilidade global** $\Delta f$ da consulta:

$$\text{Laplace}(\mu, \Delta f / \varepsilon) \quad \text{e} \quad \mathcal{N}(0, (\Delta f \cdot \sigma)^2)$$

Datasets com distribuições de cauda pesada (alta kurtosis) ou *outliers* extremos disparam $\Delta f$, forçando o mecanismo a injetar tanto ruído que a utilidade do dado privatizado colapsa. Meta-features estáticas clássicas não capturam esse fenômeno.

**Meta-features criadas:** `dp_max_kurtosis`, `dp_clipping_loss_estimate`, `dp_ratio_heavy_tails`, `dp_mean_max_median_ratio`.

### 1.2 O Paradoxo do Orçamento de Privacidade ($\varepsilon$)

O parâmetro $\varepsilon$ (epsilon) governa o trade-off fundamental entre privacidade e utilidade. Um mecanismo $A$ pode ser ótimo para $\varepsilon = 0.1$ (alta privacidade, muito ruído) e desastroso para $\varepsilon = 5.0$ (baixa privacidade, pouco ruído). Este efeito decorre da diferença na taxa de decaimento da utilidade de cada mecanismo em função de $\varepsilon$.

**Consequência:** $\varepsilon$ e o tipo de tarefa devem ser **variáveis de contexto obrigatórias** no vetor de entrada do meta-modelo — não features deriváveis dos dados. Sem elas, o F1-macro do classificador estagna em 0.70.

**Meta-features criadas:** `ctx_epsilon`, `ctx_log_epsilon`, `ctx_task_classification`, `ctx_task_regression`, `ctx_task_queries`.

### 1.3 Impacto Desproporcional (*Disparate Impact*)

Mecanismos como DP-SGD aplicam *clipping* de gradientes e adicionam ruído isotrópico. Em datasets com distribuição de classes desbalanceada, esse ruído afeta desproporcionalmente os subgrupos minoritários: a informação sobre esses subgrupos é estatisticamente "apagada" na tentativa de proteger sua privacidade, produzindo modelos com viés sistemático.

**Meta-features criadas:** `dp_minority_class_ratio`, `dp_disparate_impact_risk`, `dp_gini_impurity`, `dp_class_entropy`.

---

## 2. Contribuições Técnicas

### 2.1 Extrator de Meta-Features DP-Consciente

Expansão do vetor de features de 74 (meta-features estáticas clássicas) para **112 features** distribuídas em quatro grupos:

| Grupo | Qtd. | Propósito |
|-------|:----:|-----------|
| Estatísticas clássicas | 74 | Base de comparação com literatura AutoML |
| `dp_clipping_signal` | ~14 | Prever impacto do clipping na sensibilidade |
| `dp_sparsity_dimensionality` | ~16 | Capturar colapso de informação em alta dimensionalidade |
| `dp_subgroup_entropy` | ~8 | Quantificar risco de disparate impact |
| Variáveis de contexto | 8 | $\varepsilon$, log($\varepsilon$), tipo de tarefa (one-hot) |

**Impacto:** F1-macro do classificador ExtraTrees: 0.70 → **0.87** (+17pp).

### 2.2 Meta-Dataset Estabilizado (n_runs=5)

Cada alvo de aprendizado (`utility_loss_{mechanism}`) é a **média aritmética de 5 execuções independentes** com seeds distintos. Este protocolo elimina o que denominamos de *areia movediça estatística*: o fenômeno pelo qual labels geradas com uma única rodada ($n\_runs=1$) variam ±5pp entre execuções por causa do ruído intrínseco da DP, inviabilizando o aprendizado por regressão.

**Protocolo:** `META_STABLE_PROFILE`, `n_runs=5`, 401 datasets OpenML, persistido em CSV (1147s de computação).

**Impacto:** Catastrophic Failure Rate com o regressor: **48.6% → 10.2%** (com calibração adicional).

### 2.3 Ensemble Híbrido com Disjuntor Conservador

Arquitetura em dois estágios:
1. **Estágio 1 — Filtro de Sobrevivência:** Classificador ExtraTrees (soft-voting com LR e SVM) seleciona os Top-K (`k=3`) mecanismos candidatos com maior probabilidade de classe.
2. **Estágio 2 — Fine-Tuning por Perda:** Regressor MultiOutput ordena os candidatos pelo menor `utility_loss` previsto.
3. **Disjuntor de Segurança:** Se `loss_recomendado > loss_Laplace − 0.5pp`, o sistema recua para Laplace ou para a recomendação do classificador puro.

A margem de 0.5pp foi calibrada por Grid Search offline em 40 combinações (5 valores de `top_k` × 8 valores de margem). Ver `research/tuning/tune_results_v19.csv`.

### 2.4 Interface Human-in-the-Loop (`return_top_k`)

O parâmetro `return_top_k=2` expõe as duas melhores recomendações com perda prevista de cada uma, permitindo que o engenheiro de dados tome a decisão final. Justificativa: Hit Rate Top-2 de **94.3%** — em quase todos os casos, o mecanismo ótimo está entre as duas primeiras opções.

---

## 3. Achado Científico Central: Gerenciamento de Risco em Decisões DP

### 3.1 O Trade-off Ataque vs. Defesa Constrangida

O benchmark científico (5-fold CV, 401 datasets) revelou que maximizar a precisão média (Hit Rate Top-1) e minimizar o custo do pior caso (Max Regret) são **objetivos conflitantes** na seleção de mecanismos DP:

| Métrica | Random | Always Laplace | Vanilla v16 | **v19 Hybrid** |
|---------|:------:|:--------------:|:-----------:|:--------------:|
| Hit Rate Top-1 | 13.5% | 60.8% | **75.8%** | 68.3% |
| Hit Rate Top-2 | 23.4% | 82.0% | 93.8% | **94.3%** 🏆 |
| Avg Regret (pp) | 1.66 | 0.81 | **0.50** | 0.65 |
| Max Regret (pp) | 27.31 | 15.57 | 25.73 | **14.04** 🛡️ |
| Catastrophic Failure | 68.1% | 0.0% | 8.0% | 10.2% |

### 3.2 A Ilusão do Top-1

O Vanilla v16 supera o v19 Hybrid em 7.5pp de Hit Rate Top-1 (75.8% vs 68.3%). Em ambientes acadêmicos convencionais, essa métrica seria suficiente para declarar v16 vencedor.

No entanto, ao errar, o Vanilla v16 cometia erros com custo máximo de **25.73pp de utilidade** — equivalente a destruir mais de 25% da acurácia do modelo após a aplicação da DP. Em domínios críticos (diagnóstico médico, análise de crédito, políticas governamentais), este custo é inaceitável.

### 3.3 A Lógica do Custo Assimétrico

Em sistemas de Privacidade Diferencial em produção, os custos dos erros são **assimétricos**:
- Um acerto a mais (selecionar o mecanismo ótimo) ganha em média **0.65pp** de utilidade.
- Um erro catastrófico (selecionar um mecanismo muito ruim) pode custar **>20pp** de utilidade.

A relação custo/benefício favorece o conservadorismo: ao calibrar `margin=0.5pp`, o v19 Hybrid aceita perder em média **0.15pp por recomendação** (diferença entre Avg Regret 0.65pp e 0.50pp) para reduzir o pior caso de 25.73pp para **14.04pp** — uma redução de **45%**.

### 3.4 O Valor do Top-2 para Tomada de Decisão Assistida

Com Top-2 Hit Rate de 94.3%, o v19 Hybrid demonstra que, mesmo quando o fallback conservador substitui a recomendação mais precisa do regressor pela mais segura do classificador, o mecanismo ótimo permanece entre as duas opções apresentadas ao usuário. Isso consolida o framework como ferramenta ideal para ambientes de **Human-in-the-Loop** onde o engenheiro de dados deve ter a palavra final.

---

## 4. Posicionamento em Relação à Literatura

### 4.1 Diferença em relação ao AutoML Clássico

Sistemas como Auto-sklearn, TPOT e H2O AutoML otimizam para métricas de acurácia preditiva em dados não-privatizados. Eles não possuem:
- Meta-features de sensibilidade global ou caudas longas
- Modelagem do parâmetro $\varepsilon$ como feature de contexto
- Métricas de avaliação baseadas em utilidade pós-privatização
- Estratégias de fallback conservador para minimizar Max Regret em DP

### 4.2 Relação com Meta-Learning para Seleção de Algoritmos

O framework segue o paradigma de *Algorithm Selection* de Rice (1976) e a filosofia de *meta-features* de Brazdil et al. (2008), mas estende ambos com:
1. Features específicas de domínio DP (sensibilidade, esparsidade, disparate impact)
2. Variáveis de contexto operacional como features de primeira classe
3. Função objetivo híbrida (classificação + regressão de perda de utilidade)
4. Protocolo de estabilização de labels para dados estocásticos

### 4.3 Métricas Propostas

Este trabalho introduz a aplicação de **Max Regret** como métrica primária de avaliação de seletores DP — em contraste com Hit Rate Top-1 que é a métrica padrão em AutoML. A justificativa é o custo assimétrico dos erros em deployments de DP (seção 3.3).

---

## 5. Limitações e Trabalho Futuro

### Limitações Reconhecidas

1. **Número de datasets:** O meta-dataset conta com 401 datasets — razoável para workshops, mas abaixo do padrão de venues principais (NeurIPS, ICML) que esperam 1000+.
2. **Ausência de pré-filtros no benchmark:** O benchmark offline (`benchmark_evaluator.py`) não inclui os pré-filtros hierárquicos (CAT1/DISC/GAUSS) do pipeline de produção. O F1-macro de produção (0.910) é mais alto que o reportado no benchmark (0.70–0.76).
3. **Margem de fallback fixa:** A margem de 0.5pp é calibrada na média dos 401 datasets. Datasets com geometria muito heterogênea podem exigir margens diferentes.

### Trabalho Futuro

1. **Expansão para 1000+ datasets:** Alterar `OPENML_TRAINING_TARGET=1200` em `config.py` e rodar `main.py --stable` (~57 min com checkpoint).
2. **Margens de Fallback Dinâmicas:** Substituir `_hybrid_laplace_margin=0.5` por função de `dp_mean_col_sparsity` — afrouxar em geometrias previsíveis, apertar em datasets heterogêneos.
3. **Multi-Task Meta-Learning:** Expandir para prever dinamicamente *learning rate*, *batch size* e *clipping threshold* em pipelines de aprendizado federado.
4. **Avaliação em dados reais regulados:** Validar em datasets do setor de saúde (MIMIC, eICU) e finanças (Home Credit, Fannie Mae) onde o custo assimétrico dos erros de DP é mais crítico.

---

## 6. Reprodutibilidade

| Artefato | Localização | Descrição |
|----------|-------------|-----------|
| Meta-dataset estável | `meta_datasets_v19/` | 401 datasets × 112 features + 9 targets (n_runs=5) |
| Script de benchmark | `research/benchmark_evaluator.py` | 5 seletores × 6 métricas, 5-fold CV |
| Relatório de benchmark | `research/docs/20_final_benchmark_report.md` | Tabela completa com intervalos de confiança |
| Grid Search calibração | `research/tuning/tune_meta_models.py` | 40 combinações (top_k × margin) |
| Resultados do grid | `research/tuning/tune_results_v19.csv` | CSV com Hit Rate e Catastrophic Failure por combinação |
| Decisões arquiteturais | `research/DECISIONS.md` | DEC-001 a DEC-027 |

Para reproduzir o benchmark completo:
```bash
cd dp_meta_selector
source .venv/bin/activate
python -m research.benchmark_evaluator
# Saída: research/docs/20_final_benchmark_report.md (~2 min)
```
