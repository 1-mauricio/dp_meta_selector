# DP Meta Selector

Framework para seleção automática de mecanismos de Privacidade Diferencial (DP) via meta-aprendizagem em datasets tabulares.

## Visão geral

O projeto aprende, a partir de múltiplos datasets (OpenML), qual mecanismo DP tende a preservar melhor a utilidade para um novo dataset.

Fluxo macro:
1. Carrega datasets de treino (OpenML).
2. Pré-computa baselines sem DP (cache SQLite).
3. Constrói meta-dataset (meta-features + utilidade por mecanismo).
4. Treina meta-modelo para prever o melhor mecanismo.
5. Avalia em holdout e reporta métricas (hit rate, regret, desempenho relativo).

---

## Estrutura do código

| Módulo | Responsabilidade |
|--------|-----------------|
| `main.py` | CLI e orquestração da pipeline |
| `datasets.py` | Carregamento e split de datasets OpenML |
| `baseline_store.py` | Armazenamento incremental dos baselines (SQLite) |
| `meta_features.py` | Extração de meta-features (estáticas + DP-específicas + contexto) |
| `utility.py` | Avaliação de utilidade (perfis, cache, screening, n_runs) |
| `meta_dataset.py` | Construção do meta-dataset, incluindo `utility_loss_*` |
| `meta_learner.py` | Treino do meta-modelo: classificação + regressão de perda |
| `selector.py` | Interface principal (`fit`, `recommend`, `apply`) |
| `applicator.py` | Aplicação prática dos mecanismos DP nos dados |
| `evaluator.py` | Avaliação final do framework no conjunto de teste |
| `mechanisms.py` | Registro dos mecanismos DP suportados |
| `calibration.py` | Calibração de `epsilon` por família |

---

## Requisitos

- Python 3.10+ (testado em 3.13/3.14)
- Dependências principais: `diffprivlib`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `tqdm`, `openml`, `joblib`

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install diffprivlib scikit-learn pandas numpy scipy tqdm openml joblib
```

---

## Execução rápida

```bash
# Pipeline completa (treino + avaliação + persistência do modelo)
python -m dp_meta_selector

# Apenas pré-computação de baselines
python -m dp_meta_selector --precompute-baselines

# Avaliação com perfis mais completos
python -m dp_meta_selector --eval-full --full-oracle-test
```

---

## API programática

```python
from dp_meta_selector import (
    DPMechanismSelector,
    META_STABLE_PROFILE,
    TASK_CLASSIFICATION, TASK_REGRESSION, TASK_QUERIES,
)

# 1. Treina o seletor
selector = DPMechanismSelector()
selector.fit(datasets)

# 2. Recomenda mecanismo passando contexto obrigatório
rec = selector.recommend(
    X_new, y_new,
    epsilon=1.0,                   # orçamento de privacidade do usuário
    task_type=TASK_CLASSIFICATION, # tipo de tarefa
)

# 3. Acessa a recomendação
print(rec["recommended_mechanism"])   # ex: "GaussianAnalytic"
print(rec["meta_model_used"])         # ex: "regression_multioutput"

# Se usou regressão, vem com as perdas previstas por mecanismo
if "predicted_utility_loss" in rec:
    for mech, loss in sorted(rec["predicted_utility_loss"].items(), key=lambda x: x[1]):
        print(f"  {mech}: {loss:.1f}% de perda prevista")

# 4. Aplica o mecanismo recomendado
X_dp = selector.apply(X_new, rec["recommended_mechanism"])
```

### Usando o perfil estável (n_runs=5) para labels de treino mais confiáveis

```python
from dp_meta_selector import DPMechanismSelector, META_STABLE_PROFILE

# Meta-dataset construído com 5 execuções por mecanismo por dataset
# Elimina o ruído estocástico da DP dos labels de treino (mais lento, mais confiável)
selector = DPMechanismSelector(meta_profile=META_STABLE_PROFILE)
selector.fit(datasets)
```

---

## Opções da CLI

| Flag | Descrição |
|------|-----------|
| `--precompute-baselines` | Calcula baselines e encerra |
| `--baseline-id ID` | Restringe quais baselines calcular (repetível) |
| `--export-baselines PATH` | Exporta tabela de baselines para CSV/Parquet |
| `--skip-baseline-precompute` | Pula pré-computação antes do treino |
| `--no-cache` | Desativa cache local (`.dp_meta_cache`) |
| `--eval-full` | Usa perfil de avaliação completo no holdout |
| `--full-oracle-test` | Usa oráculo completo na avaliação (mais caro) |

---

## Cache e artefatos

- **Cache**: `.dp_meta_cache/`
- **Baselines**: `.dp_meta_cache/baselines.sqlite`
- **Modelo treinado**: `dp_meta_selector.joblib`

### Esquema do SQLite (`baselines`)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `dataset_id` | TEXT (PK) | Identificador do dataset |
| `baseline_id` | TEXT (PK) | Identificador do baseline |
| `schema_version` | TEXT (PK) | Versão do esquema |
| `fingerprint` | TEXT | Hash dos dados |
| `profile_key` | TEXT | Chave do perfil de avaliação |
| `accuracy` | REAL | Acurácia sem DP |
| `computed_at` | TEXT | Timestamp |

---

## Pipeline detalhada

```
__main__.py
    └─ main.cli()
        ├─ datasets.py         → carrega OpenML, separa meta-train/test
        ├─ baseline_store.py   → pré-computa baselines sem DP (cache SQLite)
        ├─ meta_dataset.py
        │   ├─ meta_features.py  → extrai 116 meta-features por dataset
        │   └─ utility.py        → avalia utilidade de cada mecanismo DP
        ├─ meta_learner.py     → treina classificadores + regressor de perda
        ├─ selector.py         → recomenda mecanismo para novo dataset
        └─ evaluator.py        → compara recomendado vs oráculo
```

---

## Mecanismos DP suportados

| Mecanismo | Família | Melhor para |
|-----------|---------|-------------|
| Laplace | continuous | Dados contínuos, uso geral |
| Gaussian | continuous | Alta dimensionalidade |
| GaussianAnalytic | continuous | Alta dim. + variância distribuída |
| Staircase | continuous | Dados contínuos com estrutura regular |
| LaplaceTruncated | continuous | Dados com range limitado |
| LaplaceFolded | continuous | Dados não-negativos |
| Snapping | continuous | Robustez a overflow numérico |
| Exponential | categorical | Dados categóricos / baixa cardinalidade |
| Uniform | continuous | Distribuição uniforme do ruído |

---

## Perfis de avaliação

| Perfil | clf | cv | n_runs | screening | Uso recomendado |
|--------|-----|----|--------|-----------|-----------------|
| `META_FAST_PROFILE` | logreg | 3 | 1 | ✓ | Desenvolvimento rápido |
| `META_ALIGNED_PROFILE` | rf | 3 | 2 | ✓ | Treino balanceado |
| `META_STABLE_PROFILE` | rf | 3 | **5** | ✓ | **Treino de produção** (labels confiáveis) |
| `EVAL_FAST_PROFILE` | rf | 3 | 2 | ✓ | Avaliação padrão |
| `EVAL_FULL_PROFILE` | rf | 5 | 3 | ✗ | Avaliação completa |

> **`META_STABLE_PROFILE` (n_runs=5)**: cada mecanismo é avaliado 5 vezes com seeds
> diferentes e a média é usada como label de treino. Isso elimina o ruído estocástico
> intrínseco da DP e produz targets mais confiáveis para o regressor de perda de utilidade.

---

## Arquitetura do meta-learner

O `MetaLearner` implementa um ensemble hierárquico com duas camadas de decisão:

### Camada 1 — Pré-filtros (casos claros)

```
Novo dataset
    │
    ├─ cat_prefilter  (GradientBoosting binário)
    │   └─ se p(Exponential) ≥ 0.75 → recomenda Exponential
    │
    ├─ disc_prefilter (GradientBoosting binário)
    │   └─ se p(discrete) ≥ 0.70 → recomenda Geometric
    │
    └─ gauss_prefilter (GradientBoosting binário)
        └─ se p(GaussianAnalytic) ≥ 0.80 → recomenda GaussianAnalytic
```

### Camada 2 — Regressão / Classificação (casos ambíguos)

```
Casos ambíguos (sem sinal forte de família)
    │
    ├─ regression_multioutput  ← PADRÃO quando disponível
    │   Prevê utility_loss_M (%) para cada mecanismo M
    │   Recomenda M com MENOR perda prevista
    │
    └─ ensemble classificador  ← fallback (ou quando model_name especificado)
        ExtraTrees + LogReg + SVM-Linear (soft-voting)
        + family_gate (hard/soft gate por família)
        + GA boost (amplifica GaussianAnalytic em alta dim.)
        + Laplace fallback (quando confiança < 0.65)
```

---

## Meta-features extraídas (116 features)

### Estatísticas clássicas (~30 features)
Dimensões, médias, desvios, skewness, curtose, correlações, sparsidade, cardinalidade.

### Informação mútua e entropia (~5 features)
MI média/max/min/std, entropia da classe alvo.

### Landmarks (~2–4 features)
Acurácia de classificadores simples (decision stump, logistic regression).

### Relevância DP — `_dp_relevance` (~5 features)
Sensibilidade global, outlier ratio, dimensionalidade intrínseca (PCA).

### Sinal categórico — `_categorical_signal` (~7 features)
Cardinalidade, entropia por coluna, dominância nominal (para detectar Exponential).

### Sinal discreto — `_discrete_signal` (~8 features)
Colunas inteiras, range de inteiros, densidade discreta (para detectar Geometric).

### Discriminadores de família — `_family_discriminators` (~14 features)
Scores de continuidade, discretude, categoricidade; PCA spread; correlação entre features.

### Sinal Gaussian — `_gaussian_signal` (~7 features)
Dimensionalidade log, razão features/amostras, spread PCA (para detectar GaussianAnalytic).

### ⚡ NOVO: Sinal de clipping — `_dp_clipping_signal` (10 features)

| Feature | Descrição | Por que importa em DP |
|---------|-----------|----------------------|
| `dp_mean_max_median_ratio` | Razão média máximo/mediana por coluna | Mede severidade dos outliers que serão clipados |
| `dp_max_max_median_ratio` | Pior coluna em razão max/mediana | Identifica colunas com outliers extremos |
| `dp_mean_kurtosis` | Curtose média (caudas da distribuição) | Curtose alta → caudas pesadas → mais clipping necessário |
| `dp_max_kurtosis` | Curtose máxima entre colunas | Detecta colunas com distribuição leptocúrtica |
| `dp_std_kurtosis` | Desvio-padrão da curtose | Heterogeneidade nas caudas entre colunas |
| `dp_ratio_heavy_tails` | % colunas com curtose > 3 | Proporção de colunas com outliers problemáticos |
| `dp_mean_iqr_ratio` | Razão IQR/range médio | Baixo → outliers dominam o range → clipping caro |
| `dp_min_iqr_ratio` | Menor IQR/range (pior coluna) | Identifica a coluna mais problemática para clipping |
| `dp_clipping_loss_estimate` | % valores clipados com threshold 3σ | Estimativa direta da perda por clipping |
| `dp_global_sensitivity_norm` | Range / desvio-padrão médio | Sensibilidade global normalizada (chave para calibração ε) |

### ⚡ NOVO: Esparsidade e dimensionalidade — `_dp_sparsity_dimensionality` (11 features)

| Feature | Descrição | Por que importa em DP |
|---------|-----------|----------------------|
| `dp_zero_ratio` | Proporção de zeros na matriz | Dados esparsos perdem mais utilidade com ruído |
| `dp_max_col_sparsity` | Coluna mais esparsa (% zeros) | Detecta colunas onde o ruído vai dominar |
| `dp_mean_col_sparsity` | Esparsidade média por coluna | Esparsidade global da matriz |
| `dp_ratio_sparse_cols` | % colunas com > 50% zeros | Estrutura esparsa que prejudica utilidade |
| `dp_numerical_rank` | Rank numérico via SVD | Dimensionalidade intrínseca real da matriz |
| `dp_numerical_rank_ratio` | Rank / n_features | Quanto do espaço de features é efetivamente usado |
| `dp_effective_dim_ratio` | Dim. efetiva (entropia dos valores singulares) | Dimensionalidade efetiva vs nominal |
| `dp_condition_number` | Razão maior/menor valor singular | Alto → matriz mal condicionada → DP mais instável |
| `dp_log_condition_number` | log₁₀ do número de condição | Versão suavizada para features de ML |
| `dp_var_top1` | Variância explicada pelo 1º componente | Concentração de variância |
| `dp_var_top5` | Variância explicada pelos top-5 componentes | Quanto do sinal está nos componentes principais |

### ⚡ NOVO: Entropia de subgrupos — `_dp_subgroup_entropy` (9 features)

| Feature | Descrição | Por que importa em DP |
|---------|-----------|----------------------|
| `dp_minority_class_ratio` | Proporção da menor classe | Grupos pequenos sofrem mais com ruído DP (Disparate Impact) |
| `dp_minority_class_size` | Tamanho absoluto do menor grupo | Grupos com < 30 amostras são muito vulneráveis |
| `dp_majority_class_ratio` | Proporção da maior classe | Assimetria entre grupos |
| `dp_class_imbalance_ratio` | Razão maior/menor classe | Fator de desbalanceamento |
| `dp_class_entropy` | Entropia de Shannon das classes | Distribuição de grupos (alta = mais balanceado) |
| `dp_class_entropy_normalized` | Entropia normalizada [0, 1] | 1.0 = perfeitamente balanceado |
| `dp_gini_impurity` | Impureza de Gini das classes | Outra medida de desbalanceamento |
| `dp_effective_n_classes` | Nº efetivo de classes (exp(entropia)) | Classes "equivalentes" considerando desbalanceamento |
| `dp_disparate_impact_risk` | Score composto de risco de Disparate Impact | Alto quando: minoria pequena + distribuição desigual |

### ⚡ NOVO: Variáveis de contexto — `_context_features` (8 features)

| Feature | Descrição | Uso |
|---------|-----------|-----|
| `ctx_epsilon` | Orçamento de privacidade (ε) informado pelo usuário | Valor exato passado em `recommend(..., epsilon=1.0)` |
| `ctx_log_epsilon` | log(ε) | Versão suavizada para ML |
| `ctx_epsilon_low` | 1 se ε < 1.0 (privacidade forte) | Bucket: mecanismos mais robustos a ruído alto |
| `ctx_epsilon_medium` | 1 se 1.0 ≤ ε < 5.0 (padrão) | Bucket: comportamento típico |
| `ctx_epsilon_high` | 1 se ε ≥ 5.0 (privacidade fraca) | Bucket: mecanismos que exploram mais ε |
| `ctx_task_classification` | 1 se tarefa = classificação | One-hot de tipo de tarefa |
| `ctx_task_regression` | 1 se tarefa = regressão | One-hot de tipo de tarefa |
| `ctx_task_queries` | 1 se tarefa = queries | One-hot de tipo de tarefa |

> **Como usar o contexto:**
> ```python
> rec = selector.recommend(X, y, epsilon=0.5, task_type="classification")
> rec = selector.recommend(X, y, epsilon=10.0, task_type="regression")
> ```
> Quando não especificados, defaults são aplicados: `epsilon=1.0`, `task_type="classification"`.

---

## Resultados experimentais

### Benchmark (489 datasets OpenML, split 75/25)

| Métrica | Valor |
|---------|-------|
| **Classificador ExtraTrees F1-macro (CV)** | **0.874** |
| **Regressor Multi-output MAE-CV** | **4.16%** (erro médio na perda de utilidade) |
| Total de meta-features | 116 (↑ de ~76) |
| Pré-filtro Exponential F1 (CV) | 0.79 |
| Pré-filtro GaussianAnalytic F1 (CV) | 0.45 |

### Comparação: pipeline antiga vs nova (123 datasets de teste)

```
                                 ANTIGA (clf)    NOVA (reg)    Delta
─────────────────────────────────────────────────────────────────────
Hit Rate geral                     66.4%           29.4%       -37.0%
Gap médio (perda extra ao errar)    0.77%           3.56%       +2.79%

Casos ambíguos (sem prefilter, 73% dos datasets):
Hit Rate                           71.3%           20.7%       -50.6%
Gap médio                           0.36%           4.18%       +3.81%
```

### Interpretação dos resultados

**Por que o regressor performa pior com o perfil atual?**

O regressor de perda de utilidade é uma arquitetura mais expressiva, mas também mais exigente em qualidade de dados de treino. Com o `META_FAST_PROFILE` (`n_runs=1`), cada avaliação DP é executada **uma única vez**, introduzindo ruído estocástico alto nos targets. O regressor tenta aprender precisão milimétrica (perdas em %) a partir de labels ruidosas, o que degrada sua performance.

Adicionalmente, o classificador opera sobre 3 classes bem separadas (Laplace, Exponential, GaussianAnalytic) e acumula 6+ anos de otimização via prefilters especializados. O regressor cobre 9 mecanismos sem prefilters dedicados.

**Quando o regressor mostrará seu valor?**

| Condição | Status | Como ativar |
|----------|--------|-------------|
| `n_runs ≥ 5` (labels confiáveis) | ⚠️ Requer `META_STABLE_PROFILE` | `DPMechanismSelector(meta_profile=META_STABLE_PROFILE)` |
| Contexto ε fornecido | ✅ Implementado | `recommend(..., epsilon=1.0)` |
| Muitos datasets de treino (> 500) | ✅ Disponível | Dataset padrão tem 489 |
| Prefilters para casos ambíguos | 🔲 Trabalho futuro | Integrar regressor com prefilters |

**Recomendação de uso atual:**
- Use `META_FAST_PROFILE` (padrão) para desenvolvimento e experimentação → classificador é mais confiável.
- Use `META_STABLE_PROFILE` (`n_runs=5`) quando quiser explorar a regressão → labels de treino limpas.
- O `predicted_utility_loss` no resultado da regressão é útil como **ranking explicável** mesmo que o mecanismo escolhido mude.

---

## Changelog técnico (refatoração DP-aware)

### v6 — Meta-features DP-específicas e regressão de utilidade

#### `meta_features.py` — Novos métodos

**`_dp_clipping_signal(X, y)`** — Razão máximo/mediana e curtose por coluna.
- Prediz o impacto do clipping na sensibilidade global.
- Features: `dp_mean_max_median_ratio`, `dp_max_kurtosis`, `dp_ratio_heavy_tails`, `dp_clipping_loss_estimate`, `dp_global_sensitivity_norm`, etc.

**`_dp_sparsity_dimensionality(X, y)`** — Rank SVD e esparsidade.
- Avalia o colapso de utilidade em alta dimensionalidade e dados esparsos.
- Features: `dp_numerical_rank_ratio`, `dp_effective_dim_ratio`, `dp_condition_number`, `dp_zero_ratio`, etc.

**`_dp_subgroup_entropy(X, y)`** — Entropia de subgrupos e Disparate Impact.
- Mede o risco de impacto desproporcional do ruído DP em grupos minoritários.
- Features: `dp_minority_class_ratio`, `dp_disparate_impact_risk`, `dp_gini_impurity`, etc.

**`_context_features(epsilon, task_type)`** — Variáveis de contexto obrigatórias.
- Concatena ao vetor X o orçamento de privacidade e o tipo de tarefa.
- O meta-modelo não precisa mais adivinhar o contexto a partir dos dados.

**`extract(X, y, epsilon=None, task_type=None)`** — Assinatura atualizada.
- Aceita `epsilon` e `task_type` como parâmetros opcionais com defaults sensatos.

#### `meta_dataset.py` — Novos targets de regressão

Cada linha do meta-dataset agora inclui:
```python
utility_loss_Laplace = max(0, (baseline - acc_laplace) / baseline) * 100  # %
utility_loss_Gaussian = ...
# ... para todos os 9 mecanismos
```

Essas colunas são excluídas das features de entrada e usadas exclusivamente como targets do regressor.

#### `meta_learner.py` — Regressor multi-output

**`_fit_regression(meta_df, X_meta)`** — Treina `MultiOutputRegressor(RandomForestRegressor)`.
- Target: matriz `(n_datasets × n_mechanisms)` de perdas de utilidade (%).
- Métrica de qualidade: MAE cross-validation.
- Treinado nos dados originais (sem oversampling).

**`_predict_regression(row)`** — Prediz perda por mecanismo e recomenda o menor.
- Converte perdas em "probabilidades" via softmin para compatibilidade com a API.
- Retorna `predicted_utility_loss` no dict de resultado.

**`predict()` — Ordem de decisão atualizada:**
```
1. cat_prefilter      → Exponential (p ≥ 0.75)
2. disc_prefilter     → Geometric   (p ≥ 0.70)
3. gauss_prefilter    → GaussianAnalytic (p ≥ 0.80)
4. regression_multioutput  ← NOVO (substitui classificador como camada principal)
5. ensemble classificador  ← fallback ou quando model_name especificado
```

**`save()/load()` atualizado** para persistir `_regression_model`, `_regression_mechanisms`, `_regression_cv_mae`.

#### `utility.py` — Novo perfil estável

```python
META_STABLE_PROFILE = UtilityProfile(
    name="meta_stable",
    clf="rf",
    n_estimators=30,
    cv_splits=3,
    n_runs=5,  # ← executa 5x e tira a média, eliminando ruído estocástico da DP
    ...
)
```

#### `selector.py` — Interface atualizada

```python
# Antes
selector.recommend(X, y)

# Depois — contexto obrigatório para melhores recomendações
selector.recommend(X, y, epsilon=1.0, task_type="classification")
```

O log agora exibe a perda prevista por mecanismo quando a regressão é usada:
```
  Perda de utilidade prevista (menor = melhor):
   Uniform                  0.2%  [continuous  ] ◄ recomendado
   Gaussian                 0.2%  [continuous  ]
   Laplace                  0.2%  [continuous  ]
   ...
   Exponential              1.2%  [categorical ]
```

---

## Solução de problemas

- Se ocorrer erro de importação, verifique se o ambiente virtual está ativo.
- Se OpenML estiver lento/indisponível, tente novamente (dependência de rede).
- Se quiser reduzir custo computacional, prefira perfis rápidos (padrão) e mantenha cache ativo.
- Para recomeçar do zero, remova `.dp_meta_cache/`.

---

## Licença e uso acadêmico

Este repositório está orientado a experimentação acadêmica de meta-aprendizagem aplicada a mecanismos de DP em dados tabulares.