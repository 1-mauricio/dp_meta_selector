# Arquitetura Técnica do Framework

> Documentação técnica completa do DP Meta-Selector para uso em dissertação.

---

## 1. Visão Geral da Arquitetura

O DP Meta-Selector é um framework de **meta-aprendizagem** que seleciona automaticamente o mecanismo de Privacidade Diferencial (DP) mais adequado para um dataset tabular.

### 1.1 Problema Abordado

A escolha do mecanismo DP impacta diretamente a utilidade dos dados privatizados:
- **Laplace** é o mecanismo padrão, mas só é ótimo em ~28% dos casos
- **GaussianAnalytic** domina em datasets de alta dimensionalidade (~79% quando >50 features)
- **Exponential** pode dar ganhos de +20-33pp em datasets categóricos

O framework automatiza essa escolha usando meta-aprendizagem.

### 1.2 Fluxo de Processamento

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASE DE TREINAMENTO                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Datasets      Meta-Feature      Avaliação DP      Meta-Dataset │
│  de Treino  →  Extraction    →   (Oracle)      →   (X, y)      │
│  (n=350+)      (39 features)     (9 mecanismos)    (labels)     │
│                                                                 │
│                           ↓                                     │
│                                                                 │
│              ┌────────────────────────────────────┐             │
│              │     TREINAMENTO DO META-MODELO     │             │
│              ├────────────────────────────────────┤             │
│              │  1. CAT1 Prefilter (Exponential)   │             │
│              │  2. GAUSS Prefilter (GA)           │             │
│              │  3. HIER Family Classifier         │             │
│              │  4. Ensemble ExtraTrees            │             │
│              └────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    FASE DE INFERÊNCIA                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dataset     Meta-Feature    Pipeline de        Mecanismo       │
│  Novo    →   Extraction  →   Decisão        →   Recomendado     │
│                              Hierárquica                        │
│                                                                 │
│              ┌─────────────────────────────────┐                │
│              │   1. CAT1: p_exp ≥ 0.75?        │                │
│              │      └─ AND p_cat ≥ 0.15?       │                │
│              │         └─ → Exponential        │                │
│              │                                 │                │
│              │   2. GAUSS: p_ga ≥ 0.80?        │                │
│              │         └─ → GaussianAnalytic   │                │
│              │                                 │                │
│              │   3. HIER: família ≥ 0.55?      │                │
│              │         └─ Ajusta probabilidades│                │
│              │                                 │                │
│              │   4. Ensemble: argmax(proba)    │                │
│              │         └─ → Mecanismo final    │                │
│              └─────────────────────────────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Componentes do Sistema

### 2.1 Estrutura de Módulos

```
dp_meta_selector/
├── selector.py          # DPMechanismSelector - API principal
├── meta_learner.py      # MetaLearner - modelo de meta-aprendizagem
├── meta_features.py     # MetaFeatureExtractor - extração de features
├── meta_dataset.py      # MetaDatasetBuilder - construção do meta-dataset
├── mechanisms.py        # Definição dos mecanismos DP
├── calibration.py       # Calibração de epsilon por família
├── applicator.py        # DPApplicator - aplicação de DP
├── utility.py           # Avaliação de utilidade (cache, profiles)
├── diagnostics.py       # Métricas avançadas de diagnóstico
├── synthetic_datasets.py # Geradores de datasets sintéticos
└── main.py              # CLI principal
```

### 2.2 Dependências Principais

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| scikit-learn | ≥1.3 | Meta-modelos, validação cruzada |
| diffprivlib | ≥0.6 | Mecanismos DP |
| numpy | ≥1.24 | Operações numéricas |
| pandas | ≥2.0 | Manipulação de dados |
| joblib | ≥1.3 | Paralelismo e cache |
| scipy | ≥1.11 | Estatísticas |

---

## 3. Mecanismos de Privacidade Diferencial

### 3.1 Mecanismos Implementados

| Mecanismo | Família | Descrição | Caso de Uso Ideal |
|-----------|---------|-----------|-------------------|
| **Laplace** | continuous | Ruído Laplaciano clássico | Dados contínuos gerais |
| **Gaussian** | continuous | Ruído Gaussiano (ε,δ)-DP | Composição de queries |
| **GaussianAnalytic** | continuous | Gaussiano analítico Balle-Wang | Alta dimensionalidade |
| **Staircase** | continuous | Mistura geométrica | Casos especiais |
| **LaplaceTruncated** | continuous | Laplace truncado [0,1] | Dados normalizados |
| **LaplaceFolded** | continuous | Laplace com reflexão | Dados bounded |
| **Snapping** | continuous | Mironov snapping | Proteção floating-point |
| **Exponential** | categorical | Mecanismo exponencial | Dados categóricos |
| **Uniform** | continuous | Ruído uniforme δ-DP | Casos específicos |

### 3.2 Calibração de Epsilon

O sistema usa **calibração por família** para garantir nível de ruído comparável:

```python
FAMILY_EPSILON = {
    "continuous": 5.0,    # E[|noise|] ≈ 19.5% do range
    "discrete":   0.04,   # E[|noise|] ≈ 20%
    "categorical": 2.0,   # Orçamento razoável
}

MECHANISM_EPSILON = {
    "GaussianAnalytic": 19.34,  # σ calibrado para ~20% ruído
    "Gaussian": 19.34,
    "Uniform": 1.25,
}
```

**Objetivo:** Todos os mecanismos produzem aproximadamente **20% de ruído relativo** ao range da coluna. Isso permite comparação justa entre mecanismos.

### 3.3 Implementação do Aplicador

```python
class DPApplicator:
    def apply(self, name: str, X: np.ndarray) -> np.ndarray:
        # Normaliza para [0,1] por coluna
        for j in range(X.shape[1]):
            col = X[:, j]
            c_min, c_max = col.min(), col.max()
            col_n = (col - c_min) / (c_max - c_min + 1e-9)
            
            # Aplica mecanismo DP
            noisy_n = self._apply_mechanism(name, col_n, epsilon)
            
            # Desnormaliza
            X_out[:, j] = noisy_n * (c_max - c_min) + c_min
        
        return X_out
```

---

## 4. Extração de Meta-Features

### 4.1 Categorias de Meta-Features

O sistema extrai **39 meta-features** organizadas em 7 categorias:

#### 4.1.1 Features Estatísticas (15 features)

| Feature | Descrição | Fórmula |
|---------|-----------|---------|
| `n_samples` | Número de amostras | n |
| `n_features` | Número de features | d |
| `n_classes` | Número de classes | \|C\| |
| `samples_per_feature` | Razão amostras/features | n/d |
| `samples_per_class` | Média de amostras por classe | n/\|C\| |
| `class_imbalance` | Desvio padrão das contagens | σ(count_c)/n |
| `mean_mean` | Média das médias por coluna | μ(μ_j) |
| `std_mean` | Desvio das médias | σ(μ_j) |
| `mean_std` | Média dos desvios | μ(σ_j) |
| `std_std` | Desvio dos desvios | σ(σ_j) |
| `mean_skew` | Assimetria média | μ(\|skew_j\|) |
| `max_skew` | Assimetria máxima | max(\|skew_j\|) |
| `mean_kurt` | Curtose média | μ(kurt_j) |
| `max_kurt` | Curtose máxima | max(kurt_j) |
| `mean_corr` | Correlação média | μ(\|corr_{ij}\|) |

#### 4.1.2 Features de Discretização (7 features)

| Feature | Descrição |
|---------|-----------|
| `ratio_discrete` | % de colunas com ≤10 valores únicos |
| `ratio_integer_cols` | % de colunas com valores inteiros |
| `ratio_binary_cols` | % de colunas binárias |
| `mean_log_unique_ratio` | log(unique/n) médio |
| `std_log_unique_ratio` | Desvio de log(unique/n) |
| `median_unique_per_col` | Mediana de valores únicos |
| `max_unique_per_col` | Máximo de valores únicos |

#### 4.1.3 Features de Informação (5 features)

| Feature | Descrição |
|---------|-----------|
| `mean_mi` | Informação mútua média X→y |
| `max_mi` | MI máxima |
| `min_mi` | MI mínima |
| `std_mi` | Desvio da MI |
| `class_entropy` | Entropia do target |

#### 4.1.4 Features de Landmark (2 features)

| Feature | Descrição |
|---------|-----------|
| `lm_stump` | Acurácia de DecisionStump |
| `lm_lin` | Acurácia de LogisticRegression |

#### 4.1.5 Features de Relevância DP (5 features)

| Feature | Descrição |
|---------|-----------|
| `mean_sensitivity` | Sensibilidade média (max-min) |
| `max_sensitivity` | Sensibilidade máxima |
| `outlier_ratio` | % de outliers (>3σ) |
| `pca_intrinsic_dim_ratio` | Dim. intrínseca via PCA |
| `pca_top1_var` | Variância explicada pelo PC1 |

#### 4.1.6 Features Categóricas - CAT1 (7 features)

| Feature | Descrição |
|---------|-----------|
| `cat_ratio_low_cardinality` | % colunas com ≤10 valores |
| `cat_ratio_very_low_cardinality` | % colunas com ≤5 valores |
| `cat_mean_col_entropy` | Entropia média por coluna |
| `cat_max_col_entropy` | Entropia máxima |
| `cat_target_entropy` | Entropia do target |
| `cat_target_entropy_ratio` | Entropia/Entropia máxima |
| `cat_ratio_dominant_cols` | % colunas com valor dominante |

#### 4.1.7 Features de Família (12 features v16)

| Feature | Descrição |
|---------|-----------|
| `fam_continuity_score` | Score de continuidade |
| `fam_discreteness_score` | Score de discretude |
| `fam_categoricity_score` | Score de categoricidade |
| `fam_mean_gini` | Gini impurity média |
| `fam_p_continuous` | Prob. soft-max contínuo |
| `fam_p_discrete` | Prob. soft-max discreto |
| `fam_p_categorical` | Prob. soft-max categórico |
| `fam_max_cardinality` | Cardinalidade máxima |
| `fam_mean_feature_corr` | Correlação média entre features |
| `fam_pca_var_top3` | Variância top-3 PCA |
| `fam_is_high_dim` | Flag: >50 features |
| `fam_ga_score` | Score composto para GA |

### 4.2 Implementação

```python
class MetaFeatureExtractor:
    def extract(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        f = {}
        f.update(self._stat(X, y))           # Estatísticas básicas
        f.update(self._info(X, y))           # Teoria da informação
        f.update(self._land(X, y))           # Landmarks
        f.update(self._dp_relevance(X, y))   # Relevância para DP
        f.update(self._categorical_signal(X, y))  # Sinal categórico
        f.update(self._discrete_signal(X, y))     # Sinal discreto
        f.update(self._family_discriminators(X, y))  # Discriminadores
        return f  # 39 features
```

---

## 5. Pipeline de Meta-Aprendizagem

### 5.1 Construção do Meta-Dataset

```
Para cada dataset D_i no conjunto de treino:
    1. Extrair meta-features: X_meta[i] = extract(D_i.X, D_i.y)
    2. Avaliar todos os mecanismos:
       Para cada mecanismo M_j:
           acc[j] = cross_val_score(
               classifier, 
               M_j.apply(D_i.X), 
               D_i.y, 
               cv=3
           )
    3. Selecionar melhor mecanismo: y_meta[i] = argmax(acc)
    4. Aplicar desempate por família se necessário
```

### 5.2 Seleção do Melhor Mecanismo (Oracle)

O algoritmo de seleção usa **desempate por família**:

```python
def _select_best_mechanism(dp_results, rel_acc, meta_features, margin=0.005):
    best_rel = max(rel_acc.values())
    candidates = [m for m in MECHANISMS if rel_acc[m] >= best_rel - margin]
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Infere família preferida do dataset
    if meta_features["cat_ratio_low_cardinality"] >= 0.7:
        preferred = "categorical"
    elif meta_features["ratio_integer_cols"] >= 0.8:
        preferred = "discrete"
    else:
        preferred = "continuous"
    
    # Filtra por família preferida
    family_candidates = [m for m in candidates 
                         if FAMILY_OF[m] == preferred]
    
    if family_candidates:
        return max(family_candidates, key=lambda m: rel_acc[m])
    
    return max(candidates, key=lambda m: rel_acc[m])
```

### 5.3 Treinamento do Meta-Modelo

O meta-modelo é um **ensemble hierárquico** com múltiplos componentes:

#### 5.3.1 Pré-filtro Categórico (CAT1)

```python
class MetaLearner:
    def _fit_categorical_prefilter(self, X_meta, y_meta):
        # Classificador binário: Exponential vs. resto
        y_binary = (y_meta == EXP_IDX).astype(int)
        
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, 
                max_depth=3,
                learning_rate=0.1
            ))
        ])
        clf.fit(X_meta, y_binary)
        self._cat_prefilter = clf
```

#### 5.3.2 Pré-filtro Gaussiano (GAUSS)

```python
def _fit_gaussian_prefilter(self, X_meta, y_meta):
    # Features específicas para GA vs Laplace
    GA_FEATURES = [
        "n_features", "pca_top1_var", "pca_intrinsic_dim_ratio",
        "mean_sensitivity", "mean_corr", "samples_per_feature"
    ]
    
    # Filtra apenas datasets contínuos
    cont_mask = (y_meta == GA_IDX) | (y_meta == LAP_IDX)
    X_cont = X_meta[cont_mask][:, GA_FEATURE_IDX]
    y_cont = (y_meta[cont_mask] == GA_IDX).astype(int)
    
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3)
    clf.fit(X_cont, y_cont)
```

#### 5.3.3 Classificador de Família (HIER)

```python
def _fit_family_classifier(self, X_meta, y_meta):
    # Mapeia mecanismos para famílias
    y_fam = [FAMILY_OF[CLASSES[y]] for y in y_meta]
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", probability=True, 
                    class_weight="balanced"))
    ])
    clf.fit(X_meta, y_fam)
    self._family_classifier = clf
```

#### 5.3.4 Ensemble Principal

```python
def _fit_ensemble(self, X_meta, y_meta):
    # Oversampling das classes minoritárias
    X_os, y_os = self._oversample(X_meta, y_meta, target_ratio=0.8)
    
    # Treina múltiplos modelos
    models = {
        "ExtraTrees": RandomForestClassifier(
            n_estimators=200, 
            class_weight="balanced"
        ),
        "LogReg": LogisticRegression(
            max_iter=500, 
            class_weight="balanced"
        ),
        "SVM-Linear": SVC(
            kernel="linear", 
            probability=True, 
            class_weight="balanced"
        )
    }
    
    # Seleciona melhor por F1-macro
    scores = {name: cross_val_score(m, X_os, y_os, 
                                    scoring="f1_macro").mean()
              for name, m in models.items()}
    
    best = max(scores, key=scores.get)
    self.best_model = models[best]
    
    # Calibração de probabilidades (Platt scaling)
    calibrated = CalibratedClassifierCV(
        self.best_model, cv="prefit", method="isotonic"
    )
    calibrated.fit(X_os, y_os)
```

---

## 6. Pipeline de Inferência

### 6.1 Fluxo de Decisão Hierárquico

```python
def predict(self, X, y) -> Dict:
    # 1. Extrai meta-features
    features = self.extractor.extract(X, y)
    row = np.array([[features[c] for c in self.META_FEATURE_COLS]])
    
    # 2. Obtém probabilidades de família
    if self._family_classifier:
        family_probs = self._get_family_probs(row)
    
    # 3. CAT1: verifica pré-filtro categórico
    if self._cat_prefilter:
        p_exp = self._cat_prefilter.predict_proba(row)[0, 1]
        p_cat = family_probs.get("categorical", 0)
        
        # Dual-gate: precisa passar ambos os thresholds
        if p_exp >= 0.75 and p_cat >= 0.15:
            return {"recommended_mechanism": "Exponential",
                    "confidence": p_exp,
                    "meta_model_used": "cat_prefilter"}
    
    # 4. GAUSS: verifica pré-filtro gaussiano
    if self._gauss_prefilter:
        p_ga = self._gauss_prefilter.predict_proba(row)[0, 1]
        if p_ga >= 0.80:
            return {"recommended_mechanism": "GaussianAnalytic",
                    "confidence": p_ga,
                    "meta_model_used": "gauss_prefilter"}
    
    # 5. Ensemble com portão de família
    proba = self.best_model.predict_proba(row)[0]
    all_proba = {c: p for c, p in zip(self.CLASSES, proba)}
    
    # 6. HIER: aplica portão de família
    if self._family_classifier:
        all_proba = self._apply_family_gate(all_proba, family_probs)
    
    # 7. Boost para GaussianAnalytic em alta dimensionalidade
    if features.get("pca_top1_var", 1) < 0.45:
        all_proba["GaussianAnalytic"] *= 2.8
    
    # 8. Normaliza e retorna
    total = sum(all_proba.values())
    all_proba = {k: v/total for k, v in all_proba.items()}
    
    best = max(all_proba, key=all_proba.get)
    return {
        "recommended_mechanism": best,
        "confidence": all_proba[best],
        "all_proba": all_proba,
        "meta_model_used": self.best_model_name
    }
```

### 6.2 Portão Hierárquico de Família (HIER)

```python
def _apply_family_gate(self, all_proba, family_probs, threshold=0.55):
    """
    Se uma família tem probabilidade >= threshold,
    zera probabilidades de mecanismos de outras famílias.
    """
    best_family = max(family_probs, key=family_probs.get)
    
    if family_probs[best_family] >= threshold:
        for mech, prob in all_proba.items():
            if FAMILY_OF[mech] != best_family:
                all_proba[mech] = 0.0
    
    return all_proba
```

---

## 7. Hiperparâmetros do Sistema

### 7.1 Thresholds de Decisão

| Parâmetro | Valor (v16) | Descrição |
|-----------|-------------|-----------|
| `_cat_prefilter_threshold` | 0.75 | Confiança mínima CAT1 |
| `_cat_prefilter_family_min` | 0.15 | Família mínima (dual-gate) |
| `_gauss_prefilter_threshold` | 0.80 | Confiança mínima GAUSS |
| `_family_gate_threshold` | 0.55 | Confiança mínima HIER |
| `_ga_boost_pca_threshold` | 0.45 | PCA var para boost GA |
| `_ga_boost_factor` | 2.8 | Fator de boost GA |
| `_oversample_target_ratio` | 0.80 | Razão de oversampling |

### 7.2 Configurações de Validação

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| CV splits (meta) | 3-5 | Validação cruzada |
| n_estimators (RF) | 200 | Árvores do ExtraTrees |
| max_depth (GBC) | 3 | Profundidade do GradientBoosting |
| learning_rate (GBC) | 0.1 | Taxa de aprendizado |

---

## 8. Sistema de Cache

### 8.1 Cache de Resultados

O sistema usa cache em dois níveis:

```python
class UtilityResultCache:
    def __init__(self, cache_dir=".dp_meta_cache"):
        self.cache_dir = Path(cache_dir)
        self._memory_cache = {}  # Cache em memória
        
    def get(self, data_fp, mech, profile_key, epsilon):
        key = f"{data_fp}|{mech}|{profile_key}|{epsilon}"
        
        # 1. Tenta memória
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # 2. Tenta disco
        cache_file = self.cache_dir / f"{hash(key)}.joblib"
        if cache_file.exists():
            result = joblib.load(cache_file)
            self._memory_cache[key] = result
            return result
        
        return None
```

### 8.2 Fingerprint de Dados

```python
def _data_fingerprint(X, y, sample_size=1000):
    """Gera hash único para identificar dataset."""
    rng = np.random.RandomState(42)
    n = X.shape[0]
    
    if n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X_sample, y_sample = X[idx], y[idx]
    else:
        X_sample, y_sample = X, y
    
    content = np.concatenate([
        X_sample.ravel(), 
        y_sample.ravel()
    ]).tobytes()
    
    return hashlib.sha256(content).hexdigest()[:16]
```

---

## 9. Interface de Uso

### 9.1 API Python

```python
from dp_meta_selector import DPMechanismSelector

# Inicialização
selector = DPMechanismSelector(
    delta=1e-5,
    use_cache=True,
    fast_meta_models=True
)

# Treinamento
selector.fit(training_datasets)

# Recomendação
result = selector.recommend(X_new, y_new)
print(f"Mecanismo: {result['recommended_mechanism']}")
print(f"Confiança: {result['confidence']:.2%}")

# Aplicação de DP
X_private = selector.apply(X_new, result['recommended_mechanism'])

# Avaliação
metrics = selector.evaluate(X_new, y_new)
```

### 9.2 CLI

```bash
# Treinamento e avaliação completa
python -m dp_meta_selector --verbose

# Com diagnósticos avançados
python -m dp_meta_selector --diagnostics

# Modo rápido
python -m dp_meta_selector --fast
```

### 9.3 Persistência

```python
# Salvar modelo treinado
selector.save("dp_meta_selector.joblib")

# Carregar modelo
selector = DPMechanismSelector.load_from("dp_meta_selector.joblib")
```

---

## 10. Métricas de Avaliação

### 10.1 Métricas Principais

| Métrica | Fórmula | Descrição |
|---------|---------|-----------|
| **Hit Rate** | correct / total | Taxa de acerto do oracle |
| **Regret** | E[acc_oracle - acc_model] | Perda vs. escolha ótima |
| **Model Accuracy** | E[acc(model_choice)] | Acurácia média do modelo |
| **Relative Performance** | model_acc / oracle_acc | Performance relativa |

### 10.2 Métricas por Família

| Métrica | Descrição |
|---------|-----------|
| `cat_hit` | Hit rate em datasets categóricos |
| `cont_hit` | Hit rate em datasets contínuos |
| `disc_hit` | Hit rate em datasets discretos |

### 10.3 Métricas de Diagnóstico

```python
from dp_meta_selector import run_full_diagnostics

diagnostics = run_full_diagnostics(meta_df, y_pred, y_true)
# - F1-macro por família
# - Confusion matrix
# - Expected Calibration Error (ECE)
# - K-fold cross-validation
# - Ablation study
```
