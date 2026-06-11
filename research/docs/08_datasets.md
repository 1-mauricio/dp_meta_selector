# Datasets Utilizados

> Detalhes dos 489 datasets do OpenML utilizados nos experimentos.

---

## Visão Geral

| Estatística | Valor |
|-------------|-------|
| **Total de datasets** | 489 |
| **Fonte** | OpenML |
| **Tipo** | Classificação tabular |

---

## Distribuição por Tamanho

### Por Número de Features

| Categoria | n_features | Quantidade | % |
|-----------|------------|------------|---|
| Pequenos | < 10 | 203 | 41.5% |
| Médios | 10-50 | 218 | 44.6% |
| Grandes | > 50 | 68 | 13.9% |

### Estatísticas Descritivas

| Métrica | n_samples | n_features |
|---------|-----------|------------|
| Mínimo | 150 | 1 |
| Máximo | 3,000 | 4,096 |
| Média | 963 | 63.9 |
| Mediana | 800 | 12 |

---

## Distribuição por Melhor Mecanismo

| Mecanismo | Datasets | % |
|-----------|----------|---|
| GaussianAnalytic | 255 | 52.1% |
| Laplace | 135 | 27.6% |
| Exponential | 99 | 20.2% |

---

## Datasets Onde Exponential Domina

Datasets categóricos com maior ganho sobre Laplace:

| Dataset | Gap vs Laplace | n_features | Acc Laplace | Acc Exponential |
|---------|----------------|------------|-------------|-----------------|
| pc3 | **+32.3pp** | 37 | 57.4% | 89.8% |
| kc2 | **+31.7pp** | 21 | 47.8% | 79.5% |
| PizzaCutter3 | **+29.1pp** | 37 | 58.7% | 87.8% |
| PieChart3 | **+29.1pp** | 37 | 58.4% | 87.5% |
| pc1 | **+28.3pp** | 21 | 64.7% | 93.1% |
| jm1[sub=3000] | **+25.9pp** | 21 | 54.5% | 80.3% |
| mental-health-in-tech-survey | **+24.1pp** | 26 | 51.7% | 75.8% |
| PieChart1 | **+22.7pp** | 37 | 68.6% | 91.3% |
| hill-valley | **+21.8pp** | 21 | 62.7% | 84.5% |
| total_score | **+21.8pp** | 10 | 36.2% | 58.0% |

**Característica comum:** Features com baixa cardinalidade (valores discretos/categóricos).

---

## Datasets Onde GaussianAnalytic Domina

Datasets com maior ganho de GaussianAnalytic sobre Laplace:

| Dataset | Gap vs Laplace | n_features | Acc Laplace | Acc GA |
|---------|----------------|------------|-------------|--------|
| Smartphone_Recognition | +9.2pp | 66 | 85.2% | 94.4% |
| JuanFeldmanIris | +8.3pp | 4 | 90.4% | 98.7% |
| darwin | +7.8pp | 450 | 42.2% | 50.0% |
| fishcatch | +6.9pp | 7 | 72.1% | 79.0% |
| machine_cpu | +6.7pp | 6 | 84.3% | 91.0% |
| arrhythmia | +6.0pp | 279 | 48.5% | 54.5% |
| hayes-roth_clean | +5.6pp | 4 | 72.8% | 78.4% |
| planning-relax | +5.5pp | 12 | 57.3% | 62.8% |
| heart-long-beach | +5.3pp | 13 | 68.2% | 73.5% |
| mental_health_detection | +5.2pp | 15 | 69.8% | 75.0% |

---

## Datasets de Alta Dimensionalidade (>100 features)

| Dataset | n_features | Melhor Mecanismo |
|---------|------------|------------------|
| Olivetti_Faces | 4,096 | GaussianAnalytic |
| CIFAR_10[sub=3000] | 3,072 | GaussianAnalytic |
| Bioresponse[sub=3000] | 1,776 | GaussianAnalytic |
| Internet-Advertisements | 1,558 | GaussianAnalytic |
| micro-mass | 1,300 | GaussianAnalytic |
| toxicity | 1,203 | Exponential |
| Devnagari-Script | 1,024 | GaussianAnalytic |
| GAMETES_Epistasis | 1,000 | Laplace |
| cnae-9 | 856 | GaussianAnalytic |
| mnist_784[sub=3000] | 784 | GaussianAnalytic |
| Fashion-MNIST[sub=3000] | 784 | GaussianAnalytic |
| isolet[sub=3000] | 617 | GaussianAnalytic |
| har[sub=3000] | 561 | GaussianAnalytic |
| madelon | 500 | GaussianAnalytic |
| darwin | 450 | GaussianAnalytic |

**Padrão:** 14/15 datasets de alta dimensionalidade favorecem GaussianAnalytic (93.3%).

---

## Categorias de Datasets

### Datasets de Imagens (Flattened)

| Dataset | n_features | Melhor |
|---------|------------|--------|
| Olivetti_Faces | 4,096 | GA |
| CIFAR_10 | 3,072 | GA |
| Devnagari-Script | 1,024 | GA |
| mnist_784 | 784 | GA |
| Fashion-MNIST | 784 | GA |

### Datasets de Software Engineering

| Dataset | n_features | Melhor | Gap |
|---------|------------|--------|-----|
| pc3 | 37 | Exp | +32.3pp |
| kc2 | 21 | Exp | +31.7pp |
| pc1 | 21 | Exp | +28.3pp |
| jm1 | 21 | Exp | +25.9pp |

### Datasets de Saúde

| Dataset | n_features | Melhor |
|---------|------------|--------|
| arrhythmia | 279 | GA |
| mental-health-in-tech-survey | 26 | Exp |
| mental_health_detection | 15 | GA |
| heart-long-beach | 13 | GA |

### Datasets de Atividade Humana

| Dataset | n_features | Melhor |
|---------|------------|--------|
| har | 561 | GA |
| Smartphone_Recognition | 66 | GA |

---

## Critérios de Seleção

Os datasets foram selecionados do OpenML com os seguintes critérios:

1. **Tipo:** Classificação supervisionada
2. **Tamanho:** 150-3000 amostras
3. **Features:** Numéricas (contínuas ou discretas)
4. **Qualidade:** Sem missing values excessivos
5. **Diversidade:** Diferentes domínios e características

---

## Arquivos de Referência

| Arquivo | Descrição |
|---------|-----------|
| `research/dp_comparison_full.csv` | Resultados completos (489 datasets) |
| `research/datasets_summary.csv` | Resumo dos datasets |

### Colunas do CSV

| Coluna | Descrição |
|--------|-----------|
| `dataset` | Nome do dataset (prefixo openml:) |
| `n_samples` | Número de amostras |
| `n_features` | Número de features |
| `best_mechanism` | Mecanismo com melhor acurácia |
| `acc_Laplace` | Acurácia com Laplace |
| `acc_GaussianAnalytic` | Acurácia com GaussianAnalytic |
| `acc_Exponential` | Acurácia com Exponential |
| `best_acc` | Melhor acurácia obtida |
| `gap_vs_laplace` | Diferença vs Laplace |
| `baseline_no_dp` | Acurácia sem DP |
| `gap_vs_baseline` | Perda devido ao DP |
