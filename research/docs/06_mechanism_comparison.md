# Estudo Comparativo de Mecanismos DP

> Análise em 489 datasets reais para justificar a necessidade do framework.

---

## Objetivo

Responder à pergunta fundamental:
**"Existe um mecanismo DP universalmente melhor, ou depende do dataset?"**

---

## Metodologia

### Configuração Experimental

| Parâmetro | Valor |
|-----------|-------|
| Datasets | 489 (OpenML) |
| Mecanismos | Laplace, GaussianAnalytic, Exponential |
| Runs por mecanismo | 2 |
| Classificador | LogisticRegression |
| Validação | 3-fold CV |
| Epsilon (continuous) | 5.0 |
| Epsilon (categorical) | 2.0 |

### Script

```bash
python scripts/compare_dp_mechanisms.py --n-datasets 500 --n-runs 2
```

### Métricas

- **gap_vs_laplace**: Diferença de acurácia entre melhor mecanismo e Laplace
- **best_mechanism**: Qual mecanismo teve maior acurácia
- **gap_vs_baseline**: Perda de acurácia devido ao DP

---

## Resultados Principais

### 1. Distribuição do Melhor Mecanismo

| Mecanismo | Datasets | % | Visualização |
|-----------|----------|---|--------------|
| **GaussianAnalytic** | 255 | 52.1% | ██████████████████████████ |
| **Laplace** | 135 | 27.6% | █████████████ |
| **Exponential** | 99 | 20.2% | ██████████ |

**Conclusão:** Laplace, o mecanismo mais comum e default, é o melhor em **apenas 27.6%** dos datasets.

### 2. Gap de Performance vs Laplace Fixo

| Estatística | Valor |
|-------------|-------|
| Gap médio | **+2.18pp** |
| Gap mediano | +0.60pp |
| Gap máximo | +32.3pp |
| Gap mínimo | 0.00pp |
| Desvio padrão | 4.74pp |

**Interpretação:** Um seletor perfeito (oracle) proporcionaria **+2.18pp de acurácia média** sobre usar Laplace fixo.

### 3. Padrões por Dimensionalidade

| Categoria | n_features | n_datasets | GA | Laplace | Exp |
|-----------|------------|------------|----|----|---------|
| SMALL | <10 | 217 | **46.5%** | 35.5% | 18.0% |
| MEDIUM | 10-50 | 204 | **49.0%** | 25.0% | 26.0% |
| LARGE | >50 | 61 | **78.7%** | 11.5% | 9.8% |

**Padrões identificados:**
- **Datasets pequenos:** GaussianAnalytic lidera, mas Laplace é competitivo
- **Datasets médios:** GaussianAnalytic e Exponential dominam
- **Datasets grandes (alta dim):** GaussianAnalytic domina com **78.7%**

---

## Casos Extremos

### Top 10 - Maior Ganho sobre Laplace

| Dataset | Melhor Mecanismo | Gap vs Laplace | Acc Laplace | Acc Melhor |
|---------|------------------|----------------|-------------|------------|
| pc3 | Exponential | **+32.3pp** | 57.4% | 89.8% |
| kc2 | Exponential | **+31.7pp** | 47.8% | 79.5% |
| PizzaCutter3 | Exponential | **+29.1pp** | 58.7% | 87.8% |
| PieChart3 | Exponential | **+29.1pp** | 58.4% | 87.5% |
| pc1 | Exponential | **+28.3pp** | 64.7% | 93.1% |
| jm1[sub=3000] | Exponential | **+25.9pp** | 54.5% | 80.3% |
| mental-health-survey | Exponential | **+24.1pp** | 51.7% | 75.8% |
| PieChart1 | Exponential | **+22.7pp** | 68.6% | 91.3% |
| hill-valley | Exponential | **+21.8pp** | 62.7% | 84.5% |
| total_score | Exponential | **+21.8pp** | 36.2% | 58.0% |

**Padrão:** Datasets categóricos mostram ganhos de **+20-33pp** com Exponential.

---

## Performance Relativa

### Comparação com Laplace

| Mecanismo | Melhor que Laplace | Igual | Pior |
|-----------|-------------------|-------|------|
| GaussianAnalytic | 287 (58.7%) | 45 (9.2%) | 157 (32.1%) |
| Exponential | 110 (22.5%) | 31 (6.3%) | 348 (71.2%) |

**Conclusão:** GaussianAnalytic supera Laplace em **58.7%** dos datasets.

### Impacto do DP na Acurácia

| Métrica | Valor |
|---------|-------|
| Queda média com DP | 7.31pp |
| Queda mediana | 2.94pp |
| Melhor caso | -15.9pp (DP melhorou!) |
| Pior caso | 56.4pp |

---

## Justificativa Quantitativa do Framework

### Cenário 1: Sempre usar Laplace (baseline ingênuo)

```
• Laplace é o melhor em apenas 27.6% dos datasets (135/489)
• Perda média por não escolher corretamente: 2.18pp de acurácia
• Perda máxima em casos extremos: 32.3pp
```

### Cenário 2: Usar seletor perfeito (oracle)

```
• Acerta 100% das vezes
• Ganho médio sobre Laplace: +2.18pp
• Quando outro mecanismo vence, ganho médio: +3.01pp
```

### Cenário 3: Usar nosso framework (v16)

```
• Hit rate atual: 62-68%
• F1-macro: 0.70
• Ganho estimado: ~0.6-1.0pp sobre Laplace fixo
```

---

## Conclusões

### Por que o framework se justifica

1. **Laplace não é universal** — só é o melhor em 27.6% dos casos (489 datasets)

2. **O ganho potencial é significativo** — até +32pp em casos extremos

3. **Existe padrão aprendível:**
   - Alta dimensionalidade → GaussianAnalytic (78.7%)
   - Dados categóricos → Exponential (+20-33pp de ganho)
   - Baixa dimensionalidade → Laplace é competitivo

4. **Mesmo um seletor imperfeito agrega valor** — nosso framework com 62-68% de hit rate captura parte significativa do ganho potencial de 2.18pp

### Implicações para a Dissertação

| Afirmação | Evidência |
|-----------|-----------|
| "Não existe mecanismo DP universal" | Laplace melhor em apenas 27.6% |
| "A escolha impacta significativamente" | Gap médio de 2.18pp, máximo de 32pp |
| "Padrões são aprendíveis" | Dimensionalidade prediz GA (78.7% em alta dim) |
| "Meta-learning é viável" | Framework atinge 62-68% hit rate |

---

## Arquivos do Estudo

| Arquivo | Descrição |
|---------|-----------|
| `scripts/compare_dp_mechanisms.py` | Script de comparação |
| `research/dp_comparison_full.csv` | Resultados detalhados (489 datasets) |
