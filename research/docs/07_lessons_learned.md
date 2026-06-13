# Lições Aprendidas

> Compilação de todas as lições aprendidas durante o desenvolvimento do DP Meta-Selector.

---

## Arquitetura do Meta-modelo

### 1. Treinar auxiliares pré-oversample

**Contexto:** DEC-008 (HIER gate)

Todos os classificadores auxiliares (prefilters, family classifier) devem ser treinados nos dados *originais*, não nos dados após oversample.

**Por quê:** O oversample altera a distribuição de classes. Um classificador treinado após oversample verá Geometric com 159 exemplos (vs 7 reais) e vai predizer "discrete" em excesso.

**Regra:** Qualquer classificador que tome decisões de roteamento deve ver a distribuição real.

---

### 2. Mudanças de perfil invalidam o cache

**Contexto:** DEC-005 (Aligned Profile)

Alterar `clf`, `cv_splits` ou `n_runs` no perfil de meta-build gera labels diferentes para os mesmos datasets.

**Consequência:** Um modelo treinado com um perfil não é comparável a um modelo treinado com outro perfil. O cache deve ser invalidado.

**Regra:** Documente o perfil exato usado para cada experimento.

---

### 3. Poucos exemplos = impossível de modelar

**Contexto:** DEC-009 (DISC prefilter)

Com apenas 7 exemplos positivos (Geometric), nenhum classificador binário consegue aprender o padrão.

**Limiar prático:** ~20 exemplos positivos mínimo para viabilidade.

**Solução:** Datasets sintéticos (DEC-015) ou remoção da classe (DEC-010).

---

## Engenharia de Features

### 4. Adicionar features globais é arriscado

**Contexto:** DEC-011 (GAUSS prefilter)

`_gaussian_signal()` adicionado às features globais causou colapso:
- CAT1 absorveu as novas features
- Disparou 141x Exponential (vs 37 oracle)

**Regra:** Features específicas de um prefilter devem ficar isoladas nesse prefilter, não no espaço global.

---

### 5. Filtrar datasets por feature é arriscado

**Contexto:** DEC-010 (tentativa revertida)

Filtrar datasets com `disc_composite_score > 0.25` parecia seguro, mas afetou 183/342 datasets porque colunas inteiras são comuns em datasets categóricos codificados.

**Regra:** Sempre verificar o impacto de um filtro em todo o dataset antes de aplicar.

---

### 6. Dados sintéticos são essenciais para classes raras

**Contexto:** DEC-015

Com apenas 7 exemplos de Geometric no treino real, o prefilter não aprendia. Com sintéticos, temos cobertura suficiente.

**Regra:** Para classes com < 20 exemplos, gerar sintéticos ou aceitar que o modelo não vai aprender.

---

## Thresholds e Trade-offs

### 7. Prefilter com precision < 35% é net negativo

**Contexto:** DEC-013 (GAUSS desabilitado)

GAUSS prefilter tinha 5 TPs / 14 FPs (precision=26%). Cada FP gera 1 miss que seria 1 hit se Laplace.

**Cálculo:** Net impact = TPs - FPs = 5 - 14 = -9 hits.

**Regra:** Se precision < 35%, desabilitar o prefilter.

---

### 8. Threshold elevado ≠ menos recall

**Contexto:** DEC-014

Elevar T1 de 0.65 → 0.90 bloqueou FPs com p_exp moderado enquanto manteve TPs (que têm p_exp muito alto, mediana ~0.97).

**Insight:** Se TPs têm confiança muito maior que FPs, elevar o threshold melhora net.

---

### 9. CAT1 intercepta antes do HIER

**Contexto:** DEC-012 (dual-gate)

O retorno early do prefilter impede que o HIER veja o Exponential. Portanto, o HIER não é proteção suficiente para FPs do CAT1.

**Solução:** Dual-gate dentro do CAT1 (verificar família antes de retornar).

---

### 10. Desempate por família > desempate por acurácia

**Contexto:** DEC-016

Quando mecanismos têm acurácias similares (diferença < 0.5%), a família do dataset é um sinal mais robusto.

**Regra:** Usar características do dataset para desempate, não frações decimais de acurácia.

---

### 11. Thresholds otimizados para precision prejudicam recall

**Contexto:** DEC-018

Thresholds altos (v13: T1=0.90) maximizam precision mas causam recall de 2.8% em categóricos.

**Trade-off:** Com mais dados de treino, podemos relaxar thresholds para melhor recall.

---

### 12. F1-macro vs hit rate é um trade-off

**Contexto:** v16

Melhorar recall de classes minoritárias (Exponential, GA) piora o hit rate geral porque Laplace é a maioria.

| Métrica | v15 | v16 |
|---------|-----|-----|
| hit_rate | 67.6% | 61.9% |
| F1-macro | 0.55 | 0.70 |

**Decisão:** Depende do objetivo. Para dissertação, F1-macro balanceado é mais defensável.

---

## Meta-aprendizagem

### 13. Headroom real é pequeno

**Contexto:** DEC-003 (diagnóstico)

Oracle supera Laplace em apenas 65/147 datasets com ganho médio de 0.8pp. O framework é viável mas a margem é apertada.

**Implicação:** Qualquer ruído introduzido é prejudicial. Cada decisão de design importa.

---

### 14. Laplace não é universal

**Contexto:** Estudo comparativo (489 datasets)

Laplace é o melhor em apenas 27.6% dos datasets. GaussianAnalytic e Exponential são melhores em 72.4% dos casos.

**Implicação:** O problema de seleção de mecanismo é real e significativo.

---

### 15. Padrões são aprendíveis

**Contexto:** Estudo comparativo

- Alta dimensionalidade → GaussianAnalytic (78.7%)
- Dados categóricos → Exponential (+20-33pp de ganho)
- Baixa dimensionalidade → Laplace competitivo

**Implicação:** Meta-features capturam informação suficiente para predição.

---

## Resumo Visual

```
DECISÕES DE ARQUITETURA
├── Treinar auxiliares pré-oversample
├── Manter features isoladas por prefilter
└── Dual-gate para early returns

THRESHOLDS
├── Precision < 35% → desabilitar
├── TPs têm alta confiança → elevar threshold
└── Mais dados → relaxar thresholds

TRADE-OFFS
├── Precision vs Recall
├── Hit rate vs F1-macro
└── Simplicidade vs Performance

META-LEARNING
├── Headroom é pequeno (~2pp)
├── Laplace não é universal (27.6%)
└── Padrões são aprendíveis (dimensionalidade, categoricidade)
```

---

## Privacidade Diferencial e Meta-Learning (v17)

### 16. Features DP-específicas têm alto retorno

**Contexto:** DEC-023 (v17)

Meta-features genéricas (média, correlação, forma) não capturam o que importa para DP: sensibilidade global, impacto de clipping e disparate impact em subgrupos.

**Resultado:** Adicionar 40 features DP-específicas elevou o F1-macro de 0.70 → 0.87 (+17pp) sem alterar nenhum threshold ou arquitetura do ensemble.

**Regra:** Domínio específico do problema → features específicas do problema. Features genéricas têm teto de performance.

---

### 17. Contexto do usuário (ε, task_type) é obrigatório

**Contexto:** DEC-023 (v17)

O mecanismo ideal para ε=0.1 é frequentemente diferente do mecanismo ideal para ε=10. Um meta-modelo sem contexto de ε está fazendo uma decisão sem informação fundamental.

**Exemplo:** Com ε pequeno, o ruído é alto e mecanismos com menor sensibilidade (Laplace vs Gaussian) têm vantagem. Com ε grande, a escolha depende mais da natureza dos dados.

**Regra:** Inclua variáveis de contexto operacional no vetor de features. O modelo não pode adivinhar o que o usuário não forneceu.

---

### 18. Regressão de perda precisa de labels confiáveis

**Contexto:** DEC-023 (v17)

A abordagem de regressão (prever "perda de utilidade %") é teoricamente superior à classificação ("qual mecanismo é melhor") porque produz um ranking contínuo em vez de uma decisão binária.

**Porém:** Com `n_runs=1`, o label `utility_loss_laplace` para um dataset varia ±5pp entre execuções por causa do ruído estocástico da DP. O regressor tenta aprender padrões precisos de dados imprecisos.

**O classificador é robusto** porque só precisa saber "A > B", não "A é 3.2pp melhor". Empates e ruído não mudam a classificação.

**Regra:** Use `META_STABLE_PROFILE` (n_runs=5) ao gerar meta-datasets para regressão. Para classificação, `n_runs=1` é aceitável.

**Hierarquia:** meta-dataset com labels confiáveis → regressor bem calibrado → melhor hit rate. Com labels ruidosas, o classificador vence.

---

### 19. Treinar regressão no dado original (pré-oversample)

**Contexto:** DEC-023 (v17)

O oversample foi introduzido para balancear classes raras (ex: Geometric com 7 exemplos) no classificador. Mas o regressor não precisa de balanceamento — ele aprende a magnitude da perda, não a frequência da classe.

**Risco:** Treinar o regressor em dados sobre-amostrados infla artificialmente a importância de datasets sintéticos e distorce as perdas previstas.

**Regra:** Salvar `X_meta_orig` (pré-oversample) e treinar o regressor nele. O classificador usa `X_meta` (pós-oversample).

---

### 20. Softmin para converter perdas em probabilidades

**Contexto:** DEC-023 (v17)

O `_predict_regression()` retorna perdas (ex: `[Laplace: 3.1%, Gaussian: 5.2%, Exponential: 8.7%]`). Para manter API compatível (`all_proba` dict) e permitir análise de confiança, as perdas são convertidas em "probabilidades" via softmin com temperatura T=10:

```python
p_i = exp(-loss_i / T) / Σ exp(-loss_j / T)
```

Temperatura T=10 normaliza perdas em escala de % para probabilidades sem colapso (T muito baixo colapsaria para one-hot).

**Regra:** Softmin é preferível a normalização linear porque é diferenciável e respeita a relação relativa entre perdas.

---

## Resumo Visual (Atualizado v17)

```
DECISÕES DE ARQUITETURA (v17)
├── Treinar auxiliares pré-oversample
├── Treinar REGRESSOR pré-oversample (novo)
├── Manter features isoladas por prefilter
├── Dual-gate para early returns
└── Contexto obrigatório (ε, task_type)

THRESHOLDS
├── Precision < 35% → desabilitar
├── TPs têm alta confiança → elevar threshold
└── Mais dados → relaxar thresholds

TRADE-OFFS
├── Precision vs Recall
├── Hit rate vs F1-macro
├── Simplicidade vs Performance
├── n_runs=1 (rápido) vs n_runs=5 (confiável)  ← novo
└── Classificação (robusto ao ruído) vs Regressão (informativo)  ← novo

META-LEARNING
├── Headroom é pequeno (~2pp)
├── Laplace não é universal (27.6%)
├── Padrões são aprendíveis (dimensionalidade, categoricidade)
├── Features DP-específicas: +17pp F1-macro  ← novo
└── Contexto ε/task_type é não-opcional  ← novo
```
