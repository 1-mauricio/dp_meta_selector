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

## Estrutura do código

- `main.py`: CLI e orquestração da pipeline.
- `datasets.py`: carregamento e split de datasets OpenML.
- `baseline_store.py`: armazenamento incremental dos baselines (`SQLite`).
- `meta_features.py`: extração de meta-features.
- `utility.py`: avaliação de utilidade (perfis rápido/completo, cache, screening).
- `meta_dataset.py`: construção do meta-dataset.
- `meta_learner.py`: treino do meta-modelo de seleção.
- `selector.py`: interface principal (`fit`, `recommend`, `apply`).
- `applicator.py`: aplicação prática dos mecanismos DP nos dados.
- `evaluator.py`: avaliação final do framework no conjunto de teste.
- `mechanisms.py`: registro dos mecanismos DP suportados.
- `calibration.py`: calibração de `epsilon` por família.

## Requisitos

- Python 3.10+ (testado em 3.13)
- Dependências principais:
  - `diffprivlib`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `scipy`
  - `tqdm`
  - `openml`
  - `joblib`

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install diffprivlib scikit-learn pandas numpy scipy tqdm openml joblib
```

## Execução rápida

Rodar pipeline completa (treino + avaliação + persistência do modelo):

```bash
python -m dp_meta_selector
```

Rodar apenas pré-computação dos baselines e encerrar:

```bash
python -m dp_meta_selector --precompute-baselines
```

Testar rapidamente o recomendador carregando o modelo salvo (`dp_meta_selector.joblib`):

```bash
/Users/1mmauricio/Workspace/college/mestrado/dp_meta_selector/venv/bin/python test.py
```

Se o ambiente virtual já estiver ativado, também funciona:

```bash
python test.py
```

O script `test.py` valida o fluxo ponta a ponta:
- carrega o modelo salvo;
- executa `recommend()` em dados sintéticos;
- aplica o mecanismo recomendado com `apply()`;
- verifica formato e sanidade da saída (shape e valores finitos).

## Opções da CLI

- `--precompute-baselines`: calcula baselines e encerra.
- `--baseline-id ID` (repetível): restringe/define quais baselines calcular.
- `--export-baselines PATH`: exporta tabela de baselines para CSV/Parquet.
- `--skip-baseline-precompute`: pula pré-computação antes do treino.
- `--no-cache`: desativa cache local (`.dp_meta_cache`).
- `--eval-full`: usa perfil de avaliação completo no holdout.
- `--full-oracle-test`: usa oráculo completo na avaliação (mais caro).

Exemplo com avaliação completa:

```bash
python -m dp_meta_selector --eval-full
```

## Cache e artefatos

- Diretório de cache padrão: `.dp_meta_cache/`
- Banco de baselines: `.dp_meta_cache/baselines.sqlite`
- Modelo treinado (pipeline padrão): `dp_meta_selector.joblib`

### Esquema do SQLite (`baselines`)

Colunas:
- `dataset_id` (`TEXT`, PK composta)
- `baseline_id` (`TEXT`, PK composta)
- `schema_version` (`TEXT`, PK composta)
- `fingerprint` (`TEXT`)
- `profile_key` (`TEXT`)
- `accuracy` (`REAL`)
- `computed_at` (`TEXT`)

Chave composta: `(dataset_id, baseline_id, schema_version)`.

## Pipeline detalhada

1. `__main__.py` chama `main.cli()`.
2. `main.py` configura perfis e cache.
3. `datasets.py` carrega datasets OpenML e separa meta-train/test.
4. `baseline_store.py` garante baselines sem DP.
5. `meta_dataset.py` constrói meta-dataset:
   - meta-features (`meta_features.py`)
   - utilidade por mecanismo DP (`utility.py` + `applicator.py`)
6. `meta_learner.py` treina o preditor de melhor mecanismo.
7. `selector.py` recomenda mecanismo para novo dataset.
8. `evaluator.py` compara recomendado vs oráculo/baselines.

## Reprodução sugerida

1) Pré-computar baselines:

```bash
python -m dp_meta_selector --precompute-baselines
```

2) Rodar pipeline completa:

```bash
python -m dp_meta_selector
```

3) Rodar com avaliação mais pesada:

```bash
python -m dp_meta_selector --eval-full --full-oracle-test
```

## Solução de problemas

- Se ocorrer erro de importação, verifique se o ambiente virtual está ativo.
- Se OpenML estiver lento/indisponível, tente novamente (dependência de rede).
- Se quiser reduzir custo computacional, prefira perfis rápidos (padrão) e mantenha cache ativo.
- Para recomeçar do zero, remova `.dp_meta_cache/`.

## Licença e uso acadêmico

Este repositório está orientado a experimentação acadêmica de meta-aprendizagem aplicada a mecanismos de DP em dados tabulares.