# dp_meta_selector

Código e resultados da dissertação de mestrado sobre seleção de mecanismos de
privacidade diferencial (DP). O framework recomenda, para cada conjunto de
dados e orçamento ε, uma combinação de mecanismo DP e modelo de um portfólio
fixo de 15 opções. Ele não altera o treinamento das opções: só escolhe entre
elas, usando o schema público do conjunto e o ε.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `dpmeta/` | Biblioteca: mecanismos, classificadores, contabilidade de privacidade, metafeatures, seletor (`dissertation_selector.py`), confirmação. |
| `scripts/` | Execução do benchmark, análises da dissertação, holdout confirmatório e externo. |
| `experiments/` | Experimentos que definiram o portfólio final e a regra pública de λ do DP-SGD. |
| `protocols/` | Protocolos confirmatórios congelados (com SHA-256). |
| `data/` | Manifestos dos conjuntos OpenML, UCI e Kaggle (metadados, não os dados). |
| `results/dissertation/` | Tabelas e números da dissertação; `NUMBERS.md` reúne todos os valores citados. |
| `results/holdout-openml/` | Holdout confirmatório com famílias novas do OpenML. |
| `results/holdout-external/` | Confirmação externa em UCI e Kaggle. |
| `results/development-strata/` | Ganho do seletor por estrato público. |
| `figures/` | Figuras finais (`dissertation/`, `external/`). |
| `tests/` | Testes de contabilidade de privacidade e do framework. |

## Nomenclatura dos seletores

- **GBM-15**: seletor final, GBM residual sobre 15 opções, com fallback b8u (τ = 0,02).
- **GBM-13**: seletor intermediário com 13 opções, usado como comparação.
- **RF-12**: seletor inicial, floresta aleatória com 12 opções, usado como comparação.

## Instalação

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## Reprodução

Os resultados vêm de execuções longas em cluster. Os dados brutos de
OpenML, UCI e Kaggle não são redistribuídos aqui: os manifestos em `data/`
listam identificadores e versões para baixá-los das fontes. As condições
completas do benchmark (`results/portfolio-984-sgd/conditions_15.csv`), as
escolhas LODO e os modelos congelados ficam fora do repositório por tamanho.

Ponto de entrada das análises: `scripts/reproduce_dissertation.py`.
Holdouts: `scripts/confirmatory.py` e `scripts/external_*.py`.
