# Emenda ao protocolo confirmatório final — confirmação em repositórios externos

**Estado:** congelada em 2026-11-12, ANTES de listar, baixar ou avaliar qualquer
dataset das fontes abaixo. O protocolo base é `protocols/CONFIRMATORY_PROTOCOL_FINAL.md`
(sha256 66933d771548a8048b38c5952dbc4a9e1675c31694421c0b39d5063af5dfc183),
que permanece inalterado. Motivo: o catálogo OpenML tinha só 2 famílias novas
elegíveis (`results/holdout-openml/SUMMARY.md`).

## O que NÃO muda
- Seletores congelados: exatamente os modelos salvos em
  `results/holdout-openml/models/` (hashes em `results/holdout-openml/frozen.json`).
  Nenhum retreino, nenhum ajuste de τ, fallback, features ou portfólio.
- Portfólio de 15 opções, 8 ε, 3 repetições, transformação `tanh`, regras de
  split e sementes do benchmark; auditoria de ledger em todo ajuste.
- Elegibilidade: 200–50.000 instâncias, 2–30 classes, ≤ 512 atributos
  codificados, schema válido pelos validadores existentes.
- Análise: desfecho primário, bootstrap por família (20.000, semente 20261111),
  critério de sucesso (limite inferior do IC95% > 0 do ganho sobre `..._b8u`),
  caudas e secundários com Holm, exatamente como no protocolo base.

## Fontes e ordem (pré-especificadas)
1. **UCI Machine Learning Repository** (API pública `archive.ics.uci.edu`),
   somente tarefas de classificação.
2. **Kaggle Datasets**, somente se a etapa 1 render menos de 30 famílias
   elegíveis: datasets públicos tabulares de classificação, ordenados por
   uma consulta fixa (`tags: tabular, classification`; arquivos CSV; ordem da
   API por `votes`), primeiros 600 candidatos, com a mesma elegibilidade.
- Alvo total: **40 famílias**; parar ao atingir 40, na ordem sorteada.
- As 2 famílias novas do OpenML já avaliadas entram na análise primária
  combinada (n = 2 + externas). Relatar também a análise só das externas.

## Seleção e disjunção
- Ordem: permutação com semente `20261112` dentro de cada fonte.
- Disjunção obrigatória por nome normalizado, família canônica e família
  semântica (`dpmeta.families`), e por origem declarada (página/DOI de origem
  quando disponível), contra todos os manifestos de `data/` e contra o
  catálogo OpenML inteiro (nomes), para excluir cópias da mesma base.
  Na dúvida sobre ser a mesma base, excluir.
- Uma família por dataset; substituição só pela ordem sorteada em caso de
  falha de validação/carregamento, sem olhar resultados.
- Alvo: a coluna declarada como alvo pela fonte (UCI: variável `target`;
  Kaggle: só datasets com alvo inequívoco declarado na descrição; caso
  contrário, excluir). Sem limpeza além do que o loader já faz (remoção de
  linhas com alvo ausente; atributos com tipo declarado ou inferido pelo dtype).

## Ressalvas registradas antes dos dados
- O schema (domínios nominais, classes) é obtido da própria base pública, como
  no OpenML; vale para benchmark com dados públicos, não para uso com dados
  sensíveis.
- Bases de repositórios diferentes podem ter qualidade e pré-processamento
  diferentes do OpenML; essa é uma mudança de distribuição esperada e faz parte
  do teste de transferência.
