"""Human-readable report rendering from benchmark outputs."""

from __future__ import annotations

import pandas as pd


def _pp(value: float) -> str:
    return f"{100 * value:.2f} p.p."


def _direction(selector_value: float, fixed_value: float) -> str:
    change = selector_value - fixed_value
    if abs(change) < 1e-8:
        return "empatou"
    return f"{'piorou' if change > 0 else 'melhorou'} {_pp(abs(change))}"


def render_validation_report(summary: dict, results: pd.DataFrame,
                             manifest: pd.DataFrame,
                             predictions: pd.DataFrame) -> str:
    family_map = manifest.drop_duplicates("dataset_id").set_index("dataset_id").family_id
    conditions = results.groupby(
        ["dataset_id", "epsilon", "mechanism", "allocation"], as_index=False
    ).regret.mean()
    conditions["family_id"] = conditions.dataset_id.map(family_map)
    per_dataset_action = conditions.groupby(
        ["family_id", "dataset_id", "mechanism", "allocation"], as_index=False
    ).regret.mean()
    per_family_action = per_dataset_action.groupby(
        ["family_id", "mechanism", "allocation"], as_index=False
    ).regret.mean()
    action_means = (per_family_action.groupby(["mechanism", "allocation"], as_index=False)
                    .regret.mean().sort_values("regret"))
    action_range = float(action_means.regret.max() - action_means.regret.min())
    action_lines = "\n".join(
        f"| {row.mechanism} | {row.allocation} | {_pp(row.regret)} |"
        for row in action_means.itertuples(index=False)
    )

    mode_labels = {
        "public": "schema público + ε",
        "dp_aware": "públicas + calibração DP",
        "private_research_only": "públicas + calibração DP + dados de treino (não-DP)",
    }
    mode_lines = []
    for mode, mode_summary in summary["feature_modes"].items():
        low, high = mode_summary["gain_ci95_family_bootstrap"]
        mode_lines.append(
            f"| {mode_labels[mode]} | {_pp(mode_summary['selector']['mean'])} "
            f"| {_pp(mode_summary['selector']['median'])} | "
            f"{_pp(mode_summary['selector']['q90'])} | "
            f"{_pp(mode_summary['selector']['cvar10'])} | "
            f"{_pp(mode_summary['mean_gain'])} [{_pp(low)}, {_pp(high)}] | "
            f"{mode_summary['sign_flip_p_two_sided']:.3f} |"
        )
    mode_table = "\n".join(mode_lines)

    public_predictions = predictions[predictions.feature_mode == "public"]
    selected = public_predictions.selector_action.value_counts().sort_index()
    selected_lines = "\n".join(
        f"| {action} | {int(count)} / {len(public_predictions)} |"
        for action, count in selected.items()
    )
    source_counts = manifest.selection_source.value_counts().to_dict()
    fixed_actions = ", ".join(sorted(public_predictions.fixed_action.unique()))
    public_summary = summary["feature_modes"]["public"]
    selector, fixed = public_summary["selector"], public_summary["best_fixed"]
    fixed_eps, oracle = public_summary["best_fixed_by_epsilon"], public_summary["oracle_reference"]
    ci_low, ci_high = public_summary["gain_ci95_family_bootstrap"]
    conclusion = (
        "O seletor público passou os critérios pré-especificados para esta validação exploratória."
        if public_summary["success"] else
        "O seletor público não demonstrou superioridade sobre o melhor mecanismo fixo."
    )
    l2_summary = summary.get("sufficient_statistic_l2_selector")
    if l2_summary:
        l2_public = l2_summary["feature_modes"]["public"]
        l2_utility = l2_summary["classifier_utility_of_l2_selected_action"]["public"]
        l2_low, l2_high = l2_public["gain_ci95_family_bootstrap"]
        l2_utility_low, l2_utility_high = l2_utility["gain_ci95_family_bootstrap"]
        l2_section = f"""## Seletor treinado para regret L2 das estatísticas suficientes

A métrica é o RMS dos erros L2 de contagens, somas e segundos momentos,
normalizado pela sensibilidade L2 pública de cada bloco. O regret relativo de
cada mecanismo é sua distorção média dividida pela menor distorção na mesma
condição dataset/ε. Este alvo é treinado e avaliado separadamente do regret de
classificação.

| Resumo por família | Seletor L2 público | Melhor fixo L2 |
|---|---:|---:|
| Regret relativo médio | {l2_public['selector']['mean']:.4f} | {l2_public['best_fixed']['mean']:.4f} |
| Mediana | {l2_public['selector']['median']:.4f} | {l2_public['best_fixed']['median']:.4f} |
| q90 | {l2_public['selector']['q90']:.4f} | {l2_public['best_fixed']['q90']:.4f} |

Ganho médio contra o melhor fixo selecionado no treino: **{l2_public['mean_gain']:.4f}**;
IC95% bootstrap por família **[{l2_low:.4f}, {l2_high:.4f}]**; teste de inversão
de sinais *p*={l2_public['sign_flip_p_two_sided']:.4g}; Top-1 auxiliar
{100 * l2_public['top1_auxiliary']:.1f}%. Critério de L2 passou: **{l2_public['success']}**.

Ao avaliar o modelo escolhido pelo alvo L2, o ganho de regret de classificação
contra o melhor fixo escolhido para balanced accuracy foi
{_pp(l2_utility['mean_gain'])} (IC95% por família
[{_pp(l2_utility_low)}, {_pp(l2_utility_high)}], *p*={l2_utility['sign_flip_p_two_sided']:.3f});
critério de utilidade: **{l2_utility['success']}**. Menor erro L2 das estatísticas
não implica melhor classificação.
"""
    else:
        l2_section = ""
    return f"""# Validação OpenML — seletor de mecanismos DP

## Resultado principal

{conclusion}

| Resumo por família | Seletor público | Melhor fixo | Fixo por ε |
|---|---:|---:|---:|
| Regret médio | {_pp(selector['mean'])} | {_pp(fixed['mean'])} | {_pp(fixed_eps['mean'])} |
| Regret mediano | {_pp(selector['median'])} | {_pp(fixed['median'])} | {_pp(fixed_eps['median'])} |
| Regret q90 | {_pp(selector['q90'])} | {_pp(fixed['q90'])} | {_pp(fixed_eps['q90'])} |
| CVaR-10% | {_pp(selector['cvar10'])} | {_pp(fixed['cvar10'])} | {_pp(fixed_eps['cvar10'])} |

Ganho médio pareado: **{_pp(public_summary['mean_gain'])}**; IC95% bootstrap por
família **[{_pp(ci_low)}, {_pp(ci_high)}]**; teste de inversão de sinais
*p*={public_summary['sign_flip_p_two_sided']:.3f}. Com ROPE de 1 p.p., foram
{public_summary['wins_ties_losses_rope_0_01']['wins']} vitórias,
{public_summary['wins_ties_losses_rope_0_01']['ties']} empates e
{public_summary['wins_ties_losses_rope_0_01']['losses']} derrotas. Top-1 foi
{100 * public_summary['top1_auxiliary']:.1f}% e permanece uma métrica auxiliar.

## Comparação dos meta-seletores

| Seletor | Regret médio | Mediana | q90 | CVaR-10% | Ganho vs. melhor fixo (IC95%) | *p* |
|---|---:|---:|---:|---:|---:|---:|
{mode_table}

`dp_aware` acrescenta apenas custos, sensibilidades e escalas de ruído
determinados pelo schema público, ε, δ e mecanismo; não consulta linhas e não
consome orçamento adicional. O ramo `private_research_only` usa descritores
sem ruído dos dados/rótulos de treino. É um diagnóstico de sinal potencial,
**não é implantável sob a garantia atual**: seleção adaptativa baseada em
features privadas fica fora do accounting do ajuste final.

O oráculo por condição teve regret médio de {_pp(oracle['mean'])}, contra
{_pp(fixed['mean'])} do melhor fixo. A ação fixa nos folds externos foi
`{fixed_actions}`.

## Dados e configuração

- {summary['n_datasets']} datasets e {summary['n_families']} famílias em cinco
  folds externos agrupados; fontes: {source_counts.get('OpenML-CC18', 0)}
  OpenML-CC18 e {source_counts.get('OpenML-active-catalog', 0)} do catálogo ativo.
- {summary['repeats']} repetições, ε={', '.join(str(value) for value in summary['epsilons'])},
  {len(summary['actions'])} ações, até
  {summary['n_datasets'] * summary['repeats'] * len(summary['epsilons']) * len(summary['actions']):,}
  treinamentos privados; falhas finais: {len(summary['failed_datasets'])}.
- Laplace e staircase puro-DP; Gaussiano analítico com δ=10⁻⁶; Gaussiano
  calibrado por zCDP e convertido para (ε,δ). Cada ajuste faz três liberações;
  o ledger valida composição básica ou soma de ρ, conforme o mecanismo.
- A transformação é fixa: `tanh` para numéricas e one-hot pelo domínio
  nominal. Baseline limpo e versões privadas usam a mesma representação.
- Seletor e mecanismo fixo são ajustados somente nos grupos de treino de cada
  fold; nenhum resultado externo foi usado para ajuste.

### Regret médio por ação

| Mecanismo | Alocação de ε/ρ | Regret macro por família |
|---|---|---:|
{action_lines}

### Ações do seletor público

| Ação | Condições externas |
|---|---:|
{selected_lines}

{l2_section}

## Interpretação e limites

A comparação principal é entre o seletor público e o melhor mecanismo fixo
escolhido dentro de cada fold. O critério exige IC95% favorável e não piora da
mediana/cauda; Top-1 não decide sucesso. As médias das ações cobrem
{_pp(action_range)} no total, o que mede a oportunidade disponível para
seleção neste benchmark.

OpenML é um benchmark publicamente acessível. Estes resultados medem utilidade
sob o contrato de schema público e não provam privacidade empírica de dados
reais. A garantia formal para cada ajuste usa adjacência replace-one, domínio
limitado e accounting definidos no código. Protocolo informado:
`{summary.get('protocol') or 'nenhum; análise exploratória'}`. O meta-modelo foi treinado
com resultados públicos; treiná-lo com resultados privados exigiria proteger
também o próprio meta-dataset.

O mecanismo staircase usa parâmetro fixo γ=0,5, composição coordenada a
coordenada e sensitivity L∞ por coordenada. Não alegamos que essa configuração
é ótima para este problema: os resultados ótimos da literatura dependem da
função de custo. O Gaussiano zCDP é uma nova calibração/contabilidade para a
mesma família de ruído Gaussiano, não uma nova distribuição.

Manifest SHA-256: `{summary['manifest_sha256']}`. Protocolo SHA-256:
`{summary['protocol_sha256'] or 'ausente'}`. Código SHA-256:
`{summary['implementation_sha256']}`.
"""
