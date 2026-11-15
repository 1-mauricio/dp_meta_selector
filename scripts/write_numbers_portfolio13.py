"""Audit-ready Portuguese numerical crosswalk from dissertation RF-12 text to GBM-13 artifacts."""
from __future__ import annotations
import argparse,json,re,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from dpmeta import selector_portfolio13 as sel,portfolio_benchmark as pb
from dpmeta.provenance import file_sha256,runtime_fingerprint
from scripts.dissertation_analysis_portfolio13 import fam,mean_ci,dump


def fmt(v):
    if v is None or (isinstance(v,float) and np.isnan(v)):return 'não se aplica'
    if isinstance(v,(float,np.floating)):return f'{v:.10g}'.replace('.',',')
    return str(v).replace('|','\\|').replace('\n',' ')


def mdtable(df):
    columns=list(df.columns)
    return '\n'.join(['| '+' | '.join(map(str,columns))+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])


def flatten(obj,prefix=''):
    result=[]
    if isinstance(obj,dict):
        for key,val in obj.items():result.extend(flatten(val,f'{prefix}.{key}' if prefix else key))
    elif isinstance(obj,list):
        if all(not isinstance(v,(dict,list)) for v in obj):result.append((prefix,json.dumps(obj,ensure_ascii=False)))
        else:
            for i,val in enumerate(obj):result.extend(flatten(val,f'{prefix}[{i}]'))
    else:result.append((prefix,obj))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dissertation',type=Path,default=Path('../mestrado-papers/latex-dissertacao'))
    parser.add_argument('--output',type=Path,default=Path('results/dissertation-portfolio13'))
    args=parser.parse_args();root=args.output
    required=['summary.json','extras/metamodels.csv','extras/shapley_primitives.csv','extras/importance_columns.csv',
              'modes/modes.csv','modes/importance_groups.csv','learning_curve/summary.csv','holdout/summary.json',
              'size/sensitivity_by_size.csv','strategies/summary.json']
    missing=[str(root/f) for f in required if not (root/f).exists()]
    if missing:raise RuntimeError(f'Análises incompletas: {missing}')
    get=lambda p:json.loads((root/p).read_text())
    s=get('summary.json');h=get('holdout/summary.json');corpus=get('corpus_summary.json')
    old=json.loads(Path('results/portfolio-984-shared-selector-lodo/summary.json').read_text())['primary_result']
    oldextra=json.loads(Path('results/portfolio-984-dissertation-extras/summary.json').read_text())
    oldhold=json.loads(Path('results/openml-holdout-41/summary.json').read_text())
    rows=[]
    def metric(key,oldvalue,newvalue,source,ci=None,note=''):
        rows.append(dict(metrica=key,RF-12=oldvalue,GBM-13=newvalue,IC95_gbm13='' if ci is None else
                         '['+fmt(ci[0])+'; '+fmt(ci[1])+']',fonte_gbm13=source,nota=note))
    for key,ov,nv in [('datasets',984,s['datasets']),('opções',12,s['options']),('avaliações',283392,s['evaluations']),
          ('condições dataset–epsilon',7872,s['conditions']),('famílias na validação principal',984,s['families']),
          ('falhas',0,11),('repetições',3,3),('orçamentos',8,8)]:
        metric(key,ov,nv,'summary.json',note='contagem; sem IC amostral')
    shares=pd.read_csv(root/'win_share_by_epsilon.csv',index_col=0)['Total']
    metric('opção mais vencedora: fração com empates divididos',.229,float(shares.max()),'win_share_by_epsilon.csv',
           note='RF-12: NB antigo; GBM-13: '+str(shares.idxmax()))
    ci=s['best_fixed_ba'];metric('BA_privada_melhor_fixa',.4356,ci['mean'],'summary.json#best_fixed_ba',
                              (ci['ci_low'],ci['ci_high']))
    noise=pd.read_csv(root/'noise/q2_result.csv',index_col=0)
    for key,value in [('floor_private_option_rep_vs_rep',.556),('winner_2reps_predicts_3rd',.443),
                      ('reliability_mean3',.67),('floor_clean_model_rep_vs_rep',.255),('dp_changes_model_single_rep',.455)]:
        metric('ruído.'+key,value,float(noise.loc[key,'todos']),'noise/q2_result.csv',note='diagnóstico da grade pública; ponto, sem IC na análise de origem')
    metric('DP muda o modelo vencedor na média de três repetições',.507,get('extras/summary.json')['dp_changes_best_model']['overall'],
           'extras/summary.json#dp_changes_best_model')
    pairs={'tied_fraction':.133,'all_tied':443,'constant_unique_datasets':25,'median_distinct_winners':5,
           'friedman_dataset_chi2':998.55,'kendall_W_dataset':.09,'spread_median':.098,'spread_p90':.318,
           'spread_gt_10pp':.495,'top_two_gap_median':.011,'near_chance_clean_datasets':238,
           'near_chance_private_fraction':.30}
    for key,ov in pairs.items():metric('H1.'+key,ov,s['h1'][key],'summary.json#h1')
    for stat in ['mean','median','q90','cvar90']:
        metric('seletor.regret.'+stat,old['selector_regret'][stat],s['best_fixed_regret']['selector'][stat],
               'summary.json#best_fixed_regret',note='RF-12: 984 famílias canônicas; GBM-13: 667 grupos semânticos')
        metric('melhor_fixa.regret.'+stat,old['best_fixed_regret'][stat],s['best_fixed_regret']['reference'][stat],
               'summary.json#best_fixed_regret',note='referência muda de NB antigo para b8u')
        gain=s['best_fixed_regret']['gain'][stat]
        ov=(old['mean_regret_gain_vs_best_fixed'] if stat=='mean' else old.get('regret_tail_gains',{}).get(stat,{}).get('gain'))
        metric('ganho_vs_melhor_fixa.'+stat,ov,gain['mean'],'summary.json#best_fixed_regret.gain',
               (gain['ci_low'],gain['ci_high']))
    for key,ov,new in [('BA_privada_seletor',.4421,s['selector_ba']['mean']),
          ('Top1_estrito_condições',.2744,s['top1_strict']),('Top1_estrito_melhor_fixa',.2174,s['fixed_top1_strict']),
          ('oráculo.folga_média_dataset',.0436,s['headroom']['mean']),('oráculo.folga_mediana_dataset',.0244,s['headroom']['median']),
          ('oráculo.folga_menor_0005',210,s['headroom']['below_005']),('oráculo.folga_maior_01',129,s['headroom']['above_10']),
          ('oráculo.fração_recuperada',.15,s['headroom']['captured_family'])]:metric(key,ov,new,'summary.json')
    audit=pd.read_csv(root/'top1_definition_audit.csv').set_index('run')
    metric('Top1_realmente_vencedora_única',audit.loc['v4_984','true_unique_winner'],audit.loc['gbm13_973','true_unique_winner'],
           'top1_definition_audit.csv',(audit.loc['gbm13_973','unique_ci_low'],audit.loc['gbm13_973','unique_ci_high']),
           note='ERRATA: 27,44% da versão anterior era argmax com desempate; a taxa de vencedora única era 23,8821%')
    for ref,ov in [('fixed_h27',.0780),('fixed_b8u',.0025),('fixed_old_nb',.0025),('selector_rf12',0.)]:
        gain=h['comparisons'][ref]['gain']['mean']
        metric('holdout.ganho_vs_'+ref,ov,gain['mean'],'holdout/summary.json#comparisons',
               (gain['ci_low'],gain['ci_high']),note='EXPLORATÓRIO; RF-12 melhor fixa era NB antigo')
    for stat,ov in [('mean',.0719),('median',.0355),('q90',.1874),('cvar90',.2267)]:
        metric('holdout.seletor.regret.'+stat,ov,h['comparisons']['fixed_b8u']['selector'][stat],
               'holdout/summary.json',note='EXPLORATÓRIO')
    for key,ov,new in [('famílias dentro da faixa',14,h['strata']['inside_dev_range']['families']),
                      ('famílias fora da faixa',27,h['strata']['outside_dev_range']['families']),
                      ('famílias semânticas com parentes',44,corpus['multi_dataset_families']),
                      ('datasets com parentes',349,corpus['datasets_with_relatives']),
                      ('datasets sem parentes',635,corpus['independent_datasets'])]:metric(key,ov,new,'corpus_summary.json / holdout/summary.json')
    extra=get('extras/summary.json')
    for key,ov in [('recommendation_latency_ms',oldextra['recommendation_latency_ms'])]:
        metric(key,json.dumps(ov),json.dumps(extra[key]),'extras/summary.json',note='medição de tempo local, sem IC; sob carga de outras análises')
    for group,ov in [('attributes',.459),('n',.320),('epsilon',.163),('classes',.057)]:
        sh=pd.read_csv(root/'extras/shapley_primitives.csv').set_index('primitive').loc[group]
        metric('Shapley.parcela.'+group,ov,sh['share'],'extras/shapley_primitives.csv')
        metric('Shapley.phi.'+group,{'attributes':.0064,'n':.0045,'epsilon':.0023,'classes':.0008}[group],
               sh['mean'],'extras/shapley_primitives.csv',(sh.ci_low,sh.ci_high))
    metric('largura do vetor de features',84,97,'selector_replay.json',note='72 públicas/calibração + 12 indicadores legados (nova opção herda NB) + 13 one-hot próprios; reprodução exata do experimento')
    metric('features públicas/calibração sem indicadores de opção',72,72,'extras/importance_columns.csv',note='17 schema + 55 calibração; custo adicional ε=0')
    metric('metamodelo', 'RF: 1000 árvores, profundidade 6, max_samples=0,5',
        'HistGradientBoostingRegressor: 400 iterações, learning_rate=0,05; max_leaf_nodes=31; min_samples_leaf=40; l2=1; seed=20261110; alvo BA_opção−BA_NB_antigo; tau=0,01',
        '../../dpmeta/selector_portfolio13.py',note='peso total de treino = número de famílias; fallback para melhor fixa do treino')
    metric('NB_categórico', '16 bins; prior empírico ruidoso', 'nova opção: 8 bins; prior uniforme na decisão; consulta ruidosa de prior retida',
           '../portfolio-984-nb8u/SUMMARY.md',note='p+1 consultas, sensibilidade L1=2 e mesmo ledger; calibração do seletor herdada do NB antigo')
    core=pd.DataFrame(rows);core.to_csv(root/'numeric_crosswalk.csv',index=False)
    lines=['# Números da dissertação GBM-13 — Fase A','',
      'Este arquivo é um registro para a reescrita, sem editar o LaTeX. Valores em escala BA; multiplicar por 100 para pontos percentuais. Os valores RF-12 são os efetivamente impressos no texto ou os valores de precisão integral das fontes antigas indicadas. Quando o comparador mudou, a linha identifica a mudança; não se deve substituir apenas o algarismo na frase antiga.','',
      '**Holdout GBM-13: EXPLORATÓRIO.** A b8u e o seletor GBM-13 foram escolhidos depois da abertura do holdout. Nenhum resultado GBM-13 neste holdout confirma H2a/H2b segundo o protocolo confirmatório inicial. O RF-12 e os outros métodos históricos permanecem comparadores congelados das execuções originais; as buscas históricas continuam operando com 12 opções.','',
      '**Errata de Top-1 na RF-12:** os 27,44% impressos não exigiam vencedora única; o código usava argmax com desempate pela ordem do registro. Recalculando vencedora única nas mesmas 7.872 condições, a taxa RF-12 é 23,8821%. A GBM-13 principal usa vencedora única; as três definições e seus denominadores estão em top1_definition_audit.csv. Não interpretar 27,44%→25,40% como queda entre métricas idênticas.','',
      'A validação principal usa os picks existentes do LODO por família semântica (`results/experiments/lodo_13/lodo_picks.csv`, GBM residual, fallback=sim). As análises secundárias usam cinco partições semânticas. A estimativa principal dá peso igual por família, depois por dataset e orçamento. A RF-12 principal dava peso igual às 984 famílias canônicas (uma por dataset): a variação RF-12→GBM-13 inclui mudança do estimando, redução para 973 datasets, expansão do portfólio e troca do seletor. `portfolio_expansion_vs_attrition.csv` isola a expansão da perda de datasets.','',
      'Os ICs são percentis de 95%, com 20.000 reamostragens pareadas das famílias, salvo a reprodução literal do LODO original (5.000) e as tabelas diagnósticas que explicitamente usam bootstrap de datasets. Contagens, hiperparâmetros e descrições do corpus não possuem IC amostral. Os CIs são pontuais; os testes pareados recebem Holm na respectiva família de contrastes. `p=0` em CSV indica subfluxo numérico, a escrever `p<10⁻³⁰⁰`, nunca probabilidade nula.','',
      'Dois datasets (41526 e 45939) não têm `semantic_family_id`. O LODO fornecido converteu NaN para a chave literal `nan`, formando um grupo compartilhado. A reprodução mantém 667 grupos. Há 666 famílias com nome declarado, ou 668 se os dois IDs sem nome recebessem chaves exclusivas. O loader pode preservar esse campo como string vazia; ela representa o mesmo grupo compartilhado. Ao combinar arquivos, o rótulo é alinhado explicitamente para não perder essa família na inferência.','',
      'A referência de regret nos CSVs permanece `best_clean_portfolio_ba` do portfólio original, conforme o benchmark b8u; não foi ampliada silenciosamente. O critério perto do acaso e o apêndice usam o melhor clean das 13 opções (média dos máximos por repetição); a referência original de regret é preservada em coluna separada. A tabela de transformações calcula explicitamente o máximo clean das 13 opções dentro da transformação. O ganho de BA e a redução de regret pareados são invariantes a uma referência clean comum.','',
      '## Valores centrais: RF-12 → GBM-13','',mdtable(core),'',
      '## Tabelas: valores antigos e novos, linha a linha','',
      'Cada bloco transcreve as linhas numéricas da tabela antiga e, em seguida, apresenta integralmente os valores GBM-13. As colunas `ci_low`/`ci_high` ou equivalentes pertencem ao ponto da mesma linha. `não se aplica` não significa zero. A tabela de 973 datasets inclui também os metadados e resultados individuais para refazer o apêndice.','']
    tablemap={
     'tab:opcoes':([], '13 opções: as 12 linhas antigas permanecem; acrescentar categorical_naive_bayes__laplace_histograms_b8u, 8 bins e prior uniforme. Mesmo ledger do NB antigo; hiperparâmetros completos na API GBM-13 e SUMMARY do benchmark.'),
     'tab:vitorias-eps':(['win_share_by_epsilon.csv','winner_share_ci.csv'], 'Vitórias divididas entre empates; frações 0–1, converter para %.'),
     'tab:resultados-opcoes':(['option_performance.csv'],'Peso igual por família semântica. Cada média tem seu IC.'),
     'tab:resultados-seletor':(['method_summary.csv','strategies/summary.json'],'LODO principal; estratégia por ε usa cinco partições semânticas. family_scores são dados individuais, não tabela de IC.'),
     'tab:regret-eps':(['regret_by_epsilon.csv','strategies/by_epsilon.csv'],'Top-1 exige vencedora única; comparar separadamente do acerto que admite empate.'),
     'tab:estratos':(['extras/gain_by_stratum.csv'],'N de datasets e N de famílias separados.'),
     'tab:modos':(['modes/modes.csv','modes/pairwise.csv'],'Todos usam GBM residual+fallback e as mesmas cinco partições; descritores privados históricos preservados.'),
     'tab:orcamento-total':(['modes/equal_total.csv'],'Custo ε_meta=1. Referência usa maior ε na grade que não exceda o orçamento total.'),
     'tab:importancia':(['modes/importance_groups.csv'],'Impureza não se aplica a HistGradientBoosting; remover a coluna da floresta, não preencher com zero.'),
     'tab:shapley':(['extras/shapley_primitives.csv','extras/summary.json'],'Três permutações por partição. Efeito total em shapley_total; opções one-hot não permutadas.'),
     'tab:metamodelos':(['extras/metamodels.csv','extras/metamodel_pairwise.csv'],'RF-12 é sua arquitetura reestimada nas 13 opções; GBM residual+fallback GBM-13; GBM abs; Extra-Trees; Ridge; kNN; melhor fixa. kNN não aceita pesos.'),
     'tab:curva-aprendizado':(['learning_curve/summary.csv'],'A quantidade de treino agora é de famílias semânticas; dois sorteios por fração, um para 100%. RF e GBM-13 nas mesmas famílias.'),
     'tab:busca-idealizada':(['extras/idealized_private_search.csv'],'Seletor LODO em ε; oráculo e melhor fixa b8u em ε/2 e ε/4.'),
     'tab:tamanho':(['size/sensitivity_by_size.csv','size/coverage_provenance.json'],'Bootstrap de famílias dentro de cada faixa. A cobertura de classes foi reaproveitada dos mesmos splits, filtrando aos IDs retidos.'),
     'tab:transformacao':(['transformations/summary.csv','transformations/provenance.json'],'199 datasets comparáveis; 391 rejeitado, sem substituição. Novas execuções b8u em asinh10 e public_bounds; tanh reutiliza os mesmos seeds do benchmark completo. IC do contraste NB b8u−H27 por dataset, como a tabela original.'),
     'tab:holdout':(['holdout/comparisons.csv','holdout/summary.json'],'EXPLORATÓRIO. Referências: b8u (dev best), H27, NB antigo, RF-12 congelada. CIs/tails por família; Holm inclui todos os contrastes GBM-13, sem contraste confirmatório isento.'),
     'tab:colapso-softmax':(['holdout/softmax_diagnostic.csv','holdout/diagnostic_summary.json'],'As colunas da softmax limpa, saturação e classe majoritária são diagnósticos históricos inalterados; novas colunas NB b8u limpa e gap privado. Não reexecutar a mesma softmax equivale a reutilizar o ajuste idêntico, sem tratá-lo como novo resultado.'),
     'tab:regra-limiar':(['rules/summary.csv','rules/frozen_rules.json','rules/holdout_exploratory.json'],'A configuração é reajustada por LODO semântico com b8u como opção de alto orçamento; treino nunca inclui a família testada. Holdout é exploratório. A regra antiga 10^2,2 não deve ser transposta como se ainda fosse a regra ótima.'),
     'tab:ap-origem':(['appendix_origins.csv'],'Corpus efetivamente comparado: 973.'),
     'tab:ap-resumo':(['appendix_public_dimensions.csv'],'Quantis e extremos descritivos, sem IC.'),
     'tab:ap-faixas':(['appendix_bands.csv'],'Contagens e percentuais no corpus retido.'),
     'tab:ap-semanticas':(['appendix_semantic_families.csv','corpus_summary.json'],'Agrupamento efetivamente usado no LODO, incluindo a chave compartilhada nan.'),
     'tab:ap-excluidas':(['appendix_cost_exclusions.csv','appendix_validation_exclusions.csv'],'Os 16 excluídos por custo permanecem históricos; listar separadamente os 11 excluídos por validação na extensão GBM-13.'),
     'tab:ap-vencedoras':(['appendix_modal_winners.csv','appendix_distinct_winners.csv'],'Modal divide empates nas vitórias e usa BA média como desempate, como o gerador do apêndice.'),
     'tab:ap-datasets':(['appendix_datasets.csv'],'973 linhas; winner, distinct_winners, best_clean, best_private, headroom, metadados públicos. IDs excluídos não recebem imputação.'),
    }
    texfiles=[args.dissertation/'capitulos/3-metodologia.tex',args.dissertation/'capitulos/4-resultados.tex',args.dissertation/'postextual/apendice-a.tex']
    tables={}
    for p in texfiles:
        text=p.read_text()
        for match in re.finditer(r'\\begin\{(table|longtable)\}.*?\\end\{\1\}',text,re.S):
            block=match.group();labels=re.findall(r'\\label\{(tab:[^}]+)\}',block)
            if labels:tables[labels[0]]=(p,1+text[:match.start()].count('\n'),block)
    absent=set(tables)-set(tablemap)
    if absent:raise RuntimeError(f'Tabelas sem cobertura: {absent}')
    coverage=[]
    for label,(paths,note) in tablemap.items():
        if label not in tables:continue
        p,lineno,block=tables[label]
        oldrows=[line for line in block.splitlines() if '&' in line and re.search(r'\d',line)]
        lines += [f'### {label}','',f'Fonte antiga: `{p}:{lineno}`. {note}','',
                  'Valores RF-12 impressos:','','```tex',*oldrows,'```','']
        for rel in paths:
            target=root/rel
            if not target.exists():raise RuntimeError(f'Fonte GBM-13 ausente: {target}')
            lines += [f'Valores GBM-13 — [{rel}]({rel}):','']
            if target.suffix=='.csv':
                df=pd.read_csv(target)
                if label=='tab:ap-datasets':
                    keep=['dataset_id','dataset_name','dataset_version','n_instances','n_features','n_encoded_features','n_classes',
                          'best_clean','best_private','winner','distinct_winners','headroom']
                    df=df[[col for col in keep if col in df]]
                elif label in ['tab:ap-excluidas']:
                    keep=['dataset_id','dataset_name','n_instances','n_features','n_encoded_features','n_classes']
                    df=df[[col for col in keep if col in df]]
                elif rel=='family_scores.csv':
                    # Summary of all rows is in summary.json; preserve paired observations in file.
                    df=pd.DataFrame(flatten(s['best_fixed_regret']),columns=['métrica','gbm13'])
                elif label=='tab:colapso-softmax':df=df[df.dataset_id.isin([41014,46361,42680,156,1597,1241])]
                lines += [mdtable(df),'']
            else:lines += [mdtable(pd.DataFrame(flatten(json.loads(target.read_text())),columns=['métrica','gbm13'])),'']
        coverage.append(dict(table=label,source_rf12=str(p),line=lineno,source_portfolio13=';'.join(paths),status='regenerada' if paths else 'portfólio estendido; parâmetros documentados'))
    pd.DataFrame(coverage).to_csv(root/'table_coverage.csv',index=False)
    lines += ['## Números narrativos e estatísticas complementares','',
       'As fontes abaixo cobrem números citados no corpo que não aparecem numa tabela. Os diagnósticos do seletor são GBM-13; os diagnósticos de engenharia que não mudam com o seletor são identificados como históricos.','']
    supplemental=['summary.json','corpus_summary.json','portfolio_expansion_vs_attrition.csv','portfolio_contribution_ci.csv',
          'portfolio_group_contribution.csv','extras/selector_vs_each_option.csv','extras/option_rank_correlation.csv',
          'noise/q2_result.csv','noise/informative_by_epsilon.csv','noisy_oracle.csv','methods_nemenyi.json',
          'extras/topk.csv','extras/importance_columns.csv','extras/winner_map_bins.csv','holdout/diagnostic_summary.json',
          'saturation/summary.csv','saturation/config.json','top1_definition_audit.csv','lodo_12_13_comparison.csv','lodo_common_datasets_comparison.json','selector_replay.json','holdout/alignment.json','holdout/run_config.json']
    for rel in supplemental:
        p=root/rel
        if not p.exists() and rel.startswith('lodo_'):continue
        lines += [f'### {rel}','',f'Fonte: [{rel}]({rel}).','']
        if p.suffix=='.csv':lines += [mdtable(pd.read_csv(p)),'']
        else:lines += [mdtable(pd.DataFrame(flatten(json.loads(p.read_text())),columns=['métrica','gbm13'])),'']
    lines += ['## Inventário integral das afirmações numéricas antigas','',
      'Este inventário transcreve todas as linhas narrativas com algarismos nos capítulos 4 e 5, resumo, abstract e introdução. Citações bibliográficas e referências de seção não são resultados. O Top-1 histórico usava desempate pela ordem do registro; a taxa GBM-13 principal exige vencedora única, e top1_definition_audit.csv permite distinguir as definições. O vínculo aponta o conjunto de dados GBM-13 usado para reescrever a afirmação inteira; as correspondências diretas já estão no quadro central e nos blocos de tabelas. Não fazer substituições globais: por exemplo, “12” pode ser número de opções, ID bibliográfico ou quantidade de métodos históricos.','']
    sections={
       'Heterogeneidade':['summary.json','noise/q2_result.csv','win_share_by_epsilon.csv'],
       'Variação com o orçamento':['win_share_by_epsilon.csv','regret_by_epsilon.csv'],
       'Magnitude da escolha':['summary.json','transformations/summary.csv'],
       'Desempenho de cada opção':['option_performance.csv','saturation/summary.csv'],
       'Interdependência':['extras/summary.json','noise/q2_result.csv'],
       'Composição do portfólio':['portfolio_contribution_ci.csv','portfolio_group_contribution.csv','extras/option_rank_correlation.csv'],
       'Seletor sem custo':['summary.json'], 'Ganho por orçamento':['regret_by_epsilon.csv','strategies/by_epsilon.csv'],
       'Comparação entre métodos':['methods_nemenyi.json'], 'Comparação com cada opção':['extras/selector_vs_each_option.csv'],
       'Ganho por perfil':['extras/gain_by_stratum.csv'], 'Estratégia do seletor':['strategies/summary.json'],
       'Comparação entre modos':['modes/modes.csv','modes/pairwise.csv','modes/equal_total.csv'],
       'Importância':['modes/importance_groups.csv','extras/importance_columns.csv'],
       'Contribuição das grandezas':['extras/shapley_primitives.csv','extras/summary.json'],
       'Regime de ruído':['extras/winner_map_bins.csv'], 'Recomendação em lista':['extras/topk.csv'],
       'Metamodelos alternativos':['extras/metamodels.csv','extras/summary.json'],
       'Curva de aprendizado':['learning_curve/summary.csv'],
       'Relevância prática':['summary.json','noisy_oracle.csv'],
       'Escolha gratuita':['extras/idealized_private_search.csv'],
       'Análises de sensibilidade':['size/sensitivity_by_size.csv','summary.json','corpus_summary.json'],
       'Transformação numérica':['transformations/summary.csv','transformations/provenance.json'],
       'Avaliação em famílias novas':['holdout/summary.json','holdout/comparisons.csv'],
       'Comparação primária':['holdout/summary.json','holdout/diagnostic_summary.json'],
       'Todas as variantes':['holdout/comparisons.csv'], 'Dentro e fora':['holdout/summary.json'],
       'Por que a opção':['holdout/softmax_diagnostic.csv','holdout/diagnostic_summary.json'],
       'Leitura do resultado':['holdout/summary.json'], 'Uma regra':['rules/summary.csv','rules/frozen_rules.json','rules/holdout_exploratory.json'],
    }
    files=[args.dissertation/'capitulos/4-resultados.tex',args.dissertation/'capitulos/5-conclusao.tex',
           args.dissertation/'capitulos/1-introducao.tex',args.dissertation/'pretextual/resumo.tex',args.dissertation/'pretextual/abstract.tex']
    inventory=[]
    for file in files:
        section='Escopo / síntese';source=['numeric_crosswalk.csv','summary.json','holdout/summary.json'];in_table=False
        for i,line in enumerate(file.read_text().splitlines(),1):
            if re.search(r'\\begin\{(?:table|longtable)\}',line):in_table=True
            if re.search(r'\\end\{(?:table|longtable)\}',line):in_table=False;continue
            match=re.search(r'\\(?:sub)?section\{([^}]+)',line)
            if match:
                section=match[1]
                for prefix,paths in sections.items():
                    if section.startswith(prefix):source=paths;break
            if in_table or line.lstrip().startswith(('%','\\label','\\includegraphics','\\caption','\\begin','\\end','\\section','\\subsection','\\chapter')):continue
            no_cite=re.sub(r'\\(?:cite|citeonline|ref|eqref)\{[^}]*\}','',line)
            if not re.search(r'\d',no_cite):continue
            tokens=re.findall(r'[-−]?(?:\d+(?:\{,\}|[.,])\d+|\d+)',no_cite)
            if not tokens:continue
            status='GBM-13 recalculado; consultar métricas e tabelas vinculadas'
            if '80 conjuntos' in line or '5000' in line and '39\\%' in line:
                status='NÃO REEXECUTÁVEL: diagnóstico de 80 datasets sem script/manifesto no repositório; manter apenas como histórico RF-12 ou retirar'
            inventory.append(dict(arquivo=str(file),linha=i,secao=section,valores_rf12='; '.join(tokens),
               afirmacao_rf12=line.strip(),fontes_gbm13='; '.join(source),status=status))
            lines += [f'**{file.name}:{i} — {section}.**','',f'RF-12: {line.strip()}','',
                      'GBM-13: '+', '.join(f'[{p}]({p})' for p in source)+f'. {status}.','']
    pd.DataFrame(inventory).to_csv(root/'narrative_number_inventory.csv',index=False)
    lines += ['## Limites e decisões para a reescrita','',
      '- O diagnóstico antigo em 80 datasets (39% e perda de 4,7 p.p.) não tem script/manifesto neste repositório, segundo `scripts/defense/README.md`, e não foi apresentado como número GBM-13. A correlação foi reconstruída nos 580 datasets só numéricos, com limiar |tanh(x)|>0,995 explicitamente declarado; como o limiar original não foi preservado, não se afirma equivalência exata ao diagnóstico RF-12. As análises novas de saturação e transformações são reproduzíveis.','',
      '- A fração de impureza da RF não existe no HistGradientBoosting; retirar essa coluna e a interpretação de que 90% da impureza vem das classes. Permutação e Shapley foram recalculados com fallback ativo.','',
      '- A softmax e seus ajustes diagnósticos de regularização não mudaram. Foram reutilizados os valores limpos históricos e atualizadas as comparações com b8u, sem inventar uma nova reexecução desses mesmos ajustes.','',
      '- O LODO de 12 opções em `results/experiments/lodo_12` não foi interrompido. Seu estado está no manifesto final; a comparação do portfólio de 12 opções no corpus comum foi recalculada independentemente.','',
      '- Tempos de recomendação foram medidos localmente sob carga simultânea de análises; não sustentam comparação de velocidade entre RF-12 e GBM-13.','',
      '## Reprodutibilidade','',
      'Execução completa/retomada: `PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=$PWD/.cache/matplotlib XDG_CACHE_HOME=$PWD/.cache .venv/bin/python scripts/regenerate_v5_dissertation.py`. A extensão do holdout verifica 5 GiB livres antes e depois de cada dataset; todos os novos arquivos ficam em results/dissertation-portfolio13 e figures/GBM-13. Nenhum comando git é necessário.','']
    (root/'NUMBERS_gbm13.md').write_text('\n'.join(lines))
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in ['artifact_manifest.json']]
    manifest=dict(runtime=runtime_fingerprint(),tables=len(coverage),narrative_claims=len(inventory),
        status='completo, exceto diagnósticos históricos explicitamente não reproduzíveis',
        lodo12_finished=Path('results/experiments/lodo_12/lodo_meta.json').exists(),
        artifacts={str(p):dict(bytes=p.stat().st_size,sha256=file_sha256(p)) for p in files},
        figures={str(p):file_sha256(p) for p in Path('figures/GBM-13').glob('*.pdf')})
    dump(root/'artifact_manifest.json',manifest)
    print(json.dumps(dict(tables=len(coverage),narrative_claims=len(inventory),report=str(root/'NUMBERS_gbm13.md')),indent=2))

if __name__=='__main__':main()
