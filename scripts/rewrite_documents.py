
# A versão oficial revisada não pode ser sobrescrita pelo gerador histórico.
from pathlib import Path as _OfficialPath
if (_OfficialPath(__file__).resolve().parents[2] / "mestrado-papers" / "latex-dissertacao" / "VERSAO_OFICIAL.md").exists():
    raise SystemExit("Gerador histórico desativado: use mestrado-papers/latex-dissertacao/correcao-oficial-2026-11-13/compilar.py; fontes revisados são autoritativos.")

"""Render source-backed final tables and rewrite only the three authorized GBM-13 folders."""
from pathlib import Path
import json,re,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];PAPERS=ROOT.parent/'mestrado-papers';OUT=ROOT/'results/dissertation';OLD=ROOT/'results/dissertation-portfolio13';SNAP=OUT/'source_portfolio13';D=PAPERS/'latex-dissertacao'
CODE={'categorical_naive_bayes__laplace_histograms':'NBC-Lap','categorical_naive_bayes__laplace_histograms_b8u':'NBC-b8u','gnb__laplace_equal':'NBG-Lap','gnb__analytic_gaussian_equal':'NBG-GA','gnb__gaussian_zcdp_equal':'NBG-zC','gnb__staircase_vector_equal':'NBG-St','softmax__analytic_gaussian':'SM-GA','softmax__dp_sgd':'SGD-25','softmax__dp_sgd_steps100':'SGD-100','softmax__dp_sgd_clip2_reg003':'H27','softmax__dp_sgd_clip2_lamrule':'SGD-regra','softmax__dp_sgd_clip2_lam0003':'SGD-0003','random_forest__laplace_parallel_shards_25':'AA-Lap','random_forest__analytic_gaussian_parallel_shards':'AA-GA','random_forest__exponential_majority_parallel_shards':'AA-Exp'}
MODE={'public_schema_only':'Só schema','public_plus_dp':'Público + calibração','dp_research_rich':'Descritores DP','research_privileged':'Privilegiado (sem DP)','research_rich':'Rico (sem DP)'}
GROUP={'schema':'Schema','public':'Público','all_calibration':'Toda calibração','all_meta_features':'Todos os descritores'}
TABLES={}
def read(rel,old=False):return pd.read_csv((OLD if old else OUT)/rel)
def js(rel):return json.loads((OUT/rel).read_text())
def tex(x):
    mapping={'&':r'\&','%':r'\%','_':r'\_','<':r'$<$','>':r'$>$','#':r'\#','$':r'\$','{':r'\{','}':r'\}','\\':r'\textbackslash{}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','≥':r'$\geq$','≤':r'$\leq$','–':'--','−':'-','ε':r'$\varepsilon$'}
    return ''.join(mapping.get(c,c) for c in str(x))

def f(x,n=4):return f'{float(x):.{n}f}'.replace('.',',')
def pp(x,n=2):return f(float(x)*100,n)
def ci(r,point='mean',lo='ci_low',hi='ci_high',scale=100,n=2):return f(r[point]*scale,n)+' ['+f(r[lo]*scale,n)+'; '+f(r[hi]*scale,n)+']'
def table(label,caption,headers,rows,source,align=None):
    align=align or 'l'+'r'*(len(headers)-1)
    body='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    tab=r'\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(headers)+r' \\\midrule'+'\n'+body+'\n\\bottomrule\n\\end{tabular}'
    if len(headers)>=4:tab=r'\resizebox{\textwidth}{!}{%'+ '\n'+tab+'}\n'
    content='\n'.join([r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.12}',tab,r'\par\smallskip{\footnotesize Fonte: elaboração do autor; '+tex(source)+'.}',r'\end{table}',''])
    TABLES[label]=(content,source);return r'\input{tables/'+label.replace('tab:','')+'}\n'
def figure(name,caption,label,width=r'\textwidth'):
    return '\n'.join([r'\begin{figure}[htbp]',r'\centering',r'\includegraphics[width='+width+']{figures/'+name+'.pdf}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\par\smallskip{\footnotesize Fonte: elaboração do autor a partir dos resultados final.}',r'\end{figure}',''])

def results():
    s=js('summary.json');h=s['h1'];base=s['best_fixed_regret'];pr=s['practical'];perf=read('option_performance.csv');t=[]
    t.append(r'''\chapter{Resultados e Discussão}
\label{chap:resultados}

A especificação principal é a final: 15 opções, GBM residual e retorno à b8u quando a vantagem prevista não supera $\tau=0{,}02$. O desenvolvimento contém 973 datasets, 667 grupos semânticos, 7.784 condições dataset--$\varepsilon$ e 350.280 avaliações. As médias principais dão peso igual à família, depois ao dataset e orçamento. As frequências de vitórias e de perdas por condição usam peso igual por dataset. Escolha de portfólio, fallback e limiar foi exploratória, após o protocolo histórico e com o holdout antigo já aberto.

O ganho previamente resumido como 1,42 p.p. [1,13; 1,74] foi auditado: a leitura padrão seguida de \texttt{groupby(fam)} descarta o grupo sem nome. Os ajustes LODO têm 667 grupos, mas essa média usa 666. Incluindo explicitamente os IDs 41526 e 45939 como grupo compartilhado, esta regeneração obtém 1,426 p.p. [1,124; 1,745], com 20.000 reamostragens. A grade exploratória de fallback registrava [1,13; 1,76], com outra reamostragem. Os registros de origem são preservados; as tabelas seguintes usam todos os 667 grupos. Essa diferença pequena não muda a interpretação.

\section{Heterogeneidade das opções e H1}
\label{sec:resultado-h1}
\label{sec:vitorias-eps}
\label{sec:magnitude}
\label{sec:opcoes}
''')
    shares=read('win_share_by_epsilon.csv').rename(columns={'Unnamed: 0':'pipeline'})
    lead=shares.loc[shares.Total.idxmax()]
    t.append(f"Nenhuma opção domina: {CODE[lead.pipeline]} reúne {pp(lead.Total,1)}\\% das vitórias com empates divididos. Há empates no topo em {pp(h['tied_fraction'],1)}\\% das condições; todas as 15 opções empatam em {h['all_tied']} condições. Somente {h['constant_unique_datasets']} datasets mantêm uma vencedora única nos oito orçamentos; a mediana é de {int(h['median_distinct_winners'])} vencedoras distintas. H1 é sustentada como heterogeneidade empírica, sem que isso implique previsibilidade da vencedora.\n\n")
    t.append(f"O Friedman por família dá $\\chi^2={f(h['friedman_family_chi2'],2)}$ e $W={f(h['kendall_W_family'],4)}$. A amplitude melhor--pior tem mediana de {pp(h['spread_median'])} p.p. e P90 de {pp(h['spread_p90'])} p.p.; supera dez pontos em {pp(h['spread_gt_10pp'],1)}\\% das condições. A diferença mediana entre as duas melhores é {pp(h['top_two_gap_median'])} p.p. A amplitude total inclui opções fracas e não é o ganho disponível sobre a b8u.\n\n")
    t.append(figure('fig1_winner_dispersion_12opcoes','Vitórias nas 7.784 condições do portfólio final de 15 opções. Empates dividem a vitória; peso igual por dataset.','fig:vencedores'))
    t.append(table('tab:vitorias-eps','Vitórias por orçamento (\\%); 15 opções, empates divididos.', ['Opção',*['0,1','0,25','0,5','1','2','4','8','16'],'Total'],[[CODE[r.iloc[0]],*[pp(x,1) for x in r.iloc[1:]]] for _,r in shares.iterrows()],'win_share_by_epsilon.csv'))
    t.append('A regra coincide com H27 quando escolhe lambda 0,03 e com a constante quando escolhe 0,0003; parte dos empates é estrutural, além da instabilidade entre repetições.\n\n')
    t.append('NBC designa o NB categórico; NBG, o Gaussiano; AA, árvores aleatórias; SM, softmax por perturbação de saída; SGD, DP-SGD. H27 é a variante histórica com 100 passos, clip 2 e $\\lambda=0{,}03$; SGD-regra e SGD-0003 alteram apenas a regularização.\n\n')
    t.append(table('tab:resultados-opcoes','BA privada e regret das 15 opções; médias e IC95\\% por família.',['Opção','BA privada (\\%) [IC95\\%]','Regret [IC95\\%]'],[[CODE[r.pipeline],ci(r,'private_ba_mean','private_ba_ci_low','private_ba_ci_high'),ci(r,'portfolio_regret_mean','portfolio_regret_ci_low','portfolio_regret_ci_high',scale=1,n=4)] for _,r in perf.iterrows()],'option_performance.csv'))
    t.append(r'''A b8u é a melhor fixa do benchmark com peso igual por dataset: BA 44,7675\%, frente a 44,7637\% da constante $\lambda=0{,}0003$, diferença de apenas 0,0038 p.p. Sob peso igual por família, a constante passa a ser a melhor fixa: BA 48,6480\%, frente a 48,3701\% da b8u. Essa ordenação retrospectiva não demonstra superioridade estatística. A referência primária permanece a b8u congelada; a melhor fixa no estimando semântico é reportada separadamente.

\subsection{Regularização DP-SGD: E6/E7 e benchmark completo}
\label{sec:desenvolvimento-sgd}

No benchmark completo, a regra pública e a constante ganham, respectivamente, 1,582 e 1,739 p.p. sobre H27 na média dos datasets e dos oito orçamentos; ambas melhoram a média em cada orçamento. Os ICs por orçamento são pontuais e o corpus se sobrepõe ao desenvolvimento E6/E7, sem confirmação independente da escolha. A regra não demonstrou superioridade sobre a constante.

E6 usou 55 datasets e exclusão de um dataset no ajuste da regra: ganho de 1,83 p.p. [0,66; 3,21] sobre H27. A constante 0,0003 teve ganho descritivo de 2,04 p.p. [0,88; 3,41] na mesma amostra. Em E7, sob a regra LODO e clip 2, elevar de 100 para 400 passos reduziu o gap clean--sem ruído de 3,34 para 0,04 p.p., mas o ganho privado sobre H27 caiu de 1,23 para 0,37 p.p. Sob orçamento total fixo, mais passos elevam o ruído por passo. Isso motivou manter 100 passos e incluir ambas as políticas de regularização, com o mesmo ledger de H27. E6/E7 são exploratórios, com ICs pontuais e seleção posterior na mesma amostra.

\subsection{Composição e instabilidade do portfólio}
\label{sec:interdependencia}
\label{sec:portfolio}
''')
    extra=js('extras/summary.json');noise=read('noise/q2_result.csv').set_index('Unnamed: 0')
    t.append(f"A DP muda a família de modelo vencedora em {pp(extra['dp_changes_best_model']['overall'],1)}\\% das condições. O piso de instabilidade do modelo limpo entre repetições é {pp(float(noise.loc['floor_clean_model_rep_vs_rep','todos']),1)}\\%, enquanto a opção privada vencedora muda em {pp(float(noise.loc['floor_private_option_rep_vs_rep','todos']),1)}\\%. Há sinal de interação mecanismo--modelo e ruído relevante no alvo; três repetições não identificam separadamente todas as fontes de variância.\n\n")
    t.append(figure('fig7_portfolio_redundancy','Contribuição marginal ao oráculo e redundância de postos no portfólio de 15 opções. Contribuições individuais não se somam.','fig:redundancia'))
    t.append(r'''\section{Seletor GBM-15 e H2a no desenvolvimento}
\label{sec:resultado-h2}
\label{sec:resultado-h2a}
\label{sec:seletor}
''')
    t.append(f"O LODO ganha {ci(base['gain']['mean'],n=3)} p.p. sobre b8u, com teste pareado de permutação de sinais $p={f(base['sign_flip_p'],5)}$. A mediana, P90 e CVaR$_{{90}}$ do regret caem de {f(base['reference']['median'])}, {f(base['reference']['q90'])} e {f(base['reference']['cvar90'])} para {f(base['selector']['median'])}, {f(base['selector']['q90'])} e {f(base['selector']['cvar90'])}. A evidência interna favorece H2a, condicionada à configuração escolhida no desenvolvimento; não é um IC pós-seleção.\n\n")
    t.append(f"O seletor tem BA média por família de {pp(s['selector_ba']['mean'])}\\%. A perda supera um ponto percentual em {pp(pr['conditions_worse_over_1pp'],2)}\\% das condições; o CVaR$_{{90}}$ da perda assinada por condição é {pp(pr['cvar90_condition_loss'],3)} p.p. Essas caudas da perda frente à b8u são distintas das caudas da distribuição do regret. O Top-1 de vencedora única é {pp(s['top1_strict'],2)}\\%, frente a {pp(s['fixed_top1_strict'],2)}\\% da b8u.\n\n")
    empirical=s['empirical_fixed_regret']
    t.append('Contra a constante, melhor fixa retrospectiva com peso por família, o ganho é '+ci(empirical['gain']['mean'],n=3)+' p.p.; mediana, P90 e CVaR90 do regret também diminuem. A confirmação externa continua sendo contra b8u, como congelado no protocolo.\n\n')
    methods=read('method_summary.csv')
    t.append(table('tab:resultados-seletor','Métodos e estratégias final; regret por família. Ganho sobre b8u em p.p.',['Método','Validação','Média','Mediana','P90','CVaR90','Ganho [IC95\\%]'],[[tex(r.method),tex(r.validation),f(r.regret),f(r['median']),f(r.q90),f(r.cvar90),ci(r,'gain')] for _,r in methods.iterrows()],'method_summary.csv'))
    comparison=read('lodo_portfolio_comparison.csv');comparison=comparison[comparison.convention=='all_667_groups']
    t.append(table('tab:comparacao-portfolios','LODO residual com fallback; comparação sobre referência b8u comum, incluindo o grupo sem nome.',['Origem','Grupos','Ganho vs b8u (p.p.) [IC95\\%]','Ganho vs NB antigo (p.p.)','Perda maior que 1 p.p. (\\%)'],[[tex(r.portfolio),str(int(r.families)),ci(r,'vs_b8u'),pp(r.vs_old_nb),pp(r.loss_gt_1pp)] for _,r in comparison.iterrows()],'lodo_portfolio_comparison.csv'))
    t.append('O ganho da final frente ao NB antigo é '+pp(s['old_nb_fixed_regret']['gain']['mean']['mean'],3)+' p.p. na média dos 667 grupos (o resumo inicial informava 2,28 p.p.). As versões 13, 13b, 14 e 15 são comparadas contra a mesma b8u; a versão final usa fallback b8u e tau 0,02. As demais preservam seus picks de origem e escolhas de fallback/limiar. A tabela descreve políticas diferentes, sem atribuir a diferença apenas ao número de opções.\n\n')
    t.append(r'''\subsection{Resultados por orçamento e métodos executáveis}
\label{sec:regret-eps}
\label{sec:estrategia}
''')
    eps=read('regret_by_epsilon.csv');t.append(table('tab:regret-eps','LODO por orçamento; regret médio e Top-1 estrito por condição.',['$\\varepsilon$','Seletor','b8u','Ganho (p.p.) [IC95\\%]','Top-1 (\\%)'],[[f(r.epsilon,2),f(r.selector_regret_mean),f(r.best_fixed_regret_mean),ci(r,'gain_mean','gain_ci_low','gain_ci_high'),pp(r.top1)] for _,r in eps.iterrows()],'regret_by_epsilon.csv'))
    t.append(figure('fig2_regret_and_top1_curves_12opcoes','Curvas LODO GBM-15: regret com peso por família; Top-1 exige vencedora única entre 15 opções.','fig:regret-curvas'))
    t.append('O seletor compartilhado e os modelos separados por orçamento são reestimados nas mesmas cinco partições semânticas; seus resultados constam da Tabela~\\ref{tab:resultados-seletor}. O diagrama de métodos executáveis usa essas cinco partições e não o LODO principal.\n\n')
    t.append(figure('fig3_critical_difference_nemenyi_12opcoes','Métodos executáveis nas mesmas cinco partições semânticas; diagrama de Nemenyi descritivo.','fig:cd'))
    t.append(figure('fig4_regret_distribution_and_headroom_12opcoes','Ganho sobre b8u e distribuições de regret em cinco partições; médias por família.','fig:regret-distribuicao'))
    t.append(r'''\subsection{Cada opção fixa e perfis do corpus}
\label{sec:vs-opcoes}
\label{sec:estratos}
''')
    t.append(figure('fig5_selector_vs_each_option','Ganho LODO sobre cada opção fixa, com IC95\\% e contagens por família. Testes pareados recebem Holm.','fig:vs-opcoes',r'0.98\textwidth'))
    strata=read('extras/gain_by_stratum.csv')
    # Column layout is taken from the generated source, without inventing new estimates.
    headers=['Dimensão','Estrato','Datasets','Famílias','Ganho (p.p.) [IC95\\%]'];rows=[[tex(r.dimension),tex(r.stratum),str(int(r.datasets)),str(int(r.families)),ci(r)] for _,r in strata.iterrows()]
    t.append(table('tab:estratos','Ganho LODO por perfil em p.p.; estratos exploratórios.',headers,rows,'extras/gain_by_stratum.csv'))
    t.append(figure('fig6_rank_heatmap_by_stratum','Postos médios por estrato no portfólio final; estratos exploratórios.','fig:estratos'))
    t.append(r'''\section{Descritores públicos e ablações}
\label{sec:modos}

Todas as ablações seguintes reestimam o GBM residual com fallback b8u e $\tau=0{,}02$, nas mesmas cinco partições por família. Os modos que leem treino sem DP são tetos diagnósticos; o modo com descritores DP paga $\varepsilon_{\mathrm{meta}}=1$. Nenhum ganho desses modos pode ser confundido com uma recomendação gratuita.
''')
    modes=read('modes/modes.csv');t.append(table('tab:modos','Modos de descritores; regret e ganho sobre b8u (p.p.), cinco partições.',['Modo','$\\varepsilon_{meta}$','Média','P90','CVaR90','Ganho [IC95\\%]'],[[MODE[r['mode']],'-' if pd.isna(r.metadata_epsilon) else f(r.metadata_epsilon,0),f(r.regret_mean),f(r.regret_q90),f(r.regret_cvar90),ci(r,'gain_vs_best_fixed','gain_ci95_low','gain_ci95_high')] for _,r in modes.iterrows()],'modes/modes.csv'))
    equal=read('modes/equal_total.csv');eq=equal[equal.exact_budget_match.astype(str).str.lower().eq('true')]
    if eq.empty:eq=equal
    t.append(table('tab:orcamento-total','Comparações no orçamento total; ganho do modo que paga descritores em p.p.',['$\\varepsilon_m$','$\\varepsilon_{total}$','Referência','$\\varepsilon_{ref}$','Ganho [IC95\\%]'],[[f(r.model_epsilon,2),f(r.total_epsilon,2),tex(MODE.get(r.reference,r.reference)),f(r.reference_epsilon,2),ci({'mean':r.gain,'ci_low':float(str(r.ci95).strip('[]').split(',')[0]),'ci_high':float(str(r.ci95).strip('[]').split(',')[1])})] for _,r in eq.iterrows()],'modes/equal_total.csv'))
    t.append('A tabela mostra os encontros exatos da grade; o CSV inclui também referências com orçamento menor. Quando não há igualdade, apenas a derrota do modo que gasta é conclusiva nessa comparação. Os ICs são marginais; os testes entre modos recebem Holm no conjunto próprio.\n\n')
    importance=read('modes/importance_groups.csv');imp=importance[(importance['mode']=='public_plus_dp') & importance.group.isin(['schema','public','all_calibration','all_meta_features'])]
    if imp.empty:imp=importance[importance['mode']=='public_plus_dp']
    t.append(table('tab:importancia','Importância por permutação em public\\_plus\\_dp; aumento do regret em p.p.',['Grupo','Descritores','Aumento [IC95\\%]'],[[tex(GROUP.get(r.group,r.group)),str(int(r.n_features)),ci(r,'regret_increase','ci95_low','ci95_high')] for _,r in imp.iterrows()],'modes/importance_groups.csv'))
    t.append(r'''\subsection{Grandezas primitivas e regime de ruído}
\label{sec:importancia}
\label{sec:shapley}
\label{sec:regime}

A importância por permutação mede a mudança no modelo ajustado. A decomposição de Shapley troca tamanho, orçamento, classes e estrutura dos atributos e recalcula todas as features derivadas, inclusive a regularização pública. A leitura é preditiva; não identifica efeitos causais de cada grandeza. Parcelas podem ser negativas ou exceder cem por cento quando há compensação entre contribuições.
''')
    shap=read('extras/shapley_primitives.csv');names={'attributes':'Estrutura dos atributos','n':'Instâncias','epsilon':'Orçamento','classes':'Classes'}
    t.append(table('tab:shapley','Shapley do aumento de regret, cinco partições, três permutações por partição.',['Grandeza','$\phi$ (p.p.) [IC95\\%]','Parcela (\\%)','Escolhas alteradas (\\%)'],[[names[r.primitive],ci(r),pp(r.share),pp(r.changed)] for _,r in shap.iterrows()],'extras/shapley_primitives.csv'))
    top_shap=shap.set_index('primitive')
    t.append('A estrutura dos atributos responde por '+pp(top_shap.loc['attributes','share'],1)+'\\% do efeito conjunto e o tamanho por '+pp(top_shap.loc['n','share'],1)+'\\%. São parcelas do efeito de perturbação do preditor, sem interpretação causal ou garantia de validade em novas famílias.\n\n')
    t.append(figure('fig10_metafeature_importance','Shapley por grandeza pública e importância de colunas; seletor GBM-15 em cinco partições.','fig:importancia'))
    t.append(figure('fig8_winner_map_n_eps_p','Mapa das famílias de modelo vencedoras por regime público; empates divididos, 15 opções.','fig:regime'))
    t.append(figure('fig12_metafeature_space_tsne','Projeção t-SNE dos metadados públicos; cores pela família modal. Visualização exploratória.','fig:tsne',r'0.85\textwidth'))
    t.append(r'''\section{Listas, metamodelos e tamanho do treino}
\label{sec:topk}
\label{sec:metamodelos}

Listas são diagnósticos da melhor BA observada entre as primeiras $k$ recomendações. Avaliar várias opções em uma base privada exige composição; a garantia de custo zero aplica-se à recomendação de uma opção. O empate com a vencedora conta como acerto na lista, diferentemente do Top-1 estrito do resultado principal.
''')
    t.append(figure('fig9_topk_recommendation','Listas final em cinco partições; regret@k e cobertura da vencedora, com empates aceitos no acerto.','fig:topk'))
    meta=read('extras/metamodels.csv');t.append(table('tab:metamodelos','Metamodelos reestimados com 15 opções e cinco partições semânticas.',['Modelo','Regret','Ganho sobre b8u (p.p.) [IC95\\%]'],[[tex(str(r.metamodel).replace('(GBM-13)','(GBM-15)')),f(r.regret),ci(r)] for _,r in meta.iterrows()],'extras/metamodels.csv'))
    t.append('RF-12 nesta tabela significa sua arquitetura reestimada nas 15 opções; não é o modelo congelado da confirmação histórica. O k-NN não aceita os pesos de treino por família, limitação da comparação. As alternativas usam BA absoluta; o seletor principal usa o alvo residual e fallback.\n\n')
    t.append(figure('fig11_metamodel_cd','Metamodelos nas mesmas cinco partições; Nemenyi descritivo e contrastes pareados/Holm no artefato numérico.','fig:metamodelos'))
    learning=read('learning_curve/summary.csv');t.append(table('tab:curva-aprendizado','Curva de aprendizado final: subamostras de famílias de treino, ganho sobre b8u em p.p.',['Modelo','Fração','Famílias treino','Datasets treino','Ganho [IC95\\%]'],[[tex(str(r.model).replace('gbm13','final')),pp(r.fraction,0)+r'\%',f(r.train_families,1),f(r.train_datasets,1),ci(r,'gain')] for _,r in learning.iterrows()],'learning_curve/summary.csv'))
    t.append(r'''\section{Relevância prática e sensibilidades}
\label{sec:relevancia}
\label{sec:sensibilidade}
\label{sec:busca-idealizada}

O ganho médio não é uma promessa individual. A b8u continua adequada em muitas condições; o retorno conservador limita, mas não elimina, perdas. O oráculo usa o máximo de estimativas ruidosas no teste e fornece um teto otimista. Os diagnósticos de troca de repetições foram refeitos para 15 opções e estão no registro numérico, distinguindo o máximo das três repetições da escolha avaliada em repetições não usadas na seleção.
''')
    search=read('extras/idealized_private_search.csv');t.append(table('tab:busca-idealizada','Busca privada idealizada; seletor em $\\varepsilon$ contra referência com orçamento fracionado. Ganho em p.p.',['$\\varepsilon$','Divisor','Referência','Seletor regret','Ref. regret','Ganho [IC95\\%]'],[[f(r.epsilon,2),str(int(r.factor)),tex({'oracle':'Oráculo','best_fixed':'b8u'}.get(r.reference,r.reference)),f(r.selector),f(r.reference_regret),ci(r)] for _,r in search.iterrows()],'extras/idealized_private_search.csv'))
    size=read('size/sensitivity_by_size.csv');t.append(table('tab:tamanho','Sensibilidade LODO por tamanho; bootstrap por família dentro da faixa.',['Faixa de $n$','Datasets','Famílias','Ganho (p.p.) [IC95\\%]'],[[tex(r.size_band),str(int(r.n_datasets)),str(int(r.families)),ci(r)] for _,r in size.iterrows()],'size/sensitivity_by_size.csv'))
    t.append(r'''\subsection{Diagnósticos históricos não transportados}
\label{sec:transformacao}
\label{sec:sensibilidade-transformacao}

A comparação de transformações tem 199 datasets comuns e 13 opções GBM-13. Não há scores das duas novas variantes sob as transformações alternativas. A Tabela~\ref{tab:transformacao} é preservada como diagnóstico histórico GBM-13 e não estima o desempenho final. Limites observados nessa simulação não equivalem a limites públicos declarados numa implantação. Os ajustes limpos auxiliares das novas variantes também não estão completos no holdout antigo; nenhuma célula ausente foi imputada.
''')
    oldtext=(SNAP/'latex-dissertacao/capitulos/4-resultados.tex').read_text()
    for label in ['tab:transformacao','tab:colapso-softmax']:
        match=next(m for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',oldtext,re.S) if '\\label{'+label+'}' in m.group())
        raw=match.group().replace(r'\caption{',r'\caption{Diagnóstico histórico GBM-13 (13 opções), sem atualização final: ',1)
        TABLES[label]=(raw,'results/dissertation-portfolio13 (histórico)')
        if label=='tab:transformacao':t.append(r'\input{tables/transformacao}'+'\n')
    t.append(r'''\section{Avaliação externa: protocolo congelado final}
\label{sec:holdout}
\label{sec:resultado-h2b}

O protocolo confirmatório final foi congelado antes de selecionar o novo holdout, incluindo modelo treinado no desenvolvimento, 15 opções, b8u como comparador primário e fallback, e $\tau=0{,}02$. Foram esgotados 869 candidatos da ordem sorteada no catálogo ativo: só duas famílias elegíveis e disjuntas de todos os manifestos prévios permaneceram. O alvo de 40 famílias não pôde ser alcançado; o protocolo autorizava usar todas as elegíveis. São \texttt{marketing-income} (8.993 linhas, nove classes) e \texttt{NovAsia\_Property\_Listing\_Dedup\_Synthetic} (4.800 linhas, duas classes). A auditoria registra 720 avaliações e ledgers aprovados, sem substituição baseada em desempenho.

\textbf{O critério confirmatório não foi atingido.} O ganho primário final sobre b8u é 1,080 p.p. [\mbox{$-0,278$}; 2,438]; o limite inferior não é maior que zero. Com duas famílias, não há poder adequado nem base para generalização ampla. O resultado não confirma transferência e tampouco demonstra ausência de efeito. O nome H2a final no protocolo identifica esse teste externo frente à b8u, distinto do rótulo histórico H2a para o LODO.

Nos secundários do novo holdout, a correção de Holm cobre exatamente quatro contrastes; todos têm $p_{Holm}=1$. Seus ICs são marginais. A perda superior a um ponto ocorre em 6,25\% das condições [0,00; 12,50]\%, e o CVaR$_{90}$ da perda é 2,694 p.p. [0,000; 2,694]. Com duas unidades de reamostragem, esses intervalos são frágeis.

\subsection{Holdout antigo: exclusivamente exploratório}

No holdout de 41 famílias já aberto, a final ganha 0,677 p.p. [0,020; 1,618] sobre b8u. A perda superior a um ponto ocorre em 0,30\% das condições [0,00; 0,91]\%; o CVaR$_{90}$ da perda é 0,079 p.p. [0,000; 0,227]. A extensão acrescentou as duas opções sem alterar os scores antigos. Esses resultados não constituem confirmação independente: configuração, fallback e limiar foram escolhidos com esse holdout já aberto.
''')
    hold=read('holdout_comparisons.csv');refs={'b8u':'b8u (primária)','H27':'H27','old_NB':'NB antigo','gbm13_selector':'Seletor GBM-13','rf12_selector':'RF-12'}
    t.append(table('tab:holdout','final vs referências: duas famílias novas e 41 antigas em blocos separados.',['População','Referência','Ganho (p.p.) [IC95\\%]','$p_{Holm}$'],[[('Nova (2)' if r.cohort=='confirmatory' else 'Antiga (41), expl.'),refs[r.reference],ci(r,'gain'),'-' if pd.isna(r.p_holm) else f(r.p_holm,4)] for _,r in hold.iterrows()],'holdout-openml/*/analysis.json',align='llrr'))
    t.append(figure('fig14_holdout_gains_vs_h27','Ganhos da final: confirmação com duas famílias novas (critério primário não atingido) e exploração em 41 famílias antigas. Cada linha nomeia sua referência; ICs marginais.','fig:holdout-ganhos'))
    t.append(figure('fig13_holdout_cd','Postos de métodos executáveis, separando duas famílias novas e 41 antigas. Diagramas descritivos; a amostra nova não sustenta inferência por postos.','fig:holdout-cd'))
    t.append(r'''\subsection{História confirmatória RF-12 e diagnóstico de H27}
\label{sec:colapso-softmax}

H2b conserva o significado histórico: o RF-12 congelado foi comparado à H27 pré-especificada, com ganho de 7,8 p.p. [3,9; 12,4] em 41 famílias. A ampliação de 30 para 41 ocorreu após observar as primeiras; o teste nas onze adicionais teve $p=0{,}086$. Esse registro não confirma a configuração final. H27 é uma opção de engenharia específica, cuja regularização fixa pode prejudicar a utilidade em bases grandes; o diagnóstico limpo histórico abaixo não avalia as novas regularizações.
''')
    t.append(r'\input{tables/colapso-softmax}'+'\n')
    t.append(r'''\section{Regra simples e discussão}
\label{sec:regra-limiar}

As regras de limiar foram reestimadas por exclusão de família semântica, com 15 opções disponíveis na regra de um limiar e b8u como retorno. A regra de dois limiares preserva a subgrade histórica de opções intermediárias. Os limiares não são escolhidos na família de teste; a concepção da grade permanece exploratória. A regra é uma referência interpretável, não uma confirmação independente da política selecionada.
''')
    rules=read('rules/summary.csv');t.append(table('tab:regra-limiar','Regras de limiar LODO: regret e ganho (p.p.) sobre b8u e sobre final.',['Regra','Regret','Ganho vs b8u [IC95\\%]','Ganho vs seletor [IC95\\%]'],[[tex(r.rule),f(r.regret),ci(r,'gain_mean','gain_ci_low','gain_ci_high'),ci(r,'vs_selector_mean','vs_selector_ci_low','vs_selector_ci_high')] for _,r in rules.iterrows()],'rules/summary.csv'))
    t.append(r'''\section{Limitações da interpretação}
\label{sec:discussao}

H1 descreve heterogeneidade no portfólio observado. H2a tem evidência interna favorável, mas a configuração foi escolhida no desenvolvimento; os ICs condicionam aos picks e não refazem toda a seleção em cada bootstrap. H2b RF-12 permanece história. O novo teste congelado final não atingiu seu critério e contém apenas duas famílias; a exploração nas 41 antigas não supre essa lacuna.

A amostra do desenvolvimento tem datasets pequenos, famílias relacionadas e exclusões por validação e custo. A taxonomia semântica não garante independência substantiva. A transformação fixa pode perder sinal; os escores limpos e a BA em teste não são liberações DP certificadas. As 15 opções não esgotam mecanismos possíveis. O ganho médio e o controle das caudas observadas não garantem segurança individual de cada recomendação. A prioridade de validação é obter famílias de outras fontes, mantendo o protocolo confirmatório final congelado.
''')
    return '\n'.join(t)

RESUMO=r'''A Privacidade Diferencial (DP) oferece garantia formal, mas não determina qual par mecanismo-modelo preserva mais utilidade para uma base tabular e um orçamento $\varepsilon$. Este trabalho apresenta o framework final, que recomenda usando somente schema e calibração públicos, sem custo adicional de privacidade sob um contrato público explícito. O portfólio tem 15 opções, incluindo NB categórico com oito intervalos e prior uniforme na decisão (b8u), DP-SGD com regularização pela regra pública $n\varepsilon$ e o controle $\lambda=0{,}0003$. As duas variantes DP-SGD preservam a contabilidade H27. O benchmark reúne 973 datasets, oito orçamentos e três repetições, totalizando 350.280 avaliações. O GBM residual prevê a vantagem sobre o NB original e retorna à b8u salvo vantagem prevista superior a $\tau=0{,}02$. No LODO de 667 grupos semânticos, o ganho sobre b8u é 1,426 p.p. (IC95\% [1,124; 1,745]), próximo dos 1,42 p.p. do resumo inicial; a auditoria explicita o grupo sem nome e o denominador. A perda excede um ponto em 2,4\% das condições, com CVaR$_{90}$ da perda de 1,03 p.p. A configuração e a escolha de fallback/limiar são exploratórias no desenvolvimento, posteriores ao protocolo histórico. O protocolo confirmatório final foi então congelado antes da seleção externa, mas o OpenML ofereceu somente duas famílias novas elegíveis: ganho de 1,08 p.p. [$-0,28$; 2,44], critério confirmatório não atingido e poder insuficiente. Nas 41 famílias antigas, já abertas, o ganho foi 0,68 p.p. [0,02; 1,62], com 0,3\% de perdas maiores que um ponto, exclusivamente exploratório. H1 e H2a recebem evidência interna; H2b RF-12 permanece história. A transferência da final exige confirmação em outros repositórios com o protocolo congelado.'''
ABSTRACT=r'''Differential Privacy (DP) provides a formal guarantee but does not determine which mechanism--model pair preserves the most utility for a tabular dataset and budget $\varepsilon$. This work presents framework final, which recommends using public schema and calibration features with no additional privacy cost under an explicit public-information contract. The portfolio contains 15 options, including categorical NB with eight bins and a uniform decision-time prior (b8u), DP-SGD with a public $n\varepsilon$ regularization rule, and a $\lambda=0.0003$ control. Both DP-SGD variants retain H27 accounting. The benchmark comprises 973 datasets, eight budgets and three repetitions, yielding 350,280 evaluations. A residual GBM predicts advantage over the original NB and falls back to b8u unless its predicted advantage exceeds $\tau=0.02$. Leave-one-family-out validation over 667 semantic groups yields a 1.426 percentage-point gain over b8u (95\% CI [1.124; 1.745]), close to the initial 1.42-point summary; an audit explains the unnamed group and denominator. Loss exceeds one point in 2.4\% of conditions; loss CVaR$_{90}$ is 1.03 points. Configuration and fallback/threshold selection are exploratory development choices made after the historical protocol. The final protocol was subsequently frozen before external selection, but OpenML supplied only two eligible new families: gain 1.08 points [$-0.28$; 2.44], failing the confirmatory criterion with insufficient power. On the previously opened 41-family holdout, gain is 0.68 points [0.02; 1.62], with 0.3\% of conditions losing over one point; this evidence is exclusively exploratory. H1 and H2a receive internal support; RF-12 H2b remains historical. Transfer of final requires confirmation using other repositories under the frozen protocol.'''
CONCLUSION=r'''\chapter{Conclusão}
\label{chap:conclusao}

\section{Síntese dos achados}
\label{sec:sintese-achados}

O framework final recomenda entre 15 pares mecanismo-modelo usando apenas informações públicas. Em 973 datasets, oito orçamentos e três repetições, foram realizadas 350.280 avaliações. O GBM residual usa b8u como retorno fixo e $\tau=0{,}02$. As novas regularizações DP-SGD melhoram H27 no benchmark completo sem alterar sua contabilidade: 1,582 p.p. para a regra e 1,739 p.p. para a constante, na média dos datasets. A regra não demonstrou vantagem sobre a constante.

H1 é sustentada pela ausência de dominância e pela heterogeneidade dos ganhos. H2a recebe evidência favorável no desenvolvimento: o LODO com todos os 667 grupos ganha 1,426 p.p. [1,124; 1,745] sobre b8u e reduz mediana, P90 e CVaR$_{90}$ do regret. A perda superior a um ponto ocorre em 2,4\% das condições, e o CVaR$_{90}$ da perda assinada é 1,03 p.p. A seleção de portfólio, fallback e limiar foi exploratória, pós-protocolo no corpus de desenvolvimento; esses ICs não corrigem essa seleção.

A confirmação final foi executada sob protocolo congelado antes da seleção externa, mas encontrou apenas duas famílias novas elegíveis no OpenML, em vez do alvo de 40. O ganho primário de 1,08 p.p. [$-0,28$; 2,44] não atingiu o critério. A amostra não oferece poder adequado nem sustenta uma conclusão geral de transferência ou de ausência de efeito. Nas 41 famílias antigas, o ganho de 0,68 p.p. [0,02; 1,62] e a taxa de 0,3\% de perdas maiores que um ponto são exclusivamente exploratórios.

H2b conserva a história do RF-12 congelado contra H27, com a ressalva de ampliação do holdout após a análise inicial. Esse resultado não é uma confirmação final. A nomenclatura H2a final do novo protocolo refere-se ao contraste externo frente à b8u, e não muda a distinção entre evidência interna e externa.

\section{Resposta ao problema de pesquisa}
\label{sec:resposta-problema}

Metadados públicos permitem recomendar uma opção com ganho médio interno sobre uma referência fixa forte, sem consultar células confidenciais. O custo zero da escolha depende de tamanho, domínios, classes e transformação declarados previamente. O modelo escolhido continua sujeito à sua contabilidade DP; a evidência externa da vantagem final permanece insuficiente. As ablações, listas, metamodelos e regras simples foram reestimados com essa definição em partições semânticas e permanecem diagnósticos exploratórios.

\section{Premissas e limitações}
\label{sec:limitacoes}

O escopo é classificação tabular centralizada. As 15 opções não esgotam os mecanismos disponíveis. A transformação $\tanh$ pode perder informação; onze candidatos falharam na validação da extensão e dezesseis conjuntos maiores haviam sido excluídos por custo. Existem datasets pequenos e famílias relacionadas; a taxonomia semântica não garante independência substantiva. Os dois IDs sem família compartilham um grupo; a auditoria preserva a diferença entre 667 grupos nos ajustes e 666 na síntese anterior.

Os ICs são pontuais e condicionados aos picks, sem incorporar toda a escolha adaptativa ou a dependência decorrente de treinos LODO sobrepostos. A decomposição de Shapley e as permutações medem efeitos no preditor, sem interpretação causal. As análises de cinco partições têm escopo distinto do LODO. A comparação de transformações permanece histórica GBM-13 porque não há scores das duas novas opções sob as transformações alternativas; não foram fabricados resultados. O mesmo vale para ajustes limpos auxiliares ausentes na extensão exploratória antiga.

BA de teste, oráculos e ajustes limpos são diagnósticos em bases públicas. O orçamento DP é por treinamento; liberar várias variantes sobre uma mesma base exige composição adicional. Sementes fixas e aritmética do benchmark não são uma implementação endurecida para produção \cite{ref86}. O protocolo é interno e versionado, sem pré-registro público. Somente duas famílias externas novas, uma delas sintética, limitam poder e validade externa; o holdout antigo não resolve essa limitação por já ter sido aberto.

\section{Trabalhos futuros}
\label{sec:trabalhos-futuros}

A prioridade é recrutar famílias independentes em outras fontes de dados e repositórios, preservando a configuração, os comparadores, o desfecho primário, as caudas e o critério do protocolo confirmatório final congelado. O desenho deve planejar tamanho amostral e poder antes de abrir resultados; qualquer adaptação necessária à fonte precisa ser registrada antes dessa abertura, sem reutilizar a confirmação pequena como tuning.

Outras direções são comparar pré-processamentos DP sob orçamento total equivalente, avaliar descritores privados mais informativos, reduzir o portfólio sem seleção retrospectiva no teste, endurecer a implementação numérica e estender o estudo a regressão. Nenhuma dessas extensões substitui a confirmação externa da versão atual.

\section{Considerações finais}
\label{sec:consideracoes-finais}

A final melhora a seleção interna sob um contrato público explícito. A confirmação externa permaneceu sem poder; a próxima validação deve recrutar famílias em outros repositórios, preservando o protocolo congelado.
'''

def introduction():
    text=(SNAP/'latex-dissertacao/capitulos/1-introducao.tex').read_text()
    for a,b in [('13 opções','15 opções'),('13 pares','15 pares'),('303.576','350.280'),('0{,}01','0{,}02'),('melhor opção fixa do treino','b8u congelada')]:text=text.replace(a,b)
    text=text.replace('mantendo a mesma contabilidade de privacidade do original.','mantendo a mesma contabilidade de privacidade do original. Acrescenta também DP-SGD com regularização pela regra pública $n\\varepsilon$ e com $\\lambda=0{,}0003$, ambos com a contabilidade H27.')
    start=text.index('A Hipótese 1 responde');end=text.index(r'\section{Contribuições}',start)
    text=text[:start]+r'''A Hipótese 1 responde às Questões 3 e 4; H2a mede a vantagem interna sobre uma referência fixa forte, a b8u. Na final, a configuração e a escolha de fallback/limiar são exploratórias, posteriores ao protocolo histórico e feitas no desenvolvimento. H2b mantém a história da comparação RF-12 congelada contra H27; não se transporta essa confirmação ao novo seletor.

O protocolo confirmatório final congelou a comparação externa primária contra b8u antes de selecionar novas famílias. Ele denomina esse critério H2a final: limite inferior do IC95\% do ganho maior que zero. Apenas duas famílias novas elegíveis existiam no catálogo OpenML examinado, em vez das 40 pretendidas; o ganho de 1,08 p.p. [$-0,28$; 2,44] não atingiu o critério e não dispõe de poder adequado. Nas 41 famílias antigas, a avaliação é exclusivamente exploratória. Confirmar transferência requer outras fontes de dados sob o protocolo congelado.

'''+text[end:]
    text=text.replace('identificando como exploratória a reutilização do \\textit{holdout} na GBM-13','distinguindo a tentativa confirmatória final sem poder da reutilização exploratória do \\textit{holdout} antigo')
    return text

def methodology():
    text=(SNAP/'latex-dissertacao/capitulos/3-metodologia.tex').read_text()
    text=text[:text.index(r'\subsection{Confirmação em famílias novas}')]
    for a,b in [('13 opções','15 opções'),('13 pares','15 pares'),('303.576','350.280'),('0{,}01','0{,}02'),('melhor fixa do treino','b8u congelada'),('97 posições','99 posições'),('13 indicadores próprios','15 indicadores próprios'),('seletor principal GBM-13','seletor principal final'),('A GBM-13 usa','A final usa')]:text=text.replace(a,b)
    text=text.replace('A b8u herda a calibração do NB original nos indicadores legados, conforme o experimento de origem.','A final usa a calibração pública própria de cada opção. As variantes DP-SGD compartilham a identidade legada de H27 e alteram o descritor público de regularização; a b8u usa sua calibração de oito intervalos. Os 15 indicadores próprios distinguem as ações. O vetor anterior GBM-13 tinha 97 posições e usava o alias legado do NB para b8u.')
    old='25 passos; 100 passos; 100 passos com \\textit{clipping} 2 e $\\lambda = 0{,}03$'
    text=text.replace(old,old+'; 100 passos/clip 2 com regra pública de $\\lambda$; 100 passos/clip 2 com $\\lambda=0{,}0003$')
    start=text.index('Seja $b_T$');end=text.index('\n\nA formulação',start)
    text=text[:start]+r'''Seja $b$ a b8u congelada e $\hat a_o$ a vantagem residual prevista. A política final escolhe $\arg\max_o\hat a_o$ somente quando $\max_o\hat a_o-\hat a_b>\tau$, com $\tau=0{,}02$; caso contrário, retorna à b8u. O retorno é explícito, inclusive nas análises secundárias, e não muda com a melhor fixa observada em uma partição. A escolha conjunta de fallback e limiar usou a grade exploratória no desenvolvimento, depois do protocolo histórico; os ICs condicionam à configuração selecionada. A b8u foi mantida como referência congelada. O RF-12 permanece comparador histórico; sua arquitetura reestimada em análises secundárias é identificada separadamente.'''+text[end:]
    text=text.replace('A melhor opção fixa, principal \\textit{baseline}, é escolhida em cada partição usando apenas as famílias de treino.','A referência primária e o fallback são a b8u congelada. Referências auxiliares de melhor opção por orçamento são escolhidas somente no treino da partição. A ordenação retrospectiva por família é separada da ordenação por dataset que motivou a referência b8u.')
    text=text.replace('A GBM-13 usa a família semântica','A final usa a família semântica').replace('Como nunca foram avaliados, permaneceram elegíveis para a confirmação externa, e cinco deles integram o \\textit{holdout}', 'Na história RF-12, a exclusão por custo deixou esses conjuntos elegíveis à confirmação e cinco integraram o \\textit{holdout} antigo; a final exige disjunção contra todos os manifestos anteriores')
    text=text.replace('973 \\times 13 \\times 8', '973 \\times 15 \\times 8').replace('ou melhor fixa', 'ou b8u')
    text=text.replace('com um classificador da opção vencedora e com outros regressores', 'com outros regressores').replace('regressão Ridge e um classificador que prevê diretamente a opção vencedora', 'regressão Ridge e a arquitetura RF histórica reestimada').replace('com os hiperparâmetros padrão', 'com os mesmos hiperparâmetros do GBM residual, mas alvo absoluto').replace('; o classificador é uma floresta de 500 árvores, com profundidade máxima 12 e pelo menos 3 exemplos por folha', '. Não se reajustou um classificador direto da vencedora nesta regeneração final')
    text += r'''
\subsection{Regularização pública e contabilidade das novas opções}
\label{sec:desenvolvimento}

O estudo E6/E7 motivou duas opções com DP-SGD em lote completo, 100 passos, clip 2 e $\delta=10^{-6}$. A regra final usa o tamanho total público $n$, e não o tamanho da partição de treino:
\[
\lambda(n,\varepsilon)=\begin{cases}
0{,}03,& n\varepsilon<100,\\
0{,}01,&100\leq n\varepsilon<1000,\\
0{,}0003,&n\varepsilon\geq1000.
\end{cases}
\]
O controle usa $\lambda=0{,}0003$ em todas as condições. O JSON final E6 preserva cinco faixas originais; as três últimas escolhem o mesmo valor, justificando a expressão compacta. O tamanho usado na calibração do ruído continua sendo $n_{treino}$. A regra não lê BA, rótulos nem células da nova base para escolher $\lambda$.

A contabilidade é idêntica à H27: $\rho_{total}=(\sqrt{\log(1/\delta)+\varepsilon}-\sqrt{\log(1/\delta)})^2$, $\rho_{passo}=\rho_{total}/100$, sensibilidade $2\,clip/n_{treino}$ e $\sigma=(2\,clip/n_{treino})/\sqrt{2\rho_{passo}}$. Mudar somente $\lambda$ não altera o ledger. Não há amplificação por subamostragem. Foram auditadas 46.704 avaliações novas, sem falhas. O orçamento é por treinamento individual; liberar todas as variantes exige composição. E6/E7 e a escolha da política são desenvolvimento exploratório, sem pretensão de confirmação independente.

\subsection{Protocolo congelado e confirmação final}
\label{sec:protocolo-confirmatorio}

O protocolo confirmatório final foi congelado antes de selecionar, baixar ou avaliar o novo holdout. O modelo foi treinado uma vez nos 973 datasets e salvo com hashes, usando 15 opções, fallback b8u e $\tau=0{,}02$. A seleção OpenML usa semente 20261111, exige 200--50.000 linhas, 2--30 classes, até 512 atributos codificados e disjunção por ID, família canônica e semântica contra todos os manifestos de dados anteriores. O alvo é 40 famílias; o protocolo autoriza todas as elegíveis se houver menos. Falhas são substituídas na ordem sorteada, sem olhar desempenho.

O desfecho primário é BA da recomendação menos BA da b8u, promediada por orçamento e repetição, depois por dataset e família. O IC95\% é percentil bootstrap de famílias, 20.000 reamostragens, semente 20261111. O critério congelado, denominado H2a final, exige limite inferior maior que zero. As caudas são restrições descritivas: fração de condições com perda superior a um ponto e CVaR$_{90}$ da perda assinada $\mathrm{BA}_{b8u}-\mathrm{BA}_{seletor}$, média das maiores $\max(1,\lfloor0{,}1N\rfloor)$ perdas. Essas caudas por condição não são as caudas do regret por família.

Os quatro secundários comparam H27, NB antigo, seletor GBM-13 de 13 opções e RF-12 congelado; o teste é sign-flip bilateral pareado por família, com 50.000 sorteios e Holm exatamente sobre esses quatro contrastes. Os ICs secundários são marginais. O resultado nas 41 famílias antigas e todos os estratos são exploratórios. Nenhuma decisão é retocada após a abertura do novo holdout.

O catálogo examinado ofereceu apenas duas famílias novas elegíveis, após esgotar a ordem de 869 candidatos. A tentativa foi executada como congelada e não atingiu o critério primário; sua dimensão impede confirmação com poder adequado. Trata-se de protocolo interno versionado, não de pré-registro público. A confirmação histórica H2b RF-12 contra H27, incluindo sua expansão pós-análise de 30 para 41 famílias, é mantida como história e não como validação da nova configuração.

\subsection{Escopo das análises secundárias}

Modos de descritores, permutação, Shapley, listas, metamodelos, curva de aprendizado e modelos por orçamento usam a definição final, com a calibração própria das opções e o retorno explícito à b8u. A curva usa 10\%, 25\%, 50\%, 75\% e 100\% das famílias de treino, com dois sorteios nas frações menores e um em 100\%. Regras simples são reajustadas por exclusão semântica; concepção da grade e comparação de políticas são exploratórias.

A comparação histórica de transformações em 199 datasets tem somente as 13 opções GBM-13, pois os scores das duas novas variantes sob transformações alternativas não estão disponíveis. É mantida com esse rótulo e não extrapolada para final. Diagnósticos limpos históricos e buscas privadas congeladas de 12 opções também não são rebatizados como resultados novos. A ausência de scores é registrada, sem imputação.

\section{Ambiente computacional e reprodutibilidade}
\label{sec:ambiente}

A regeneração reutiliza a estrutura da Fase A, em scripts novos, com saídas isoladas em \texttt{results/GBM-13\_1-dissertation/} e figuras em \texttt{figures/GBM-13\_1/}. Condições, picks, JSON da regra pública, protocolo, hashes e confirmação são fontes preservadas. A API de produção e os defaults de \texttt{dpmeta} não são alterados. O arquivo \texttt{NUMBERS\_gbm13\_1.md} registra valores anteriores, novos, proveniência, denominadores e itens não atualizáveis. As três versões LaTeX recebem os mesmos gráficos e tabelas; a compilação e a inspeção visual são registradas no changelog.
'''
    return text

def appendix():
    old=(SNAP/'latex-dissertacao/postextual/apendice-a.tex').read_text();prefix=old[:old.index(r'\section{Comportamento dos pares em cada conjunto}')]
    prefix=prefix.replace('13 opções','15 opções').replace('matriz GBM-13','matriz final').replace('artefatos GBM-13.','artefatos final.')
    ap=read('appendix_datasets.csv');modal=read('appendix_modal_winners.csv');distinct=read('appendix_distinct_winners.csv')
    prefix+=r'''\section{Comportamento dos pares em cada conjunto}
\label{sec:ap-comportamento}

A vencedora por condição maximiza a BA privada média de três repetições; empates dividem a vitória. A vencedora modal soma as vitórias nos oito orçamentos, com desempate pela BA média. O número de vencedoras distintas inclui empates e pode chegar a 15. A folga compara o oráculo privado à b8u e mantém o otimismo do máximo em teste. A coluna limpa usa o máximo das 15 opções por repetição; o regret principal mantém a referência limpa original, preservada no CSV separado.

'''
    rows=[]
    for i in range(max(len(modal),len(distinct))):
        l=modal.iloc[i] if i<len(modal) else None;r=distinct.iloc[i] if i<len(distinct) else None
        rows.append([CODE.get(l.iloc[0],tex(l.iloc[0])) if l is not None else '',str(int(l.iloc[1])) if l is not None else '',pp(l.iloc[1]/973,1) if l is not None else '',str(int(r.iloc[0])) if r is not None else '',str(int(r.iloc[1])) if r is not None else '',pp(r.iloc[1]/973,1) if r is not None else ''])
    prefix+=table('tab:ap-vencedoras','Vencedoras modais e distintas na final; denominador 973.',['Modal','Datasets','\\%','Distintas','Datasets','\\%'],rows,'appendix_modal_winners.csv; appendix_distinct_winners.csv')
    prefix+=r'''\section{Relação completa dos conjuntos}
\label{sec:ap-lista}

Códigos: NBC-b8u e NBC-Lap são os dois NB categóricos; NBG são os Gaussianos; AA são árvores; SM-GA é perturbação de saída. SGD-25 e SGD-100 são as variantes históricas; H27 tem clip 2 e $\lambda=0{,}03$; SGD-regra usa a regra pública e SGD-0003 a constante 0,0003. $V$ conta vencedoras distintas; BA limpa e privada são máximos descritivos, com peso igual por dataset. Nomes e IDs permitem localizar cada resultado no CSV integral.

\begingroup
\scriptsize
\setlength{\tabcolsep}{1.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{longtable}{r>{\raggedright\arraybackslash}p{3.3cm}rrrr>{\raggedright\arraybackslash}p{1.4cm}rrrr}
\caption{Os 973 datasets validados no portfólio final de 15 opções.}\label{tab:ap-datasets}\\
\toprule
ID & Nome & $n$ & $d$ & $p$ & $k$ & Modal & $V$ & Limpa & Privada & Folga\\\midrule
\endfirsthead
\multicolumn{11}{l}{\textit{Continuação da Tabela~\thetable}}\\
\toprule
ID & Nome & $n$ & $d$ & $p$ & $k$ & Modal & $V$ & Limpa & Privada & Folga\\\midrule
\endhead
\midrule\multicolumn{11}{r}{\textit{Continua na próxima página}}\\\endfoot
\bottomrule\endlastfoot
'''
    def breakname(name):
        s=tex(name);return re.sub(r'([A-Za-z0-9]{10})',r'\1\\allowbreak{}',s).replace('-',r'-\allowbreak{}').replace(r'\_',r'\_\allowbreak{}')
    for _,r in ap.sort_values(['n_instances','dataset_id']).iterrows():
        prefix+=' & '.join([str(int(r.dataset_id)),breakname(r.dataset_name),str(int(r.n_instances)),str(int(r.n_features)),str(int(r.n_encoded_features)),str(int(r.n_classes)),CODE[r.winner].replace('-',r'-\allowbreak{}'),str(int(r.distinct_winners)),f(r.best_clean,3),f(r.best_private,3),f(r.headroom,3)])+r' \\'+'\n'
    return prefix+'\\end{longtable}\n\\endgroup\n'

def convert(text,appendix=True):
    text=re.sub(r'\\section\{Estrutura da dissertação\}.*?(?=\\chapter|\Z)','',text,flags=re.S)
    text=text.replace(r'\subsection{',r'\subsubsection{').replace(r'\section{',r'\subsection{').replace(r'\chapter{',r'\section{').replace('\\SingleSpacing\n','')
    for a,b in [('Capítulo~','Seção~'),('capítulo~','seção~'),('Esta dissertação','Este trabalho'),('esta dissertação','este trabalho'),('desta dissertação','deste trabalho'),('nesta dissertação','neste trabalho')]:text=text.replace(a,b)
    return text

def assemble():
    intro=introduction();method=methodology();res=results();ap=appendix()
    (D/'capitulos/1-introducao.tex').write_text(intro);(D/'capitulos/3-metodologia.tex').write_text(method);(D/'capitulos/4-resultados.tex').write_text(res);(D/'capitulos/5-conclusao.tex').write_text(CONCLUSION);(D/'postextual/apendice-a.tex').write_text(ap)
    (D/'pretextual/resumo.tex').write_text('\\begin{resumo}\n'+RESUMO+'\n\n\\vspace{\\onelineskip}\n\\noindent\\textbf{Palavras-chave}: Privacidade Diferencial. Meta-Learning. Seleção de algoritmos. Metadados públicos.\n\\end{resumo}\n')
    (D/'pretextual/abstract.tex').write_text('\\begin{resumo}[Abstract]\n\\begin{otherlanguage*}{english}\n'+ABSTRACT+'\n\n\\vspace{\\onelineskip}\n\\noindent\\textbf{Keywords}: Differential Privacy. Meta-learning. Algorithm selection. Public metadata.\n\\end{otherlanguage*}\n\\end{resumo}\n')
    article=SNAP/'artigo/main.tex';old=article.read_text();preamble=old[:old.index(r'\begin{document}')]
    preamble=preamble.replace(r'\usepackage{enumitem}',r'\usepackage{enumitem,longtable}')
    fundamentals=old[old.index(r'\section{Fundamentos e trabalhos relacionados}'):old.index(r'\section{Questões de pesquisa e hipóteses}')].replace('13 opções','15 opções')
    article_intro=r'''\section{Introdução}
\label{sec:introducao}

A escolha do par mecanismo-modelo sob Privacidade Diferencial depende do orçamento e do dataset \cite{ref02,ref16,ref17}. Avaliar candidatos nos dados privados consome orçamento; meta-features convencionais também leem células \cite{ref26,ref33,ref64,ref65}. Propomos recomendar usando apenas schema e calibração públicos. A final reúne 15 opções e um GBM residual com fallback b8u e $\tau=0{,}02$. O desenvolvimento contém 973 datasets e 667 grupos semânticos; a seleção da configuração é exploratória, posterior ao protocolo histórico.

H1 pergunta se nenhuma opção domina o portfólio; H2a avalia o ganho interno sobre b8u sem piorar caudas. H2b mantém a história do RF-12 congelado contra H27. O novo protocolo confirmatório final denomina H2a final seu critério externo contra b8u. Esse teste foi congelado antes da seleção, mas somente duas famílias novas foram elegíveis; a falta de poder impede confirmação ampla. O holdout antigo de 41 famílias é exclusivamente exploratório para a nova configuração. A contribuição é uma recomendação sem custo adicional de privacidade, com evolução contábil preservada e separação explícita entre desenvolvimento e confirmação.

'''
    dims=read('appendix_public_dimensions.csv');dim_names={'n_instances':'Instâncias','n_features':'Atributos originais','n_encoded_features':'Colunas codificadas','n_nominal_features':'Atributos nominais','n_classes':'Classes'}
    article_corpus=table('tab:corpus','Perfil público dos 973 datasets do desenvolvimento final.',['Grandeza','Mínimo','Mediana','Máximo'],[[dim_names[r.variable],f(r['min'],0),f(r['median'],1),f(r['max'],0)] for _,r in dims.iterrows()],'appendix_public_dimensions.csv')
    body=preamble+'\\begin{document}\n\\maketitle\n\\begin{abstract}\n\\noindent\\textbf{Resumo.} '+RESUMO+'\n\n\\noindent\\textbf{Palavras-chave:} Privacidade Diferencial; Meta-Learning; seleção de algoritmos; metadados públicos.\n\\end{abstract}\n\\noindent\\textbf{Abstract.} '+ABSTRACT+'\n\n\\noindent\\textbf{Keywords:} Differential Privacy; meta-learning; algorithm selection; public metadata.\n\n'+article_intro+fundamentals+convert(method)+article_corpus+convert(res)+convert(CONCLUSION)
    body=re.sub(r'Apêndice~\\ref\{ap:datasets\}', 'CSV individual do corpus', body)
    body=body.replace('Com base na literatura da Seção~\\ref{sec:calibracao-epsilon}, o trabalho', 'O trabalho').replace('que cobre as faixas da Seção~\\ref{sec:calibracao-epsilon}:', 'que cobre duas faixas de estudo:')
    body+='\n\\section*{Disponibilidade e uso de inteligência artificial}\nOs scores, picks e registros numéricos sustentam as tabelas e figuras. Ferramentas de IA auxiliaram a implementação e a revisão; interpretação e responsabilidade são do autor.\n\\bibliography{referencias}\n\\end{document}\n'
    (PAPERS/'artigo/main.tex').write_text(body)
    fundamental=(SNAP/'latex-dissertacao/capitulos/2-fundamentacao.tex').read_text().replace('13 opções','15 opções')
    (D/'capitulos/2-fundamentacao.tex').write_text(fundamental)
    for filename in ['main.tex','main-resultados.tex']:
        target=PAPERS/'latex-problema-pesquisa'/filename;old=(SNAP/'latex-problema-pesquisa'/filename).read_text()
        toc=old.index(r'\tableofcontents');cut=old.index('\n',old.index(r'\newpage',toc))+1;head=old[:cut]
        if r'\begin{abstract}' in head:head=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:r'\begin{abstract}'+'\n'+RESUMO+'\n\n'+r'\noindent\textbf{Palavras-chave:} Privacidade Diferencial; Meta-Learning; metadados públicos.'+'\n'+r'\end{abstract}',head,flags=re.S)
        else:
            marker=r'\addcontentsline{toc}{section}{Resumo}';i=head.index(marker)+len(marker);j=head.index(r'\noindent\textbf{Palavras-chave',i);head=head[:i]+'\n'+RESUMO+'\n\n'+head[j:]
        head=re.sub(r'\bv5\b(?!\.1)', 'final',head)
        cron=re.search(r'\\subsection\{Cronograma\}.*?\\end\{table\}',old,re.S).group()
        parts=[convert(intro)+cron,convert(fundamental),convert(method)]
        if filename=='main-resultados.tex':parts.extend([convert(res),convert(CONCLUSION)])
        else:
            summary=r'''\section{Síntese dos resultados final}

O desenvolvimento LODO de 667 grupos ganha 1,426 p.p. [1,124; 1,745] sobre b8u; o registro inicial de 1,42 p.p. é preservado e a auditoria explicita o grupo sem nome descartado na síntese anterior. A perda superior a um ponto ocorre em 2,4\% das condições; CVaR$_{90}$ da perda de 1,03 p.p. H1 e H2a têm evidência interna, condicionada às escolhas exploratórias.

O protocolo externo congelado final encontrou somente duas famílias novas no OpenML: ganho de 1,08 p.p. [$-0,28$; 2,44], critério não atingido, poder insuficiente. No holdout antigo de 41 famílias, a final ganha 0,68 p.p. [0,02; 1,62], com 0,3\% de condições perdendo mais de um ponto; esse resultado é exclusivamente exploratório. A história confirmatória RF-12 não é transferida à final. O próximo passo é confirmação em novos repositórios com o protocolo confirmatório final preservado.

'''
            parts.extend([summary,r'\input{tables/resultados-seletor}',r'\input{tables/holdout}',convert(CONCLUSION)])
        tail='\n\\clearpage\n\\bibliography{referencias}\n'
        if filename=='main-resultados.tex':tail+='\n\\clearpage\n\\appendix\n'+convert(ap)
        tail+='\n\\end{document}\n'
        assembled=head+'\n'+'\n'.join(parts)+tail
        if r'\usepackage{longtable}' not in assembled:assembled=assembled.replace(r'\usepackage{array}',r'\usepackage{array,longtable}')
        # Proposal has no detailed results: remove only references to absent result labels.
        if filename=='main.tex':
            assembled=assembled.replace('(Seções~\\ref{sec:resultado-h1} e~\\ref{sec:sensibilidade-transformacao})', '(análise detalhada na versão completa de resultados)')
            assembled=re.sub(r'Apêndice~\\ref\{ap:datasets\}', 'CSV individual do corpus', assembled)
            available=set(re.findall(r'\\label\{([^}]+)\}',assembled))|set(TABLES)
            assembled=re.sub(r'(?:Seção|Tabela|Figura)~\\ref\{([^}]+)\}',lambda m:m.group() if m.group(1) in available else 'a análise detalhada de resultados',assembled)
        target.write_text(assembled)
    for folder in ['latex-dissertacao','artigo','latex-problema-pesquisa']:
        directory=PAPERS/folder;(directory/'tables').mkdir(exist_ok=True)
        for label,(content,source) in TABLES.items():(directory/'tables'/f'{label.replace("tab:","")}.tex').write_text(content)
        for figure in (ROOT/'figures/dissertation').glob('fig*.pdf'):shutil.copyfile(figure,directory/'figures'/figure.name)
    # Safe regeneration route for this GBM-13 folder; never read original folders.
    gen=PAPERS/'latex-problema-pesquisa/gen_from_dissertacao.py'
    gen.write_text('"""Regenera somente as versões final autorizadas, usando snapshots e artefatos da Fase C."""\nfrom pathlib import Path\nimport subprocess\nroot=Path(__file__).resolve().parents[2]/"dp_meta_selector_gbm13"\nsubprocess.run([str(root/".venv/bin/python"),str(root/"scripts/rewrite_documents.py")],cwd=root,check=True)\n')
    coverage=[dict(label=label,source=source) for label,(_,source) in TABLES.items()];pd.DataFrame(coverage).to_csv(OUT/'rendered_tables.csv',index=False)
    with (PAPERS/'V5_1_CHANGELOG.md').open('a') as log:
        log.write('\n## Etapa 2 concluída — reescrita final\n\n- Dissertação: resumo/abstract, introdução, portfólio, features, seletor, protocolo, resultados completos, discussão, conclusão e apêndice individual regenerados.\n- Artigo: resumo/abstract, introdução/hipóteses, método, resultados, limitações e conclusão reescritos; mesmas fontes numéricas e gráficos.\n- Problema de pesquisa: main.tex e main-resultados.tex atualizados, síntese/resultado completo e protocolo confirmatório final; gerador local redirecionado exclusivamente às pastas autorizadas.\n- Números: opções 13 → 15; avaliações 303.576 → 350.280; vetor 97 → 99; tau 0,01 → 0,02; ganho sobre b8u 0,9767 → 1,4263 pp no estimando de 667 grupos; perda >1pp 2,7% → 2,4%; regret médio 0,1444 → 0,1399; confirmação nova sem antecessor: 2 famílias, +1,080 pp [-0,278; 2,438], critério não atingido. Holdout antigo +0,4571 → +0,6772 pp, só exploratório.\n- Auditoria: resumo anterior 1,42 pp excluía implicitamente o grupo sem nome (666 grupos na média, 667 nos ajustes). Mantidos o registro publicado e a estimativa com todos os grupos; detalhes em NUMBERS.md.\n- Itens não atualizáveis: transformações alternativas sem scores das duas novas opções; diagnósticos limpos novos incompletos no holdout antigo; buscas RF-12 de 12 opções. Tabelas de transformação/colapso mantidas explicitamente como história GBM-13; sem imputação.\n- Build/QA: pendentes.\n')
    print('Rewritten three GBM-13 document folders; tables',len(TABLES))

if __name__=='__main__':assemble()
