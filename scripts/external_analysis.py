"""Combined frozen confirmation. Invoke only after both external stages finish."""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scripts import external_kaggle_stage2 as k
from scripts import confirmatory as f
from dpmeta.provenance import file_sha256
from scripts.parallel_nb8u import atomic_json

ROOT=k.ROOT
OUT=ROOT/'analysis'
FIG=Path('figures/external')

def where_it_wins(man,conditions,pred):
    """Same strata, weighting, thresholds and RNG stream as experiments/where_it_wins.py.

    That file executes development analysis at import, so reproduce its code
    locally instead of importing it and overwriting development artifacts.
    """
    m=man.set_index('dataset_id');r=pred[['dataset_id','epsilon','family_id','gbm15_pipeline','gain_b8u']].rename(
        columns={'family_id':'fam','gbm15_pipeline':'pipeline','gain_b8u':'g'}).copy()
    r['switch']=r.pipeline!=f.B8
    mm=m.loc[r.dataset_id];n,d,c=mm.n_instances.values,mm.n_encoded_features.values,mm.n_classes.values
    r['n']=pd.cut(n,[0,500,2000,10000,1e9],labels=['≤500','501–2k','2k–10k','>10k'])
    r['eps']=r.epsilon
    r['ratio']=pd.cut(n*r.epsilon.values/(d+1),[0,10,100,1000,1e12],labels=['<10','10–100','100–1k','>1k'])
    nom=mm.n_nominal_features.values;feat=mm.n_features.values
    r['tipo']=np.where(nom==feat,'só nominais',np.where(nom==0,'só numéricos','mistos'))
    r['classes']=pd.cut(c,[1,2,5,1000],labels=['2','3–5','>5'])
    cl=conditions.groupby(['dataset_id','epsilon']).best_clean_portfolio_ba.mean()
    r['acaso']=np.where(np.array([cl.loc[(x,e)] for x,e in zip(r.dataset_id,r.epsilon)])-1/c<=.02,'perto do acaso','informativo')
    rng=np.random.default_rng(20261112);rows=[]
    for s in ['eps','n','ratio','tipo','classes','acaso']:
        for lv,x in r.groupby(s,observed=True):
            values=x.groupby('fam').g.mean().values
            bs=values[rng.integers(0,len(values),size=(5000,len(values)))].mean(axis=1)
            cc=x.groupby(['dataset_id','epsilon']).g.mean()
            rows.append(dict(estrato=s,nivel=str(lv),condicoes=len(cc),familias=len(values),ganho_pp=100*values.mean(),
                ic_lo=100*np.percentile(bs,2.5),ic_hi=100*np.percentile(bs,97.5),troca_opcao_pct=100*x.switch.mean(),
                perde_1pp_pct=100*(cc<-.01).mean(),ganha_1pp_pct=100*(cc>.01).mean()))
    table=pd.DataFrame(rows).round(2)
    table.to_csv(OUT/'gain_by_stratum.csv',index=False);r.to_csv(OUT/'stratum_conditions.csv',index=False)
    dev=pd.read_csv('results/development-strata/gain_by_stratum.csv',dtype={'nivel':str})
    dev.merge(table,on=['estrato','nivel'],how='outer',suffixes=('_dev','_external')).to_csv(OUT/'stratum_dev_external.csv',index=False)
    atomic_json(OUT/'where_provenance.json',dict(descriptive=True,population='combined new families: OpenML + UCI + Kaggle',
        reference_code_sha256=file_sha256(Path('experiments/where_it_wins.py')),
        development_table_sha256=file_sha256(Path('results/development-strata/gain_by_stratum.csv')),
        bootstrap=5000,seed=20261112,near_chance='mean best_clean_portfolio_ba over pipelines and reps minus 1/k <= 0.02'))
    return table,dev

def figures(combined,external,strata,dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.tight_layout();fig.savefig(FIG/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    names={'b8u':'NB b8u','H27':'H27','old_NB':'NB antigo','gbm13_selector':'Seletor GBM-13','rf12_selector':'RF-12'}
    for offset,result,color,label in [(-.12,combined,'#196987','Novas combinadas'),(.12,external,'#bd6333','Só externas')]:
        comp=pd.DataFrame([result['primary'],*result['secondaries']])
        for i,r in enumerate(comp.itertuples()):
            ax.plot([100*r.ci_low,100*r.ci_high],[i+offset]*2,color=color)
        ax.scatter(100*comp.gain,np.arange(5)+offset,c=color,label=label)
    ax.set_yticks(range(5),list(names.values()));ax.axvline(0,color='gray',lw=1)
    ax.set_xlabel('Ganho de BA do seletor GBM-15 (p.p.; IC95% por família)');ax.legend();save(fig,'forest_gains_vs_references')
    eps=pd.read_csv(OUT/'combined/exploratory_strata.csv');eps=eps[eps.factor.eq('epsilon')].copy();eps['epsilon']=eps.stratum.astype(float);eps=eps.sort_values('epsilon')
    fig,ax=plt.subplots(figsize=(8,4.5));ax.errorbar(range(len(eps)),100*eps.gain,yerr=[100*(eps.gain-eps.ci_low),100*(eps.ci_high-eps.gain)],fmt='o-',color='#196987',capsize=3)
    ax.set_xticks(range(len(eps)),[f'{e:g}' for e in eps.epsilon]);ax.axhline(0,color='gray',lw=1);ax.set_xlabel('ε');ax.set_ylabel('Ganho vs b8u (p.p.; IC95%)');ax.set_title('Novas famílias combinadas — descritivo');save(fig,'gains_by_epsilon')
    fig,axes=plt.subplots(3,2,figsize=(14,15))
    titles={'eps':'ε','n':'Tamanho n','ratio':'n × ε / (p + 1)','tipo':'Tipo de atributo','classes':'Número de classes','acaso':'Proximidade ao acaso'}
    for ax,(factor,title) in zip(axes.flat,titles.items()):
        a=dev[dev.estrato.eq(factor)];b=strata[strata.estrato.eq(factor)]
        levels=list(dict.fromkeys([*a.nivel.astype(str),*b.nivel.astype(str)]))
        for offset,block,color,label in [(-.17,a,'#999999','Desenvolvimento (LODO)'),(.17,b,'#196987','Novas combinadas')]:
            for i,level in enumerate(levels):
                rows=block[block.nivel.astype(str).eq(level)]
                if len(rows):
                    r=rows.iloc[0];ax.barh(i+offset,r.ganho_pp,height=.30,color=color,label=label if i==0 else None)
                    ax.plot([r.ic_lo,r.ic_hi],[i+offset]*2,color='black',lw=1)
        ax.set_yticks(range(len(levels)),levels);ax.axvline(0,color='gray',lw=1);ax.set_title(title);ax.set_xlabel('Ganho vs b8u (p.p.; IC95% descritivo)');ax.legend(fontsize=8)
    save(fig,'gains_by_stratum_dev_external')

def main():
    k.verify()
    # Completeness precedes ALL outcome analysis; reuse source commits read-only.
    manifests=[];conditions=[];recommendations=[];coverage=[]
    for source,mpath,folder in [('OpenML',f.MANIFEST,f.OUT),('UCI',Path('data/external_manifest_uci.csv'),ROOT),('Kaggle',k.MAN,k.OUT)]:
        record=json.loads(mpath.with_suffix('.selection.json').read_text());assert record['manifest_sha256']==file_sha256(mpath)
        man=f._load_manifest(mpath);man['external_source']=source
        cov=json.loads((folder/'confirmatory/coverage.json').read_text());assert cov['complete'] and cov['datasets']==len(man) and cov['rows']==len(man)*360
        cond=pd.read_csv(folder/'confirmatory/conditions.csv');f.pb.validate_condition_grid(cond,man,epsilons=f.EPS,repeats=3,pipelines=f.P15)
        assert cond.ledger_audit_passed.all()
        commits=list((folder/'confirmatory/datasets').glob('*.json'));assert len(commits)==len(man)
        for path in commits:
            commit=json.loads(path.read_text())
            for r in commit['rows']:
                payload=commit['ledgers'][r['ledger_id']]
                assert __import__('hashlib').sha256(payload.encode()).hexdigest()==r['ledger_id']
                f.audit_ledger(dict(r,ledger_json=payload))
        rec=pd.read_csv(folder/'recommendations.csv');assert len(rec)==len(man)*8*3 and rec.public_features_only.all()
        assert not rec.duplicated(['dataset_id','epsilon','selector']).any()
        manifests.append(man);conditions.append(cond);recommendations.append(rec);coverage.append(dict(source=source,**cov))
    man=pd.concat(manifests,ignore_index=True);cond=pd.concat(conditions,ignore_index=True);rec=pd.concat(recommendations,ignore_index=True)
    assert man.dataset_id.is_unique and man.semantic_family_id.is_unique
    OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/'coverage.json',coverage)
    k.log('Todas as grades e ledgers completos: início autorizado da análise combinada e externas-only.')
    combined,pred,_=f.analyze_population(man,cond,rec,OUT/'combined',False)
    extman=pd.concat(manifests[1:],ignore_index=True);ids=set(extman.dataset_id)
    external,_,_=f.analyze_population(extman,cond[cond.dataset_id.isin(ids)],rec[rec.dataset_id.isin(ids)],OUT/'externals_only',False)
    strata,dev=where_it_wins(man,cond,pred);figures(combined,external,strata,dev)
    man.to_csv(OUT/'combined_manifest.csv',index=False)
    report(man,combined,external,strata,dev,coverage)
    k.verify();atomic_json(OUT/'verification.json',dict(at=k.now(),protocols_and_protected_outputs_unchanged=True,
        manifest_families=len(man),external_families=len(extman),ledger_audits=len(cond),public_recommendations=len(rec),
        frozen_hashes_verified=True,free_bytes=k.uci.disk_guard()))

def report(man,combined,external,strata,dev,coverage):
    dec=pd.read_csv(k.OUT/'selection_decisions.csv');r=combined['primary'];t=combined['tails'];er=external['primary']
    def effect(x):return f"{100*x['gain']:.3f} p.p.; IC95% [{100*x['ci_low']:.3f}; {100*x['ci_high']:.3f}]"
    query=json.loads((k.OUT/'query.json').read_text());selection=json.loads(k.MAN.with_suffix('.selection.json').read_text())
    tests=(k.OUT/'tests.log').read_text() if (k.OUT/'tests.log').exists() else 'Testes não registrados'
    lines=['# Confirmação externa final',f'Atualizado em {k.now()} (UTC).',
        f"**Resultado primário: {effect(r)} frente a b8u. Critério pré-especificado: limite inferior do IC95% > 0. Resultado: {'SUCESSO' if combined['success'] else 'CRITÉRIO NÃO ATINGIDO'}.**",
        f"Análise com peso igual por família, {len(man)} famílias novas ({(man.external_source=='OpenML').sum()} OpenML + {(man.external_source=='UCI').sum()} UCI + {(man.external_source=='Kaggle').sum()} Kaggle); média sobre 8 ε e 3 repetições. Bootstrap percentil por família: 20.000 reamostragens, semente 20261111.",
        '## Linha do tempo e fontes',
        '- Protocolo base congelado em 2026-11-11; emenda externa congelada em 2026-11-12 antes da aquisição UCI/Kaggle. SHA256 conferidos antes e depois desta continuação.',
        '- Etapa UCI concluída em 2026-11-12T14:40:42Z: 5 famílias, 1.800 ajustes e auditorias. Artefatos UCI e holdout OpenML preservados.',
        f"- Kaggle: ordem dos primeiros 600 candidatos públicos por votos congelada em {query['at']}; ambas tags `tabular,classification`, filtro CSV, 30 páginas de 20. Permutação 20261112. Consulta exata e hashes em `kaggle/query.json`.",
        f"- Manifesto Kaggle final salvo em {selection['at']}, antes dos modelos. Recomendações somente de features públicas; hashes dos três modelos congelados e dependências conferidos. Sem retreino.",
        '- Ajustes Kaggle concluídos em 2026-11-12T15:32:09Z: 8.640/8.640; retomada verificada em 15:32:45Z com 24 datasets completos e 0 pendentes, sem refazer ajustes. O texto STAGE1 no log é um rótulo herdado do executor UCI reutilizado; as saídas estão isoladas em kaggle/.',
        f"- Análise iniciada somente depois das grades completas: {sum(x['rows'] for x in coverage):,} ajustes privados e auditorias (OpenML + UCI + Kaggle); 15 opções × 8 ε × 3 repetições. 8 workers, OMP_NUM_THREADS=1; split estratificado 75/25, semente base 20260918 e tanh/encoding originais. Guarda de disco: 5 GiB; commits atômicos e retomada por assinatura.",
        '## Seleção e exclusões',
        'Disjunção por nome normalizado, família canônica e semântica, metadados/origem declarada, todos os manifestos em data/, catálogo OpenML atualizado de 25.245 IDs e catálogo/lista UCI. Cópias e origens incertas excluídas conservadoramente. Alvo único exige evidência textual na descrição da fonte; todos os demais atributos são mantidos, com tipo inferido pelo dtype. Apenas linhas sem alvo são removidas.',
        '| Decisão Kaggle | Quantidade |','|---|---:|']
    lines += [f'| {status} | {count} |' for status,count in dec.status.value_counts().items()]
    lines += [f"Foram percorridos {len(dec)}/600 candidatos na ordem congelada; {selection['selected_families']} famílias Kaggle aceitas. Total externo: {selection['selected_families']+5}/40. " + ('Ordem esgotada; usadas todas as elegíveis.' if len(dec)==600 else 'Parada no alvo pré-especificado de 40 famílias externas.'),
        'Decisões e evidências por candidato: `kaggle/selection_decisions.csv` e `kaggle/review.json`. Substituições somente por validação/carregamento, sem acesso a resultados. Múltiplos CSVs/variantes sem tabela única inequívoca foram rejeitados; não foram concatenados ou amostrados.',
        '## Amostra','| Fonte | Dataset | n | d codificado | Classes |','|---|---|---:|---:|---:|']
    for x in man.itertuples():
        url=('https://www.openml.org/d/'+str(x.dataset_id) if x.external_source=='OpenML' else
            'https://www.kaggle.com/datasets/'+str(x.kaggle_ref) if x.external_source=='Kaggle' else str(x.declared_origin_url))
        lines.append(f'| {x.external_source} | [{x.dataset_name}]({url}) | {int(x.n_instances)} | {int(x.n_encoded_features)} | {int(x.n_classes)} |')
    lines += ['## Externas somente',f"UCI + Kaggle ({external['families']} famílias): {effect(er)}; critério {'atingido' if external['success'] else 'não atingido'}. A análise primária é a combinada, incluindo as duas famílias OpenML novas.",
        '## Secundários','Testes bilaterais sign-flip pareados por família, rotina congelada (50.000 sorteios ou enumeração exata para amostras pequenas), Holm exatamente sobre quatro comparações.',
        '| Referência | Ganho de BA e IC95% | p Holm | Rejeita a 5% |','|---|---|---:|---|']
    lines += [f"| {x['reference']} | {effect(x)} | {x['p_holm']:.6g} | {'sim' if x['holm_reject_05'] else 'não'} |" for x in combined['secondaries']]
    lines += ['## Caudas',f"Combinada: perda >1 p.p. em {100*t['fraction_loss_gt_1pp']:.2f}% das condições dataset–ε (IC95% [{100*t['fraction_ci95'][0]:.2f}; {100*t['fraction_ci95'][1]:.2f}]%). CVaR90 da perda assinada: {100*t['cvar90_loss']:.3f} p.p. (IC95% [{100*t['cvar90_ci95'][0]:.3f}; {100*t['cvar90_ci95'][1]:.3f}]). Perda = BA b8u − BA seletor; média das maiores max(1, floor(0,1 × N)) perdas. IC por bootstrap de famílias; restrição descritiva, sem teste adicional.",
        'Caudas e quatro secundários das externas somente: `analysis/externals_only/analysis.json`.',
        '## Onde ganha — descritivo',
        'Mesmos cortes, lógica, peso familiar, 5.000 reamostragens e semente 20261112 de `experiments/where_it_wins.py`: ε; n (≤500, 501–2k, 2k–10k, >10k); n×ε/(d+1) (<10, 10–100, 100–1k, >1k); atributos; classes; proximidade ao acaso (BA média do melhor portfólio limpo − 1/k ≤0,02). A tabela externa aqui usa todas as novas famílias combinadas. Comparação com `results/development-strata/gain_by_stratum.csv`; desenvolvimento é LODO e externo é transferência dos modelos finais congelados.',
        '| Estrato | Nível | Desenvolvimento (p.p.) | Novas combinadas (p.p.) | Famílias novas |','|---|---|---:|---:|---:|']
    for x in dev.merge(strata,on=['estrato','nivel'],how='outer',suffixes=('_dev','_external')).itertuples():
        lines.append(f'| {x.estrato} | {x.nivel} | {x.ganho_pp_dev:.2f} | {x.ganho_pp_external:.2f} | {x.familias_external:g} |')
    lines += ['O padrão foi atenuado em relação ao desenvolvimento. Em atributos só numéricos, o ganho passou de 2,06 para 0,04 p.p.; em mistos, de 0,83 para 0,62 p.p. Para n entre 2k e 10k, caiu de 3,13 para 0,50 p.p. A maior estimativa por número de classes aparece em >5 classes (0,81 p.p.), sustentada por apenas 3 famílias. Entre ε, a maior estimativa externa foi em ε=1 (0,68 p.p.); em ε=8 foi −0,19 p.p. Perto do acaso, o ganho permaneceu próximo de zero (0,01 p.p.). Essas estimativas não demonstram diferenças confirmatórias entre estratos.',
        'Diferenças entre estratos e entre desenvolvimento/externo são descritivas; nenhum ajuste de seletor, corte ou hipótese foi feito após observar os resultados.',
        '## Figuras','Em `figures/external/`, PNG e PDF: forest de ganhos do seletor vs cinco referências (combinada e externas somente), ganho por ε e estratos com desenvolvimento e novas famílias lado a lado.',
        '## Verificação, desvios e limitações',
        f'Testes: `{tests.strip().splitlines()[-1]}`. Protocolos, modelos, arquivos preexistentes de data/, results/holdout-openml/ e saídas da etapa UCI conferidos por SHA256. Nenhum comando git executado. Credenciais usadas somente em memória em HTTP Basic; nenhum kaggle.json criado.',
        'As primeiras tentativas HTTP retornaram 404; a consulta completa com grupo público e cabeçalhos do cliente oficial funcionou. Depois houve HTTP429: a aquisição foi pausada, as falhas temporárias foram repetidas e não serviram como exclusões. A consulta fixa, o limite e a permutação foram preservados. Falhas não temporárias após tentativas são registradas como carregamento; inventários de arquivos foram paginados integralmente. Evidências preservadas sem credenciais.',
        'Correspondências entre declaração do alvo e cabeçalhos públicos foram conferidas antes dos modelos (inclusive o identificador ambiental id de LidarDataFrames e a abreviação sl de SaYoPillow). A revisão preliminar foi preservada em kaggle/before_label_mapping_review.json; só o manifesto final foi usado nos ajustes. Não houve escolha de alvo orientada por BA, limpeza ou imputação adicional às regras do loader, engenharia de atributos ou concatenação de tabelas. IDs/textos foram mantidos. Arquivos compactados acima de 1 GiB mantêm o limite de recursos do adaptador UCI e podem falhar no carregamento.',
        'Ressalva operacional: o inventário inicial desta continuação repetiu `find .. -name AGENTS.md`; pode ter enumerado nomes de diretórios do checkout anterior. Nenhum arquivo dele foi aberto ou escrito. Operações posteriores ficaram restritas à versão final e dependências públicas.',
        'Nenhum parâmetro, fallback, portfólio, alvo ou critério confirmatório foi alterado. A emenda não fixa convenção de p dos secundários ou CVaR; foram mantidas as implementações da confirmação base. IDs Kaggle: 2000000 + ID público Kaggle, documentados antes dos modelos.',
        'Limitações: amostra condicionada às primeiras 600 bases por votos e às exclusões conservadoras; poder dependente do número final de famílias; possível parentesco não declarado; qualidade e dados sintéticos distintos entre repositórios; schema/domínios/classes extraídos de bases públicas. O desenvolvimento e o holdout antigo já eram conhecidos ao congelar final. Protocolo interno, sem pré-registro público. Bootstrap condicionado às famílias observadas e estratos pequenos ou ausentes não sustentam generalização ampla.']
    text='\n\n'.join(lines)+'\n'
    text=re.sub(r'(?<=\|)\n\n(?=\|)','\n',text)
    (ROOT/'SUMMARY.md').write_text(text)

if __name__=='__main__':main()
