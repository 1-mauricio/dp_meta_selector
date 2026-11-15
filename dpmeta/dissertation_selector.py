"""Explicit GBM-13 selector API. Historical portfolio defaults are never mutated.

Features and unnormalised weights reproduce experiments/lodo_gbm_residual.py.
The sum of weights is the number of semantic families, which matters for GBM L2.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from dpmeta import portfolio_benchmark as pb

NB = 'categorical_naive_bayes__laplace_histograms'
NB8U = pb.NB8U_PIPELINE
SEED = 20261110
TAU = .02
SELECTORS = ('rf12', 'gbm_residual_fallback')


def semantic_manifest(manifest):
    m = manifest.copy()
    # Reproduce the completed LODO exactly, including pandas' shared 'nan' key.
    m['family_id'] = (m.semantic_family_id if 'semantic_family_id' in m else m.family_id).astype(str)
    return m


def prepare(conditions, manifest):
    m = semantic_manifest(manifest)
    m = m[m.dataset_id.isin(conditions.dataset_id)].drop_duplicates('dataset_id')
    a = pb._aggregate(conditions, m)
    expanded_clean=(conditions.groupby(['dataset_id','epsilon','repetition']).clean_ba.max()
                    .groupby(['dataset_id','epsilon']).mean())
    a['best_clean_portfolio_ba_gbm13'] = expanded_clean.loc[list(zip(a.dataset_id,a.epsilon))].to_numpy()
    pipes = tuple(sorted(a.pipeline.unique()))
    if not a.groupby(['dataset_id','epsilon']).pipeline.nunique().eq(len(pipes)).all():
        raise ValueError('Incomplete portfolio')
    return a, m, pipes


def weights(a):
    w = 1. / a.groupby('family_id').dataset_id.transform('nunique')
    return (w / a.groupby('dataset_id').pipeline.transform('size')).to_numpy()


def empirical_best(a):
    return a.groupby(['family_id','pipeline']).private_ba.mean().groupby('pipeline').mean().idxmax()


def best_fixed(a):
    # Frozen comparator/fallback specified in protocols/CONFIRMATORY_PROTOCOL_FINAL.
    return NB8U if NB8U in set(a.pipeline) else empirical_best(a)


def feature_matrix(a, manifest, pipes, mode='public_plus_dp'):
    meta = manifest.drop_duplicates('dataset_id').set_index('dataset_id')
    cache = {}
    from dpmeta.meta_features import PRIVATE_META_COLUMNS, DP_META_COLUMNS
    private = {int(d): g.iloc[0][list(PRIVATE_META_COLUMNS)].to_numpy(float)
               for d,g in a.groupby('dataset_id')} if set(PRIVATE_META_COLUMNS) <= set(a) else {}
    dp = {int(d): g.iloc[0][list(DP_META_COLUMNS)].to_numpy(float)
          for d,g in a.groupby('dataset_id')} if set(DP_META_COLUMNS) <= set(a) else {}
    rows = []
    for r in a.itertuples(index=False):
        alias = r.pipeline
        base = pb._cached_candidate_features(meta.loc[int(r.dataset_id)], float(r.epsilon),
                                             alias, mode, private, dp, cache)
        rows.append(np.r_[base, [float(r.pipeline == q) for q in pipes]])
    return np.vstack(rows)


def feature_names(pipes, mode='public_plus_dp'):
    return (*pb.candidate_feature_names(mode), *((f'gbm13_option={p}', 'option') for p in pipes))


def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.,
                                         random_state=SEED)


class FallbackPredictor:
    """Return decision scores; fallback affects argmax and top-k/importance together.

    Input contains consecutive complete condition blocks in a consistent option
    order. Final one-hot columns identify options, so any order is supported.
    """
    def __init__(self, estimator, pipes, fixed, tau=TAU):
        self.estimator, self.pipes, self.fixed, self.tau = estimator, tuple(pipes), NB8U, tau

    def set_params(self, **kwargs):
        if 'n_jobs' in self.estimator.get_params():
            self.estimator.set_params(**kwargs)
        return self

    def predict(self, x):
        scores = self.estimator.predict(x)
        k = len(self.pipes)
        if len(x) % k:
            raise ValueError('Fallback needs complete condition blocks')
        blocks = scores.reshape(-1, k).copy()
        actions = x[:, -k:].reshape(-1, k, k).argmax(axis=2)
        if not np.all(np.sort(actions, axis=1) == np.arange(k)):
            raise ValueError('Each fallback block needs each option exactly once')
        pos = (actions == self.pipes.index(self.fixed)).argmax(axis=1)
        mx = blocks.max(axis=1)
        abstain = mx - blocks[np.arange(len(pos)), pos] <= self.tau
        blocks[np.flatnonzero(abstain), pos[abstain]] = np.nextafter(mx[abstain], np.inf)
        return blocks.ravel()


def fit(train, manifest, pipes=None, *, residual=True, fallback=True, mode='public_plus_dp'):
    pipes = tuple(pipes or sorted(train.pipeline.unique()))
    x = feature_matrix(train, manifest, pipes, mode)
    y = train.private_ba.to_numpy().copy()
    if residual:
        nb = train[train.pipeline == NB].set_index(['dataset_id','epsilon']).private_ba
        y -= nb.loc[list(zip(train.dataset_id, train.epsilon))].to_numpy()
    estimator = gbm().fit(x, y, sample_weight=weights(train))
    return FallbackPredictor(estimator, pipes, NB8U, TAU if fallback else -np.inf)


def ordered(a, pipes):
    return a.assign(_o=a.pipeline.map({p:i for i,p in enumerate(pipes)})).sort_values(
        ['dataset_id','epsilon','_o']).reset_index(drop=True)


def pick(test, scores):
    d = test.assign(_score=scores)
    return d.loc[d.groupby(['dataset_id','epsilon'])._score.idxmax()].copy()


def predictions(test, chosen, train, fold=0):
    fixed = best_fixed(train)
    keys = ['dataset_id','epsilon']
    rows = chosen[keys+['family_id','pipeline','private_ba','portfolio_regret']].rename(columns={
        'pipeline':'selector_pipeline','private_ba':'selector_ba','portfolio_regret':'selector_regret'})
    baselines = {'best_fixed':fixed, 'h27_fixed':pb.H27_PIPELINE, 'gnb_fixed':pb.PIPELINES[0]}
    for label, p in baselines.items():
        b = test[test.pipeline == p][keys+['portfolio_regret']].rename(columns={
            'portfolio_regret':f'{label}_regret'})
        rows = rows.merge(b, on=keys, validate='one_to_one')
    best = test.groupby(keys).private_ba.max()
    counts = test.assign(_top=test.private_ba >= test.groupby(keys).private_ba.transform('max')-1e-12)
    counts = counts.groupby(keys)._top.sum()
    unique = counts.eq(1)
    rows['selector_top1'] = [float(b >= best.loc[(d,e)]-1e-12 and unique.loc[(d,e)]) for
                            d,e,b in zip(rows.dataset_id, rows.epsilon, rows.selector_ba)]
    best_rows = test.loc[test.groupby(keys).private_ba.idxmax()].set_index(keys).pipeline
    rows['selector_top1_argmax'] = [float(p == best_rows.loc[(d,e)]) for d,e,p in
                                    zip(rows.dataset_id,rows.epsilon,rows.selector_pipeline)]
    rows['selector_top1_tie_inclusive'] = [float(b >= best.loc[(d,e)]-1e-12) for d,e,b in
                                           zip(rows.dataset_id,rows.epsilon,rows.selector_ba)]
    fba = test[test.pipeline == fixed].set_index(keys).private_ba
    rows['fixed_top1'] = [float(fba.loc[(d,e)] >= best.loc[(d,e)]-1e-12 and unique.loc[(d,e)])
                         for d,e in zip(rows.dataset_id,rows.epsilon)]
    for label, op in [('oracle','min'),('random_pipeline','mean'),('worst_pipeline','max')]:
        b = test.groupby(keys).portfolio_regret.agg(op).rename(f'{label}_regret').reset_index()
        rows = rows.merge(b,on=keys)
    eps_pipes = {float(e):empirical_best(train[train.epsilon == e]) for e in test.epsilon.unique()}
    score_lookup = test.set_index(keys+['pipeline']).portfolio_regret
    rows['epsilon_fixed_pipeline'] = rows.epsilon.map(eps_pipes)
    rows['epsilon_fixed_regret'] = [score_lookup.loc[(d,e,eps_pipes[float(e)])]
                                   for d,e in zip(rows.dataset_id,rows.epsilon)]
    rows['best_fixed_pipeline'] = fixed
    rows['feature_mode'] = 'public_plus_dp'
    rows['fold'] = fold
    return rows


def from_lodo(path, a):
    picks = pd.read_csv(path)
    p = picks[(picks.variant == 'GBM residual') & (picks.fallback == 'sim')].copy()
    keys = ['dataset_id','epsilon','pipeline']
    if p.duplicated(keys[:2]).any() or len(p) != a.groupby(keys[:2]).ngroups:
        raise ValueError('LODO picks do not cover conditions exactly')
    chosen = p[keys].merge(a,on=keys,validate='one_to_one')
    out = predictions(a,chosen,a)
    bf = p.set_index(keys[:2]).best_fixed
    if not (bf == best_fixed(a)).all():
        raise ValueError('LODO uses varying training fixed options; supply fold-specific references')
    return out


def grouped(conditions, manifest, *, mode='public_plus_dp', n_splits=5, sink=None):
    a,m,pipes = prepare(conditions,manifest)
    ids = m.sort_values('dataset_id')
    rows=[]
    splits = (GroupKFold(n_splits=n_splits).split(ids,groups=ids.family_id) if n_splits else
              ((np.flatnonzero(ids.family_id.to_numpy()!=f),np.flatnonzero(ids.family_id.to_numpy()==f))
               for f in sorted(ids.family_id.unique())))
    for fold,(tr,te) in enumerate(splits):
        train = a[a.dataset_id.isin(ids.iloc[tr].dataset_id)]
        test = ordered(a[a.dataset_id.isin(ids.iloc[te].dataset_id)],pipes)
        model=fit(train,m,pipes,mode=mode)
        chosen=pick(test,model.predict(feature_matrix(test,m,pipes,mode)))
        rows.append(predictions(test,chosen,train,fold))
        if sink is not None:
            sink[fold]={'test_ids':set(test.dataset_id),'selectors':{'public_plus_dp':model}}
    pred=pd.concat(rows,ignore_index=True)
    gain=pred.assign(gain=pred.best_fixed_regret-pred.selector_regret).groupby('family_id').gain.mean()
    mean,lo,hi=pb._bootstrap_ci(gain.to_numpy(),SEED)
    summary={'feature_modes':{'public_plus_dp':{'mean_regret_gain_vs_best_fixed':mean,
              'gain_ci95_family_bootstrap':[lo,hi]}},'semantic_families':int(m.family_id.nunique()),
              'selector':'gbm_residual_fallback','tau':TAU,'seed':SEED}
    return summary,pred
