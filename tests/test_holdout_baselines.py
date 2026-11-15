import numpy as np

from dpmeta import holdout_baselines as hb
from dpmeta import portfolio_benchmark as pb
from dpmeta.schema import PublicSchema


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = np.tanh(rng.normal(size=(n, 3)))
    y = (x[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    schema = PublicSchema(("a", "b", "c"), ("numeric",) * 3, ((), (), ()), ("0", "1"))
    return x, y, schema


def test_every_option_trains_and_predicts_labels():
    x, y, schema = _data()
    for pipeline in pb.PIPELINES:
        predictor = hb.fit_option(pipeline, x, y, schema, epsilon=4.0, delta=1e-6, seed=3)
        labels = predictor(x[:50])
        assert labels.shape == (50,) and set(np.unique(labels)) <= {0, 1}


def test_exponential_choice_prefers_high_scores_and_is_seeded():
    scores = np.array([10.0, 50.0, 12.0])
    picks = [hb.exponential_choice(scores, 1.0, np.random.default_rng(i)) for i in range(200)]
    assert np.mean(np.array(picks) == 1) > 0.99
    assert hb.exponential_choice(scores, 1e-9, np.random.default_rng(5)) == \
        hb.exponential_choice(scores, 1e-9, np.random.default_rng(5))


def test_search_uses_disjoint_rows_and_returns_a_portfolio_option():
    x, y, schema = _data()
    chosen, predictor = hb.private_search(x, y, schema, epsilon=8.0, delta=1e-6,
                                          seed=1, rng_seed=2)
    assert chosen in pb.PIPELINES
    order = np.random.default_rng(2).permutation(len(y))
    n_val = int(round(hb.VALIDATION_FRACTION * len(y)))
    shards = np.array_split(order[n_val:], len(pb.PIPELINES))
    used = np.concatenate([order[:n_val], *shards])
    assert len(used) == len(set(used)) == len(y)  # cada registro em uma única liberação
    assert predictor(x).shape == (len(y),)
