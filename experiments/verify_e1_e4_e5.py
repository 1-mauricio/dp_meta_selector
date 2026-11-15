"""Focused checks for the experiment wrappers, including exact baseline reuse."""
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.e1_e4_e5 import c, g, nb_stats, zero_nb, zero_gnb, zero_sgd
from dpmeta.schema import PublicSchema


def main():
    rng = np.random.default_rng(20261110)
    x = rng.uniform(-1, 1, (48, 3))
    y = np.tile([0, 1, 2], 16)
    schema = PublicSchema(("a", "b", "c"), ("numeric",)*3, ((),)*3, ("0", "1", "2"))
    stats = nb_stats(x, y, 3, schema, 16)
    native_stats = c.categorical_nb_statistics(x, y, 3, schema)
    assert np.array_equal(stats.categories, native_stats.categories)
    assert all(np.array_equal(a,b) for a,b in zip(stats.feature_counts, native_stats.feature_counts))
    for eps in (.1, 1., 4.):
        model, _, ledger = c.fit_private_categorical_nb(x, y, 3, schema, epsilon=eps, seed=7, statistics=stats)
        native, _, native_ledger = c.fit_private_categorical_nb(x, y, 3, schema, epsilon=eps, seed=7)
        assert np.array_equal(model.class_counts, native.class_counts)
        assert all(np.array_equal(a,b) for a,b in zip(model.feature_counts, native.feature_counts))
        assert ledger.as_dict() == native_ledger.as_dict()
        for bins in (4,8):
            _, _, variant_ledger = c.fit_private_categorical_nb(
                x, y, 3, schema, epsilon=eps, seed=7, statistics=nb_stats(x,y,3,schema,bins))
            assert ledger.as_dict() == variant_ledger.as_dict()
        zero, categories, _ = zero_nb(x, y, 3, schema, epsilon=eps, seed=7, statistics=stats)
        clean, _ = c.fit_clean_categorical_nb(x, y, 3, schema)
        assert np.array_equal(zero.predict(categories), clean.predict(categories))
        zgnb, _ = zero_gnb(x, y, 3, epsilon=eps, mechanism="gaussian_zcdp", allocation="equal", seed=7)
        assert np.array_equal(g.predict(x,zgnb), g.predict(x,g.fit_clean(x,y,3)))
        base, base_ledger = c.fit_private_sgd_softmax(
            x,y,3,epsilon=eps,delta=1e-6,seed=7,n_steps=100,gradient_clip=2.,regularization=.03)
        for reg in (.01,.003,.001):
            _, ll = c.fit_private_sgd_softmax(x,y,3,epsilon=eps,delta=1e-6,seed=7,
                                           n_steps=100,gradient_clip=2.,regularization=reg)
            assert ll.as_dict() == base_ledger.as_dict()
        z1,_ = zero_sgd(x,y,3,epsilon=eps,delta=1e-6,seed=7,n_steps=100,gradient_clip=2.,regularization=.03)
        z2,_ = zero_sgd(x,y,3,epsilon=eps,delta=1e-6,seed=8,n_steps=100,gradient_clip=2.,regularization=.03)
        assert np.array_equal(z1.weights,z2.weights)
    protected = json.loads(Path("results/experiments/protected_hashes_E1_E4_E5.json").read_text())
    changed = [p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed, changed
    result = dict(passed=True, protected_files_verified=len(protected),
                  checks=["baseline NB exact statistics/predictions/ledger", "NB bins identical ledger",
                          "SGD lambda identical ledger", "zero-noise NB/GNB equal clean",
                          "zero-noise SGD independent of seed", "protected hashes unchanged"])
    Path("results/experiments/wrapper_verification_E1_E4_E5.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
