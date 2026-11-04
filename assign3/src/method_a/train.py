"""
Method A: score(d, q) = sum_i g_i * s_i(d, q), with sum_i g_i = 1.
The weights g are found by grid search over the simplex, minimising either the
squared error to 0/1 relevance (as in the lecture) or maximising training MAP.
"""
import itertools

import numpy as np


def simplex_grid(n=4, step=0.05):
    m = round(1 / step)
    return np.array([c for c in itertools.product(range(m + 1), repeat=n) if sum(c) == m]) / m


def squared_error(W, coll, feats, qids, neg_ratio=3, seed=0):
    """Total (r - score)^2 over all relevant docs plus `neg_ratio` sampled non-relevant matching docs per relevant."""
    rng = np.random.default_rng(seed)
    err = np.zeros(len(W))
    for q in qids:
        r, F = coll.relevance(q), feats[q]
        neg = np.flatnonzero((r == 0) & (F.sum(1) > 0))
        idx = np.r_[np.flatnonzero(r), rng.choice(neg, min(len(neg), neg_ratio * int(r.sum())), replace=False)]
        err += ((F[idx] @ W.T - r[idx, None]) ** 2).sum(0)
    return err


def mean_ap(W, coll, feats, qids):
    """MAP of the ranking induced by every weight vector in W (ties count half)."""
    total = np.zeros(len(W))
    for q in qids:
        S = feats[q] @ W.T                                   # (N, |W|)
        Sr = S[coll.relevance(q) > 0]                        # (R, |W|)
        rank = 1 + (S > Sr[:, None]).sum(1) + 0.5 * ((S == Sr[:, None]).sum(1) - 1)
        total += ((rank[None] <= rank[:, None]).sum(1) / rank).mean(0)
    return total / len(qids)


def learn_weights(coll, feats, qids, objective='map', step=0.05):
    W = simplex_grid(step=step)
    if objective == 'mse':
        values = squared_error(W, coll, feats, qids)
        return W[values.argmin()], W, values
    values = mean_ap(W, coll, feats, qids)
    return W[values.argmax()], W, values
