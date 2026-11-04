"""
Method B features (the list from the assignment), computed for each stream
(title, body, whole document) -> 22 x 3 = 66 features per query-document pair,
min-max normalised per query.
"""
from functools import lru_cache

import numpy as np

from src.dataset import STREAMS

AGGS = ['sum', 'min', 'max', 'mean', 'var']
BASE = (['qt_number', 'qt_ratio', 'stream_length', 'idf']
        + [f'{a}_{x}' for x in ('tf', 'ntf', 'tfidf') for a in AGGS]
        + ['boolean', 'vsm', 'bm25'])
FEATURE_NAMES = [f'{s}:{f}' for s in STREAMS for f in BASE]
K1, B = 1.2, 0.75


@lru_cache(maxsize=None)
def doc_norms(coll, s):
    tf = coll.tf[s]
    return np.linalg.norm((np.log1p(np.maximum(tf - 1, 0)) + (tf > 0)) * coll.idf[s][:, None], axis=0)


def stream_features(coll, ids, s):
    tf, L, idf = coll.tf[s][ids], coll.length[s], coll.idf[s][ids, None]
    present = tf > 0
    aggregates = [f(m, axis=0) for m in (tf, tf / np.maximum(L, 1), tf * idf)
                  for f in (np.sum, np.min, np.max, np.mean, np.var)]

    # cosine between (1 + log tf) * idf document vector and idf query vector
    w = np.log1p(np.maximum(tf - 1, 0)) + present
    vsm = (w * idf ** 2).sum(0) / (np.linalg.norm(idf) * doc_norms(coll, s) + 1e-9)

    bm25 = (idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * L / coll.avg_len[s]))).sum(0)

    return np.stack([present.sum(0), present.mean(0), L, (present * idf).sum(0),
                     *aggregates, present.any(0), vsm, bm25], axis=1)


def extract(coll, qid):
    if len(coll.queries[qid]) == 0:      # no query term in the vocabulary: every document scores 0
        return np.zeros((coll.N, len(FEATURE_NAMES)))
    F =np.hstack([stream_features(coll, coll.queries[qid], s) for s in STREAMS])
    lo, hi = F.min(0), F.max(0)
    return (F - lo) / np.where(hi > lo, hi - lo, 1)
