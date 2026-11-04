import numpy as np

from src.dataset import ZONES


def zone_scores(coll, qid, mode='term'):
    """
    (N, 4) Boolean match scores of the query against title, author, bib, body.
    'or':   1 if any query term occurs in the zone (textbook definition)
    'term': fraction of query terms occurring in the zone (per-term Boolean, averaged)
    """
    ids = coll.queries[qid]
    if len(ids) == 0:                    # no query term in the vocabulary: every document scores 0
        return np.zeros((coll.N, len(ZONES)))
    present = [coll.tf[z][ids] > 0 for z in ZONES]
    agg = np.any if mode == 'or' else np.mean
    return np.stack([agg(p, axis=0) for p in present], axis=1).astype(float)
