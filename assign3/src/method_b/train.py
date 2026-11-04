"""
Method B: learn w with a linear SVM and rank by w . psi(d, q).
  pointwise: classify each (psi(d,q), relevant?) pair
  pairwise:  Ranking SVM on psi(d_i,q) - psi(d_j,q) for docs with different graded relevance
"""
import numpy as np
from sklearn.svm import LinearSVC


def pointwise_data(coll, feats, qids):
    X = np.vstack([feats[q] for q in qids])
    y = np.concatenate([coll.relevance(q) for q in qids])
    return X, y


def pairwise_data(coll, feats, qids, max_pairs=2000, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for q in qids:
        g = coll.relevance(q, graded=True)
        i, j = np.nonzero(g[:, None] > g[None, :])       # d_i preferred over d_j
        keep = rng.choice(len(i), min(len(i), max_pairs), replace=False)
        sign = rng.choice([-1, 1], len(keep))             # flip half so both classes occur
        X.append((feats[q][i[keep]] - feats[q][j[keep]]) * sign[:, None])
        y.append(sign)
    return np.vstack(X), np.concatenate(y)


def train_svm(coll, feats, qids, kind='pairwise', C=1e-3):
    if kind == 'pointwise':
        svm = LinearSVC(C=C, class_weight='balanced', max_iter=20000)
        svm.fit(*pointwise_data(coll, feats, qids))
    else:
        svm = LinearSVC(C=C, fit_intercept=False, max_iter=20000)
        svm.fit(*pairwise_data(coll, feats, qids))
    return svm.coef_.ravel()
