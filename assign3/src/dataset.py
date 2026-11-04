import os
import random
from collections import Counter

import numpy as np

from src.parser_qq import parse_documents, parse_queries, parse_qrels
from src.preprocessing import tokenize

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
ZONES = ['title', 'author', 'bib', 'body']      # Method A
STREAMS = ['title', 'body', 'all']              # Method B


class Collection:
    """Cranfield docs, queries and qrels with a (vocab x docs) term-count matrix per zone."""

    def __init__(self, data_dir=DATA_DIR):
        docs = parse_documents(os.path.join(data_dir, 'cran.all.1400'))
        self.qrels = parse_qrels(os.path.join(data_dir, 'cranqrel'))
        self.doc_ids = sorted(docs)
        self.N = len(self.doc_ids)

        toks = {z: [tokenize(docs[d][z if z != 'body' else 'text']) for d in self.doc_ids] for z in ZONES}
        toks['all'] = [sum((toks[z][i] for z in ZONES), []) for i in range(self.N)]
        self.vocab = {t: i for i, t in enumerate(sorted({t for doc in toks['all'] for t in doc}))}

        self.tf, self.length, self.idf, self.avg_len = {}, {}, {}, {}
        for z, docs_z in toks.items():
            m = np.zeros((len(self.vocab), self.N), dtype=np.float32)
            for j, doc in enumerate(docs_z):
                for t, c in Counter(doc).items():
                    m[self.vocab[t], j] = c
            self.tf[z] = m
            self.length[z] = m.sum(axis=0)
            self.avg_len[z] = self.length[z].mean()
            self.idf[z] = np.log((self.N + 1) / ((m > 0).sum(axis=1) + 0.5))

        self.set_queries({q: v['text'] for q, v in parse_queries(os.path.join(data_dir, 'cran.qry')).items()})

    def set_queries(self, texts):
        """texts: {qid: text} -> self.queries: {qid: array of vocab ids of the distinct query terms}"""
        self.queries = {q: np.array([self.vocab[t] for t in dict.fromkeys(tokenize(s)) if t in self.vocab],
                                    dtype=int)
                        for q, s in texts.items()}

    def relevance(self, qid, graded=False):
        """Vector over docs: 1 for judged docs (or gain 5 - code if graded), 0 otherwise."""
        r = np.zeros(self.N)
        for d, code in self.qrels.get(qid, {}).items():
            r[self.doc_ids.index(d)] = 5 - code if graded else 1
        return r

    def rank(self, scores):
        """scores: {qid: score vector} -> (run {qid: ranked docids}, {qid: {docid: score}}); ties by doc id."""
        run = {q: [self.doc_ids[i] for i in np.lexsort((np.arange(self.N), -s))] for q, s in scores.items()}
        return run, {q: dict(zip(self.doc_ids, s.tolist())) for q, s in scores.items()}


def shuffled(qids, seed=42):
    qids = sorted(qids)
    random.Random(seed).shuffle(qids)
    return qids


def split_queries(qids, test_fraction=0.3):
    q = shuffled(qids)
    n = round(len(q) * test_fraction)
    return sorted(q[n:]), sorted(q[:n])


def kfold(qids, k=5):
    q = shuffled(qids)
    for i in range(k):
        test = set(q[i::k])
        yield sorted(set(q) - test), sorted(test)
