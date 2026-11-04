"""
Learning to rank on Cranfield.
  python main.py                                         # run all experiments
  python main.py predict A|B <queries.qry> <out.txt>     # rank unseen queries with the final models
"""
import json
import os
import sys

import numpy as np

from src.dataset import Collection, ZONES, split_queries, kfold
from src.evals import evaluate, best_cutoff, write_run, table
from src.method_a.boolean_features import zone_scores
from src.method_a.train import learn_weights
from src.method_b.feature_extraction import extract, FEATURE_NAMES
from src.method_b.train import train_svm
from src.parser_qq import parse_queries

METRICS = ['P@5', 'P@10', 'R@10', 'F1@10', 'P@20', 'R@20', 'MAP', 'nDCG@10', 'P', 'R', 'F1']
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1]
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
log_file = None


def log(*args):
    print(*args)
    print(*args, file=log_file)


def score(feats, w, qids):
    return {q: feats[q] @ w for q in qids}


def cv_map(coll, feats, qids, fit, k=5):
    """Mean MAP over k query folds; fit(train_qids) returns a weight vector."""
    return np.mean([evaluate(coll.rank(score(feats, fit(tr), te))[0], coll.qrels)['MAP'] for tr, te in kfold(qids, k)])


def select_C(coll, feats, qids, kind):
    maps = {C: cv_map(coll, feats, qids, lambda tr: train_svm(coll, feats, tr, kind, C)) for C in CS}
    log(f'  {kind}: CV MAP by C', {C: round(float(m), 4) for C, m in maps.items()})
    return max(maps, key=maps.get)


def experiments():
    global log_file
    log_file = open('results/experiment_log.txt', 'w')
    coll = Collection()
    qids = sorted(coll.qrels)
    train, test = split_queries(qids)
    log(f'{coll.N} docs, {len(coll.vocab)} terms, {len(train)} train / {len(test)} test queries')

    zone = {m: {q: zone_scores(coll, q, m) for q in qids} for m in ('or', 'term')}
    feats = {q: extract(coll, q) for q in qids}
    bm25 = np.eye(len(FEATURE_NAMES))[FEATURE_NAMES.index('all:bm25')]
    summary = {'test': {}, 'weights_a': {}, 'runs': {}}

    def report(name, run, k):
        summary['test'][name] = evaluate(run, coll.qrels, cutoff=k)
        summary['runs'][name] = {q: r[:50] for q, r in run.items()}

    log('\nMethod A: zone weights learned on training queries')
    for mode in ('or', 'term'):
        for obj in ('mse', 'map'):
            w, W, values = learn_weights(coll, zone[mode], train, obj)
            name = f'A {mode}/{obj}'
            log(f'  {name}:', dict(zip(ZONES, w.round(2).tolist())))
            summary['weights_a'][name] = w.tolist()
            summary[f'grid {mode}/{obj}'] = [W.tolist(), values.tolist()]
            k = best_cutoff(coll.rank(score(zone[mode], w, train))[0], coll.qrels)
            run, scores = coll.rank(score(zone[mode], w, test))
            report(name, run, k)
            if name == 'A term/map':
                summary['k_a'] = k
                write_run(run, scores, 'results/method_a_test_results.txt', 'methodA')
                write_run(run, scores, 'results/method_a_test_relevant.txt', 'methodA', depth=k)

    log('\nMethod B: C chosen by 5-fold CV on training queries')
    for kind in ('pointwise', 'pairwise'):
        C = select_C(coll, feats, train, kind)
        w = train_svm(coll, feats, train, kind, C)
        k = best_cutoff(coll.rank(score(feats, w, train))[0], coll.qrels)
        run, scores = coll.rank(score(feats, w, test))
        report(f'B {kind} SVM', run, k)
        summary[f'C_{kind}'], summary[f'weights_{kind}'] = C, dict(zip(FEATURE_NAMES, w.tolist()))
        if kind == 'pairwise':
            summary['k_b'] = k
            write_run(run, scores, 'results/method_b_test_results.txt', 'methodB')
            write_run(run, scores, 'results/method_b_test_relevant.txt', 'methodB', depth=k)
    report('BM25 baseline', coll.rank(score(feats, bm25, test))[0], summary['k_b'])

    log(f'\nTest queries (P/R/F1 = top-k result set, k tuned on training: A k={summary["k_a"]}, B k={summary["k_b"]})')
    log(table(summary['test'], METRICS))

    log('\n5-fold CV MAP over all 225 queries')
    summary['cv'] = {
        'A term/map': cv_map(coll, zone['term'], qids, lambda tr: learn_weights(coll, zone['term'], tr)[0]),
        'B pointwise SVM': cv_map(coll, feats, qids, lambda tr: train_svm(coll, feats, tr, 'pointwise', summary['C_pointwise'])),
        'B pairwise SVM': cv_map(coll, feats, qids, lambda tr: train_svm(coll, feats, tr, 'pairwise', summary['C_pairwise'])),
        'BM25 baseline': cv_map(coll, feats, qids, lambda tr: bm25),
    }
    for name, m in summary['cv'].items():
        log(f'  {name:18s} {m:.4f}')

    # final models on all judged queries, used by `predict`
    w_a = learn_weights(coll, zone['term'], qids)[0]
    w_b = train_svm(coll, feats, qids, 'pairwise', summary['C_pairwise'])
    json.dump({'weights': w_a.tolist(), 'cutoff': best_cutoff(coll.rank(score(zone['term'], w_a, qids))[0], coll.qrels)},
              open('models/method_a.json', 'w'))
    json.dump({'weights': w_b.tolist(), 'cutoff': best_cutoff(coll.rank(score(feats, w_b, qids))[0], coll.qrels)},
              open('models/method_b.json', 'w'))
    log('\nFinal Method A weights:', dict(zip(ZONES, w_a.round(2).tolist())))
    json.dump(summary, open('results/summary.json', 'w'))


def predict(method, query_file, out_file):
    coll = Collection()
    coll.set_queries({q: v['text'] for q, v in parse_queries(query_file).items()})
    model = json.load(open(f'models/method_{method.lower()}.json'))
    feature_fn = (lambda q: zone_scores(coll, q)) if method.upper() == 'A' else (lambda q: extract(coll, q))
    run, scores = coll.rank({q: feature_fn(q) @ np.array(model['weights']) for q in coll.queries})
    write_run(run, scores, out_file, f'method{method.upper()}')
    print(f'wrote top-100 for {len(run)} queries to {out_file} (F1-optimal cut-off k={model["cutoff"]})')


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == 'predict':
        predict(*sys.argv[2:])
    else:
        experiments()
