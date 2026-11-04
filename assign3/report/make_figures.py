"""Report figures from results/summary.json (run main.py first)."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.parser_qq import parse_qrels
from src.evals import evaluate, average_precision

os.chdir(ROOT)
os.makedirs('figures', exist_ok=True)
S = json.load(open('results/summary.json'))
qrels = parse_qrels('data/cranqrel')
runs = {n: {int(q): r for q, r in run.items()} for n, run in S['runs'].items()}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})


def save(name):
    plt.tight_layout()
    plt.savefig(f'figures/{name}.png', dpi=150)
    plt.close()


# objective along the title/body edge of the weight simplex (author = bib = 0)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
for ax, obj in zip(axes, ('mse', 'map')):
    W, v = map(np.array, S[f'grid term/{obj}'])
    edge = (W[:, 1] == 0) & (W[:, 2] == 0)
    ax.plot(W[edge, 0], v[edge], 'o-', ms=3)
    ax.set(xlabel='g_title  (g_body = 1 - g_title)', ylabel='total squared error' if obj == 'mse' else 'training MAP')
save('method_a_grid')

plt.figure(figsize=(6, 2.3))
for i, (name, w) in enumerate(S['weights_a'].items()):
    plt.bar(np.arange(4) + (i - 1.5) * 0.2, w, 0.2, label=name)
plt.xticks(range(4), ['title', 'author', 'bib', 'body'])
plt.ylabel('learned weight')
plt.legend(fontsize=7)
save('method_a_weights')

# precision / recall / F1 against the cut-off k
curves = ['A term/map', 'B pointwise SVM', 'B pairwise SVM', 'BM25 baseline']
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
for name in curves:
    res = evaluate(runs[name], qrels, ks=range(1, 51))
    for ax, m in zip(axes, ('P', 'R', 'F1')):
        ax.plot(range(1, 51), [res[f'{m}@{k}'] for k in range(1, 51)], label=name)
        ax.set(title=f'{m}@k', xlabel='k')
axes[0].legend(fontsize=6)
save('prf_curves')

top = sorted(S['weights_pairwise'].items(), key=lambda kv: abs(kv[1]))[-18:]
plt.figure(figsize=(6, 3.5))
plt.barh(*zip(*top), color=['tab:blue' if v > 0 else 'tab:red' for _, v in top])
plt.xlabel('Ranking SVM weight')
save('method_b_weights')

plt.figure(figsize=(7, 2.3))
ap = {n: {q: average_precision(r, set(qrels[q])) for q, r in runs[n].items()} for n in curves}
for name in ('B pairwise SVM', 'A term/map'):
    d = sorted(ap[name][q] - ap['BM25 baseline'][q] for q in ap[name])
    plt.plot(d, label=f'{name}: {sum(x > 0 for x in d)} better, {sum(x < 0 for x in d)} worse')
plt.axhline(0, color='k', lw=0.6)
plt.xlabel('test queries (sorted)')
plt.ylabel('AP@50 - AP@50 of BM25')
plt.legend(fontsize=7)
save('per_query_ap')
