import numpy as np


def prf(ranked, relevant, k):
    hits = len(set(ranked[:k]) & relevant)
    p, r = hits / k, hits / len(relevant)
    return p, r, 2 * p * r / (p + r) if hits else 0.0


def average_precision(ranked, relevant):
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def ndcg10(ranked, gains):
    dcg = sum((2 ** gains.get(d, 0) - 1) / np.log2(i + 2) for i, d in enumerate(ranked[:10]))
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(sorted(gains.values(), reverse=True)[:10]))
    return dcg / idcg


def evaluate(run, qrels, ks=(5, 10, 20), cutoff=None):
    """Macro-averaged P/R/F1@k, MAP and nDCG@10. `cutoff` adds P/R/F1 of the returned top-`cutoff` set."""
    rows = []
    for q, ranked in run.items():
        rel = set(qrels[q])
        row = {'MAP': average_precision(ranked, rel), 'nDCG@10': ndcg10(ranked, {d: 5 - c for d, c in qrels[q].items()})}
        for k in ks:
            row[f'P@{k}'], row[f'R@{k}'], row[f'F1@{k}'] = prf(ranked, rel, k)
        if cutoff:
            row['P'], row['R'], row['F1'] = prf(ranked, rel, cutoff)
        rows.append(row)
    return {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}


def best_cutoff(run, qrels, max_k=50):
    """Cut-off k maximising mean F1@k (tuned on training queries)."""
    f1 = evaluate(run, qrels, ks=range(1, max_k + 1))
    return max(range(1, max_k + 1), key=lambda k: f1[f'F1@{k}'])


def write_run(run, scores, path, tag, depth=100):
    with open(path, 'w') as f:
        for q in sorted(run):
            for rank, d in enumerate(run[q][:depth], 1):
                f.write(f'{q} Q0 {d} {rank} {scores[q][d]:.6f} {tag}\n')


def table(results, metrics):
    w = max(map(len, results)) + 2
    lines = ['Run'.ljust(w) + ''.join(m.rjust(9) for m in metrics)]
    lines += [n.ljust(w) + ''.join(f'{r[m]:9.4f}' for m in metrics) for n, r in results.items()]
    return '\n'.join(lines)
