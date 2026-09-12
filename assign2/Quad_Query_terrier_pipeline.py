"""
terrier_pipeline.py
PyTerrier version of the PA2 pipeline: indexing, ranked retrieval with
TF_IDF and BM25 (both sparse vector-space-family models, per the
assignment's constraints), parameter tuning, and TREC-style evaluation
on the Cranfield collection.

Requirements (on a machine with normal internet access):
    pip install python-terrier
    (needs a JVM; PyTerrier downloads its Terrier jar from Maven Central
    the first time pt.init() runs, so you need unrestricted internet the
    first time you run this)

This mirrors src/run_experiments.py (the from-scratch implementation) so
you can sanity-check the two against each other. It reuses parser.py from
that folder for reading cran.all.1400 / cran.qry / cranqrel, including the
query-ID-by-position fix (see Section 3.1 of the report) — Terrier expects
qids/docnos as strings, so we cast them.

NOTE: This file was written against PyTerrier 1.1.2's documented API
(class signatures verified locally) but could not be executed end-to-end
in the sandbox this assignment was built in, because PyTerrier's runtime
jar-resolution step needs to reach Maven Central, which was network-
blocked there. Run and check the output before you trust it.
"""
import os
import re
import sys
import time
import json
import pandas as pd

sys.path.insert(0, '../src')  # reuse the existing parser.py
from Quad_Query_parser import parse_documents, parse_queries, parse_qrels  # noqa: E402

import pyterrier as pt

DATA_DIR = './data'
INDEX_DIR = './terrier_index'
RESULTS_DIR = './results_pyterrier'
os.makedirs(RESULTS_DIR, exist_ok=True)


def clean_query(text):
    """Strip characters that Terrier's query parser treats as syntax
    (parentheses, colons, etc.) so natural-language queries don't error out."""
    return re.sub(r'[^a-zA-Z0-9 ]', ' ', text)


def build_dataframes():
    docs = parse_documents(f'{DATA_DIR}/cran.all.1400')
    queries = parse_queries(f'{DATA_DIR}/cran.qry')
    qrels = parse_qrels(f'{DATA_DIR}/cranqrel')

    docs_df = pd.DataFrame([
        {'docno': str(docid), 'text': d['title'] + ' ' + d['text']}
        for docid, d in docs.items()
    ])

    topics_df = pd.DataFrame([
        {'qid': str(qid), 'query': clean_query(text)}
        for qid, text in queries.items()
    ])

    qrels_rows = []
    for qid, docmap in qrels.items():
        for docid, code in docmap.items():
            qrels_rows.append({'qid': str(qid), 'docno': str(docid), 'label': 5 - code})  # graded gain, higher=better
    qrels_df = pd.DataFrame(qrels_rows)

    return docs_df, topics_df, qrels_df


def build_index(docs_df, use_stopwords=True, use_stemming=True, index_dir=INDEX_DIR):
    """
    use_stopwords / use_stemming toggle Terrier's English stopword list and
    Porter stemmer at indexing time (mirrors our own preprocess.py pipeline
    and is what the preprocessing-ablation experiment sweeps over). The
    tokeniser itself (word splitting) stays at Terrier's default.
    """
    tag = f"sw{int(use_stopwords)}_st{int(use_stemming)}"
    path = f'{index_dir}_{tag}'
    indexer = pt.IterDictIndexer(
        path, meta={'docno': 20}, overwrite=True,
        stemmer=pt.TerrierStemmer.porter if use_stemming else pt.TerrierStemmer.none,
        stopwords=pt.TerrierStopwords.terrier if use_stopwords else pt.TerrierStopwords.none,
    )
    t0 = time.time()
    index_ref = indexer.index(docs_df.to_dict(orient='records'))
    indexing_time = time.time() - t0
    return pt.IndexFactory.of(index_ref), indexing_time


def main():
    print('Initialising PyTerrier (downloads Terrier jar on first run)...')
    if not pt.started():
        pt.init()

    docs_df, topics_df, qrels_df = build_dataframes()
    print(f'{len(docs_df)} documents, {len(topics_df)} queries, {qrels_df.qid.nunique()} queries with judgments')

    eval_metrics = ['map', 'P_5', 'P_10', 'ndcg_cut_10', 'Rprec']

    # ------------------------------------------------------------------
    # Experiment 1: preprocessing ablation (stopword removal x Porter stemming)
    # ------------------------------------------------------------------
    print('\n=== Experiment 1: preprocessing ablation ===')
    pp_rows = []
    for use_stopwords in [True, False]:
        for use_stemming in [True, False]:
            index_pp, idx_time = build_index(docs_df, use_stopwords=use_stopwords, use_stemming=use_stemming)
            bm25 = pt.terrier.Retriever(index_pp, wmodel='BM25')
            t0 = time.time()
            exp = pt.Experiment([bm25], topics_df, qrels_df, eval_metrics=eval_metrics,
                                 names=[f'BM25_sw{use_stopwords}_st{use_stemming}'])
            search_time = time.time() - t0
            row = exp.iloc[0].to_dict()
            row['stopwords_removed'] = use_stopwords
            row['stemming'] = use_stemming
            row['indexing_time_s'] = round(idx_time, 4)
            row['search_time_s'] = round(search_time, 4)
            pp_rows.append(row)
            print(row)
    pd.DataFrame(pp_rows).to_csv(f'{RESULTS_DIR}/preprocessing_ablation.csv', index=False)

    # Use the stemmed + stopword-removed index from here on, matching the
    # "best preprocessing" choice found in the from-scratch experiments.
    index, indexing_time = build_index(docs_df, use_stopwords=True, use_stemming=True)

    # ------------------------------------------------------------------
    # Experiment 2: TF_IDF vs BM25 (default params)
    # ------------------------------------------------------------------
    print('\n=== Experiment 2: TF_IDF vs BM25 (defaults) ===')
    tfidf = pt.terrier.Retriever(index, wmodel='TF_IDF')
    bm25_default = pt.terrier.Retriever(index, wmodel='BM25')
    exp2 = pt.Experiment(
        [tfidf, bm25_default], topics_df, qrels_df,
        eval_metrics=eval_metrics, names=['TF_IDF', 'BM25_default']
    )
    print(exp2)
    exp2.to_csv(f'{RESULTS_DIR}/tfidf_vs_bm25.csv', index=False)

    # ------------------------------------------------------------------
    # Experiment 3: BM25 parameter tuning (k1, b grid) via Terrier controls
    # ------------------------------------------------------------------
    print('\n=== Experiment 3: BM25 parameter tuning ===')
    k1_values = [1.0, 1.2, 1.5, 1.8, 2.0]
    b_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    systems, names = [], []
    for k1 in k1_values:
        for b in b_values:
            r = pt.terrier.Retriever(
                index, wmodel='BM25',
                controls={'bm25.k_1': str(k1), 'bm25.b': str(b)}
            )
            systems.append(r)
            names.append(f'BM25_k1={k1}_b={b}')

    t0 = time.time()
    exp3 = pt.Experiment(systems, topics_df, qrels_df, eval_metrics=eval_metrics, names=names)
    grid_time = time.time() - t0
    exp3 = exp3.sort_values('map', ascending=False)
    print(exp3.head(10))
    exp3.to_csv(f'{RESULTS_DIR}/bm25_grid.csv', index=False)
    print(f'Grid search over {len(systems)} configs took {grid_time:.2f}s '
          f'({grid_time/len(systems)/len(topics_df)*1000:.2f} ms/query/config)')

    best_row = exp3.iloc[0]
    best_name = best_row['name']
    best_k1 = float(best_name.split('k1=')[1].split('_')[0])
    best_b = float(best_name.split('b=')[1])
    print(f'Best BM25 config: k1={best_k1}, b={best_b}, MAP={best_row["map"]:.4f}')

    # ------------------------------------------------------------------
    # Write TREC-format run files for the test queries (best BM25, TF_IDF)
    # ------------------------------------------------------------------
    best_bm25 = pt.terrier.Retriever(index, wmodel='BM25', controls={'bm25.k_1': str(best_k1), 'bm25.b': str(best_b)})
    best_bm25_results = best_bm25.transform(topics_df)
    pt.io.write_results(best_bm25_results, f'{RESULTS_DIR}/BestBM25.trec', format='trec', run_name=f'bm25_k1_{best_k1}_b_{best_b}')

    tfidf_results = tfidf.transform(topics_df)
    pt.io.write_results(tfidf_results, f'{RESULTS_DIR}/TFIDF.trec', format='trec', run_name='tfidf')

    summary = {
        'indexing_time_s': indexing_time,
        'best_bm25_params': {'k1': best_k1, 'b': best_b},
        'best_bm25_MAP': float(best_row['map']),
        'tfidf_MAP': float(exp2[exp2.name == 'TF_IDF']['map'].iloc[0]),
    }
    with open(f'{RESULTS_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n=== DONE. Results written to results_pyterrier/ ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
