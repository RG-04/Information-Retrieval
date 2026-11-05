# Learning to Rank on the Cranfield Collection

IR Programming Assignment III. Two learned ranking methods on the Cranfield collection:

- **Method A – Boolean weighted zone scoring.** Weights for the title, author, bibliography and body zones are learned by grid search.
- **Method B – Linear SVM.** Weights for 66 query-document features are learned with a pointwise linear SVM and a pairwise Ranking SVM.

Both methods are evaluated with precision, recall and F1 (also MAP and nDCG@10) on held-out test queries, and compared with a BM25 baseline.

---

## 1. Requirements

- Python **3.9 or newer** (tested with Python 3.14.5 on macOS)
- `pip` and the built-in `venv` module
- About 200 MB of disk space for the virtual environment

Python packages (listed in `requirements.txt`):

| Package | Used for | Tested version |
|---|---|---|
| numpy | term matrices, feature computation | 2.5.4 |
| scikit-learn | `LinearSVC` (Method B) | 1.9.1 |
| nltk | Porter stemmer only (no corpus downloads needed) | 3.10.4 |
| matplotlib | figures (optional script) | 3.11.2 |

`matplotlib` is only needed to regenerate the figures. The experiments themselves need only numpy, scikit-learn and nltk.

---

## 2. Directory layout

```
Quad_Query_A3/
├── README.md
├── requirements.txt
├── main.py                         # entry point: experiments + prediction
├── data/                           # Cranfield collection (must be present)
│   ├── cran.all.1400               #   1400 documents (.I .T .A .B .W)
│   ├── cran.qry                    #   225 queries
│   └── cranqrel                    #   relevance judgments
├── src/
│   ├── parser_qq.py                # parsers for the three data files
│   ├── preprocessing.py            # tokenising, stopwords, Porter stemming
│   ├── dataset.py                  # term-count matrices, idf, ranking, query splits
│   ├── evals.py                    # P/R/F1@k, MAP, nDCG@10, TREC run files
│   ├── method_a/
│   │   ├── boolean_features.py     # Boolean zone match scores
│   │   └── train.py                # grid search of zone weights
│   └── method_b/
│       ├── feature_extraction.py   # the 22 features x 3 streams
│       └── train.py                # pointwise SVM and Ranking SVM
├── report/
│   ├── report.pdf                  # the written report
│   ├── report.tex                  # LaTeX source of the report
│   └── make_figures.py             # regenerates figures/ (optional)
├── results/                        # created by main.py
├── models/                         # created by main.py
└── figures/                        # plots used in the report
```

---

## 3. Setting up the environment

Run every command from the project root (the folder that holds `main.py`).

### Step 1 – Open a terminal in the project folder

```bash
cd path/to/Quad_Query_A3
```

### Step 2 – Check your Python version

```bash
python3 --version
```

It must print 3.9 or higher. On Windows, use `py --version` instead, and write `py` wherever these instructions say `python3`.

### Step 3 – Create a virtual environment

```bash
python3 -m venv irenv
```

This creates a folder `irenv/` holding an isolated Python installation. If an old `irenv/` from another machine is already there, delete it first, because virtual environments can't be moved between machines:

```bash
rm -rf irenv
```

### Step 4 – Activate the virtual environment

macOS / Linux:

```bash
source irenv/bin/activate
```

Windows (Command Prompt):

```bat
irenv\Scripts\activate.bat
```

Windows (PowerShell):

```powershell
irenv\Scripts\Activate.ps1
```

Once it's active, your prompt starts with `(irenv)`. Every later command assumes the environment is active. To leave it at the end, run `deactivate`.

### Step 5 – Install the dependencies

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### Step 6 – Verify the installation

```bash
python -c "import numpy, sklearn, nltk, matplotlib; print('all packages OK')"
```

### Step 7 – Check the data files

```bash
ls data
```

You should see `cran.all.1400`, `cran.qry` and `cranqrel`. If you only have `cran.tar.gz`, extract it into `data/`:

```bash
tar -xzf cran.tar.gz -C data
```

---

## 4. Running the code

### Step 1 – Run all experiments

```bash
python main.py
```

The script will:

1. Parse and index the collection (1400 documents, 225 judged queries).
2. Split the queries 70/30 into 157 training and 68 test queries, using a fixed seed of 42.
3. **Method A:** learn the zone weights for two Boolean functions (`or`, `term`) × two objectives (`mse`, `map`) and evaluate each on the test queries.
4. **Method B:** choose C by 5-fold cross-validation on the training queries, then train and evaluate the pointwise SVM and the Ranking SVM.
5. Evaluate the BM25 baseline.
6. Run 5-fold cross-validation over all 225 queries for the main systems.
7. Retrain the final models on all 225 queries and save them to `models/`.

It prints the results tables and saves the same text to `results/experiment_log.txt`.

**Output files**

| File | Contents |
|---|---|
| `results/method_a_test_results.txt` | Method A: top 100 ranked documents per test query |
| `results/method_a_test_relevant.txt` | Method A: the documents returned as relevant (top-k, k tuned on training queries) |
| `results/method_b_test_results.txt` | Method B (Ranking SVM): top 100 per test query |
| `results/method_b_test_relevant.txt` | Method B: the documents returned as relevant |
| `results/experiment_log.txt` | learned weights, chosen C, all results tables |
| `results/summary.json` | all numbers, used to draw the figures |
| `models/method_a.json` | final Method A zone weights and cut-off k |
| `models/method_b.json` | final Ranking SVM weight vector and cut-off k |

Run files are in TREC format, one line per retrieved document:

```
<query_id> Q0 <doc_id> <rank> <score> <run_tag>
1 Q0 51 1 2.859279 methodB
```

### Step 2 – Rank new (unseen) queries

After Step 1 has created `models/`, rank any query file with either method:

```bash
python main.py predict B path/to/queries.qry results/run_B.txt
```

```bash
python main.py predict A path/to/queries.qry results/run_A.txt
```

- The query file must be in **Cranfield format**: each query starts with a `.I <id>` line, then a `.W` line, then the query text.
- Queries are numbered **1, 2, 3, … in the order they appear in the file**, which is the numbering `cranqrel` uses. The number after `.I` is ignored.
- The command writes two files in TREC format: the given output file with the top 100 documents per query, and a `_relevant` file next to it (e.g. `results/run_B_relevant.txt`) with only the top k documents per query, the set returned as relevant. k is the F1-optimal cut-off learned on the training data (stored in `models/`).
- A query with no term in the collection vocabulary (e.g. only stopwords) gets a score of 0 for every document, so its ranking is by document id.

To check that it works, rank the original queries:

```bash
python main.py predict B data/cran.qry results/run_B_all.txt
```

### Step 3 – (Optional) Regenerate the figures

```bash
python report/make_figures.py
```

This redraws the five PNG files in `figures/` from `results/summary.json`, so run it after `main.py`. It isn't needed to run or evaluate the code.

---

## 5. Expected results

On the 68 test queries (seed 42):

| System | P@10 | R@10 | F1@10 | MAP | nDCG@10 |
|---|---|---|---|---|---|
| Method A (term / map) | 0.187 | 0.349 | 0.221 | 0.270 | 0.337 |
| Method B pointwise SVM | 0.227 | 0.410 | 0.264 | 0.319 | 0.394 |
| Method B Ranking SVM | 0.229 | 0.413 | 0.268 | 0.325 | 0.399 |
| BM25 baseline | 0.224 | 0.394 | 0.261 | 0.315 | 0.383 |

Results are deterministic. Small differences (in the third decimal place) can appear with other numpy or scikit-learn versions.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run `python main.py` from the project root, not from inside `src/`. |
| `ModuleNotFoundError: numpy` / `sklearn` / `nltk` | The virtual environment isn't active (Step 4 of section 3), or the dependencies weren't installed (Step 5). |
| `FileNotFoundError: .../data/cran.all.1400` | The data files are missing or named differently. See Step 7 of section 3. |
| `FileNotFoundError: models/method_a.json` in `predict` | Run `python main.py` first. It creates the models. |
| `ConvergenceWarning` from scikit-learn | Harmless. It doesn't affect the results. |
| PowerShell won't run `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again. |
