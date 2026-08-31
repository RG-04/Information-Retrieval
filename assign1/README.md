# PA1 - Preprocessing, Indexing, and Boolean Search

This assignment implements a simple information retrieval pipeline in a single Python CLI file: [assign1/pa1.py](assign1/pa1.py). The Porter stemmer used is [assign1/porter.py](assign1/porter.py).

## Group

- QuadQuery

The default output filenames therefore follow the naming pattern:

- QuadQuery_processed.all
- QuadQuery_cran.index
- QuadQuery_result.txt

If you pass an explicit output path, the program will use that instead.

## Files

- `QuadQuery_porter.py`: Porter stemming implementation
- `QuadQuery_pa1.py`: preprocessing, index construction, and Boolean search CLI
- `cran.all.1400`: Cranfield document collection
- `stopwords.txt`: stopword list
- `QuadQuery_processed.all`: processed output file (generated)
- `QuadQuery_cran.index`: inverted index output file (generated)

## Pipeline

The implementation follows this order:

1. Tokenize using the regex `[A-Za-z]+`
2. Normalize to lowercase
3. Drop tokens of length 1
4. Remove stopwords
5. Apply Porter stemming

Important: stopwords must be removed before stemming because the stopword list is lowercase and unstemmed.

## Usage

```bash
# 1) preprocess the Cranfield collection
python3 QuadQuery_pa1.py preprocess cran.all.1400 stopwords.txt

# or with custom output name
python3 QuadQuery_pa1.py preprocess cran.all.1400 stopwords.txt QuadQuery_processed.all

# 2) build the inverted index
python3 QuadQuery_pa1.py index QuadQuery_processed.all

# or with custom output name
python3 QuadQuery_pa1.py index QuadQuery_processed.all QuadQuery_cran.index

# 3) run a 2-term Boolean query
python3 QuadQuery_pa1.py search QuadQuery_cran.index stopwords.txt "aerodynamic AND wing"
python3 QuadQuery_pa1.py search QuadQuery_cran.index stopwords.txt "boundary OR shock"

# or with custom output file
python3 QuadQuery_pa1.py search QuadQuery_cran.index stopwords.txt "aerodynamic AND wing" QuadQuery_result.txt
```

## Supported query format

The search command accepts exactly two query terms joined by either `AND` or `OR`:

```text
term1 AND term2
term1 OR term2
```

Examples:

```bash
python3 QuadQuery_pa1.py search QuadQuery_cran.index stopwords.txt "wing AND aerodynamic"
python3 QuadQuery_pa1.py search QuadQuery_cran.index stopwords.txt "boundary OR shock"
```

## Index format

The generated index file has:

- first line: `vocab_size, max_docid`
- remaining lines: `token postings`

Example:

```text
1234, 1400
aerodynam 2,7,15,88
wing 1,5,12,15
```

Postings are stored as sorted document IDs.

## Search method

The Boolean retrieval step uses sorted postings lists and a linear merge strategy:

- `merge_and` for logical AND
- `merge_or` for logical OR

This is efficient for sorted postings and avoids generic Python set operations.

## Notes

- Only `.T` (title) and `.W` (abstract) fields are indexed.
- `.A` and `.B` fields are ignored as required by the assignment.
- The project uses only the Python standard library.
- The stemmer follows the classic Porter stemming algorithm as provided in `QuadQuery_porter.py`.
