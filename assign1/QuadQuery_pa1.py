import re
import argparse
from collections import defaultdict

from QuadQuery_porter import stem_word

GROUPNAME = "QuadQuery"

TOKEN_RE = re.compile(r"[A-Za-z]+")


# 1. PREPROCESSING

# The four required steps: tokenize, normalize, remove stopwords, stem.

# The assignment lists them as tokenize / stem / stopword-remove /
# normalize, but that ordering doesn't actually work: the stopword list
# is lowercase and unstemmed, so stopwords have to be removed *before*
# stemming, and normalizing first keeps the stemmer's input clean. So the
# real order below is: tokenize -> normalize -> remove stopwords -> stem.

def tokenize(text):
    """Required step 1: split raw text into a list of alphabetic tokens."""
    return TOKEN_RE.findall(text)


def normalize(tokens):
    """Required step 2: lowercase tokens, drop single-character noise."""
    return [t.lower() for t in tokens if len(t) >= 2]


def remove_stopwords(tokens, stopwords):
    """Required step 3: drop any token that appears in the stopword set."""
    return [t for t in tokens if t not in stopwords]


def stem_tokens(tokens):
    """Required step 4: reduce each token to its Porter stem."""
    return [stem_word(t) for t in tokens]


# ---- helpers used only by cmd_preprocess -----------------------------

def _load_stopwords(path):
    """Read a stopword file (one word per line) into a lowercase set."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return set(line.strip().lower() for line in f if line.strip())


def _preprocess_text(text, stopwords):
    """Chain the four required steps together for one chunk of text."""
    tokens = tokenize(text)
    tokens = normalize(tokens)
    tokens = remove_stopwords(tokens, stopwords)
    tokens = stem_tokens(tokens)
    return tokens


def _parse_cran(path):
    """Parse cran.all into a list of (docid, title_text, abstract_text).

    Only .T (title) and .W (abstract) are kept
    """
    docs = []
    docid = None
    field = None
    title_lines, abstract_lines = [], []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith(".I "):
                # Starting a new document - flush the previous one first.
                if docid is not None:
                    docs.append((docid, " ".join(title_lines), " ".join(abstract_lines)))
                docid = int(line.split()[1])
                title_lines, abstract_lines = [], []
                field = None
            elif line.startswith(".T"):
                field = "T"
            elif line.startswith(".A"):
                field = "A"
            elif line.startswith(".B"):
                field = "B"
            elif line.startswith(".W"):
                field = "W"
            else:
                if field == "T":
                    title_lines.append(line)
                elif field == "W":
                    abstract_lines.append(line)
                # .A / .B content is intentionally ignored.

    if docid is not None:
        docs.append((docid, " ".join(title_lines), " ".join(abstract_lines)))

    return docs


def cmd_preprocess(args):
    stopwords = _load_stopwords(args.stopwords_file)
    docs = _parse_cran(args.cran_file)

    with open(args.out_file, "w", encoding="utf-8") as out:
        for docid, title, abstract in docs:
            tokens = _preprocess_text(title + " " + abstract, stopwords)
            out.write(f".I {docid}\n")
            out.write(".S\n")
            out.write(" ".join(tokens) + "\n")

    print(f"Processed {len(docs)} documents -> {args.out_file}")


# 2. INDEXING
def build_index(processed_path):
    """Required: build an inverted index (token -> sorted list of docids)
    from a *_processed.all file.
    """
    index = defaultdict(set)
    max_docid = 0

    for docid, tokens in _read_processed(processed_path):
        max_docid = max(max_docid, docid)
        for tok in tokens:
            index[tok].add(docid)

    for tok in index:
        index[tok] = sorted(index[tok])

    return index, max_docid


# ---- helpers used only by cmd_index -----------------------------------

def _read_processed(path):
    """Yield (docid, tokens) pairs from a *_processed.all file.

    Each document is exactly one .I line, one .S line, and one line of
    space-separated tokens, so we just wait for the token line and yield
    it against whatever docid we most recently saw.
    """
    docid = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(".I "):
                docid = int(line.split()[1])
            elif line.startswith(".S"):
                continue
            elif docid is not None and line.strip():
                yield docid, line.split()
                docid = None  # each doc has exactly one token line


def _write_index(index, max_docid, out_path):
    """Write the index file: header line, then one 'token postings' line each.
    """
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"{len(index)}, {max_docid}\n")
        for tok in sorted(index.keys()):
            postings = ",".join(str(d) for d in index[tok])
            out.write(f"{tok} {postings}\n")


def cmd_index(args):
    index, max_docid = build_index(args.processed_file)
    _write_index(index, max_docid, args.out_file)
    print(f"Indexed {len(index)} unique tokens, max docid {max_docid} -> {args.out_file}")


# 3. BOOLEAN SEARCH

def merge_and(list1, list2):
    """(bonus): intersect two sorted postings lists in one linear pass."""
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1
    return result


def merge_or(list1, list2):
    """(bonus): union two sorted postings lists in one linear pass."""
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    # Whichever list still has leftovers is already sorted, so just append it.
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result


# ---- helpers used only by cmd_search -----------------------------------

def _load_index(path):
    """Load an index file back into memory as (index_dict, vocab_size, max_docid)."""
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        vocab_size, max_docid = (int(x.strip()) for x in first_line.split(","))

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tok, postings = line.split(" ", 1)
            ids = [int(x) for x in postings.split(",") if x]
            index[tok] = ids

    return index, vocab_size, max_docid


def _stem_query_term(term, stopwords):
    """Run a single query word through the same normalize/stopword/stem
    pipeline used for indexing, so it matches what's actually in the index.
    Returns None if the term is empty or turns out to be a stopword.
    """
    tokens = normalize(tokenize(term))
    tokens = remove_stopwords(tokens, stopwords)
    tokens = stem_tokens(tokens)
    return tokens[0] if tokens else None


def _parse_query(query):
    """Split a query string of the form 'term1 AND term2' / 'term1 OR term2'."""
    parts = query.strip().split()
    if len(parts) != 3 or parts[1].upper() not in ("AND", "OR"):
        raise ValueError('Query must be of the form: "term1 AND term2" or "term1 OR term2"')
    return parts[0], parts[1].upper(), parts[2]


def _run_query(query, index, stopwords):
    term1_raw, connective, term2_raw = _parse_query(query)
    term1 = _stem_query_term(term1_raw, stopwords)
    term2 = _stem_query_term(term2_raw, stopwords)

    list1 = index.get(term1, [])
    list2 = index.get(term2, [])

    return merge_and(list1, list2) if connective == "AND" else merge_or(list1, list2)


def cmd_search(args):
    index, vocab_size, max_docid = _load_index(args.index_file)
    stopwords = _load_stopwords(args.stopwords_file)
    result = _run_query(args.query, index, stopwords)

    with open(args.out_file, "w", encoding="utf-8") as out:
        out.write(",".join(str(d) for d in result) + "\n")

    print(f"Query: {args.query}")
    print(f"Matched {len(result)} documents -> {args.out_file}")
    print(result)


# CLI
# Every out_file defaults to a GROUPNAME-prefixed name
# You can still override it by passing a path explicitly.

def main():
    parser = argparse.ArgumentParser(description="PA1: preprocessing, indexing, Boolean search")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("preprocess", help="Tokenize/normalize/stopword-remove/stem cran.all")
    p1.add_argument("cran_file")
    p1.add_argument("stopwords_file")
    p1.add_argument("out_file", nargs="?", default=f"{GROUPNAME}_processed.all")
    p1.set_defaults(func=cmd_preprocess)

    p2 = sub.add_parser("index", help="Build inverted index from processed file")
    p2.add_argument("processed_file")
    p2.add_argument("out_file", nargs="?", default=f"{GROUPNAME}_cran.index")
    p2.set_defaults(func=cmd_index)

    p3 = sub.add_parser("search", help="Run a 2-term Boolean AND/OR query")
    p3.add_argument("index_file")
    p3.add_argument("stopwords_file")
    p3.add_argument("query")
    p3.add_argument("out_file", nargs="?", default=f"{GROUPNAME}_result.txt")
    p3.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()