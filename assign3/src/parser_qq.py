"""
parser.py
Parsers for the Cranfield collection files: cran.all.1400, cran.qry, cranqrel.
"""
import re
from collections import defaultdict


def parse_documents(path):
    """
    Parse cran.all.1400 into a dict: {docid(int): {'title':str,'author':str,'bib':str,'text':str}}
    Fields are tagged with .I (id), .T (title), .A (author), .B (bibliography), .W (text/abstract)
    """
    with open(path, 'r', encoding='latin-1') as f:
        content = f.read()

    # Split on the .I marker (each document starts with .I <id>)
    chunks = re.split(r'\n(?=\.I )', content)
    docs = {}
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r'\.I\s+(\d+)', chunk)
        if not m:
            continue
        docid = int(m.group(1))

        def extract(tag_start, tag_end_pattern):
            pat = re.compile(re.escape(tag_start) + r'\n(.*?)(?=' + tag_end_pattern + r'|\Z)', re.S)
            mm = pat.search(chunk)
            return mm.group(1).strip() if mm else ''

        title = extract('.T', r'\n\.A')
        author = extract('.A', r'\n\.B')
        bib = extract('.B', r'\n\.W')
        text = extract('.W', r'\Z')

        # Normalize whitespace (collapse newlines within a field)
        def norm(s):
            return re.sub(r'\s+', ' ', s).strip()

        docs[docid] = {
            'title': norm(title),
            'author': norm(author),
            'bib': norm(bib),
            'text': norm(text),
        }
    return docs


def parse_queries(path):
    """
    Parse cran.qry into a dict: {qid(int): text(str)}
    Queries are tagged with .I (id) and .W (text).

    IMPORTANT: the .I numbers inside cran.qry are NOT a clean 1..225
    sequence (they skip values). The relevance judgments in cranqrel
    instead reference queries by their 1-based POSITION/order of
    appearance in cran.qry, not by the .I label. So we deliberately
    ignore the .I value and assign sequential ids 1..225 by order of
    appearance, which is what makes qrels line up with the queries.
    """
    with open(path, "r", encoding="latin-1") as f:
        content = f.read()

    chunks = re.split(r"\n(?=\.I )", content)

    queries = {}

    for position, chunk in enumerate(chunks, start=1):
        chunk = chunk.strip()

        if not chunk:
            continue

        match = re.match(r"\.I\s+(\d+)", chunk)
        if not match:
            continue

        original_id = int(match.group(1))

        wm = re.search(r"\.W\n(.*)", chunk, re.S)
        text = wm.group(1).strip() if wm else ""
        text = re.sub(r"\s+", " ", text).strip()

        queries[position] = {
            "original_id": original_id,
            "text": text
        }

    return queries


def parse_qrels(path, drop_placeholder=True):
    """
    Parse cranqrel into a dict: {qid(int): {docid(int): relevance_code(int)}}
    Relevance codes (Cleverdon scale): 1=best ... 4=minimum interest.
    A code of -1 is a per-query placeholder line present in the raw file
    (one per query, docid==itself is NOT guaranteed but the code is a marker,
    not a genuine positive judgment) and is dropped by default so it is
    never counted as relevant.
    """
    qrels = defaultdict(dict)
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            qid, docid, code = int(parts[0]), int(parts[1]), int(parts[2])
            if code == -1 and drop_placeholder:
                continue
            qrels[qid][docid] = code
    return dict(qrels)


if __name__ == '__main__':
    docs = parse_documents("../data/cran.all.1400")
    queries = parse_queries("../data/cran.qry")
    qrels = parse_qrels("../data/cranqrel")

    print("Documents:", len(docs))
    print("Queries:", len(queries))
    print("Queries with judgments:", len(qrels))

    print("Document ID range:", min(docs), max(docs))
    print("Query ID range:", min(queries), max(queries))
    print("QREL query ID range:", min(qrels), max(qrels))

    # Check whether all qrel document IDs exist in the corpus
    qrel_doc_ids = {
        docid
        for judgments in qrels.values()
        for docid in judgments
    }

    print("QREL documents missing from corpus:",
        qrel_doc_ids - docs.keys())

    # Check whether all qrel query IDs exist
    print("QREL queries missing from parsed queries:",
        qrels.keys() - queries.keys())
    
