import re
from functools import lru_cache

from nltk.stem import PorterStemmer

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here
hers him his how i if in into is it its itself may me more most must my no nor not of off on once only or other
our out over own same shall she should so some such than that the their them then there these they this those
through thus to too under until up upon us very via was we were what when where whether which while who whom why
will with within without would you your
""".split())

stem = lru_cache(maxsize=None)(PorterStemmer().stem)


def tokenize(text):
    return [stem(t) for t in re.findall(r'[a-z0-9]+', text.lower())
            if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]
