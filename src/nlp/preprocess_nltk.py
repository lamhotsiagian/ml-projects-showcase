import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP = set(stopwords.words("english"))

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"<[^>]+>", " ", s)        # remove html
    s = re.sub(r"[^a-z0-9\s]", " ", s)    # keep alnum
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_keep(s: str) -> str:
    tokens = word_tokenize(s)
    tokens = [t for t in tokens if t not in STOP and len(t) > 1]
    return " ".join(tokens)

def ensure_nltk():
    resources = ["punkt", "punkt_tab", "stopwords"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}" if "punkt" in r else f"corpora/{r}")
        except LookupError:
            nltk.download(r)

ensure_nltk()