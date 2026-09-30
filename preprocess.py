"""Text preprocessing for the FAQ chatbot.

Pipeline: lowercase -> remove punctuation -> tokenize -> remove stopwords -> lemmatize.
Required NLTK data is downloaded automatically the first time this module is imported.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# (resource path used by nltk.data.find, package name for nltk.download)
_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]


def ensure_nltk_data() -> None:
    """Download any missing NLTK resources (runs once, then is a no-op)."""
    for path, package in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()

# Standard English stopwords, minus negations, which change the meaning of a
# question ("package not received").
STOPWORDS = set(stopwords.words("english")) - {"not", "no", "nor"}

# Domain synonyms mapped to one canonical lemma so that paraphrases share
# vocabulary with the FAQ questions ("parcel" ~ "package", "money back" ~ "refund").
SYNONYMS = {
    "parcel": "package",
    "shipment": "package",
    "delivery": "shipping",
    "deliver": "ship",
    "dispatch": "ship",
    "arrive": "ship",
    "postage": "shipping",
    "abroad": "internationally",
    "overseas": "internationally",
    "international": "internationally",
    "worldwide": "internationally",
    "purchase": "order",
    "buy": "order",
    "money": "refund",
    "reimburse": "refund",
    "reimbursement": "refund",
    "guarantee": "warranty",
    "broken": "damage",
    "break": "damage",
    "damaged": "damage",
    "faulty": "defective",
    "never": "not",
    "login": "log",
    "signup": "sign",
    "register": "create",
    "passcode": "password",
    "pwd": "password",
    "installment": "emi",
    "instalment": "emi",
    "voucher": "discount",
    "coupon": "discount",
    "promo": "discount",
    "pay": "payment",
    "card": "payment",
    "remove": "delete",
    "close": "delete",
}

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to the WordNet POS the lemmatizer expects."""
    if treebank_tag.startswith("V"):
        return "v"
    if treebank_tag.startswith("J"):
        return "a"
    if treebank_tag.startswith("R"):
        return "r"
    return "n"


def preprocess(text: str) -> str:
    """Normalize text and return the cleaned tokens joined by spaces."""
    return " ".join(tokenize(text))


def tokenize(text: str) -> list[str]:
    """Return the list of cleaned, lemmatized tokens for ``text``."""
    text = text.lower()
    text = text.replace("'", " ")  # "can't" -> "can t", "I'm" -> "i m"
    text = _PUNCT_RE.sub(" ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and t not in STOPWORDS and len(t) > 1]
    tagged = nltk.pos_tag(tokens)
    lemmas = [_lemmatizer.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in tagged]
    return [SYNONYMS.get(lemma, lemma) for lemma in lemmas]


if __name__ == "__main__":
    for sample in ["How can I track my order?", "I've been charged TWICE for the same order!!", "Returning items that arrived damaged"]:
        print(f"{sample!r:50} -> {preprocess(sample)!r}")
