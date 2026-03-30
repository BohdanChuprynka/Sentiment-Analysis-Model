"""Text preprocessing pipeline for sentiment analysis on tweets.

Two strategies:
- preprocess_text()    — Heavy preprocessing for traditional ML (NB, Logistic Regression).
- preprocess_text_dl() — Light preprocessing for deep learning (USE, Hybrid).
"""

import string
import regex as re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

CHAT_WORDS = {
    "brb": "be right back",
    "btw": "by the way",
    "omg": "oh my goodness",
    "ttyl": "talk to you later",
    "omw": "on my way",
    "smh": "shaking my head",
    "smdh": "shaking my darn head",
    "lol": "laugh out loud",
    "tbd": "to be determined",
    "imho": "in my humble opinion",
    "imo": "in my opinion",
    "hmu": "hit me up",
    "iirc": "if I remember correctly",
    "lmk": "let me know",
    "og": "original gangsters",
    "ftw": "for the win",
    "nvm": "nevermind",
    "ootd": "outfit of the day",
    "ngl": "not gonna lie",
    "rq": "real quick",
    "iykyk": "if you know, you know",
    "ong": "on god/I swear",
    "yaaas": "yeah",
    "brt": "be right there",
    "sm": "so much",
    "ig": "i guess",
    "wya": "where you at",
    "istg": "i swear to god",
    "hbu": "how about you",
    "atm": "at the moment",
    "asap": "as soon as possible",
    "fyi": "for your information",
}


def delete_html_tags(text):
    """Remove HTML tags from text."""
    return re.sub(r"<.*?>", "", text)


def delete_url(text):
    """Remove URLs from text."""
    return re.sub(r"http\S+", "", text)


def remove_mention(text):
    """Replace @username mentions with @mention."""
    return re.sub(r"@\w+", "@mention", text)


def remove_punctuation(text):
    """Remove all punctuation from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def to_lowercase(text):
    """Convert text to lowercase."""
    return text.lower()


def replace_chat_words(text):
    """Expand common chat abbreviations to full phrases."""
    for short, full in CHAT_WORDS.items():
        text = text.replace(short, full)
    return text


def remove_stopwords(text):
    """Remove English stopwords from text."""
    words = text.split()
    filtered = [w for w in words if w not in STOP_WORDS]
    return " ".join(filtered)


def remove_duplicate_whitespace(text):
    """Collapse multiple whitespace characters into single spaces."""
    return " ".join(text.split())


def preprocess_text(text):
    """Heavy preprocessing for traditional ML models (NB, Logistic Regression).
    Strips everything down to clean tokens for bag-of-words approaches."""
    text = delete_html_tags(text)
    text = delete_url(text)
    text = remove_mention(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_chat_words(text)
    text = remove_stopwords(text)
    text = remove_duplicate_whitespace(text)
    return text


def preprocess_text_dl(text):
    """Light preprocessing for deep learning models.
    Preserves casing, punctuation patterns, and natural language structure
    that neural networks can learn from. Only removes noise (HTML, URLs)."""
    text = delete_html_tags(text)
    text = delete_url(text)
    text = re.sub(r"@\w+", "@user", text)
    text = replace_chat_words(text)
    text = remove_duplicate_whitespace(text)
    return text.strip()


def split_chars(text):
    """Split text into space-separated characters (for character-level embeddings)."""
    return " ".join(list(text))
