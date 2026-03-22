"""
Text Cleaner & Normaliser
Removes noise, expands contractions, and normalises feedback text
"""
import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)

_CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "they're": "they are", "we're": "we are", "you're": "you are",
    "i'm": "i am", "he's": "he is", "she's": "she is",
    "i've": "i have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "we'll": "we will", "they'll": "they will",
    "i'd": "i would", "we'd": "we would",
}


def expand_contractions(text: str) -> str:
    for contracted, expanded in _CONTRACTIONS.items():
        text = text.replace(contracted, expanded)
    return text


def clean_text(text: str) -> str:
    """
    Clean a single feedback string.

    Steps:
        1. Lowercase
        2. Remove URLs and emails
        3. Expand contractions
        4. Strip HTML tags
        5. Remove non-alphanumeric chars (keep sentence punctuation)
        6. Collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = expand_contractions(text)
    text = re.sub(r"<[^>]+>", " ", text)                   # HTML tags
    text = re.sub(r"[^\w\s\.\!\?]", " ", text)             # keep word chars + basic punct
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text to the entire DataFrame.

    Adds a 'clean_text' column and drops rows whose cleaned text
    is shorter than 4 characters.

    Args:
        df: DataFrame with a 'text' column

    Returns:
        DataFrame with added 'clean_text' column
    """
    if df.empty:
        return df

    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    before = len(df)
    df = df[df["clean_text"].str.len() >= 4].reset_index(drop=True)
    logger.info("Cleaned %d records (%d dropped as empty)", len(df), before - len(df))
    return df
