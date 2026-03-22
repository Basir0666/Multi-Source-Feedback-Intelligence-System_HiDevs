"""
Multi-Method Sentiment Analyser
VADER (rule-based, great for reviews) + TextBlob (ML-based) ensemble
with confidence scoring.
"""
import logging
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load heavy models so imports don't slow Streamlit startup
# ---------------------------------------------------------------------------
_vader_analyzer = None
_nltk_ready = False


def _get_vader():
    global _vader_analyzer, _nltk_ready
    if _vader_analyzer is None:
        import nltk
        if not _nltk_ready:
            nltk.download("vader_lexicon", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("wordnet", quiet=True)
            _nltk_ready = True
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


# ---------------------------------------------------------------------------
# Core analysis helpers
# ---------------------------------------------------------------------------

def _vader_scores(text: str) -> Dict:
    vader = _get_vader()
    return vader.polarity_scores(text)


def _textblob_scores(text: str) -> Dict:
    from textblob import TextBlob
    blob = TextBlob(text)
    return {
        "polarity": round(blob.sentiment.polarity, 4),
        "subjectivity": round(blob.sentiment.subjectivity, 4),
    }


def get_sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def compute_confidence(vader_compound: float, tb_polarity: float) -> float:
    """
    Confidence = agreement between both methods × signal strength.
    Ranges [0, 1].
    """
    agreement = 1.0 - abs(vader_compound - tb_polarity) / 2.0
    magnitude = (abs(vader_compound) + abs(tb_polarity)) / 2.0
    return round(min(agreement * 0.5 + magnitude * 0.5, 1.0), 4)


def analyze_text(text: str) -> Dict:
    """
    Full sentiment analysis for one text string.

    Returns a dict with:
        sentiment  : 'Positive' | 'Neutral' | 'Negative'
        score      : ensemble score [-1, 1]
        confidence : confidence score [0, 1]
        + raw VADER and TextBlob component scores
    """
    if not text or not text.strip():
        return {
            "sentiment": "Neutral", "score": 0.0, "confidence": 0.0,
            "vader_compound": 0.0, "vader_pos": 0.0,
            "vader_neg": 0.0, "vader_neu": 1.0,
            "textblob_polarity": 0.0, "textblob_subjectivity": 0.0,
        }

    v = _vader_scores(text)
    t = _textblob_scores(text)

    # Weighted ensemble: VADER 65% + TextBlob 35%
    combined = round(v["compound"] * 0.65 + t["polarity"] * 0.35, 4)

    return {
        "sentiment": get_sentiment_label(combined),
        "score": combined,
        "confidence": compute_confidence(v["compound"], t["polarity"]),
        "vader_compound": v["compound"],
        "vader_pos": v["pos"],
        "vader_neg": v["neg"],
        "vader_neu": v["neu"],
        "textblob_polarity": t["polarity"],
        "textblob_subjectivity": t["subjectivity"],
    }


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sentiment analysis to a full DataFrame.

    Reads 'clean_text' if present, otherwise 'text'.
    Appends sentiment columns in-place and returns the updated DataFrame.
    """
    if df.empty:
        return df

    df = df.copy()
    col = "clean_text" if "clean_text" in df.columns else "text"

    results = df[col].apply(analyze_text)
    sentiment_df = pd.DataFrame(list(results))
    df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()
    neu = (df["sentiment"] == "Neutral").sum()
    logger.info("Sentiment done — Pos: %d | Neu: %d | Neg: %d", pos, neu, neg)
    return df
