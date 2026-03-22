"""
Feedback Categoriser & Priority Scorer
Auto-tags feedback by type, topic, and urgency using keyword rules
"""
import re
import logging
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category keyword rules
# ---------------------------------------------------------------------------
CATEGORY_RULES: Dict[str, List[str]] = {
    "Bug / Crash": [
        "crash", "bug", "error", "broken", "freeze", "freezes", "freezing",
        "not working", "does not work", "doesn't work", "failed", "failure",
        "glitch", "stuck", "hangs", "hang", "blank screen", "force close",
        "won't open", "keeps closing", "corrupted", "unresponsive",
    ],
    "Performance": [
        "slow", "lag", "lagging", "speed", "fast", "battery", "drain",
        "memory", "loading", "load time", "takes forever", "too long",
        "delay", "performance", "heavy", "resource", "optimize",
    ],
    "UI / Design": [
        "design", "interface", "ui", "ux", "layout", "button", "icon",
        "color", "theme", "dark mode", "font", "ugly", "beautiful", "nice",
        "intuitive", "confusing", "cluttered", "simple", "clean", "look",
    ],
    "Feature Request": [
        "add", "feature", "wish", "would be nice", "please add", "suggestion",
        "request", "option", "ability", "allow", "support", "integrate",
        "need", "missing", "should have", "enhancement", "improvement",
    ],
    "Pricing / Subscription": [
        "price", "pricing", "cost", "expensive", "cheap", "subscription",
        "premium", "free", "pay", "payment", "worth", "value", "refund",
        "money", "billing", "charge", "cancel", "trial",
    ],
    "Customer Support": [
        "support", "help", "response", "reply", "team", "service",
        "contact", "ignored", "no response", "customer service",
        "reached out", "ticket", "assist", "resolve", "unresponsive",
    ],
    "Positive Experience": [
        "love", "great", "excellent", "amazing", "fantastic", "perfect",
        "awesome", "best", "wonderful", "superb", "recommend", "happy",
        "satisfied", "thank", "five stars", "brilliant", "outstanding",
    ],
}

# ---------------------------------------------------------------------------
# Urgency / priority keyword boosters
# ---------------------------------------------------------------------------
_URGENCY = {
    "critical": [
        "crash", "broken", "data loss", "security", "hack", "stolen",
        "emergency", "urgent", "critical", "serious", "dangerous", "scam",
    ],
    "high": [
        "bug", "error", "not working", "fails", "freeze", "can't login",
        "unable to", "cannot", "corrupted", "lost",
    ],
    "medium": [
        "issue", "problem", "weird", "strange", "incorrect", "wrong", "slow",
    ],
    "low": ["suggestion", "request", "wish", "nice to have", "minor"],
}


def categorize_text(text: str) -> str:
    """Assign the best-matching category using keyword scoring."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}

    for category, keywords in CATEGORY_RULES.items():
        hit = sum(1 for kw in keywords if kw in text_lower)
        if hit:
            scores[category] = hit

    return max(scores, key=scores.get) if scores else "General"


def score_priority(text: str, sentiment_score: float) -> Tuple[str, int]:
    """
    Compute priority 0-100 from sentiment + keyword urgency.

    Returns (priority_label, priority_score).
    """
    text_lower = text.lower()
    pts = 50

    # Sentiment contribution
    if sentiment_score < -0.6:
        pts += 30
    elif sentiment_score < -0.3:
        pts += 20
    elif sentiment_score < 0:
        pts += 10

    # Keyword contribution
    for level, kws in _URGENCY.items():
        hits = sum(1 for kw in kws if kw in text_lower)
        if hits:
            pts += {"critical": 30, "high": 20, "medium": 10, "low": -5}[level]

    pts = max(0, min(100, pts))

    if pts >= 80:
        label = "🔴 Critical"
    elif pts >= 60:
        label = "🟠 High"
    elif pts >= 40:
        label = "🟡 Medium"
    else:
        label = "🟢 Low"

    return label, pts


def extract_top_keywords(texts: List[str], top_n: int = 25) -> List[Tuple[str, int]]:
    """Return the top-N most frequent meaningful words across all texts."""
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "it", "this", "that",
        "are", "be", "has", "had", "have", "not", "i", "my", "me", "we",
        "you", "your", "he", "she", "they", "app", "use", "using", "used",
        "can", "would", "just", "so", "get", "got", "its", "also", "very",
        "will", "one", "all", "more", "their", "been", "when", "what", "which",
    }
    word_counts: Counter = Counter()
    for text in texts:
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        word_counts.update(w for w in words if w not in STOPWORDS)
    return word_counts.most_common(top_n)


def categorize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'category', 'priority_label', and 'priority_score' columns.

    Args:
        df: DataFrame with 'clean_text' and optionally 'score' columns

    Returns:
        Updated DataFrame
    """
    if df.empty:
        return df

    df = df.copy()
    col = "clean_text" if "clean_text" in df.columns else "text"

    df["category"] = df[col].apply(categorize_text)

    if "score" in df.columns:
        priority_results = df.apply(
            lambda row: score_priority(row[col], row["score"]), axis=1
        )
        df["priority_label"] = [r[0] for r in priority_results]
        df["priority_score"] = [r[1] for r in priority_results]
    else:
        df["priority_label"] = "🟡 Medium"
        df["priority_score"] = 50

    logger.info(
        "Categorised %d records into %d categories",
        len(df), df["category"].nunique(),
    )
    return df
