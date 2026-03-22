"""
Trend Detector
Identifies emerging patterns, sentiment trends, anomalies, and rising issues
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rolling sentiment helpers
# ---------------------------------------------------------------------------

def compute_rolling_sentiment(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Aggregate feedback into daily buckets and compute a rolling mean.

    Args:
        df     : Processed feedback DataFrame
        window : Rolling window size in days

    Returns:
        Daily DataFrame with columns:
        date_only, avg_sentiment, count, positive, negative, neutral, rolling_sentiment
    """
    if df.empty or "date" not in df.columns or "score" not in df.columns:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy["date_only"] = pd.to_datetime(df_copy["date"]).dt.normalize()

    daily = (
        df_copy.groupby("date_only")
        .agg(
            avg_sentiment=("score", "mean"),
            count=("score", "count"),
            positive=("sentiment", lambda x: (x == "Positive").sum()),
            negative=("sentiment", lambda x: (x == "Negative").sum()),
            neutral=("sentiment", lambda x: (x == "Neutral").sum()),
        )
        .reset_index()
        .sort_values("date_only")
    )

    w = min(window, len(daily))
    daily["rolling_sentiment"] = (
        daily["avg_sentiment"].rolling(window=w, min_periods=1).mean()
    )
    return daily


# ---------------------------------------------------------------------------
# Trend direction
# ---------------------------------------------------------------------------

def detect_sentiment_trend(df: pd.DataFrame) -> Dict:
    """
    Fit a linear trend to rolling sentiment and return direction stats.

    Returns dict with keys:
        direction, slope, magnitude, description, current_avg,
        period_start_avg, change
    """
    daily = compute_rolling_sentiment(df)

    if len(daily) < 3:
        return {
            "direction": "stable",
            "slope": 0.0,
            "magnitude": 0.0,
            "description": "Not enough data to detect a trend (need ≥ 3 days).",
            "current_avg": 0.0,
            "period_start_avg": 0.0,
            "change": 0.0,
        }

    y = daily["rolling_sentiment"].values
    x = np.arange(len(y))
    slope = float(np.polyfit(x, y, 1)[0])

    if slope > 0.005:
        direction, emoji = "improving", "📈"
    elif slope < -0.005:
        direction, emoji = "declining", "📉"
    else:
        direction, emoji = "stable", "➡️"

    description = (
        f"{emoji} Sentiment is {direction} "
        f"(slope={slope:+.5f}/day). "
        f"Period change: {y[-1] - y[0]:+.3f}"
    )

    return {
        "direction": direction,
        "slope": round(slope, 6),
        "magnitude": round(abs(slope), 6),
        "description": description,
        "current_avg": round(float(y[-1]), 4),
        "period_start_avg": round(float(y[0]), 4),
        "change": round(float(y[-1] - y[0]), 4),
    }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Flag days with abnormally high feedback volume using Z-score.

    Args:
        df          : Processed feedback DataFrame
        z_threshold : Z-score above which a day is flagged

    Returns:
        DataFrame of anomalous days (may be empty)
    """
    daily = compute_rolling_sentiment(df)
    if len(daily) < 5:
        return pd.DataFrame()

    mu = daily["count"].mean()
    sigma = daily["count"].std()
    if sigma == 0:
        return pd.DataFrame()

    daily["z_score"] = ((daily["count"] - mu) / sigma).round(2)
    return daily[daily["z_score"].abs() > z_threshold].copy()


# ---------------------------------------------------------------------------
# Trending issues
# ---------------------------------------------------------------------------

def get_trending_issues(
    df: pd.DataFrame, days: int = 30, top_n: int = 5
) -> List[Dict]:
    """
    Identify the most-reported negative categories in the last N days.

    Returns a list of dicts with:
        category, count, avg_sentiment, avg_priority
    """
    if df.empty or "sentiment" not in df.columns:
        return []

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent = df[df["date"] >= cutoff]
    if recent.empty:
        recent = df  # fallback: use all data

    neg = recent[recent["sentiment"] == "Negative"]
    if neg.empty:
        return []

    agg = (
        neg.groupby("category")
        .agg(count=("text", "count"), avg_sentiment=("score", "mean"),
             avg_priority=("priority_score", "mean"))
        .sort_values("count", ascending=False)
        .head(top_n)
    )

    return [
        {
            "category": cat,
            "count": int(row["count"]),
            "avg_sentiment": round(row["avg_sentiment"], 3),
            "avg_priority": round(row["avg_priority"], 1),
        }
        for cat, row in agg.iterrows()
    ]
