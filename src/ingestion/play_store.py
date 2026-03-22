"""
Google Play Store Review Fetcher
Fetches reviews using google-play-scraper (no API key required)
"""
import logging
import pandas as pd
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_play_store_reviews(
    app_id: str,
    count: int = 200,
    lang: str = "en",
    country: str = "us",
) -> pd.DataFrame:
    """
    Fetch reviews from Google Play Store.

    Args:
        app_id  : App package name e.g. 'com.spotify.music'
        count   : Number of reviews to retrieve
        lang    : Language code (default 'en')
        country : Country code  (default 'us')

    Returns:
        Standardised DataFrame with columns:
        id, text, rating, date, source, author, app_version
    """
    try:
        from google_play_scraper import reviews, Sort

        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=count,
        )

        records = [
            {
                "id": r.get("reviewId", f"gp_{i}"),
                "text": r.get("content", ""),
                "rating": r.get("score", 0),
                "date": r.get("at", datetime.now()),
                "source": "Google Play Store",
                "author": r.get("userName", "Anonymous"),
                "thumbs_up": r.get("thumbsUpCount", 0),
                "app_version": r.get("reviewCreatedVersion", "Unknown"),
            }
            for i, r in enumerate(result)
        ]

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
            df = df[df["text"].str.strip() != ""].reset_index(drop=True)

        logger.info("Fetched %d Play Store reviews for %s", len(df), app_id)
        return df

    except ImportError:
        logger.error("google-play-scraper not installed. Run: pip install google-play-scraper")
        return pd.DataFrame()
    except Exception as exc:
        logger.error("Play Store fetch error for %s: %s", app_id, exc)
        return pd.DataFrame()
