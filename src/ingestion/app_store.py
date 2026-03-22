"""
Apple App Store Review Fetcher
Pulls reviews via Apple's public iTunes RSS JSON feed (no API key required)
"""
import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_BASE_URL = "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"


def fetch_app_store_reviews(
    app_id: str,
    country: str = "us",
    pages: int = 3,
) -> pd.DataFrame:
    """
    Fetch reviews from Apple App Store RSS feed.

    Args:
        app_id  : Numeric App Store ID e.g. '324684580' (Yelp)
        country : Two-letter country code
        pages   : How many pages to fetch (1-10, ~50 reviews each)

    Returns:
        Standardised DataFrame
    """
    all_records: list = []

    for page in range(1, pages + 1):
        url = _BASE_URL.format(country=country, page=page, app_id=app_id)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            entries = data.get("feed", {}).get("entry", [])
            if not entries:
                break

            # First entry is app metadata — skip it
            if "im:name" in entries[0]:
                entries = entries[1:]

            for entry in entries:
                try:
                    raw_rating = entry.get("im:rating", {}).get("label", "0")
                    all_records.append(
                        {
                            "id": entry.get("id", {}).get("label", ""),
                            "text": entry.get("content", {}).get("label", ""),
                            "rating": int(raw_rating) if str(raw_rating).isdigit() else 0,
                            "date": entry.get("updated", {}).get("label", str(datetime.now())),
                            "source": "App Store",
                            "author": entry.get("author", {}).get("name", {}).get("label", "Anonymous"),
                            "title": entry.get("title", {}).get("label", ""),
                            "app_version": entry.get("im:version", {}).get("label", "Unknown"),
                        }
                    )
                except Exception as parse_err:
                    logger.debug("Skipping malformed entry: %s", parse_err)

        except requests.RequestException as net_err:
            logger.error("Network error on page %d for app %s: %s", page, app_id, net_err)
            break
        except Exception as exc:
            logger.error("App Store parse error page %d: %s", page, exc)
            break

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["date"].fillna(pd.Timestamp.now(), inplace=True)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    logger.info("Fetched %d App Store reviews for app_id=%s", len(df), app_id)
    return df
