"""
CSV / Excel Feedback Importer
Handles survey exports and custom feedback files with auto column-mapping
"""
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

# Flexible synonym mapping for common column names
_COL_MAP = {
    "review": "text", "comment": "text", "feedback": "text",
    "message": "text", "content": "text", "body": "text", "response": "text",
    "score": "rating", "stars": "rating", "rate": "rating", "rating_value": "rating",
    "timestamp": "date", "created_at": "date", "submitted_at": "date",
    "datetime": "date", "time": "date", "created": "date",
    "user": "author", "name": "author", "reviewer": "author",
    "username": "author", "customer": "author",
}


def load_csv_feedback(file, source_name: str = "CSV Upload") -> pd.DataFrame:
    """
    Load feedback from a CSV or Excel file with intelligent column detection.

    Args:
        file        : File-like object (Streamlit UploadedFile)
        source_name : Label shown as the 'source' column value

    Returns:
        Standardised DataFrame with columns: id, text, rating, date, source, author
    """
    try:
        filename = getattr(file, "name", "upload.csv").lower()
        df = pd.read_excel(file) if filename.endswith((".xlsx", ".xls")) else pd.read_csv(file)

        # Normalise column names
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

        # Apply synonym renaming (only if target doesn't already exist)
        rename = {
            src: tgt
            for src, tgt in _COL_MAP.items()
            if src in df.columns and tgt not in df.columns
        }
        df.rename(columns=rename, inplace=True)

        # --- Ensure 'text' column ---
        if "text" not in df.columns:
            # Pick the longest-average-value string column as the text column
            str_cols = [c for c in df.columns if df[c].dtype == object]
            if str_cols:
                best = max(str_cols, key=lambda c: df[c].dropna().astype(str).str.len().mean())
                df.rename(columns={best: "text"}, inplace=True)
            else:
                raise ValueError("No suitable text column found in uploaded file.")

        # --- Defaults for missing columns ---
        if "rating" not in df.columns:
            df["rating"] = None
        if "date" not in df.columns:
            df["date"] = pd.Timestamp.now()
        if "author" not in df.columns:
            df["author"] = "Anonymous"
        if "source" not in df.columns:
            df["source"] = source_name
        if "id" not in df.columns:
            df["id"] = [f"csv_{i}" for i in range(len(df))]

        # Parse & clean
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"].fillna(pd.Timestamp.now(), inplace=True)
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 2].reset_index(drop=True)

        logger.info("Loaded %d records from %s", len(df), filename)
        return df[["id", "text", "rating", "date", "source", "author"]]

    except Exception as exc:
        logger.error("CSV load error: %s", exc)
        return pd.DataFrame()
