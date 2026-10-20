"""Headline data loader supporting CSV, JSON, and raw text collections."""

from pathlib import Path
from typing import List, Optional, Union
import json
import pandas as pd

from quantpulse.core.config import SAMPLE_DATA_DIR
from quantpulse.sentiment.models import HeadlineItem

DEFAULT_HEADLINES_CSV = SAMPLE_DATA_DIR / "financial_headlines.csv"


class HeadlineLoader:
    """Ingests and validates financial headlines from structured files or manual text lists."""

    @staticmethod
    def from_csv(file_path: Union[str, Path]) -> List[HeadlineItem]:
        """Load and validate headlines from a CSV file.

        Required column: 'headline'
        Optional columns: 'ticker', 'timestamp'
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Headlines CSV not found at: {path}")

        df = pd.read_csv(path)
        if "headline" not in df.columns:
            raise ValueError(f"CSV at {path} must contain a 'headline' column.")

        items: List[HeadlineItem] = []
        for _, row in df.iterrows():
            text = str(row["headline"]).strip()
            if not text:
                continue
            ticker = str(row["ticker"]).strip().upper() if "ticker" in row and pd.notna(row["ticker"]) else None
            timestamp = str(row["timestamp"]).strip() if "timestamp" in row and pd.notna(row["timestamp"]) else None
            items.append(HeadlineItem(headline=text, ticker=ticker, timestamp=timestamp))

        return items

    @staticmethod
    def from_json(file_path: Union[str, Path]) -> List[HeadlineItem]:
        """Load and validate headlines from a JSON file (list of objects or list of strings)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Headlines JSON not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"JSON at {path} must contain a top-level list of headlines.")

        items: List[HeadlineItem] = []
        for entry in data:
            if isinstance(entry, str):
                cleaned = entry.strip()
                if cleaned:
                    items.append(HeadlineItem(headline=cleaned))
            elif isinstance(entry, dict):
                if "headline" in entry:
                    items.append(HeadlineItem(**entry))
        return items

    @staticmethod
    def from_texts(
        texts: List[str],
        ticker: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> List[HeadlineItem]:
        """Construct HeadlineItems from a Python list of strings."""
        items: List[HeadlineItem] = []
        for t in texts:
            cleaned = str(t).strip()
            if cleaned:
                items.append(HeadlineItem(headline=cleaned, ticker=ticker, timestamp=timestamp))
        return items

    @classmethod
    def load_sample(cls, path: Optional[Union[str, Path]] = None) -> List[HeadlineItem]:
        """Load default shipped sample financial headlines."""
        target = Path(path) if path else DEFAULT_HEADLINES_CSV
        return cls.from_csv(target)
