"""Unit tests for Milestone 4: FinBERT financial news sentiment classification and aggregation."""

from pathlib import Path
import pytest

from quantpulse.sentiment import (
    AssetSentimentSummary,
    FinBERTSentimentClassifier,
    HeadlineItem,
    HeadlineLoader,
    HeadlineSentiment,
    ModelLoadError,
    SentimentBatchResult,
    aggregate_sentiment_by_asset,
    sentiment_summary_dataframe,
    sentiment_time_series,
)


@pytest.fixture
def sample_headlines_items():
    """Deterministic collection of financial headlines."""
    return [
        HeadlineItem(
            headline="Apple reports record quarterly revenue beating analyst expectations.",
            ticker="AAPL",
            timestamp="2023-10-15T09:30:00Z",
        ),
        HeadlineItem(
            headline="JPMorgan cautions on loan default risks amid commercial real estate slowdown.",
            ticker="JPM",
            timestamp="2023-10-16T11:00:00Z",
        ),
        HeadlineItem(
            headline="Treasury yields hold steady following Federal Reserve monetary policy meeting.",
            ticker="TLT",
            timestamp="2023-10-17T14:30:00Z",
        ),
    ]


def test_lazy_loading_and_memory_unload(monkeypatch):
    """Verify classifier does not load weights at instantiation and unloads cleanly."""
    import transformers

    classifier = FinBERTSentimentClassifier(use_mock=False)
    assert classifier.is_loaded is False

    # Mock pretrained loaders to test loading and unloading mechanics instantly
    class MockConfig:
        id2label = {0: "positive", 1: "negative", 2: "neutral"}

    class MockModel:
        config = MockConfig()
        def to(self, device): return self
        def eval(self): return self

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", lambda *args, **kwargs: MockModel())

    classifier.load_model(allow_mock_fallback=False)
    assert classifier.is_loaded is True

    classifier.unload_model()
    assert classifier.is_loaded is False
    assert classifier._model is None
    assert classifier._tokenizer is None


def test_model_loading_failure_raises_custom_error(monkeypatch):
    """When Hugging Face model path is invalid and mock fallback is disabled, raises ModelLoadError."""
    import transformers

    classifier = FinBERTSentimentClassifier(
        model_name="non_existent/fake_model_path_12345",
        use_mock=False,
    )

    def mock_fail(*args, **kwargs):
        raise OSError("Simulated connection timeout to Hugging Face Hub")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", mock_fail)

    with pytest.raises(ModelLoadError, match="Failed to load FinBERT model"):
        classifier.load_model(allow_mock_fallback=False)


def test_empty_headlines_handling():
    """Verify empty lists and empty text inputs are rejected or handled gracefully."""
    classifier = FinBERTSentimentClassifier(use_mock=True)

    # Empty list should return empty result container
    empty_batch = classifier.classify_batch([])
    assert empty_batch.total_headlines == 0
    assert len(empty_batch.items) == 0

    # Whitespace only should raise ValueError
    with pytest.raises(ValueError, match="empty or whitespace"):
        classifier.classify_headline("   ")

    with pytest.raises(ValueError, match="empty or whitespace"):
        classifier.classify_batch(["   ", ""])


def test_output_labels_and_confidence_bounds(sample_headlines_items):
    """Verify confidence probabilities, valid label sets, and compound scores."""
    classifier = FinBERTSentimentClassifier(use_mock=True)
    result = classifier.classify_batch(sample_headlines_items)

    assert result.total_headlines == 3
    assert len(result.items) == 3

    for item in result.items:
        assert item.label in {"positive", "negative", "neutral"}
        assert 0.0 <= item.confidence <= 1.0
        assert -1.0 <= item.sentiment_score <= 1.0

        # Probabilities sum to 1.0
        prob_sum = sum(item.probabilities.values())
        assert pytest.approx(prob_sum, rel=1e-3) == 1.0

        # Confidence matches the predicted label's probability
        assert pytest.approx(item.confidence, rel=1e-3) == item.probabilities[item.label]


def test_sentiment_classification_accuracy():
    """Test classification of distinct positive, negative, and neutral headlines."""
    classifier = FinBERTSentimentClassifier(use_mock=True)

    pos_item = classifier.classify_headline("Company reports record profit growth and higher guidance.")
    assert pos_item.label == "positive"
    assert pos_item.sentiment_score > 0.0

    neg_item = classifier.classify_headline("Firm warns of severe revenue drop and escalating default risks.")
    assert neg_item.label == "negative"
    assert neg_item.sentiment_score < 0.0

    neu_item = classifier.classify_headline("Trading volume remains flat ahead of upcoming central bank conference.")
    assert neu_item.label == "neutral"


def test_headline_loader_csv_and_text(tmp_path):
    """Test HeadlineLoader ingesting CSV, raw text, and missing column handling."""
    csv_file = tmp_path / "test_headlines.csv"
    csv_file.write_text("ticker,headline,timestamp\nAAPL,New product announcement,2023-01-01\n")

    items = HeadlineLoader.from_csv(csv_file)
    assert len(items) == 1
    assert items[0].ticker == "AAPL"
    assert items[0].headline == "New product announcement"

    # From raw text
    text_items = HeadlineLoader.from_texts(["Text A", "Text B"], ticker="MSFT")
    assert len(text_items) == 2
    assert text_items[0].ticker == "MSFT"

    # Missing required 'headline' column raises ValueError
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("ticker,content\nAAPL,Test\n")
    with pytest.raises(ValueError, match="must contain a 'headline' column"):
        HeadlineLoader.from_csv(bad_csv)


def test_sentiment_aggregation_by_asset(sample_headlines_items):
    """Test aggregation of sentiment scores and proportions by ticker."""
    classifier = FinBERTSentimentClassifier(use_mock=True)
    batch_res = classifier.classify_batch(sample_headlines_items)

    summaries = aggregate_sentiment_by_asset(batch_res)
    assert set(summaries.keys()) == {"AAPL", "JPM", "TLT"}

    aapl = summaries["AAPL"]
    assert aapl.headline_count == 1
    assert pytest.approx(aapl.bullish_ratio + aapl.bearish_ratio + aapl.neutral_ratio, rel=1e-3) == 1.0

    table_df = sentiment_summary_dataframe(summaries)
    assert isinstance(table_df, object)
    assert len(table_df) == 3


def test_sentiment_time_series_chronology(sample_headlines_items):
    """Verify chronological ordering of timestamped sentiment records."""
    classifier = FinBERTSentimentClassifier(use_mock=True)
    batch_res = classifier.classify_batch(sample_headlines_items)

    ts_df = sentiment_time_series(batch_res)
    assert len(ts_df) == 3
    assert "timestamp" in ts_df.columns
    assert "sentiment_score" in ts_df.columns
    assert ts_df["timestamp"].is_monotonic_increasing
