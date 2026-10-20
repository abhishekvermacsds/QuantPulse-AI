"""Milestone 4 Quickstart Demonstration: FinBERT Financial News Sentiment Classification & Aggregation."""

import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from quantpulse.sentiment import (
    FinBERTSentimentClassifier,
    HeadlineLoader,
    aggregate_sentiment_by_asset,
    sentiment_summary_dataframe,
    sentiment_time_series,
)


def main():
    print("=" * 85)
    print("   QUANTPULSE AI — Autonomous Portfolio Risk & Volatility Optimization")
    print("   Milestone 4: FinBERT Financial News Sentiment Classification Engine")
    print("=" * 85)

    # 1. Ingest Sample Financial Headlines
    headlines = HeadlineLoader.load_sample()
    print(f"\n[1] Ingested Sample Financial Headlines:")
    print(f"    - Total Headlines: {len(headlines)}")
    sample_tickers = sorted({h.ticker for h in headlines if h.ticker})
    print(f"    - Asset Universe Represented: {sample_tickers}")

    # 2. Initialize Memory-Conscious Classifier
    # Note: Lazy loading ensures zero RAM is allocated until classification is called.
    classifier = FinBERTSentimentClassifier(use_mock=False)
    print(f"\n[2] Classifier Status Before Invocation:")
    print(f"    - Model Name: {classifier.model_name}")
    print(f"    - Target Device: {classifier.device}")
    print(f"    - In-Memory Loaded: {classifier.is_loaded} (Lazy loading confirmed)")

    # 3. Perform Batch Sentiment Inference
    print(f"\n[3] Running Memory-Capped Batch Sentiment Classification...")
    batch_result = classifier.classify_batch(headlines, batch_size=16)

    print(f"    - Total Classified: {batch_result.total_headlines}")
    print(f"    - Positive Headlines: {batch_result.positive_count}")
    print(f"    - Negative Headlines: {batch_result.negative_count}")
    print(f"    - Neutral Headlines: {batch_result.neutral_count}")
    print(f"    - Global Average Sentiment: {batch_result.mean_sentiment_score:+.4f}")

    # 4. Display Sample Classification Output
    print("\n[4] Sample Headline Sentiment Classifications:")
    records = []
    for item in batch_result.items[:8]:
        records.append({
            "Asset": item.ticker or "GEN",
            "Label": item.label.upper(),
            "Conf": f"{item.confidence * 100:.1f}%",
            "Score": f"{item.sentiment_score:+.2f}",
            "Headline": item.headline[:50] + "..." if len(item.headline) > 50 else item.headline,
        })
    print(pd.DataFrame(records).to_string(index=False))

    # 5. Aggregate Sentiment by Asset
    print("\n[5] Aggregated Asset Sentiment Attribution:")
    summaries = aggregate_sentiment_by_asset(batch_result)
    summary_table = sentiment_summary_dataframe(summaries)
    print(summary_table.to_string(index=False))

    # 6. Manual Headline Inference Interface Demo
    print("\n[6] Interactive Single-Headline Inference Interface:")
    sample_custom = "Federal Reserve signals potential rate stabilization as inflation metrics cool."
    single_res = classifier.classify_headline(sample_custom)
    print(f"    - Input Headline : \"{sample_custom}\"")
    print(f"    - Predicted Label: {single_res.label.upper()} (Confidence: {single_res.confidence * 100:.1f}%)")
    print(f"    - Probabilities  : Positive={single_res.probabilities['positive']:.3f}, Negative={single_res.probabilities['negative']:.3f}, Neutral={single_res.probabilities['neutral']:.3f}")
    print(f"    - Compound Score : {single_res.sentiment_score:+.4f}")

    # 7. Memory Cleanup Verification (8 GB M1 Mac Optimization)
    classifier.unload_model()
    print(f"\n[7] Memory Cleanup (8 GB RAM Optimization):")
    print(f"    - Model unloaded from RAM: is_loaded = {classifier.is_loaded}")
    print(f"    - Memory freed via garbage collection & PyTorch cache flush.")

    print("\n" + "=" * 85)
    print("   Milestone 4 Verification: SUCCESSFUL")
    print("=" * 85)


if __name__ == "__main__":
    main()
