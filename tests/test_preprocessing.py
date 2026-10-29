"""Tests for data preprocessing, validation, deduplication, and quality reporting."""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing import DataPreprocessor, DataQualityReport


def test_preprocessing_cleans_and_reports():
    raw_df = pd.DataFrame({
        "transaction_id": ["TX_1", "TX_2", "TX_2", "TX_3", "TX_4"],  # Contains duplicate TX_2
        "account_id": ["A1", "A2", "A2", "A3", "A4"],
        "timestamp": [
            "2026-01-01 12:00:00",
            "2026-01-01 10:00:00",  # Unsorted
            "2026-01-01 10:00:00",
            "2026-01-01 14:00:00",
            "2026-01-01 16:00:00",
        ],
        "amount": [50.0, 100.0, 100.0, -25.0, 75.0],  # Negative amount on TX_3
        "transaction_type": ["PAYMENT", "TRANSFER", "TRANSFER", "PAYMENT", None],
    })

    preprocessor = DataPreprocessor(allow_zero_amounts=False)
    clean_df, report = preprocessor.validate_and_clean(raw_df)

    # Negative amount TX_3 dropped
    assert "TX_3" not in clean_df["transaction_id"].values
    assert report.invalid_amounts_dropped == 1

    # Duplicate TX_2 deduplicated
    assert len(clean_df[clean_df["transaction_id"] == "TX_2"]) == 1
    assert report.duplicates_detected == 1

    # Total valid retained: 3 (TX_1, TX_2, TX_4)
    assert len(clean_df) == 3
    assert report.valid_records_retained == 3

    # Timestamps sorted chronologically
    assert clean_df["timestamp"].iloc[0] <= clean_df["timestamp"].iloc[1]

    # Missing transaction_type handled
    assert clean_df[clean_df["transaction_id"] == "TX_4"]["transaction_type"].iloc[0] == "UNKNOWN"


def test_preprocessing_imputes_missing_amount():
    raw_df = pd.DataFrame({
        "transaction_id": ["TX_1", "TX_2", "TX_3"],
        "amount": [10.0, np.nan, 30.0],
    })
    preprocessor = DataPreprocessor()
    clean_df, report = preprocessor.validate_and_clean(raw_df)

    assert len(clean_df) == 3
    assert report.missing_values_handled.get("amount") == 1
    # Imputed with median = 20.0
    assert clean_df["amount"].iloc[1] == 20.0


def test_preprocessing_empty_df_raises():
    preprocessor = DataPreprocessor()
    with pytest.raises(ValueError, match="empty"):
        preprocessor.validate_and_clean(pd.DataFrame())


def test_encode_and_scale():
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    preprocessor = DataPreprocessor()
    scaled_train = preprocessor.encode_and_scale(df, ["feat_a", "feat_b"], is_training=True)

    assert scaled_train.shape == (5, 2)
    assert np.isclose(np.mean(scaled_train[:, 0]), 0.0, atol=1e-7)
    assert np.isclose(np.std(scaled_train[:, 0]), 1.0, atol=1e-7)

    # Test inference scaling
    df_test = pd.DataFrame({
        "feat_a": [3.0],
        "feat_b": [30.0],
    })
    scaled_test = preprocessor.encode_and_scale(df_test, ["feat_a", "feat_b"], is_training=False)
    assert scaled_test.shape == (1, 2)
    # 3.0 was the mean, so scaled value should be ~0.0
    assert np.isclose(scaled_test[0, 0], 0.0, atol=1e-5)
