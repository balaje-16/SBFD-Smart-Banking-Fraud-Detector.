"""Tests for dataset loading, format adapters, and synthetic generator."""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    DatasetAdapter,
    SyntheticDataGenerator,
    DATASET_TYPE_CREDIT_CARD,
    DATASET_TYPE_PAYSIM,
    DATASET_TYPE_CUSTOM,
    DATASET_TYPE_UNKNOWN,
    load_csv_safely,
)


def test_detect_credit_card_format():
    df = pd.DataFrame({
        "Time": [0.0, 1.0],
        "V1": [-1.35, 1.19],
        "V2": [-0.07, 0.26],
        "Amount": [149.62, 2.69],
        "Class": [0, 0],
    })
    fmt = DatasetAdapter.detect_format(df)
    assert fmt == DATASET_TYPE_CREDIT_CARD


def test_adapt_credit_card_dataset():
    df = pd.DataFrame({
        "Time": [0.0, 10.0, 25.0],
        "V1": [-1.35, 1.19, -0.5],
        "V2": [-0.07, 0.26, 0.1],
        "Amount": [149.62, 2.69, 50.0],
        "Class": [0, 1, 0],
    })
    adapted, meta = DatasetAdapter.adapt(df)
    assert "amount" in adapted.columns
    assert "timestamp" in adapted.columns
    assert "is_fraud" in adapted.columns
    assert adapted["account_id"].iloc[0] is None
    assert meta["has_account_id"] is False
    assert meta["has_ground_truth"] is True
    assert "V1" in adapted.columns
    assert "V2" in adapted.columns


def test_detect_and_adapt_paysim_format():
    df = pd.DataFrame({
        "step": [1, 1, 2],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [9839.64, 1864.28, 181.0],
        "nameOrig": ["C1231006815", "C1666544295", "C1305486145"],
        "oldbalanceOrg": [170136.0, 21249.0, 181.0],
        "newbalanceOrig": [160296.36, 19384.72, 0.0],
        "nameDest": ["M1979787155", "C1305486145", "C553264065"],
        "oldbalanceDest": [0.0, 0.0, 0.0],
        "newbalanceDest": [0.0, 0.0, 0.0],
        "isFraud": [0, 1, 1],
        "isFlaggedFraud": [0, 0, 0],
    })
    fmt = DatasetAdapter.detect_format(df)
    assert fmt == DATASET_TYPE_PAYSIM

    adapted, meta = DatasetAdapter.adapt(df)
    assert "amount" in adapted.columns
    assert adapted["account_id"].iloc[0] == "C1231006815"
    assert meta["has_account_id"] is True
    assert meta["has_ground_truth"] is True
    assert adapted["transaction_type"].iloc[0] == "PAYMENT"


def test_adapt_custom_dataset():
    df = pd.DataFrame({
        "tx_id": ["TX_1", "TX_2"],
        "account_id": ["ACC_A", "ACC_B"],
        "timestamp": ["2026-01-01 10:00:00", "2026-01-01 11:00:00"],
        "amount": [45.50, 120.00],
        "transaction_type": ["ONLINE", "POS"],
        "is_fraud": [0, 1],
    })
    adapted, meta = DatasetAdapter.adapt(df)
    assert len(adapted) == 2
    assert adapted["transaction_id"].iloc[0] == "TX_1"
    assert adapted["account_id"].iloc[0] == "ACC_A"
    assert adapted["amount"].iloc[0] == 45.50


def test_adapt_empty_dataframe_raises_error():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        DatasetAdapter.adapt(df)


def test_synthetic_data_generator():
    df = SyntheticDataGenerator.generate_transactions(n_records=300, random_seed=42)
    assert len(df) == 300
    assert "transaction_id" in df.columns
    assert "account_id" in df.columns
    assert "timestamp" in df.columns
    assert "amount" in df.columns
    assert "is_fraud" in df.columns
    # Check that fraud cases exist
    assert df["is_fraud"].sum() > 0
    # Check chronological ordering
    assert df["timestamp"].is_monotonic_increasing


def test_synthetic_auth_events():
    df_auth = SyntheticDataGenerator.generate_auth_events(compromised_account="ACC_TEST")
    assert len(df_auth) >= 4
    assert "event_type" in df_auth.columns
    assert "LOGIN_FAILED" in df_auth["event_type"].values
    assert "LOGIN_SUCCESS" in df_auth["event_type"].values
