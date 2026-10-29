"""Smart Banking Fraud Detector - Streamlit Financial Cybersecurity Dashboard.

A production-grade, interactive cybersecurity dashboard featuring:
- Unsupervised Isolation Forest Anomaly Detection
- Longitudinal Account Behavioural Profiling
- Multi-Signal Contextual Risk Scoring Engine
- Suspicious Transaction Burst & Sequence Detection
- Persistent SOC Analyst Alert Center
- Rigorous Out-of-Time Model Evaluation
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add workspace directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.alert_manager import (
    ALERT_STATUS_FALSE_POSITIVE,
    ALERT_STATUS_NEW,
    ALERT_STATUS_RESOLVED,
    ALERT_STATUS_UNDER_REVIEW,
    AlertManager,
)
from src.data_loader import (
    DATASET_TYPE_CREDIT_CARD,
    DATASET_TYPE_CUSTOM,
    DATASET_TYPE_PAYSIM,
    DATASET_TYPE_UNKNOWN,
    DatasetAdapter,
    SyntheticDataGenerator,
    load_csv_safely,
)
from src.pipeline import FraudDetectionPipeline, PipelineExecutionResult
from src.risk_engine import RiskWeights

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Dark Navy CSS
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Banking Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Dark Navy / Cyan Financial SOC Theme
st.markdown(
    """
    <style>
    /* Global Styles */
    .stApp {
        background-color: #0b132b;
        color: #e0e6ed;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00d2ff !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #1c2541;
        border: 1px solid #2a3b61;
        border-radius: 8px;
        padding: 12px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricLabel"] {
        color: #8da9c4 !important;
        font-size: 0.85rem !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.8rem !important;
    }

    /* Custom badges */
    .badge-high {
        background-color: rgba(255, 77, 79, 0.15);
        color: #ff4d4f;
        border: 1px solid #ff4d4f;
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-med {
        background-color: rgba(250, 173, 20, 0.15);
        color: #faad14;
        border: 1px solid #faad14;
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: rgba(82, 196, 26, 0.15);
        color: #52c41a;
        border: 1px solid #52c41a;
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #101c38;
        border-right: 1px solid #1c2e56;
    }

    /* Card Panels */
    .panel-card {
        background-color: #16223f;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #25365e;
        margin-bottom: 16px;
    }

    /* Button styles */
    .stButton>button {
        background-color: #00d2ff;
        color: #0b132b;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #33dcff;
        color: #000000;
        box-shadow: 0 0 12px rgba(0, 210, 255, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

def initialize_session_state() -> None:
    """Ensure persistent session state variables exist."""
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None

    if "current_df" not in st.session_state:
        # Load default synthetic sample if present, or generate
        sample_path = os.path.join(BASE_DIR, "data", "sample", "banking_transactions_sample.csv")
        if os.path.exists(sample_path):
            df_init, meta = load_csv_safely(sample_path)
            st.session_state.current_df = df_init
            st.session_state.dataset_meta = meta
        else:
            df_gen = SyntheticDataGenerator.generate_transactions(n_records=1200, random_seed=42)
            st.session_state.current_df = df_gen
            st.session_state.dataset_meta = {"detected_type": DATASET_TYPE_CUSTOM, "has_account_id": True}

    if "dataset_meta" not in st.session_state:
        st.session_state.dataset_meta = {"detected_type": DATASET_TYPE_CUSTOM, "has_account_id": True}

    if "config_contamination" not in st.session_state:
        st.session_state.config_contamination = 0.02
    if "config_threshold_med" not in st.session_state:
        st.session_state.config_threshold_med = 40.0
    if "config_threshold_high" not in st.session_state:
        st.session_state.config_threshold_high = 70.0

    if "risk_weights" not in st.session_state:
        st.session_state.risk_weights = RiskWeights(
            ml_anomaly=0.35,
            historical_deviation=0.25,
            velocity=0.20,
            temporal_timing=0.10,
            sequence_violation=0.10,
        )

    if "alert_manager" not in st.session_state:
        alerts_file = os.path.join(BASE_DIR, "data", "alerts.json")
        st.session_state.alert_manager = AlertManager(storage_path=alerts_file)


initialize_session_state()

# Run pipeline automatically on startup if not yet computed
if st.session_state.pipeline_result is None and st.session_state.current_df is not None:
    with st.spinner("Initializing detection model and scoring baseline transactions..."):
        pipe = FraudDetectionPipeline(
            contamination=st.session_state.config_contamination,
            risk_weights=st.session_state.risk_weights,
            threshold_medium=st.session_state.config_threshold_med,
            threshold_high=st.session_state.config_threshold_high,
            alerts_storage_path=os.path.join(BASE_DIR, "data", "alerts.json"),
        )
        st.session_state.pipeline_result = pipe.run(
            st.session_state.current_df,
            train_new_model=True,
            model_checkpoint_path=os.path.join(BASE_DIR, "models", "isolation_forest.joblib"),
        )


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Status
# -----------------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 10px 0;'>
        <h2 style='margin: 0; color: #00d2ff;'>🛡️ CyberShield</h2>
        <p style='margin: 0; font-size: 0.85rem; color: #8da9c4;'>Smart Banking Fraud Detector</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

PAGES = [
    "1. Overview",
    "2. Data Management",
    "3. Fraud Detection",
    "4. Behavioural Analysis",
    "5. Alert Centre",
    "6. Model Evaluation",
    "7. Settings & About",
]

selected_page = st.sidebar.radio("Navigation", PAGES, index=0)

st.sidebar.markdown("---")

# Quick Status Indicator
res: Optional[PipelineExecutionResult] = st.session_state.pipeline_result
if res is not None:
    total_tx = len(res.features_dataframe)
    high_count = sum(1 for r in res.evaluation_results if r.risk_category == "High Risk")
    med_count = sum(1 for r in res.evaluation_results if r.risk_category == "Medium Risk")
    st.sidebar.markdown(f"**Loaded Data:** `{total_tx}` records")
    st.sidebar.markdown(f"🔴 **High Risk:** `{high_count}`")
    st.sidebar.markdown(f"🟡 **Medium Risk:** `{med_count}`")
    st.sidebar.markdown(f"🟢 **Low Risk:** `{total_tx - high_count - med_count}`")
else:
    st.sidebar.info("No dataset analyzed yet.")

st.sidebar.markdown("---")
st.sidebar.caption("Portfolio Project | ML & Cyber Fraud Detection")


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================

if selected_page == "1. Overview":
    st.title("🛡️ Banking Fraud & Anomaly Overview")
    st.markdown(
        "Real-time operational dashboard analyzing banking transactions via unsupervised "
        "Isolation Forest anomaly detection, account-level behavioral baselines, and contextual risk scoring."
    )

    if res is None:
        st.warning("No transactions currently analyzed. Please go to **Data Management** to upload or generate data.")
    else:
        df_feat = res.features_dataframe
        total_records = len(df_feat)
        high_risk_txs = [r for r in res.evaluation_results if r.risk_category == "High Risk"]
        med_risk_txs = [r for r in res.evaluation_results if r.risk_category == "Medium Risk"]
        low_risk_txs = [r for r in res.evaluation_results if r.risk_category == "Low Risk"]

        # Top Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Analyzed Transactions", f"{total_records:,}")
        col2.metric("Flagged for Review", f"{len(high_risk_txs) + len(med_risk_txs):,}")
        col3.metric("High Risk (Severe)", f"{len(high_risk_txs):,}", delta=f"{len(high_risk_txs)/total_records*100:.1f}%", delta_color="inverse")
        col4.metric("Medium Risk", f"{len(med_risk_txs):,}")
        col5.metric("Low Risk (Approved)", f"{len(low_risk_txs):,}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Charts Row
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Risk Tier Distribution")
            donut_df = pd.DataFrame({
                "Risk Category": ["High Risk", "Medium Risk", "Low Risk"],
                "Count": [len(high_risk_txs), len(med_risk_txs), len(low_risk_txs)],
            })
            fig_donut = px.pie(
                donut_df,
                values="Count",
                names="Risk Category",
                hole=0.55,
                color="Risk Category",
                color_discrete_map={
                    "High Risk": "#ff4d4f",
                    "Medium Risk": "#faad14",
                    "Low Risk": "#52c41a",
                },
            )
            fig_donut.update_layout(
                paper_bgcolor="#16223f",
                plot_bgcolor="#16223f",
                font=dict(color="#e0e6ed"),
                margin=dict(t=30, b=30, l=30, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.subheader("Anomaly Score Distribution (0-100)")
            fig_hist = px.histogram(
                df_feat,
                x="anomaly_score",
                nbins=35,
                color="risk_category",
                color_discrete_map={
                    "High Risk": "#ff4d4f",
                    "Medium Risk": "#faad14",
                    "Low Risk": "#52c41a",
                },
                labels={"anomaly_score": "Isolation Forest Anomaly Score", "count": "Transactions"},
            )
            fig_hist.update_layout(
                paper_bgcolor="#16223f",
                plot_bgcolor="#16223f",
                font=dict(color="#e0e6ed"),
                margin=dict(t=30, b=30, l=30, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                xaxis=dict(gridcolor="#25365e"),
                yaxis=dict(gridcolor="#25365e"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Timeline Chart
        st.subheader("Transaction Activity & Temporal Risk Scatter")
        sample_plot_df = df_feat.sample(min(800, len(df_feat)), random_state=42).sort_values("timestamp")
        fig_time = px.scatter(
            sample_plot_df,
            x="timestamp",
            y="amount",
            color="risk_category",
            size="overall_risk_score",
            hover_data=["transaction_id", "overall_risk_score", "anomaly_score"],
            color_discrete_map={
                "High Risk": "#ff4d4f",
                "Medium Risk": "#faad14",
                "Low Risk": "#52c41a",
            },
            labels={"timestamp": "Detection Time", "amount": "Amount ($)"},
        )
        fig_time.update_layout(
            paper_bgcolor="#16223f",
            plot_bgcolor="#16223f",
            font=dict(color="#e0e6ed"),
            margin=dict(t=30, b=30, l=30, r=30),
            xaxis=dict(gridcolor="#25365e"),
            yaxis=dict(gridcolor="#25365e"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # Recent High Risk Alerts Table
        st.subheader("⚠️ Critical Priority Alerts")
        alerts_df = st.session_state.alert_manager.to_dataframe()
        if not alerts_df.empty:
            high_alerts = alerts_df[alerts_df["risk_category"] == "High Risk"].head(10)
            if not high_alerts.empty:
                st.dataframe(
                    high_alerts[[
                        "alert_id",
                        "transaction_id",
                        "account_id",
                        "risk_score",
                        "status",
                        "recommended_action",
                        "detection_timestamp",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No High-Risk alerts currently active.")
        else:
            st.info("No alerts generated.")


# =============================================================================
# PAGE 2: DATA MANAGEMENT
# =============================================================================

elif selected_page == "2. Data Management":
    st.title("📂 Transaction Data Management")
    st.markdown(
        "Upload real or benchmark banking CSV records, inspect schema mappings, "
        "review automated Data Quality audit reports, or generate synthetic attack scenarios."
    )

    t1, t2 = st.tabs(["📤 Upload Dataset", "🧪 Synthetic Scenario Generator"])

    with t1:
        st.markdown("#### Upload CSV File")
        uploaded_file = st.file_uploader(
            "Select banking transaction CSV (Credit Card, PaySim, or Custom Schema)",
            type=["csv"],
            help="Files up to 50 MB are parsed securely in-memory. Executables are strictly rejected.",
        )

        dataset_type_override = st.selectbox(
            "Dataset Format / Schema Adapter",
            [
                "Auto-Detect Format",
                DATASET_TYPE_CREDIT_CARD,
                DATASET_TYPE_PAYSIM,
                DATASET_TYPE_CUSTOM,
            ],
            index=0,
        )

        col_up_btn, col_demo_btn = st.columns([1, 1])

        if uploaded_file is not None:
            if st.button("Process & Analyze Uploaded File", type="primary"):
                try:
                    with st.spinner("Validating file and executing schema adapter..."):
                        override = None if dataset_type_override == "Auto-Detect Format" else dataset_type_override
                        raw_df = pd.read_csv(uploaded_file, low_memory=False)
                        adapted_df, meta = DatasetAdapter.adapt(raw_df, dataset_type=override)

                        pipe = FraudDetectionPipeline(
                            contamination=st.session_state.config_contamination,
                            risk_weights=st.session_state.risk_weights,
                            threshold_medium=st.session_state.config_threshold_med,
                            threshold_high=st.session_state.config_threshold_high,
                            alerts_storage_path=os.path.join(BASE_DIR, "data", "alerts.json"),
                        )
                        st.session_state.current_df = adapted_df
                        st.session_state.dataset_meta = meta
                        st.session_state.pipeline_result = pipe.run(
                            adapted_df,
                            train_new_model=True,
                            model_checkpoint_path=os.path.join(BASE_DIR, "models", "isolation_forest.joblib"),
                        )
                        st.success(f"Successfully processed {len(adapted_df):,} records! Detected: **{meta['detected_type']}**")
                        st.rerun()
                except Exception as e:
                    st.error(f"Upload Processing Error: {str(e)}")

    with t2:
        st.markdown("#### Generate Controlled Synthetic Demonstrations")
        st.markdown(
            "Produce reproducible, labelled test datasets containing 7 distinct financial scenarios:\n"
            "- Low-value normal retail spending\n"
            "- High-value legitimate corporate wires (testing false positive resistance)\n"
            "- Sudden single-transaction outlier spike\n"
            "- High-frequency card testing burst (4 rapid transactions)\n"
            "- Coordinated account draining sequence (3 large transfers in 10 mins)\n"
            "- Rare benign event (e.g. annual travel vacation)\n"
            "- Cyber-heist authentication kill chain (failed logins -> new device -> wire)"
        )

        cg1, cg2, cg3 = st.columns(3)
        n_records = cg1.slider("Number of Transactions", min_value=300, max_value=5000, value=1200, step=100)
        fraud_ratio = cg2.slider("Fraud Ratio", min_value=0.01, max_value=0.15, value=0.04, step=0.01)
        seed = cg3.number_input("Random Seed", value=42, step=1)

        if st.button("Generate & Run Pipeline on Synthetic Data"):
            with st.spinner("Generating reproducible scenario dataset..."):
                synth_tx = SyntheticDataGenerator.generate_transactions(
                    n_records=n_records, fraud_ratio=fraud_ratio, random_seed=int(seed)
                )
                pipe = FraudDetectionPipeline(
                    contamination=st.session_state.config_contamination,
                    risk_weights=st.session_state.risk_weights,
                    threshold_medium=st.session_state.config_threshold_med,
                    threshold_high=st.session_state.config_threshold_high,
                    alerts_storage_path=os.path.join(BASE_DIR, "data", "alerts.json"),
                )
                st.session_state.current_df = synth_tx
                st.session_state.dataset_meta = {
                    "detected_type": "Synthetic Banking Multi-Scenario",
                    "has_account_id": True,
                    "has_ground_truth": True,
                }
                st.session_state.pipeline_result = pipe.run(synth_tx, train_new_model=True)
                st.success(f"Generated and scored {len(synth_tx):,} synthetic transactions!")
                st.rerun()

    st.markdown("---")

    # Data Quality Report Card
    if res is not None:
        st.subheader("📋 Data Quality Audit Report")
        qr = res.quality_report
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Total Records Ingested", f"{qr.total_records_received:,}")
        qc2.metric("Valid Records Retained", f"{qr.valid_records_retained:,}")
        qc3.metric("Duplicates Dropped", f"{qr.duplicates_detected:,}")
        qc4.metric("Invalid Amounts Dropped", f"{qr.invalid_amounts_dropped:,}")

        st.markdown(f"**Date Range:** `{qr.date_range_start}` to `{qr.date_range_end}`")
        if qr.quality_notes:
            with st.expander("View Preprocessing Audit Logs", expanded=True):
                for note in qr.quality_notes:
                    st.markdown(f"- {note}")

        # Data Preview Table
        st.subheader("Data Records Preview")
        preview_cols = [
            c for c in ["transaction_id", "account_id", "timestamp", "amount", "transaction_type", "merchant_category", "is_fraud"]
            if c in res.cleaned_dataframe.columns
        ]
        st.dataframe(res.cleaned_dataframe[preview_cols].head(15), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 3: FRAUD DETECTION
# =============================================================================

elif selected_page == "3. Fraud Detection":
    st.title("🔍 Transaction Risk & Anomaly Detector")
    st.markdown(
        "Searchable transaction ledger showing Isolation Forest multidimensional anomaly severity, "
        "account-level historical deviation, and the composite contextual risk rating."
    )

    if res is None:
        st.warning("No data analyzed. Please load a dataset first.")
    else:
        df_display = res.features_dataframe.copy()

        # Filters Row
        f1, f2, f3, f4 = st.columns([1.5, 1.5, 2, 2])
        filter_risk = f1.multiselect("Risk Level", ["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk"])
        search_query = f2.text_input("Search (TX ID / Account)", "").strip()
        min_amt, max_amt = float(df_display["amount"].min()), float(df_display["amount"].max())
        amt_range = f3.slider("Amount Range ($)", min_value=0.0, max_value=max(100.0, max_amt), value=(0.0, max(100.0, max_amt)))
        sort_by = f4.selectbox("Sort By", ["overall_risk_score (Descending)", "amount (Descending)", "timestamp (Newest First)"])

        # Apply Filters
        filtered = df_display[df_display["risk_category"].isin(filter_risk)]
        filtered = filtered[(filtered["amount"] >= amt_range[0]) & (filtered["amount"] <= amt_range[1])]

        if search_query:
            mask = filtered["transaction_id"].astype(str).str.contains(search_query, case=False)
            if "account_id" in filtered.columns:
                mask = mask | filtered["account_id"].astype(str).str.contains(search_query, case=False)
            filtered = filtered[mask]

        if "overall_risk_score" in sort_by:
            filtered = filtered.sort_values(by="overall_risk_score", ascending=False)
        elif "amount" in sort_by:
            filtered = filtered.sort_values(by="amount", ascending=False)
        else:
            filtered = filtered.sort_values(by="timestamp", ascending=False)

        st.caption(f"Displaying **{len(filtered):,}** matching transactions.")

        # Main Table
        cols_to_show = [
            "transaction_id",
            "account_id",
            "timestamp",
            "amount",
            "anomaly_score",
            "overall_risk_score",
            "risk_category",
            "explanation",
        ]
        available_cols = [c for c in cols_to_show if c in filtered.columns]
        st.dataframe(
            filtered[available_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                "anomaly_score": st.column_config.ProgressColumn("ML Score", min_value=0, max_value=100, format="%.1f"),
                "overall_risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
            },
        )

        st.markdown("---")

        # Transaction Deep-Dive Inspector
        st.subheader("🔬 Transaction Risk Deep Dive & Signal Decomposition")
        tx_options = filtered["transaction_id"].tolist()
        if tx_options:
            selected_tx = st.selectbox("Select Transaction to Inspect", tx_options[:100])
            tx_row = df_display[df_display["transaction_id"] == selected_tx].iloc[0]
            eval_match = next((r for r in res.evaluation_results if r.transaction_id == selected_tx), None)

            if eval_match:
                ic1, ic2 = st.columns([1, 1])

                with ic1:
                    st.markdown(f"#### Transaction `{eval_match.transaction_id}`")
                    st.markdown(f"- **Account ID:** `{eval_match.account_id or 'Anonymized / None'}`")
                    st.markdown(f"- **Amount:** `${eval_match.amount:,.2f}`")
                    st.markdown(f"- **Risk Classification:** `{eval_match.risk_category}`")
                    st.markdown(f"- **Composite Risk Score:** `{eval_match.overall_risk_score:.1f} / 100`")
                    st.markdown(f"- **Isolation Forest Anomaly Score:** `{eval_match.anomaly_score:.1f} / 100`")
                    st.markdown(f"**Recommended SOC Action:**\n> 🛡️ {eval_match.recommended_action}")

                    st.markdown("**Evidence & Suspicious Signals:**")
                    for r in eval_match.reasons:
                        st.markdown(f"- {r}")

                with ic2:
                    st.markdown("#### Orthogonal Risk Signal Breakdown")
                    sig_data = eval_match.contributing_signals
                    sig_df = pd.DataFrame({
                        "Signal Dimension": [
                            "ML Anomaly",
                            "Historical Deviation",
                            "Velocity & Frequency",
                            "Temporal (Night/Weekend)",
                            "Sequence Burst",
                        ],
                        "Severity Score": [
                            sig_data.get("ml_anomaly_signal", 0.0),
                            sig_data.get("historical_deviation_signal", 0.0),
                            sig_data.get("velocity_frequency_signal", 0.0),
                            sig_data.get("temporal_risk_signal", 0.0),
                            sig_data.get("sequence_pattern_signal", 0.0),
                        ],
                    })

                    fig_bar = px.bar(
                        sig_df,
                        x="Severity Score",
                        y="Signal Dimension",
                        orientation="h",
                        color="Severity Score",
                        range_x=[0, 100],
                        color_continuous_scale=["#52c41a", "#faad14", "#ff4d4f"],
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="#16223f",
                        plot_bgcolor="#16223f",
                        font=dict(color="#e0e6ed"),
                        margin=dict(t=20, b=20, l=20, r=20),
                        xaxis=dict(gridcolor="#25365e"),
                        yaxis=dict(gridcolor="#25365e"),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# PAGE 4: BEHAVIOURAL ANALYSIS
# =============================================================================

elif selected_page == "4. Behavioural Analysis":
    st.title("👤 Account Behavioural Profiling")
    st.markdown(
        "Longitudinal statistical analysis of account spending habits. Differentiates "
        "naturally large transactions (e.g. corporate wire accounts) from suspicious account takeovers."
    )

    if res is None:
        st.warning("No data analyzed.")
    elif not res.account_profiles:
        st.info(
            "⚠️ **Account-level profiling unavailable for this dataset.**\n\n"
            "The current dataset (such as the Kaggle Credit Card dataset) does not contain account or cardholder identifiers "
            "due to cryptographic anonymization. The system fell back to global stream sequence profiling without fabricating fake account histories."
        )
    else:
        profiles = res.account_profiles
        acc_list = sorted(list(profiles.keys()))

        selected_acc = st.selectbox("Select Account ID to Inspect", acc_list)
        prof = profiles[selected_acc]

        # Account Metrics Row
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Historical Transactions", prof.transaction_count)
        b2.metric("Average Spending", f"${prof.mean_amount:,.2f}")
        b3.metric("Median Spending", f"${prof.median_amount:,.2f}")
        b4.metric("Max Transaction", f"${prof.max_amount:,.2f}")
        b5.metric("Avg Interval", f"{prof.avg_interval_hours:.1f} hrs")

        if not prof.has_sufficient_history:
            st.warning(
                f"⚠️ Account has only {prof.transaction_count} transaction(s). It is classified under **Cold-Start Mode**, "
                "meaning its deviations are calibrated against population baselines."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Account Transactions Timeline
        df_acc = res.features_dataframe[res.features_dataframe["account_id"] == selected_acc].sort_values("timestamp")

        fig_acc = px.scatter(
            df_acc,
            x="timestamp",
            y="amount",
            color="risk_category",
            size="overall_risk_score",
            hover_data=["transaction_id", "overall_risk_score", "anomaly_score"],
            color_discrete_map={
                "High Risk": "#ff4d4f",
                "Medium Risk": "#faad14",
                "Low Risk": "#52c41a",
            },
            title=f"Spending Timeline for Account: {selected_acc}",
        )
        # Add baseline horizontal line
        fig_acc.add_hline(
            y=prof.mean_amount,
            line_dash="dash",
            line_color="#00d2ff",
            annotation_text=f"Account Mean (${prof.mean_amount:.2f})",
        )
        fig_acc.update_layout(
            paper_bgcolor="#16223f",
            plot_bgcolor="#16223f",
            font=dict(color="#e0e6ed"),
            xaxis=dict(gridcolor="#25365e"),
            yaxis=dict(gridcolor="#25365e"),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        st.subheader(f"Transaction History: {selected_acc}")
        st.dataframe(
            df_acc[[
                "transaction_id",
                "timestamp",
                "amount",
                "overall_risk_score",
                "risk_category",
                "feature_freq_1h",
                "explanation",
            ]],
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# PAGE 5: ALERT CENTRE
# =============================================================================

elif selected_page == "5. Alert Centre":
    st.title("🚨 Security Operations Center (SOC) Alert Centre")
    st.markdown(
        "Manage active security incidents, assign investigation workflows, and record analyst notes. "
        "All updates are persisted locally to `data/alerts.json` and survive application reruns."
    )

    alert_mgr: AlertManager = st.session_state.alert_manager
    alerts_df = alert_mgr.to_dataframe()

    if alerts_df.empty:
        st.success("No active security alerts! All transactions currently scored as Low Risk.")
    else:
        # Status Filter
        a_f1, a_f2 = st.columns([1, 1])
        status_filter = a_f1.multiselect(
            "Filter by Status",
            [ALERT_STATUS_NEW, ALERT_STATUS_UNDER_REVIEW, ALERT_STATUS_RESOLVED, ALERT_STATUS_FALSE_POSITIVE],
            default=[ALERT_STATUS_NEW, ALERT_STATUS_UNDER_REVIEW],
        )
        category_filter = a_f2.multiselect(
            "Filter by Severity",
            ["High Risk", "Medium Risk"],
            default=["High Risk", "Medium Risk"],
        )

        filtered_alerts = alerts_df[
            alerts_df["status"].isin(status_filter) & alerts_df["risk_category"].isin(category_filter)
        ]

        st.caption(f"Showing **{len(filtered_alerts)}** alerts.")

        st.dataframe(
            filtered_alerts[[
                "alert_id",
                "transaction_id",
                "account_id",
                "risk_score",
                "risk_category",
                "status",
                "recommended_action",
                "analyst_notes",
                "detection_timestamp",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Investigation Workflow Form
        st.subheader("📝 Analyst Incident Action Panel")
        alert_ids = filtered_alerts["alert_id"].tolist()

        if alert_ids:
            col_sel, col_form = st.columns([1, 2])
            with col_sel:
                selected_alert_id = st.selectbox("Select Alert ID", alert_ids)
                current_alert = alert_mgr.alerts[selected_alert_id]

                st.markdown(f"**Transaction:** `{current_alert.transaction_id}`")
                st.markdown(f"**Risk Score:** `{current_alert.risk_score:.1f}` ({current_alert.risk_category})")
                st.markdown(f"**Current Status:** `{current_alert.status}`")
                st.markdown(f"**Detected At:** `{current_alert.detection_timestamp}`")
                st.markdown("**Evidence:**")
                for r in current_alert.reasons:
                    st.markdown(f"- {r}")

            with col_form:
                new_status = st.selectbox(
                    "Update Investigation Status",
                    [ALERT_STATUS_NEW, ALERT_STATUS_UNDER_REVIEW, ALERT_STATUS_RESOLVED, ALERT_STATUS_FALSE_POSITIVE],
                    index=[
                        ALERT_STATUS_NEW,
                        ALERT_STATUS_UNDER_REVIEW,
                        ALERT_STATUS_RESOLVED,
                        ALERT_STATUS_FALSE_POSITIVE,
                    ].index(current_alert.status),
                )
                analyst_comment = st.text_area(
                    "Analyst Investigation Notes",
                    value=current_alert.analyst_notes,
                    placeholder="Enter case summary, cardholder verification outcome, or justification...",
                )

                if st.button("Save & Update Alert", type="primary"):
                    alert_mgr.update_alert_status(
                        alert_id=selected_alert_id,
                        new_status=new_status,
                        analyst_notes=analyst_comment,
                    )
                    st.success(f"Alert `{selected_alert_id}` updated to **{new_status}** and saved to disk!")
                    st.rerun()


# =============================================================================
# PAGE 6: MODEL EVALUATION
# =============================================================================

elif selected_page == "6. Model Evaluation":
    st.title("📊 Model Performance & Statistical Evaluation")
    st.markdown(
        "Rigorous verification of detection accuracy. For labeled datasets, metrics are calculated on held-out "
        "chronological test splits without data leakage. For unlabeled datasets, descriptive anomaly statistics are reported."
    )

    if res is None:
        st.warning("No data evaluated yet.")
    elif res.metrics is not None and res.metrics.has_ground_truth:
        m = res.metrics
        st.markdown(f"#### Held-Out Chronological Test Split Performance (Threshold: `{m.decision_threshold:.1f}`/100)")

        e1, e2, e3, e4, e5, e6 = st.columns(6)
        e1.metric("Precision", f"{m.precision:.3f}")
        e2.metric("Recall", f"{m.recall:.3f}")
        e3.metric("F1-Score", f"{m.f1_score:.3f}")
        e4.metric("PR-AUC", f"{m.pr_auc:.3f}" if m.pr_auc else "N/A")
        e5.metric("ROC-AUC", f"{m.roc_auc:.3f}" if m.roc_auc else "N/A")
        e6.metric("False Positive Rate", f"{m.false_positive_rate*100:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        c_cm, c_notes = st.columns([1, 1])

        with c_cm:
            st.subheader("Confusion Matrix")
            cm_z = [[m.true_negatives, m.false_positives], [m.false_negatives, m.true_positives]]
            cm_text = [
                [f"TN: {m.true_negatives:,}", f"FP: {m.false_positives:,}"],
                [f"FN: {m.false_negatives:,}", f"TP: {m.true_positives:,}"],
            ]
            fig_cm = px.imshow(
                cm_z,
                labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Cases"),
                x=["Legitimate", "Fraud"],
                y=["Legitimate", "Fraud"],
                text_auto=False,
                color_continuous_scale="Blues",
            )
            fig_cm.update_traces(text=cm_text, texttemplate="%{text}", textfont_size=16)
            fig_cm.update_layout(
                paper_bgcolor="#16223f",
                plot_bgcolor="#16223f",
                font=dict(color="#e0e6ed"),
                margin=dict(t=30, b=30, l=30, r=30),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_notes:
            st.subheader("Evaluation Methodology & Integrity")
            st.markdown(
                """
                - **Chronological Separation:** Transactions are evaluated in strict temporal order to emulate real-world production inference.
                - **Zero Label Leakage:** The Isolation Forest model was trained strictly on feature distributions without `is_fraud` labels.
                - **Threshold Calibration:** Anomaly scores are mapped to binary decisions using the configurable operational risk threshold.
                """
            )
            for note in m.notes:
                st.markdown(f"- {note}")

    elif res.unlabeled_summary is not None:
        st.info("ℹ️ **Unlabeled Dataset Mode:** No ground-truth `is_fraud` labels present in the current dataset.")
        st.markdown(
            "Rather than inventing fabricated precision or recall metrics, we report descriptive anomaly "
            "and risk-score distribution metrics:"
        )
        u = res.unlabeled_summary
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Analyzed Records", f"{u['total_records']:,}")
        u2.metric("Mean Risk Score", f"{u['mean_risk_score']:.1f}")
        u3.metric("90th Percentile Score", f"{u['percentile_90']:.1f}")
        u4.metric("High-Risk Outliers", f"{u['high_risk_count']:,} ({u['high_risk_pct']}%)")


# =============================================================================
# PAGE 7: SETTINGS & ABOUT
# =============================================================================

elif selected_page == "7. Settings & About":
    st.title("⚙️ Detection Configuration & Project Architecture")

    st.subheader("Operational Risk Thresholds & Model Hyperparameters")

    s1, s2, s3 = st.columns(3)
    new_contam = s1.slider(
        "Isolation Forest Contamination",
        min_value=0.005,
        max_value=0.10,
        value=float(st.session_state.config_contamination),
        step=0.005,
        help="Estimated proportion of outliers in the data stream.",
    )
    new_med = s2.slider(
        "Medium Risk Threshold",
        min_value=20.0,
        max_value=60.0,
        value=float(st.session_state.config_threshold_med),
        step=5.0,
    )
    new_high = s3.slider(
        "High Risk Threshold",
        min_value=60.0,
        max_value=90.0,
        value=float(st.session_state.config_threshold_high),
        step=5.0,
    )

    st.markdown("#### Contextual Signal Fusion Weights")
    st.caption("Weights specify relative contribution of orthogonal indicators. Must sum to 1.0.")

    w1, w2, w3, w4, w5 = st.columns(5)
    w_ml = w1.slider("ML Anomaly Weight", 0.05, 0.60, float(st.session_state.risk_weights.ml_anomaly), 0.05)
    w_dev = w2.slider("Historical Deviation", 0.05, 0.60, float(st.session_state.risk_weights.historical_deviation), 0.05)
    w_vel = w3.slider("Velocity Weight", 0.05, 0.40, float(st.session_state.risk_weights.velocity), 0.05)
    w_time = w4.slider("Temporal Weight", 0.05, 0.30, float(st.session_state.risk_weights.temporal_timing), 0.05)
    w_seq = w5.slider("Sequence Weight", 0.05, 0.30, float(st.session_state.risk_weights.sequence_violation), 0.05)

    sum_w = w_ml + w_dev + w_vel + w_time + w_seq
    if not (0.98 <= sum_w <= 1.02):
        st.warning(f"⚠️ Current weights sum to **{sum_w:.2f}** (must be 1.0). Normalization will apply upon save.")

    if st.button("Apply Settings & Re-score Ledger", type="primary"):
        # Normalize weights to exactly 1.0
        norm_factor = sum_w
        st.session_state.risk_weights = RiskWeights(
            ml_anomaly=round(w_ml / norm_factor, 2),
            historical_deviation=round(w_dev / norm_factor, 2),
            velocity=round(w_vel / norm_factor, 2),
            temporal_timing=round(w_time / norm_factor, 2),
            sequence_violation=round(w_seq / norm_factor, 2),
        )
        st.session_state.config_contamination = new_contam
        st.session_state.config_threshold_med = new_med
        st.session_state.config_threshold_high = new_high

        if st.session_state.current_df is not None:
            with st.spinner("Re-training model and recalculating contextual risk..."):
                pipe = FraudDetectionPipeline(
                    contamination=new_contam,
                    risk_weights=st.session_state.risk_weights,
                    threshold_medium=new_med,
                    threshold_high=new_high,
                    alerts_storage_path=os.path.join(BASE_DIR, "data", "alerts.json"),
                )
                st.session_state.pipeline_result = pipe.run(
                    st.session_state.current_df,
                    train_new_model=True,
                    model_checkpoint_path=os.path.join(BASE_DIR, "models", "isolation_forest.joblib"),
                )
                st.success("Pipeline re-executed with updated parameters!")
                st.rerun()

    st.markdown("---")

    # Architecture Documentation & Technical Creed
    st.subheader("Technical Architecture & Design Creed")
    st.markdown(
        """
        ### 1. Algorithm: Why Isolation Forest?
        - **Unsupervised Anomaly Detection:** Real-world banking fraud is dynamic and constantly evolving. Fixed rules and supervised classifiers fail when novel fraud attacks exhibit previously unseen patterns.
        - **Isolation Tree Principles:** Anomalies are few and topologically distant in feature space. In an ensemble of random partitioning trees, anomalous points require fewer splits to isolate (short path lengths) compared to dense clusters of legitimate customer transactions.
        - **Not a Black Box:** Unlike neural networks, tree path lengths provide explainable multidimensional distance metrics.

        ### 2. Distinction: Anomaly ≠ Fraud
        - An unusual transaction (e.g. buying a plane ticket or paying annual insurance) is statistically anomalous but completely legitimate.
        - That is why **Smart Banking Fraud Detector** never relies solely on raw anomaly scores. It fuses:
          1. Isolation Forest multidimensional anomaly score
          2. Individualized account spending history (rejecting false alarms for corporate/affluent accounts)
          3. Temporal velocity ($/sec) and burst frequency
          4. Chronological sequence violations (account draining, dormant reactivation)

        ### 3. Ethical Cyber Boundaries
        - This application is an educational prototype and portfolio piece.
        - It **does not** execute real banking transactions, initiate real account freezes, or connect to production payment rails.
        """
    )
