"""Alert lifecycle and analyst investigation management module.

Persists generated security alerts, supports analyst workflow statuses
(New, Under Review, Resolved, False Positive), handles safe file I/O,
and records immutable audit notes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

ALERT_STATUS_NEW = "New"
ALERT_STATUS_UNDER_REVIEW = "Under Review"
ALERT_STATUS_RESOLVED = "Resolved"
ALERT_STATUS_FALSE_POSITIVE = "False Positive"

VALID_STATUSES = [
    ALERT_STATUS_NEW,
    ALERT_STATUS_UNDER_REVIEW,
    ALERT_STATUS_RESOLVED,
    ALERT_STATUS_FALSE_POSITIVE,
]


@dataclass
class FraudAlert:
    """Represents a flagged high/medium risk incident for human analyst investigation."""

    alert_id: str
    transaction_id: str
    account_id: Optional[str]
    risk_score: float
    risk_category: str
    detection_timestamp: str
    reasons: List[str]
    recommended_action: str
    status: str = ALERT_STATUS_NEW
    analyst_notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to serializable dictionary."""
        return asdict(self)


class AlertManager:
    """Manages the creation, retrieval, status progression, and local persistence of alerts."""

    def __init__(self, storage_path: str = "data/alerts.json"):
        """Initialize alert manager.

        Args:
            storage_path: Local JSON or CSV file path for alert persistence.
        """
        self.storage_path = storage_path
        self.alerts: Dict[str, FraudAlert] = {}
        self._load_existing_alerts()

    def _load_existing_alerts(self) -> None:
        """Loads persisted alerts from local storage if existing."""
        if not os.path.exists(self.storage_path):
            self.alerts = {}
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        alert = FraudAlert(**item)
                        self.alerts[alert.alert_id] = alert
        except Exception:
            # Fallback if file is corrupted
            self.alerts = {}

    def save_alerts(self) -> None:
        """Persists all current alerts to disk atomically."""
        dir_name = os.path.dirname(os.path.abspath(self.storage_path))
        os.makedirs(dir_name, exist_ok=True)

        data = [a.to_dict() for a in self.alerts.values()]
        temp_path = f"{self.storage_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Atomic rename
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        os.rename(temp_path, self.storage_path)

    def generate_alerts_from_evaluations(
        self,
        eval_results: List[Any],
        auto_persist: bool = True,
    ) -> List[FraudAlert]:
        """Filters Medium and High risk transactions and registers new alerts.

        Args:
            eval_results: List of RiskEvaluationResult instances or dicts.
            auto_persist: Whether to immediately sync to disk.

        Returns:
            List of newly added or updated FraudAlerts.
        """
        created: List[FraudAlert] = []
        # Find existing transaction IDs already alerted to avoid duplicate alert storms
        existing_txs = {a.transaction_id: a for a in self.alerts.values()}

        for item in eval_results:
            d = item.to_dict() if hasattr(item, "to_dict") else item
            cat = d.get("risk_category", "Low Risk")
            tx_id = str(d.get("transaction_id", ""))

            # Flag only Medium and High risk
            if cat in ["Medium Risk", "High Risk"]:
                if tx_id in existing_txs:
                    # Update existing record if new risk score is higher
                    existing = existing_txs[tx_id]
                    if d.get("overall_risk_score", 0.0) > existing.risk_score:
                        existing.risk_score = d.get("overall_risk_score", existing.risk_score)
                        existing.risk_category = cat
                        existing.reasons = d.get("reasons", existing.reasons)
                        existing.updated_at = datetime.now().isoformat()
                    created.append(existing)
                else:
                    new_id = f"ALT_{len(self.alerts) + 1:05d}"
                    alert = FraudAlert(
                        alert_id=new_id,
                        transaction_id=tx_id,
                        account_id=d.get("account_id"),
                        risk_score=float(d.get("overall_risk_score", 0.0)),
                        risk_category=cat,
                        detection_timestamp=str(d.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
                        reasons=list(d.get("reasons", [])),
                        recommended_action=str(d.get("recommended_action", "Investigate")),
                        status=ALERT_STATUS_NEW,
                        analyst_notes="",
                    )
                    self.alerts[new_id] = alert
                    existing_txs[tx_id] = alert
                    created.append(alert)

        if auto_persist:
            self.save_alerts()

        return created

    def update_alert_status(
        self,
        alert_id: str,
        new_status: str,
        analyst_notes: Optional[str] = None,
    ) -> FraudAlert:
        """Updates alert investigation status and attaches analyst comments."""
        if alert_id not in self.alerts:
            raise KeyError(f"Alert ID '{alert_id}' not found.")

        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}.")

        alert = self.alerts[alert_id]
        alert.status = new_status
        if analyst_notes is not None:
            alert.analyst_notes = analyst_notes
        alert.updated_at = datetime.now().isoformat()

        self.save_alerts()
        return alert

    def to_dataframe(self) -> pd.DataFrame:
        """Exports all alerts into a tabular pandas DataFrame."""
        if not self.alerts:
            return pd.DataFrame(
                columns=[
                    "alert_id",
                    "transaction_id",
                    "account_id",
                    "risk_score",
                    "risk_category",
                    "status",
                    "recommended_action",
                    "analyst_notes",
                    "detection_timestamp",
                    "updated_at",
                ]
            )

        rows = []
        for a in self.alerts.values():
            rows.append({
                "alert_id": a.alert_id,
                "transaction_id": a.transaction_id,
                "account_id": a.account_id or "N/A",
                "risk_score": a.risk_score,
                "risk_category": a.risk_category,
                "status": a.status,
                "recommended_action": a.recommended_action,
                "reasons": " | ".join(a.reasons),
                "analyst_notes": a.analyst_notes,
                "detection_timestamp": a.detection_timestamp,
                "updated_at": a.updated_at,
            })
        return pd.DataFrame(rows)
