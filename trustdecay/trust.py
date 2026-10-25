"""Trust Score rubric and Trust Decay Score (disclosure 8.4).

The rubric is the 'predefined security and audit factors' scoring the disclosure mentions: 100 minus capped penalties for
each audit factor. It is an assumption and is meant to be replaced by an organisation's own policy.
TDS(a, t) = max(0, TS(a, t) - TS(a, t+h)).
"""
import numpy as np

from .config import CATEGORIES, CATEGORY_MIDPOINT, TDS_CUTOFFS

# name -> (weight, cap): penalty = min(cap, weight * value)
RUBRIC = {
    "sev_sum_open": (1.3, 38.0),                 # unresolved vulnerabilities, severity-weighted
    "patch_delay_max": (0.4, 12.0),              # days the oldest open vulnerability has waited
    "failed_auth": (1.2, 12.0),
    "suspicious_access": (3.0, 12.0),
    "unauth_changes_since_audit": (2.0, 10.0),   # configuration drift
    "priv_esc": (5.0, 10.0),
    "backup_failed": (8.0, 8.0),
    "windows_since_backup_ok": (1.0, 8.0),
    "compliance_violations": (4.0, 8.0),
    "security_alerts": (2.5, 12.0),
}
CONNECTION_PENALTY = (2.0, 8.0)                  # per high-risk connected asset


def node_penalties(raw) -> dict:
    return {k: min(cap, w * float(raw[k])) for k, (w, cap) in RUBRIC.items()}


def trust_node(raw) -> float:
    """Trust from the asset's own audit state only (no graph neighbours)."""
    return float(np.clip(100.0 - sum(node_penalties(raw).values()), 0.0, 100.0))


def connection_penalty(high_risk_connections) -> float:
    w, cap = CONNECTION_PENALTY
    return min(cap, w * float(high_risk_connections))


def trust_score(raw, high_risk_connections) -> float:
    return float(np.clip(trust_node(raw) - connection_penalty(high_risk_connections), 0.0, 100.0))


def decay(ts_now, ts_future):
    return np.maximum(0.0, np.asarray(ts_now, dtype=float) - np.asarray(ts_future, dtype=float))


def category(tds: float) -> str:
    lo, mid, hi = TDS_CUTOFFS
    return CATEGORIES[0] if tds <= lo else CATEGORIES[1] if tds <= mid else CATEGORIES[2] if tds <= hi else CATEGORIES[3]


def category_index(tds) -> np.ndarray:
    a = np.asarray(tds, dtype=float)
    return np.digitize(a, TDS_CUTOFFS, right=True)   # right=True: (0,5] -> 0, (5,15] -> 1, (15,30] -> 2, >30 -> 3


def expected_decay(probs) -> np.ndarray:
    """Expected decay in trust points from class probabilities (columns ordered Low..Critical)."""
    mids = np.array([CATEGORY_MIDPOINT[c] for c in CATEGORIES])
    return np.asarray(probs) @ mids
