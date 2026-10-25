"""Audit Intelligence and Decision Support (disclosure step 6, claim 5): rank assets and assign an audit priority.

Rank score combines predicted risk with asset criticality and audit staleness: it is a purely forward-looking early-warning
score. Priority starts from the predicted risk category, is raised one level for critical assets or stale audits that are
already at least Moderate, and has a floor for assets that are ALREADY badly degraded (an asset near the bottom of the trust
scale cannot decay much further, so a decay forecast alone would ignore it; the disclosure also asks that current trust be
considered). The dashboard lists assets by priority, then rank score. All thresholds are design parameters, not fitted.
"""
import numpy as np
import pandas as pd

PRIORITIES = ["Low", "Medium", "High", "Immediate"]
CATEGORY_TO_PRIORITY = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
STALE_WINDOWS = 20
CRITICAL_ASSET = 4
DEGRADED_HIGH_TS = 15.0      # current trust below this: at least High priority
DEGRADED_MEDIUM_TS = 30.0    # current trust below this: at least Medium priority


def trend_label(trend3: float, eps: float = 2.0) -> str:
    return "Declining" if trend3 < -eps else "Increasing" if trend3 > eps else "Stable"


def rank_score(p_high: float, expected_decay: float, criticality: float, windows_since_audit: float) -> float:
    return float(100 * p_high * (0.6 + 0.1 * criticality) + expected_decay + 0.15 * min(windows_since_audit, 30))


def priority(category: str, criticality: float, windows_since_audit: float, current_ts: float = 100.0) -> str:
    level = CATEGORY_TO_PRIORITY[category]
    if level >= 1 and (criticality >= CRITICAL_ASSET or windows_since_audit >= STALE_WINDOWS):
        level = min(3, level + 1)
    if current_ts < DEGRADED_HIGH_TS:
        level = max(level, 2)
    elif current_ts < DEGRADED_MEDIUM_TS:
        level = max(level, 1)
    return PRIORITIES[level]


def prioritise(F: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """F: feature rows, P: engine.predict(F). Returns one row per asset sorted by descending rank score."""
    out = pd.DataFrame({
        "asset": F["asset"].values, "asset_type": F["asset_type"].values, "window": F["window"].values,
        "criticality": F["criticality"].astype(int).values, "current_ts": F["ts"].round(1).values,
        "pred_future_ts": P["pred_future_ts"].round(1).values, "pred_decay": P["pred_decay"].round(1).values,
        "expected_decay": P["expected_decay"].round(1).values, "category": P["category"].values,
        "p_high_or_critical": P["p_high_or_critical"].round(3).values,
        "trend": [trend_label(t) for t in F["trend3"]], "windows_since_audit": F["windows_since_audit"].astype(int).values,
    }, index=F.index)
    out["rank_score"] = [round(rank_score(p, e, c, a), 2) for p, e, c, a in zip(out.p_high_or_critical, out.expected_decay, out.criticality, out.windows_since_audit)]
    out["priority"] = [priority(c, k, a, t) for c, k, a, t in zip(out.category, out.criticality, out.windows_since_audit, out.current_ts)]
    out["already_degraded"] = out["current_ts"] < DEGRADED_HIGH_TS
    out["display_key"] = [PRIORITIES.index(q) * 1000 + s for q, s in zip(out["priority"], out["rank_score"])]
    return out.sort_values("display_key", ascending=False).drop(columns="display_key")
