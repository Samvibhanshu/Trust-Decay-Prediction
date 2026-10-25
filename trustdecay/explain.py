"""Explainable Trust Decay Assessment (claim 6).

For each prediction, which recorded audit factors are pushing the predicted decay up? Method: counterfactual substitution.
For each factor group, reset that group's inputs to a healthy reference (no vulnerabilities, no failed logins, ...),
recompute the trust score and every feature derived from it, re-predict, and report the drop in expected decay.
This says how much the MODEL's prediction depends on each factor for this asset. It is an attribution of the model,
not a proof of causation in the enterprise.
"""
import numpy as np
import pandas as pd

from . import trust
from .features import ALL_FEATURES

FACTORS = {
    "Unresolved vulnerabilities / missing patches": ["open_vulns", "max_sev_open", "sev_sum_open", "patch_delay_max", "patch_delay_mean", "new_vulns", "new_vulns_3", "vulns_delta"],
    "Suspicious authentication activity": ["failed_auth", "suspicious_access", "failed_auth_3"],
    "Configuration drift": ["config_changes", "unauth_changes_since_audit"],
    "Privilege escalation": ["priv_esc"],
    "Backup failures": ["backup_failed", "windows_since_backup_ok", "backup_fail_5"],
    "Compliance violations": ["compliance_violations"],
    "Security alert activity": ["security_alerts", "alerts_3"],
    "Exposure to high-risk connected assets": ["high_risk_connections", "nbr_open_crit_vulns", "nbr_alerts_mean", "nbr_ts_drop"],
    "Overdue audit": ["windows_since_audit"],
}
OBSERVED = {
    "Unresolved vulnerabilities / missing patches": ["sev_sum_open", "patch_delay_max"],
    "Suspicious authentication activity": ["failed_auth", "suspicious_access"],
    "Configuration drift": ["unauth_changes_since_audit"],
    "Privilege escalation": ["priv_esc"],
    "Backup failures": ["backup_failed", "windows_since_backup_ok"],
    "Compliance violations": ["compliance_violations"],
    "Security alert activity": ["security_alerts"],
}


def observed_penalties(row, min_points: float = 1.0) -> list[dict]:
    """What is actually costing this asset trust right now, straight from the rubric (observed audit evidence,
    not a model attribution): [{factor, points, recommendation}] sorted by points."""
    pen = trust.node_penalties(row)
    items = [(name, sum(pen[k] for k in keys)) for name, keys in OBSERVED.items()]
    items.append(("Exposure to high-risk connected assets", trust.connection_penalty(row["high_risk_connections"])))
    return [{"factor": n, "points": round(float(p), 1), "recommendation": RECOMMENDATION[n]} for n, p in sorted(items, key=lambda x: -x[1]) if p >= min_points]


HEALTHY = {"nbr_mean_ts_node": 100.0, "nbr_min_ts_node": 100.0}       # everything else resets to 0
RECOMMENDATION = {
    "Unresolved vulnerabilities / missing patches": "Review the open vulnerability list and patch SLA; verify remediation evidence.",
    "Suspicious authentication activity": "Review authentication logs and the accounts involved; check lockout and MFA policy.",
    "Configuration drift": "Reconcile unauthorised changes against the change-management record.",
    "Privilege escalation": "Review privilege grants and the accounts that escalated; confirm approvals.",
    "Backup failures": "Test backup and restore for this asset; confirm recovery point objectives.",
    "Compliance violations": "Review the violated controls and their remediation owners.",
    "Security alert activity": "Triage recent alerts for this asset and confirm none are unaddressed.",
    "Exposure to high-risk connected assets": "Assess the connected high-risk assets; consider segmentation or joint review.",
    "Overdue audit": "Schedule the next formal audit; the last one is stale.",
}


def recompute(r: pd.Series) -> pd.Series:
    """Recompute the trust score and every feature derived from it after raw audit columns were edited."""
    r["ts_node"] = trust.trust_node(r)
    r["ts"] = trust.trust_score(r, r["high_risk_connections"])
    r["trend1_node"], r["trend3_node"] = r["ts_node"] - r["ts_node_lag1"], r["ts_node"] - r["ts_node_lag3"]
    r["trend1"], r["trend3"] = r["ts"] - r["ts_lag1"], r["ts"] - r["ts_lag3"]
    return r


def _neutralise(row: pd.Series, cols: list) -> pd.Series:
    r = row.copy()
    for c in cols:
        r[c] = HEALTHY.get(c, 0.0)
    return recompute(r)


def explain(engine, F: pd.DataFrame, top: int = 3, min_effect: float = 0.5) -> list[list[dict]]:
    """For each row of F returns up to `top` factors [{factor, effect, recommendation}] sorted by effect (trust points of
    expected decay). Factors with effect below `min_effect` are omitted; an asset with none has no dominant factor."""
    base = engine.predict(F)["expected_decay"].to_numpy()
    effects = np.zeros((len(F), len(FACTORS)))
    for j, (name, cols) in enumerate(FACTORS.items()):
        mod = F.copy()
        neutral = pd.DataFrame([_neutralise(r, cols) for _, r in F.iterrows()], index=F.index)
        for c in ALL_FEATURES:
            mod[c] = neutral[c]
        effects[:, j] = base - engine.predict(mod)["expected_decay"].to_numpy()
    names = list(FACTORS)
    out = []
    for i in range(len(F)):
        order = np.argsort(-effects[i])[:top]
        out.append([{"factor": names[j], "effect": round(float(effects[i, j]), 2), "recommendation": RECOMMENDATION[names[j]]}
                    for j in order if effects[i, j] >= min_effect])
    return out
