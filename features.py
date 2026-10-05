"""Trust Feature Extraction (disclosure step 4, claim 3): walks the Temporal Knowledge Graph and produces one Trust
Feature Vector per (asset, window). Only information available at the END of window t is used for the vector of window t.

Pass 1 scans each asset's own event edges in time order (open vulnerabilities, backup and audit history, drift).
Pass 2 adds relationship-derived features from structurally connected assets, and the temporal history features.
"""
import numpy as np
import pandas as pd

from .config import ASSET_TYPES, CRITICAL_SEVERITY, HIGH_RISK_TS, HORIZON
from .tkg import TemporalKG
from . import trust

CAP_BACKUP, CAP_AUDIT = 10, 30

NODE_RAW = ["open_vulns", "max_sev_open", "sev_sum_open", "patch_delay_max", "patch_delay_mean", "new_vulns",
            "failed_auth", "suspicious_access", "config_changes", "unauth_changes_since_audit", "priv_esc",
            "backup_failed", "windows_since_backup_ok", "compliance_violations", "security_alerts", "windows_since_audit"]
STATIC = ["criticality"] + [f"type_{t}" for t in ASSET_TYPES]
NODE_FEATURES = NODE_RAW + ["ts_node"] + STATIC
TEMPORAL_FEATURES = ["ts_node_lag1", "ts_node_lag3", "trend1_node", "trend3_node", "failed_auth_3", "alerts_3",
                     "backup_fail_5", "new_vulns_3", "vulns_delta", "ts_node_std5"]
GRAPH_FEATURES = ["ts", "high_risk_connections", "degree", "nbr_mean_ts_node", "nbr_min_ts_node",
                  "nbr_open_crit_vulns", "nbr_alerts_mean", "nbr_ts_drop", "trend1", "trend3", "ts_lag1", "ts_lag3"]
FEATURE_SETS = {"node": NODE_FEATURES,
                "node+temporal": NODE_FEATURES + TEMPORAL_FEATURES,
                "node+temporal+graph": NODE_FEATURES + TEMPORAL_FEATURES + GRAPH_FEATURES}
ALL_FEATURES = FEATURE_SETS["node+temporal+graph"]


def _scan_asset(tkg: TemporalKG, aid: str, n_windows: int) -> list[dict]:
    evs = tkg.asset_events(aid)
    i, rows = 0, []
    open_v: dict[str, tuple] = {}                          # vuln -> (severity, detected window)
    last_bk_ok, last_audit, unauth = None, None, 0
    for w in range(n_windows):
        c = dict.fromkeys(["new_vulns", "failed_auth", "suspicious_access", "config_changes", "priv_esc", "backup_failed",
                           "compliance_violations", "security_alerts"], 0)
        bk_seen = False
        while i < len(evs) and evs[i].w <= w:
            e = evs[i]
            i += 1
            if e.w < w:
                continue                                    # cannot happen for sorted input, guards late edits
            r = e.r
            if r == "detected_on":
                open_v[e.s] = (float(e.attrs["severity"]), w)
                c["new_vulns"] += 1
            elif r == "patched_on":
                open_v.pop(e.s, None)
            elif r == "failed_login":
                c["failed_auth"] += int(e.attrs.get("count", 1))
            elif r == "suspicious_access":
                c["suspicious_access"] += int(e.attrs.get("count", 1))
            elif r == "modified":
                c["config_changes"] += 1
                if not e.attrs.get("authorized", True):
                    unauth += 1
            elif r == "privilege_escalation":
                c["priv_esc"] += 1
            elif r == "backup_result":
                bk_seen = True
                if e.attrs.get("success"):
                    last_bk_ok = w
                else:
                    c["backup_failed"] = 1
            elif r == "compliance_violation":
                c["compliance_violations"] += 1
            elif r == "security_alert":
                c["security_alerts"] += 1
            elif r == "audit_completed":
                last_audit, unauth = w, 0                   # remediation itself arrives as separate patched_on events
        sev = [s for s, _ in open_v.values()]
        ages = [w - dw + 1 for _, dw in open_v.values()]
        rows.append({
            "asset": aid, "window": w, **c,
            "open_vulns": len(sev), "max_sev_open": max(sev, default=0.0), "sev_sum_open": float(sum(sev)),
            "patch_delay_max": float(max(ages, default=0)), "patch_delay_mean": float(np.mean(ages)) if ages else 0.0,
            "unauth_changes_since_audit": unauth,
            "windows_since_backup_ok": float(min(CAP_BACKUP, (w - last_bk_ok) if last_bk_ok is not None else w + 1)),
            "windows_since_audit": float(min(CAP_AUDIT, (w - last_audit) if last_audit is not None else w + 1)),
            "open_crit_vulns": sum(1 for s in sev if s >= CRITICAL_SEVERITY),
        })
    return rows


def extract(tkg: TemporalKG, n_windows: int | None = None) -> pd.DataFrame:
    """Trust Feature Vectors for every asset and window 0..n_windows-1, sorted by (asset, window)."""
    W = n_windows if n_windows is not None else tkg.n_windows
    if W <= 0:
        return pd.DataFrame(columns=["asset", "window"] + ALL_FEATURES)
    df = pd.DataFrame([r for a in tkg.assets for r in _scan_asset(tkg, a, W)])
    df["ts_node"] = [trust.trust_node(r) for r in df[list(trust.RUBRIC)].to_dict("records")]
    meta = pd.DataFrame(tkg.assets).T
    df["criticality"] = df["asset"].map(meta["criticality"]).astype(float)
    df["asset_type"] = df["asset"].map(meta["type"])
    for t in ASSET_TYPES:
        df[f"type_{t}"] = (df["asset_type"] == t).astype(float)

    # ---------- relationship (graph) features, window by window
    idx = {(a, w): i for i, (a, w) in enumerate(zip(df["asset"], df["window"]))}
    ts_node = df["ts_node"].to_numpy()
    open_crit = df["open_crit_vulns"].to_numpy()
    alerts = df["security_alerts"].to_numpy()
    n = len(df)
    hrc, deg, nmean, nmin, ncrit, nalert, ndrop = (np.zeros(n) for _ in range(7))
    for a in tkg.assets:
        for w in range(W):
            i = idx[(a, w)]
            nb = sorted(tkg.neighbors(a, w))
            deg[i] = len(nb)
            if not nb:
                nmean[i], nmin[i] = 100.0, 100.0
                continue
            js = [idx[(b, w)] for b in nb]
            vals = ts_node[js]
            hrc[i] = float((vals < HIGH_RISK_TS).sum())
            nmean[i], nmin[i] = float(vals.mean()), float(vals.min())
            ncrit[i] = float(open_crit[js].sum())
            nalert[i] = float(alerts[js].mean())
            if w > 0:
                ndrop[i] = float(np.mean([ts_node[j] - ts_node[idx[(b, w - 1)]] for j, b in zip(js, nb)]))
    df["high_risk_connections"], df["degree"] = hrc, deg
    df["nbr_mean_ts_node"], df["nbr_min_ts_node"], df["nbr_open_crit_vulns"] = nmean, nmin, ncrit
    df["nbr_alerts_mean"], df["nbr_ts_drop"] = nalert, ndrop
    df["ts"] = [trust.trust_score(r, h) for r, h in zip(df[list(trust.RUBRIC)].to_dict("records"), hrc)]

    # ---------- temporal history (strictly backwards-looking)
    df = df.sort_values(["asset", "window"]).reset_index(drop=True)
    g = df.groupby("asset", sort=False)

    def lag(col, k):
        return g[col].shift(k).fillna(df[col] if k else 0)     # before the timeline starts, treat history as "unchanged"

    for col in ("ts_node", "ts"):
        pre = "ts_node" if col == "ts_node" else "ts"
        df[f"{pre}_lag1"], df[f"{pre}_lag3"] = lag(col, 1), lag(col, 3)
    df["trend1_node"], df["trend3_node"] = df["ts_node"] - df["ts_node_lag1"], df["ts_node"] - df["ts_node_lag3"]
    df["trend1"], df["trend3"] = df["ts"] - df["ts_lag1"], df["ts"] - df["ts_lag3"]
    roll = lambda col, k, fn: g[col].transform(lambda s: getattr(s.rolling(k, min_periods=1), fn)())
    df["failed_auth_3"], df["alerts_3"] = roll("failed_auth", 3, "mean"), roll("security_alerts", 3, "mean")
    df["backup_fail_5"], df["new_vulns_3"] = roll("backup_failed", 5, "sum"), roll("new_vulns", 3, "sum")
    df["vulns_delta"] = df["open_vulns"] - lag("open_vulns", 1)
    df["ts_node_std5"] = g["ts_node"].transform(lambda s: s.rolling(5, min_periods=2).std()).fillna(0.0)
    return df


def add_targets(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Labels for the supervised task: trust h windows ahead, the decay, and its category. Rows with no future are dropped."""
    d = df.copy()
    d["ts_future"] = d.groupby("asset")["ts"].shift(-horizon)
    d = d.dropna(subset=["ts_future"]).copy()
    d["decay"] = trust.decay(d["ts"], d["ts_future"])
    d["y"] = trust.category_index(d["decay"])
    return d
