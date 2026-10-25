from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trustdecay import features as F
from trustdecay import trust
from trustdecay.schema import Event
from trustdecay.simulator import SimConfig, simulate
from trustdecay.tkg import TemporalKG


def world():
    g = TemporalKG()
    for a, t in (("SRV-01", "server"), ("APP-01", "application"), ("DB-01", "database")):
        g.add_asset(a, t, 3)
    g.add_structural("APP-01", "runs_on", "SRV-01")
    g.add_structural("SRV-01", "hosts", "DB-01")
    return g


def add(g, s, r, o, w, **attrs):
    g.add_event(Event(s, r, o, g.window_start(w) + timedelta(hours=5), w, attrs))


def row(df, asset, w):
    return df[(df.asset == asset) & (df.window == w)].iloc[0]


def test_vulnerability_lifecycle_patch_delay_and_backup_drift_audit_features():
    g = world()
    add(g, "VULN-1", "detected_on", "SRV-01", 0, severity=8)
    add(g, "VULN-2", "detected_on", "SRV-01", 1, severity=5)
    add(g, "VULN-1", "patched_on", "SRV-01", 2)
    add(g, "BKP-1", "backup_result", "SRV-01", 0, success=True)
    add(g, "BKP-2", "backup_result", "SRV-01", 1, success=False)
    add(g, "BKP-3", "backup_result", "SRV-01", 2, success=False)
    add(g, "ADMIN-1", "modified", "SRV-01", 1, authorized=False)
    add(g, "ADMIN-1", "modified", "SRV-01", 2, authorized=True)
    add(g, "ADMIN-2", "modified", "SRV-01", 2, authorized=False)
    add(g, "AUD-3", "audit_completed", "SRV-01", 3)
    add(g, "USER-1", "failed_login", "SRV-01", 1, count=4)
    add(g, "SYSTEM", "security_alert", "SRV-01", 1, severity=3)
    add(g, "USER-1", "privilege_escalation", "SRV-01", 2)
    add(g, "CTL-1", "compliance_violation", "SRV-01", 2)
    add(g, "BKP-9", "backup_result", "DB-01", 3, success=True)                     # ensure 4 windows exist
    df = F.extract(g)
    r0, r1, r2, r3 = (row(df, "SRV-01", w) for w in range(4))
    assert (r0.open_vulns, r0.sev_sum_open, r0.patch_delay_max, r0.new_vulns) == (1, 8, 1, 1)
    assert (r1.open_vulns, r1.sev_sum_open, r1.max_sev_open, r1.patch_delay_max, r1.patch_delay_mean) == (2, 13, 8, 2, 1.5)
    assert (r2.open_vulns, r2.sev_sum_open, r2.patch_delay_max) == (1, 5, 2)          # VULN-1 patched, VULN-2 has waited 2 windows
    assert (r0.backup_failed, r1.backup_failed, r2.backup_failed) == (0, 1, 1)
    assert (r0.windows_since_backup_ok, r1.windows_since_backup_ok, r2.windows_since_backup_ok) == (0, 1, 2)
    assert (r1.unauth_changes_since_audit, r2.unauth_changes_since_audit, r3.unauth_changes_since_audit) == (1, 2, 0)   # audit resets drift
    assert (r1.failed_auth, r1.security_alerts, r2.priv_esc, r2.compliance_violations, r2.config_changes) == (4, 1, 1, 1, 2)
    assert (r3.windows_since_audit, r2.windows_since_audit) == (0, 3)
    manual = 100 - (1.3 * 13 + 0.4 * 2 + 1.2 * 4 + 2.0 * 1 + 8 * 1 + 1.0 * 1 + 2.5 * 1)
    assert r1.ts_node == pytest.approx(manual)


def test_rubric_caps_and_bounds():
    worst = {k: 1e6 for k in trust.RUBRIC}
    assert trust.trust_node(worst) == 0.0
    assert trust.trust_node({k: 0 for k in trust.RUBRIC}) == 100.0
    assert trust.connection_penalty(99) == 8.0 and trust.connection_penalty(0) == 0.0
    assert trust.trust_score({k: 0 for k in trust.RUBRIC}, 3) == 94.0


def test_relationship_features_come_from_structurally_connected_assets():
    g = world()
    for i in range(4):
        add(g, f"VULN-{i}", "detected_on", "SRV-01", 0, severity=10)
    add(g, "USER-1", "failed_login", "SRV-01", 0, count=10)
    add(g, "SYSTEM", "security_alert", "DB-01", 0, severity=2)
    df = F.extract(g)
    srv, app, db = (row(df, a, 0) for a in ("SRV-01", "APP-01", "DB-01"))
    assert srv.ts_node < 60 and app.ts_node > 90                                        # SRV-01 is the high-risk asset
    assert app.high_risk_connections == 1 and db.high_risk_connections == 1 and srv.high_risk_connections == 0
    assert app.ts == pytest.approx(app.ts_node - 2.0)                                   # 2 points per high-risk connection
    assert app.nbr_min_ts_node == pytest.approx(srv.ts_node) and app.nbr_open_crit_vulns == 4 and app.degree == 1 and srv.degree == 2
    assert srv.nbr_alerts_mean == pytest.approx(0.5)                                    # mean alerts of {APP-01: 0, DB-01: 1}


def test_structural_edge_validity_changes_relationship_features_over_time():
    g = TemporalKG()
    for a, t in (("SRV-01", "server"), ("APP-01", "application")):
        g.add_asset(a, t, 3)
    g.add_structural("APP-01", "runs_on", "SRV-01", w_start=2)
    for w in range(4):
        add(g, f"BKP-{w}", "backup_result", "APP-01", w, success=True)
        add(g, f"BKP-{w}b", "backup_result", "SRV-01", w, success=True)
    df = F.extract(g)
    assert [row(df, "APP-01", w).degree for w in range(4)] == [0, 0, 1, 1]
    assert row(df, "APP-01", 0).nbr_mean_ts_node == 100.0                               # no neighbours: neutral fill


def test_NO_LOOKAHEAD_features_for_window_t_do_not_change_when_later_events_are_added():
    """The property that matters most: the vector for window t must be computable from events up to t only."""
    full = simulate(SimConfig(n_assets=16, n_windows=30, seed=11))
    df_full = F.extract(full)
    for t in (9, 17):
        part = TemporalKG()
        for a, m in full.assets.items():
            part.add_asset(a, m["type"], m["criticality"])
        for u, v, d in full.g.edges(data=True):
            if "w_start" in d:
                part.add_structural(u, d["relation"], v, d["w_start"], d["w_end"])
        for e in full.events:
            if e.w <= t:
                part.add_event(e)
        df_part = F.extract(part, n_windows=t + 1)
        a = df_full[df_full.window <= t].sort_values(["asset", "window"]).reset_index(drop=True)
        b = df_part.sort_values(["asset", "window"]).reset_index(drop=True)
        cols = F.ALL_FEATURES
        pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9, atol=1e-9)


def test_targets_look_strictly_forward_and_drop_rows_without_a_future():
    g = simulate(SimConfig(n_assets=12, n_windows=20, seed=5))
    base = F.extract(g)
    for h in (1, 3):
        d = F.add_targets(base, h)
        assert d["window"].max() == 20 - 1 - h and len(d) == 12 * (20 - h)
        chk = d.merge(base[["asset", "window", "ts"]].rename(columns={"window": "w2", "ts": "ts_h"}).assign(window=lambda x: x["w2"] - h), on=["asset", "window"])
        assert np.allclose(chk["ts_future"], chk["ts_h"])
        assert (d["decay"] >= 0).all() and set(d["y"].unique()) <= {0, 1, 2, 3}


@pytest.mark.parametrize("tds,cat", [(0, "Low"), (5, "Low"), (5.01, "Moderate"), (15, "Moderate"), (15.01, "High"), (30, "High"), (30.01, "Critical"), (99, "Critical")])
def test_category_thresholds_match_the_disclosure_table(tds, cat):
    assert trust.category(tds) == cat and trust.CATEGORIES[int(trust.category_index(tds))] == cat


def test_decay_is_never_negative_and_improvement_counts_as_zero():
    assert list(trust.decay([80, 50, 70], [60, 55, 70])) == [20, 0, 0]


def test_feature_sets_are_nested_and_unique():
    a, b, c = (F.FEATURE_SETS[k] for k in ("node", "node+temporal", "node+temporal+graph"))
    assert set(a) < set(b) < set(c) and len(set(c)) == len(c)
