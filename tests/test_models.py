import numpy as np
import pandas as pd
import pytest

from trustdecay import dataset as D
from trustdecay import explain, features as F, prioritize, trust
from trustdecay.config import CATEGORIES
from trustdecay.evaluation import per_window_topk, topk_effectiveness
from trustdecay.models import TrustEngine
from trustdecay.persist import load_tkg
from trustdecay.simulator import SimConfig, simulate


# ---------------------------------------------------------------- simulator
def test_simulator_is_deterministic_and_the_control_world_differs_only_by_contagion():
    a = simulate(SimConfig(n_assets=14, n_windows=15, seed=4))
    b = simulate(SimConfig(n_assets=14, n_windows=15, seed=4))
    c = simulate(SimConfig(n_assets=14, n_windows=15, seed=4, contagion=0.0))
    ids = lambda g: [(e.s, e.r, e.o, e.t) for e in g.events]
    assert ids(a) == ids(b) and ids(a) != ids(c)
    assert {x["type"] for x in a.assets.values()} == {"server", "database", "application", "firewall", "network_device"} and len(a.assets) == 14


def test_simulated_events_are_valid_time_ordered_per_asset_and_windowed_correctly():
    g = simulate(SimConfig(n_assets=12, n_windows=12, seed=2))
    for aid in g.assets:
        evs = g.asset_events(aid)
        assert [(e.w, e.t) for e in evs] == sorted((e.w, e.t) for e in evs)
        assert all(g.window_of(e.t) == e.w for e in evs)
    assert {e.r for e in g.events} >= {"detected_on", "patched_on", "failed_login", "backup_result", "modified", "security_alert", "audit_completed"}
    patched = {e.s for e in g.events if e.r == "patched_on"}
    assert patched <= {e.s for e in g.events if e.r == "detected_on"}                 # nothing is patched before it exists


# ---------------------------------------------------------------- dataset
def test_chronological_split_is_by_window_disjoint_and_purged():
    ds = D.build(simulate(SimConfig(n_assets=10, n_windows=40, seed=1)), horizon=3)
    sp = D.chronological_split(ds, purge=3)
    tr, va, te = (set(x["window"]) for x in (sp.train, sp.val, sp.test))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert max(tr) + 3 < min(va) and max(va) + 3 < min(te)                             # no training label reaches a later split's inputs
    assert max(tr) < min(va) < max(va) < min(te)
    assert len(sp.train) + len(sp.val) + len(sp.test) < len(ds)                        # purged rows really removed


def test_expanding_folds_never_train_on_the_future():
    ds = D.build(simulate(SimConfig(n_assets=8, n_windows=40, seed=1)), horizon=2)
    folds = list(D.expanding_folds(ds, n_folds=3, purge=2))
    assert len(folds) == 3
    for tr, te in folds:
        assert tr["window"].max() + 2 < te["window"].min()
    assert folds[0][0]["window"].max() < folds[-1][0]["window"].max()


# ---------------------------------------------------------------- engine
def test_engine_output_is_well_formed(svc):
    f = svc.features()
    p = svc.engine.predict(f)
    assert len(p) == len(f) and set(p["category"]) <= set(CATEGORIES)
    probs = p[[f"p_{c.lower()}" for c in CATEGORIES]].to_numpy()
    assert np.allclose(probs.sum(1), 1) and (probs >= 0).all()
    assert p["pred_future_ts"].between(0, 100).all() and (p["pred_decay"] >= 0).all()
    assert np.allclose(p["p_high_or_critical"], p["p_high"] + p["p_critical"])
    assert (p["expected_decay"] >= 2.0 - 1e-9).all() and (p["expected_decay"] <= 38.0 + 1e-9).all()


def test_bundle_records_horizon_split_and_is_reloadable(artifacts):
    e = TrustEngine.load(artifacts / "model_bundle.joblib")
    m = e.bundle["meta"]
    assert m["horizon"] in (1, 3) and m["split"]["train"][1] < m["split"]["val"][0] < m["split"]["test"][0]
    assert e.version.startswith("v1-") and e.features == F.ALL_FEATURES
    with pytest.raises(FileNotFoundError):
        TrustEngine.load(artifacts / "nope.joblib")


def test_metrics_file_is_internally_consistent(artifacts):
    import json
    m = json.loads((artifacts / "metrics.json").read_text())
    t = m["test"]
    assert np.array(t["confusion_matrix"]).sum() == t["n"] == m["split"]["test_rows"]
    assert 0 <= t["accuracy"] <= 1 and t["majority_class_accuracy"] >= 0.25
    assert sum(v["support"] for v in t["per_class"].values()) == t["n"]
    assert set(m["ablation"]) == {"node", "node+temporal", "node+temporal+graph"}
    assert m["horizon_sweep"]["chosen"] == m["horizon_windows"] and "VALIDATION" in m["horizon_sweep"]["criterion"].upper()
    assert m["non_predictive_control"]["accuracy"] > 0.9                                   # non-forward-looking labels are trivially reproduced
    assert m["ranking"]["lowest current trust score"]["precision_at_k"] < m["ranking"]["_meta"]["high_or_critical_base_rate"]
    assert m["data"]["kind"].startswith("SIMULATED") and "not comparable" in m["disclosure_comparison"]["note"]


# ---------------------------------------------------------------- ranking metrics
def test_topk_metrics_on_a_hand_computed_example():
    # window 0 scores [9,1,8,2,3]: top-2 are assets 0 and 2 (y 3,2; decay 40+20) of a window total 40+20+10 = 70
    # window 1 scores [0,1,5,9,2]: top-2 are assets 3 and 2 (y 3,1; decay 35+10) of a window total 10+35+5 = 50
    df = pd.DataFrame({"window": [0] * 5 + [1] * 5, "y": [3, 0, 2, 0, 1, 0, 0, 1, 3, 0], "decay": [40, 0, 20, 0, 10, 0, 0, 10, 35, 5]})
    score = np.array([9, 1, 8, 2, 3, 0, 1, 5, 9, 2])
    r = per_window_topk(df, score, k_frac=0.4)                                             # k = 2 per window
    assert list(r["prec"]) == [1.0, 0.5]                                                   # window 1: only the y=3 asset is High/Critical
    assert list(r["rec"]) == [1.0, 1.0]                                                    # both windows: every High/Critical asset is in the top 2
    assert r["cap"].iloc[0] == pytest.approx(60 / 70)
    assert r["cap"].iloc[1] == pytest.approx(45 / 50)
    m = topk_effectiveness(df, score, k_frac=0.4)
    assert m["precision_at_k"] == pytest.approx(0.75) and m["decay_captured"] == pytest.approx((60 / 70 + 45 / 50) / 2) and m["k_per_window"] == 2


# ---------------------------------------------------------------- prioritisation
def test_priority_rules_and_rank_order():
    assert prioritize.priority("Low", 5, 50) == "Low" and prioritize.priority("Moderate", 3, 0) == "Medium"
    assert prioritize.priority("Moderate", 5, 0) == "High" and prioritize.priority("Moderate", 2, 25) == "High"
    assert prioritize.priority("High", 5, 0) == "Immediate" and prioritize.priority("Critical", 1, 0) == "Immediate"
    assert prioritize.rank_score(.8, 20, 5, 10) > prioritize.rank_score(.8, 20, 2, 10) > prioritize.rank_score(.2, 5, 2, 10)
    assert prioritize.trend_label(-5) == "Declining" and prioritize.trend_label(5) == "Increasing" and prioritize.trend_label(1) == "Stable"


def test_already_degraded_assets_get_a_priority_floor():
    assert prioritize.priority("Low", 1, 0, current_ts=10) == "High"           # severely degraded: cannot decay much, still needs attention
    assert prioritize.priority("Low", 1, 0, current_ts=20) == "Medium"
    assert prioritize.priority("Low", 1, 0, current_ts=80) == "Low"
    assert prioritize.priority("Critical", 1, 0, current_ts=10) == "Immediate"  # a floor never lowers a priority


def test_observed_penalties_are_the_rubric_points_grouped_by_factor():
    row = pd.Series({**{k: 0.0 for k in trust.RUBRIC}, "sev_sum_open": 10.0, "patch_delay_max": 5.0, "backup_failed": 1.0, "windows_since_backup_ok": 2.0,
                     "failed_auth": 2.0, "high_risk_connections": 2.0})
    got = {o["factor"]: o["points"] for o in explain.observed_penalties(row)}
    assert got == {"Unresolved vulnerabilities / missing patches": 15.0,        # 1.3*10 + 0.4*5
                   "Backup failures": 10.0,                                     # 8*1 + 1*2
                   "Exposure to high-risk connected assets": 4.0,               # 2 points per connection
                   "Suspicious authentication activity": 2.4}                   # 1.2*2
    assert list(got)[0] == "Unresolved vulnerabilities / missing patches"
    assert explain.observed_penalties(pd.Series({**{k: 0.0 for k in trust.RUBRIC}, "high_risk_connections": 0.0})) == []


def test_prioritised_table_is_sorted_and_complete(svc):
    r = svc.assess(with_factors=False, log=False)
    keys = [(prioritize.PRIORITIES.index(a["priority"]), a["rank_score"]) for a in r["assets"]]
    assert keys == sorted(keys, reverse=True) and len(r["assets"]) == len(svc.tkg.assets)      # priority first, then rank score
    need = {"asset", "asset_type", "current_ts", "pred_future_ts", "pred_decay", "category", "trend", "priority", "last_assessment", "primary_factor", "p_high_or_critical"}
    assert need <= set(r["assets"][0])                                                    # every column the disclosure lists for the dashboard


# ---------------------------------------------------------------- explanation
def test_explanation_is_model_consistent_sorted_and_empty_for_a_clean_asset(svc):
    f = svc.features()
    w = svc.latest_window
    rows = f[f["window"] == w].head(6)
    out = explain.explain(svc.engine, rows, top=3, min_effect=0.0)
    for items in out:
        eff = [i["effect"] for i in items]
        assert eff == sorted(eff, reverse=True) and len(items) <= 3
        assert all(i["factor"] in explain.FACTORS and i["recommendation"] for i in items)
    # effect definition: base expected decay minus expected decay with that factor neutralised
    r = rows.iloc[[0]]
    name, cols = "Backup failures", explain.FACTORS["Backup failures"]
    manual = svc.engine.predict(r)["expected_decay"].iloc[0] - svc.engine.predict(pd.DataFrame([explain._neutralise(r.iloc[0], cols)], index=r.index))["expected_decay"].iloc[0]
    got = {i["factor"]: i["effect"] for i in explain.explain(svc.engine, r, top=9, min_effect=-1e9)[0]}
    assert got["Backup failures"] == pytest.approx(round(manual, 2))
    clean = rows.iloc[0].copy()
    for cols in explain.FACTORS.values():
        clean = explain._neutralise(clean, cols)
    assert explain.explain(svc.engine, pd.DataFrame([clean]), min_effect=0.0001) == [[]]      # nothing to neutralise -> no factors


def test_recompute_matches_a_freshly_extracted_row(svc):
    f = svc.features()
    r = f.iloc[-1].copy()
    r["ts_node"] = -1.0                               # corrupt derived values, then recompute
    r["ts"] = -1.0
    r = explain.recompute(r)
    assert r["ts_node"] == pytest.approx(f.iloc[-1]["ts_node"]) and r["ts"] == pytest.approx(f.iloc[-1]["ts"])
