import re
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from conftest import FIXED_NOW
from trustdecay.webapp import findings


def newev(svc, w, **over):
    e = {"s": "VULN-900", "r": "detected_on", "o": "SRV-01", "t": (svc.tkg.window_start(w) + timedelta(hours=2)).isoformat().replace("+00:00", "Z"), "attrs": {"severity": 9}}
    e.update(over)
    return e


# ---------------------------------------------------------------- service
def test_assess_windows_are_labelled_by_how_the_model_saw_them(svc):
    b, h = svc.split, svc.h
    assert svc.window_split(0) == "train" and svc.window_split(b["test"][0]) == "test"
    assert svc.window_split(svc.latest_window) == "live" and svc.window_split(svc.latest_window - h + 1) == "live"
    assert svc.window_split(b["val"][0]) == "validation"
    r = svc.assess(b["test"][0], with_factors=False, log=False)
    assert r["split"] == "test" and r["horizon_windows"] == h
    with pytest.raises(ValueError):
        svc.assess(999)


def test_reconcile_shows_in_sample_is_better_than_out_of_sample(svc):
    r = svc.reconcile()
    assert r["pairs"] > 0 and "in-sample" in r["note"]
    by = r["by_split"]
    assert by["train"]["category_accuracy"] > by["test"]["category_accuracy"] and by["train"]["future_ts_mae"] < by["test"]["future_ts_mae"]


def test_ingest_extends_the_graph_and_only_affects_current_and_later_windows(svc):
    w0 = svc.latest_window
    before = svc.features().query("asset == 'SRV-01'").set_index("window")
    res = svc.ingest([newev(svc, w0 + 1), newev(svc, w0 + 1, s="BKP-900", r="backup_result", attrs={"success": False})])
    assert res["accepted"] == 2 and svc.latest_window == w0 + 1 and res["windows_touched"] == [w0 + 1]
    after = svc.features().query("asset == 'SRV-01'").set_index("window")
    assert after.loc[w0 + 1, "open_vulns"] == before.loc[w0, "open_vulns"] + 1 and after.loc[w0 + 1, "backup_failed"] == 1
    assert (after.loc[:w0, "ts"] == before.loc[:w0, "ts"]).all()                                # the past is untouched
    assert svc.store.counts()["events"] == 2


def test_a_late_event_changes_only_windows_from_its_own_onward(svc):
    before = svc.features().query("asset == 'SRV-01'").set_index("window")["open_vulns"]
    res = svc.ingest([newev(svc, 10, s="VULN-901")])
    assert res["accepted"] == 1 and svc.latest_window == 43
    after = svc.features().query("asset == 'SRV-01'").set_index("window")["open_vulns"]
    assert (after.loc[:9] == before.loc[:9]).all() and after.loc[10] >= before.loc[10] + 1


def test_duplicates_bad_events_and_batch_limits(svc):
    e = newev(svc, 5, s="VULN-902")
    r1, r2 = svc.ingest([e]), svc.ingest([e])
    assert r1["accepted"] == 1 and r2["accepted"] == 0 and r2["duplicates_ignored"] == 1
    mixed = svc.ingest([newev(svc, 5, s="VULN-903"), newev(svc, 5, o="SRV-99"), newev(svc, 5, attrs={"severity": 99}),
                        newev(svc, 5, t=(FIXED_NOW + timedelta(days=2)).isoformat())])
    assert mixed["accepted"] == 1 and {x["index"] for x in mixed["rejected"]} == {1, 2, 3}
    for bad in ([], "x", None):
        with pytest.raises(ValueError):
            svc.ingest(bad)
    with pytest.raises(ValueError):
        svc.ingest([e] * 5001)


def test_asset_detail_history_is_aligned_to_target_windows(svc):
    d = svc.asset_detail("SRV-01")
    h = svc.h
    assert len(d["history"]) == svc.latest_window + 1 and d["factors"] is not None
    row = d["history"][10]
    assert row["target_window"] == 10 + h and row["observed_future_ts"] == d["history"][10 + h]["ts"]
    assert d["history"][-1]["observed_future_ts"] is None and d["history"][-1]["split"] == "live"
    with pytest.raises(KeyError):
        svc.asset_detail("SRV-99")


def test_logged_runs_store_features_predictions_and_the_snapshot_hash(svc):
    r = svc.assess(with_factors=True, log=True)
    n = len(svc.tkg.assets)
    c = svc.store.counts()
    assert c["runs"] == 1 and c["predictions"] == n and c["features"] == n and c["snapshots"] == 1
    run = svc.store.runs()[0]
    assert run["graph_sha"] == svc.tkg.snapshot(svc.latest_window)["sha256"] == r["graph_snapshot"]["sha256"] and run["model_version"] == svc.engine.version
    p = svc.store.predictions(run["id"], limit=3)
    assert len(p) == 3 and isinstance(p[0]["factors"], list)
    svc.assess(with_factors=False, log=True)
    assert svc.store.counts()["snapshots"] == 1                                                  # same graph state: same snapshot row


# ---------------------------------------------------------------- web
def test_pages_static_and_health(client):
    for p in ("/", "/model", "/repository", "/asset/SRV-01"):
        r = client.get(p)
        assert r.status_code == 200 and b"Prototype on simulated data" in r.data
    assert client.get("/asset/SRV-99").status_code == 404 and client.get("/static/app.css").status_code == 200
    assert client.get("/figures/fig12_confusion_matrix.png").status_code == 200
    for bad in ("../metrics.json", "secret.png", "..%2fmetrics.json"):
        assert client.get(f"/figures/{bad}").status_code == 404
    assert client.get("/health").get_json()["ok"] is True


def test_page_scripts_have_valid_javascript_syntax(client, tmp_path):
    for p in ("/", "/asset/SRV-01", "/repository"):
        html = client.get(p).get_data(as_text=True)
        js = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
        f = tmp_path / "page.js"
        f.write_text(js)
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, (p, r.stderr[:300])


def test_assess_api_filters_and_validation(client):
    j = client.get("/api/assess?explain=0").get_json()
    assert j["total_assets"] == 20 and j["split"] == "live" and j["assets"][0]["rank_score"] >= j["assets"][-1]["rank_score"]
    sub = client.get("/api/assess?explain=0&type=server").get_json()
    assert 0 < len(sub["assets"]) < 20 and {a["asset_type"] for a in sub["assets"]} == {"server"}
    cats = {a["category"] for a in j["assets"]}
    assert {a["category"] for a in client.get(f"/api/assess?explain=0&category={sorted(cats)[0]}").get_json()["assets"]} == {sorted(cats)[0]}
    for q in ("window=abc", "window=-1", "window=9999", "type=toaster", "category=Extreme", "priority=Now"):
        r = client.get("/api/assess?" + q)
        assert r.status_code == 400 and r.get_json()["error"]["code"] == "BAD_REQUEST"
    full = client.get("/api/assess?window=30").get_json()
    a0 = full["assets"][0]
    assert a0["factors"] is not None and a0["primary_factor"] and isinstance(a0["observed_factors"], list) and "already_degraded" in a0


def test_asset_and_graph_endpoints(client):
    d = client.get("/api/assets/SRV-01").get_json()
    assert d["asset"] == "SRV-01" and d["history"] and "category" in d["current"] and isinstance(d["neighbors"], list) and isinstance(d["observed"], list)
    assert client.get("/api/assets/NOPE").status_code == 404
    g = client.get("/api/graph/SRV-01?window=30&back=4").get_json()
    focus = [n for n in g["nodes"] if n.get("focus")]
    assert len(focus) == 1 and (focus[0]["x"], focus[0]["y"]) == (0.0, 0.0) and all(-1.01 <= n["x"] <= 1.01 and -1.01 <= n["y"] <= 1.01 for n in g["nodes"])
    ids = {n["id"] for n in g["nodes"]}
    assert all(e["s"] in ids and e["o"] in ids for e in g["edges"])
    assert client.get("/api/graph/SRV-01?window=99999").status_code == 400 and client.get("/api/graph/SRV-01?back=0").status_code == 400
    assert client.get("/api/graph/NOPE").status_code == 404


def test_event_ingestion_endpoint_end_to_end(client):
    svc = client.application.svc
    e = newev(svc, svc.latest_window + 1, s="VULN-950")
    r = client.post("/api/events", json={"events": [e]})
    assert r.status_code == 200 and r.get_json()["accepted"] == 1
    assert client.get("/health").get_json()["latest_window"] == svc.latest_window
    again = client.post("/api/events", json=[e]).get_json()
    assert again["accepted"] == 0 and again["duplicates_ignored"] == 1
    bad = client.post("/api/events", json={"events": [dict(e, o="SRV-99")]})
    assert bad.status_code == 400 and bad.get_json()["rejected"][0]["errors"]
    assert client.post("/api/events", json={"events": []}).status_code == 400 and client.post("/api/events", data="{x", content_type="application/json").status_code == 400
    new = client.get(f"/api/assess?explain=0&window={svc.latest_window}").get_json()
    assert new["window"] == svc.latest_window


def test_run_and_log_populates_the_repository(client):
    assert client.get("/api/repository/counts").get_json()["runs"] == 0
    r = client.post("/api/assess", json={}).get_json()
    assert r["run_id"] == 1 and r["assets_assessed"] == 20 and len(r["top"]) == 5
    assert client.get("/api/repository/counts").get_json()["predictions"] == 20
    assert client.get("/api/repository/runs").get_json()[0]["id"] == 1
    assert len(client.get("/api/repository/predictions?run_id=1&limit=4").get_json()) == 4
    assert client.get("/api/repository/snapshots").get_json()[0]["sha256"]
    assert client.get("/api/repository/secrets").status_code == 404 and client.get("/api/repository/runs?limit=0").status_code == 400
    assert client.post("/api/assess", json={"window": 5000}).status_code == 400
    assert client.get("/api/reconcile").get_json()["pairs"] > 0 and client.get("/api/model/metrics").get_json()["data"]["kind"].startswith("SIMULATED")


def test_api_key_protects_writes_only(artifacts, tmp_path):
    import shutil
    from trustdecay.webapp import create_app
    shutil.copytree(artifacts, tmp_path / "a")
    c = create_app(artifacts_dir=tmp_path / "a", api_key="k3y", now_fn=lambda: FIXED_NOW).test_client()
    assert c.get("/api/assess?explain=0").status_code == 200 and c.get("/health").status_code == 200
    assert c.post("/api/assess", json={}).status_code == 401 and c.post("/api/events", json={"events": []}, headers={"X-API-Key": "no"}).status_code == 401
    assert c.post("/api/assess", json={}, headers={"X-API-Key": "k3y"}).status_code == 200


# ---------------------------------------------------------------- the page text can never claim more than the numbers
def metrics_stub(acc=0.48, maj=0.53, lo=0.32, hi=0.53, gain=0.013, ctrl=-0.004, sd=0.012, lr_auc=0.65, auc=0.68, node_auc=0.672, full_auc=0.682, recall_mod=0.03):
    return {"test": {"accuracy": acc, "majority_class_accuracy": maj, "auc_high_or_critical": auc, "f1_macro": .29,
                     "per_class": {"Low": {"recall": .8, "support": 300}, "Moderate": {"recall": recall_mod, "support": 100}}},
            "ranking": {"model: P(High or Critical)": {"precision_at_k": .43, "precision_at_k_ci95": [lo, hi]}, "lowest current trust score": {"precision_at_k": .03},
                        "_meta": {"k_fraction": .1, "high_or_critical_base_rate": .31}},
            "baselines": {"logistic_regression": {"auc_high_or_critical": lr_auc, "f1_macro": .3}},
            "ablation": {"node": {"auc_high_or_critical": node_auc}, "node+temporal+graph": {"auc_high_or_critical": full_auc}},
            "graph_signal_control": {"with_relationship_signal": {"auc_gain": {"mean": gain, "sd": sd}}, "control_no_relationship_signal": {"auc_gain": {"mean": ctrl, "sd": sd}}},
            "non_predictive_control": {"label": "past change", "accuracy": .99}}


def test_findings_adapt_to_the_numbers():
    weak = " ".join(findings(metrics_stub()))
    assert "does not beat the trivial baseline" in weak and "Moderate decay is almost never identified" in weak
    assert "interval sits above chance" in weak and "worse than random" in weak and "about as good" in weak
    assert "do not clearly improve" in weak and "inconclusive" in weak and "99%" in weak
    strong = " ".join(findings(metrics_stub(acc=.8, maj=.5, lo=.28, hi=.6, gain=.09, ctrl=0.0, sd=.01, lr_auc=.5, auc=.85, node_auc=.7, full_auc=.85, recall_mod=.5)))
    assert "does not beat" not in strong and "Sanity check passed" in strong and "about as good" not in strong and "do not clearly improve" not in strong
    assert "includes chance" in " ".join(findings(metrics_stub(lo=.2, hi=.5)))
