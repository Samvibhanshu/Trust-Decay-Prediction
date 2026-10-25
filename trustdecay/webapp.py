"""Audit Prioritization Dashboard and JSON API (disclosure step 6; 'web-based dashboard, e.g. Streamlit or Dash' -> Flask here)."""
import hmac
import json
import math
import os

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from .config import ARTIFACTS, CATEGORIES, ROOT
from .models import TrustEngine
from .persist import load_tkg
from .prioritize import PRIORITIES
from .service import TrustService
from .store import Store

FIGURES = {
    "fig12_confusion_matrix.png": "Fig 12. Confusion matrix (test)", "fig13_feature_importance.png": "Fig 13. Feature importance",
    "fig14_trust_evolution.png": "Fig 14. Trust evolution", "fig15_decay_distribution.png": "Fig 15. Decay distribution",
    "fig_horizon_sweep.png": "Horizon sweep (validation)", "fig_ablation_ranking.png": "Ablation and audit ranking"}
TYPE_ORDER = {"focus": 0, "asset": 1, "entity": 2}


def findings(m: dict) -> list[str]:
    """Plain-language conclusions computed FROM metrics.json, so the page can never claim more than the numbers show."""
    t, rk, out = m["test"], m["ranking"], []
    maj = t["majority_class_accuracy"]
    if t["accuracy"] <= maj + 0.01:
        out.append(f"As a classifier the model does not beat the trivial baseline: accuracy {t['accuracy']:.1%} versus {maj:.1%} for always predicting the most common category.")
    else:
        out.append(f"Category accuracy is {t['accuracy']:.1%} versus {maj:.1%} for always predicting the most common category.")
    weak = [c for c, v in t["per_class"].items() if v["recall"] < 0.10 and v["support"] >= 20]
    if weak:
        out.append(f"{' and '.join(weak)} decay is almost never identified (recall under 10%).")
    pk = rk["model: P(High or Critical)"]
    lo, hi, base = pk["precision_at_k_ci95"][0], pk["precision_at_k_ci95"][1], rk["_meta"]["high_or_critical_base_rate"]
    lift = pk["precision_at_k"] / base
    out.append(f"Ranking is where it helps: the top-{int(rk['_meta']['k_fraction'] * 100)}% audit list is right {pk['precision_at_k']:.0%} of the time (95% interval {lo:.0%}-{hi:.0%}) against {base:.0%} by chance, "
               f"a {lift:.1f}x lift" + (", and the interval sits above chance." if lo > base else ", but the interval includes chance, so this is not conclusive."))
    naive = rk["lowest current trust score"]["precision_at_k"]
    out.append(f"Auditing the lowest-trust assets first is worse than random ({naive:.0%}): an asset already at low trust has little left to lose.")
    disp = rk.get("dashboard display order (priority, then rank score)")
    if disp and disp["precision_at_k"] < pk["precision_at_k"] - 0.02:
        out.append(f"The dashboard's display order also lifts assets that are already badly degraded, which future-decay precision does not reward: {disp['precision_at_k']:.0%} for the displayed order versus {pk['precision_at_k']:.0%} for the pure early-warning score.")
    lr = m["baselines"]["logistic_regression"]
    if lr["auc_high_or_critical"] >= t["auc_high_or_critical"] - 0.03:
        out.append(f"A plain logistic regression is about as good (AUC {lr['auc_high_or_critical']:.2f} vs {t['auc_high_or_critical']:.2f}; macro-F1 {lr['f1_macro']:.2f} vs {t['f1_macro']:.2f}), so the Random Forest's extra flexibility is not shown to matter here.")
    ab = m["ablation"]
    d = ab["node+temporal+graph"]["auc_high_or_critical"] - ab["node"]["auc_high_or_critical"]
    if d < 0.02:
        out.append(f"Temporal-history and graph features do not clearly improve on the asset's own current state (AUC {ab['node']['auc_high_or_critical']:.3f} vs {ab['node+temporal+graph']['auc_high_or_critical']:.3f}).")
    g = m["graph_signal_control"]
    a, c = g["with_relationship_signal"]["auc_gain"], g["control_no_relationship_signal"]["auc_gain"]
    if a["mean"] - c["mean"] > 2 * max(a["sd"], c["sd"]):
        out.append(f"Sanity check passed: graph features add {a['mean']:+.3f} AUC when the simulator links connected assets and {c['mean']:+.3f} when it does not.")
    else:
        out.append(f"The graph-feature sanity check is inconclusive: gain {a['mean']:+.3f} AUC (sd {a['sd']:.3f}) with relationship signal versus {c['mean']:+.3f} (sd {c['sd']:.3f}) without, which is within seed-to-seed noise.")
    n = m["non_predictive_control"]
    out.append(f"For contrast, a label that is NOT forward-looking ({n['label']}) is reproduced at {n['accuracy']:.0%} accuracy. High accuracy is easy when the label is computed from data the model can see; it only becomes meaningful when the label is in the future.")
    return out


def layout(ego: dict) -> dict:
    """Radial layout: focus asset in the centre, connected assets on ring 1, audit entities on ring 2 grouped by type."""
    nodes = ego["nodes"]
    ring1 = [n for n in nodes if n["kind"] == "asset" and not n.get("focus")]
    ring2 = sorted([n for n in nodes if n["kind"] == "entity"], key=lambda n: (n["type"], n["id"]))
    for n in nodes:
        if n.get("focus"):
            n["x"], n["y"] = 0.0, 0.0
    for ring, radius, phase in ((ring1, 0.42, -math.pi / 2), (ring2, 0.85, -math.pi / 2 + 0.2)):
        for i, n in enumerate(ring):
            a = phase + 2 * math.pi * i / max(1, len(ring))
            n["x"], n["y"] = round(radius * math.cos(a), 3), round(radius * math.sin(a), 3)
    return ego


def create_app(artifacts_dir=None, db_path=None, service: TrustService | None = None, api_key: str | None = None, now_fn=None):
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
    art = artifacts_dir or ARTIFACTS
    if service is None:
        kw = {"now_fn": now_fn} if now_fn else {}
        service = TrustService(load_tkg(art / "enterprise.json.gz"), TrustEngine.load(art / "model_bundle.joblib"), Store(db_path or ":memory:"), **kw)
    svc = service
    app.svc = svc
    key = api_key if api_key is not None else os.environ.get("TRUSTDECAY_API_KEY")

    def metrics():
        p = art / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else None

    def err(code, message, status, **extra):
        return jsonify(error={"code": code, "message": message, **extra}), status

    @app.before_request
    def guard():
        if key and request.method in ("POST", "PUT", "DELETE") and request.path.startswith("/api/"):
            if not hmac.compare_digest(request.headers.get("X-API-Key", ""), key):
                return err("UNAUTHORIZED", "Missing or wrong X-API-Key", 401)

    for code, name in ((404, "NOT_FOUND"), (405, "METHOD_NOT_ALLOWED"), (413, "TOO_LARGE")):
        app.register_error_handler(code, lambda e, c=code, n=name: err(n, str(getattr(e, "description", n)), c))

    def int_arg(name, default=None, lo=None, hi=None):
        v = request.args.get(name)
        if v in (None, ""):
            return default
        try:
            n = int(v)
        except ValueError:
            raise ValueError(f"{name} must be a whole number")
        if (lo is not None and n < lo) or (hi is not None and n > hi):
            raise ValueError(f"{name} must be between {lo} and {hi}")
        return n

    @app.errorhandler(ValueError)
    def _bad(e):
        return err("BAD_REQUEST", str(e), 400)

    ctx = lambda: {"metrics": metrics(), "version": svc.engine.version, "h": svc.h}

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def dashboard():
        return render_template("dashboard.html", latest=svc.latest_window, categories=CATEGORIES, priorities=PRIORITIES,
                               types=sorted({a["type"] for a in svc.tkg.assets.values()}), **ctx())

    @app.get("/asset/<asset_id>")
    def asset_page(asset_id):
        if asset_id not in svc.tkg.assets:
            abort(404)
        return render_template("asset.html", asset=asset_id, latest=svc.latest_window, **ctx())

    @app.get("/model")
    def model_page():
        m = metrics()
        return render_template("model.html", m=m, figures=FIGURES, findings=findings(m) if m else [], **ctx())

    @app.get("/repository")
    def repository_page():
        return render_template("repository.html", **ctx())

    @app.get("/figures/<name>")
    def figure(name):
        if name not in FIGURES:
            abort(404)
        return send_from_directory(art / "figures", name, max_age=0)

    @app.get("/health")
    def health():
        return {"ok": True, "model_version": svc.engine.version, "latest_window": svc.latest_window, "assets": len(svc.tkg.assets)}

    # ---------------------------------------------------------------- API
    @app.get("/api/assess")
    def api_assess():
        w = int_arg("window", None, 0, svc.latest_window)
        res = svc.assess(w, with_factors=request.args.get("explain", "1") != "0", log=False)
        rows = res["assets"]
        for name, allowed in (("type", {a["type"] for a in svc.tkg.assets.values()}), ("category", set(CATEGORIES)), ("priority", set(PRIORITIES))):
            v = request.args.get(name)
            if v:
                if v not in allowed:
                    raise ValueError(f"{name} must be one of {', '.join(sorted(allowed))}")
                rows = [r for r in rows if r["asset_type" if name == "type" else name] == v]
        return jsonify({**res, "assets": rows, "total_assets": len(res["assets"])})

    @app.post("/api/assess")
    def api_assess_run():
        b = request.get_json(silent=True) or {}
        w = b.get("window")
        if w is not None and (not isinstance(w, int) or isinstance(w, bool) or not 0 <= w <= svc.latest_window):
            raise ValueError(f"window must be a whole number between 0 and {svc.latest_window}")
        res = svc.assess(w, with_factors=True, log=True)
        return jsonify({k: v for k, v in res.items() if k != "assets"} | {"assets_assessed": len(res["assets"]), "top": res["assets"][:5]})

    @app.get("/api/assets/<asset_id>")
    def api_asset(asset_id):
        try:
            return jsonify(svc.asset_detail(asset_id))
        except KeyError:
            return err("NOT_FOUND", f"unknown asset {asset_id}", 404)

    @app.get("/api/graph/<asset_id>")
    def api_graph(asset_id):
        if asset_id not in svc.tkg.assets:
            return err("NOT_FOUND", f"unknown asset {asset_id}", 404)
        w = int_arg("window", svc.latest_window, 0, svc.latest_window)
        back = int_arg("back", 3, 1, 10)
        return jsonify(layout(svc.tkg.ego(asset_id, w, back)))

    @app.post("/api/events")
    def api_events():
        b = request.get_json(silent=True)
        events = b.get("events") if isinstance(b, dict) else b
        res = svc.ingest(events, source="api")
        return jsonify(res), (200 if res["accepted"] or not res["rejected"] else 400)

    @app.get("/api/reconcile")
    def api_reconcile():
        return jsonify(svc.reconcile())

    @app.get("/api/repository/<what>")
    def api_repo(what):
        limit = int_arg("limit", 50, 1, 500)
        if what == "runs":
            return jsonify(svc.store.runs(limit))
        if what == "snapshots":
            return jsonify(svc.store.snapshots(limit))
        if what == "predictions":
            return jsonify(svc.store.predictions(int_arg("run_id"), request.args.get("asset"), limit))
        if what == "counts":
            return jsonify(svc.store.counts())
        abort(404)

    @app.get("/api/model/metrics")
    def api_metrics():
        m = metrics()
        return jsonify(m) if m else err("NOT_FOUND", "metrics.json missing; run python train.py", 404)

    return app
