"""Training and honest evaluation of the trust-decay model on the simulated enterprise.

What the disclosure describes: one Random Forest, one chronological split, accuracy / weighted P-R-F1 / confusion matrix.
This module reproduces that, and adds what is needed to interpret it: majority and persistence baselines, a logistic
regression, a horizon sweep chosen on validation only, feature-set ablations, audit-ranking effectiveness against naive
rankers, a new-asset holdout, rolling-origin cross-validation, and a control world with no relationship signal.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_recall_fscore_support, r2_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import dataset as D
from . import features as F
from . import models as M
from . import persist, prioritize, trust
from .config import ARTIFACTS, CATEGORIES
from .simulator import SimConfig, simulate

warnings.filterwarnings("ignore")
HORIZONS = (1, 2, 3, 5, 7)
K_FRAC = 0.10                      # audit capacity: top 10% of assets per window


def auc_high(y, p):
    hi = (np.asarray(y) >= 2).astype(int)
    return float(roc_auc_score(hi, p)) if 0 < hi.sum() < len(hi) else float("nan")


def clf_metrics(y, P) -> dict:
    y, pred = np.asarray(y).astype(int), P.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2, 3], zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    wrong = pred != y
    return {"n": int(len(y)), "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw, "f1_macro": float(f.mean()),
            "auc_high_or_critical": auc_high(y, P[:, 2] + P[:, 3]),
            "per_class": {c: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])} for i, c in enumerate(CATEGORIES)},
            "confusion_matrix": cm.tolist(),
            "adjacent_error_share": float((np.abs(pred - y)[wrong] == 1).mean()) if wrong.any() else None,
            "majority_class_accuracy": float(np.bincount(y, minlength=4).max() / len(y))}


def per_window_topk(df: pd.DataFrame, score, k_frac=K_FRAC) -> pd.DataFrame:
    """One row per window: precision@k, recall@k and decay captured for the k highest-scored assets."""
    d = df.assign(_s=np.asarray(score))
    rows = []
    for w, g in d.groupby("window"):
        k = max(1, int(round(len(g) * k_frac)))
        top = g.nlargest(k, "_s")
        pos = int((g["y"] >= 2).sum())
        rows.append({"window": w, "k": k, "prec": float((top["y"] >= 2).mean()),
                     "rec": float((top["y"] >= 2).sum() / pos) if pos else np.nan,
                     "cap": float(top["decay"].sum() / g["decay"].sum()) if g["decay"].sum() > 0 else np.nan})
    return pd.DataFrame(rows)


def topk_effectiveness(df: pd.DataFrame, score, k_frac=K_FRAC) -> dict:
    """Window-averaged audit-list quality. df needs window, y, decay."""
    r = per_window_topk(df, score, k_frac)
    return {"precision_at_k": float(r["prec"].mean()), "recall_at_k": float(r["rec"].mean()),
            "decay_captured": float(r["cap"].mean()), "k_per_window": int(r["k"].iloc[0])}


def boot_ci(df, score, key, n=500, seed=0):
    """95% interval over WINDOWS (assets within a window are not independent draws)."""
    col = {"precision_at_k": "prec", "recall_at_k": "rec", "decay_captured": "cap"}[key]
    v = per_window_topk(df, score)[col].dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def fit_rf(train, cols, params):
    X, y, r = D.xy(train, cols)
    return M.build_classifier(params).fit(X, y), M.build_regressor(params).fit(X, r)


def tune(train, val, cols, quick):
    Xtr, ytr, _ = D.xy(train, cols)
    Xva, yva, _ = D.xy(val, cols)
    best, best_f = M.GRID[0], -1
    for params in ([{"n_estimators": 100, "min_samples_leaf": l, "class_weight": "balanced_subsample"} for l in (1, 3, 6)] if quick else M.GRID):
        P = M.full_probs(M.build_classifier(params).fit(Xtr, ytr), Xva)
        f = f1_score(yva, P.argmax(1), average="macro")
        if f > best_f + 1e-9:
            best, best_f = params, f
    return best, best_f


def horizon_sweep(base, quick, horizons=HORIZONS):
    trees = 100 if quick else 300
    rows = []
    for h in horizons:
        ds = F.add_targets(base, h)
        sp = D.chronological_split(ds, purge=h)
        clf = M.build_classifier({"class_weight": "balanced_subsample", "n_estimators": trees}).fit(*D.xy(sp.train)[:2])
        Pv = M.full_probs(clf, sp.val[F.ALL_FEATURES])
        Pt = M.full_probs(clf, sp.test[F.ALL_FEATURES])
        rows.append({"horizon": h, "val_f1_macro": float(f1_score(sp.val["y"], Pv.argmax(1), average="macro")),
                     "val_auc_high": auc_high(sp.val["y"], Pv[:, 2] + Pv[:, 3]),
                     "test_f1_macro_informational": float(f1_score(sp.test["y"], Pt.argmax(1), average="macro")),
                     "test_auc_high_informational": auc_high(sp.test["y"], Pt[:, 2] + Pt[:, 3]),
                     "high_or_critical_rate_val": float((sp.val["y"] >= 2).mean())})
    best = max(rows, key=lambda r: (round(r["val_f1_macro"], 6), -r["horizon"]))
    return rows, best["horizon"]


def ablation(tkg_base_df, h, params, seeds_dfs=None):
    ds = F.add_targets(tkg_base_df, h)
    sp = D.chronological_split(ds, purge=h)
    out = {}
    for name, cols in F.FEATURE_SETS.items():
        clf, _ = fit_rf(sp.fit_frame(), cols, params)
        P = M.full_probs(clf, sp.test[cols])
        ph = P[:, 2] + P[:, 3]
        out[name] = {"f1_macro": float(f1_score(sp.test["y"], P.argmax(1), average="macro")),
                     "auc_high_or_critical": auc_high(sp.test["y"], ph),
                     "precision_at_k": topk_effectiveness(sp.test, ph)["precision_at_k"], "n_features": len(cols)}
    return out


def graph_signal_control(seed, h, params, n_seeds=3, cfg=None):
    """Does the graph feature group add anything, and does the check detect it? Compare a world with relationship
    contagion (0.6) to a control world without (0.0), averaged over seeds."""
    res = {}
    for label, contagion in (("with_relationship_signal", 0.6), ("control_no_relationship_signal", 0.0)):
        gains = []
        for s in range(seed, seed + n_seeds):
            base = F.extract(simulate(SimConfig(**{**(cfg.__dict__ if cfg else {}), "seed": s, "contagion": contagion})))
            ab = ablation(base, h, params)
            gains.append({"auc_gain": ab["node+temporal+graph"]["auc_high_or_critical"] - ab["node+temporal"]["auc_high_or_critical"],
                          "f1_gain": ab["node+temporal+graph"]["f1_macro"] - ab["node+temporal"]["f1_macro"],
                          "precision_at_k_gain": ab["node+temporal+graph"]["precision_at_k"] - ab["node+temporal"]["precision_at_k"]})
        res[label] = {"seeds": n_seeds, **{k: {"mean": float(np.mean([g[k] for g in gains])), "sd": float(np.std([g[k] for g in gains]))} for k in gains[0]}}
    return res


def non_predictive_control(base, params):
    """How high can 'accuracy' get if the label is NOT forward-looking? Label = category of the change in trust over the
    PAST window (max(0, TS[t-1] - TS[t])), predicted from features of window t, which include that very change (trend1).
    A model that just reproduces the rubric scores very high; that says nothing about foreseeing future decay."""
    d = base.copy()
    d["decay"] = np.maximum(0.0, d["ts_lag1"] - d["ts"])
    d["y"] = trust.category_index(d["decay"])
    d["ts_future"] = d["ts"]
    sp = D.chronological_split(d, purge=0)
    clf = M.build_classifier({**params, "n_estimators": min(params.get("n_estimators", 300), 200)}).fit(*D.xy(sp.fit_frame())[:2])
    P = M.full_probs(clf, sp.test[F.ALL_FEATURES])
    y = sp.test["y"].astype(int)
    return {"label": "category of the change in trust over the PAST window (not forward-looking)", "accuracy": float(accuracy_score(y, P.argmax(1))),
            "f1_weighted": float(f1_score(y, P.argmax(1), average="weighted")), "majority_class_accuracy": float(np.bincount(y, minlength=4).max() / len(y))}


def run(quick=False, out_dir=ARTIFACTS, make_figs=True, seed=7, sim_cfg: SimConfig | None = None, horizons=HORIZONS):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = sim_cfg or SimConfig(seed=seed)
    tkg = simulate(cfg)
    base = F.extract(tkg)
    sweep, h = horizon_sweep(base, quick, horizons)                         # horizon chosen on VALIDATION macro-F1 only

    ds = F.add_targets(base, h)
    sp = D.chronological_split(ds, purge=h)
    params, val_f1 = tune(sp.train, sp.val, F.ALL_FEATURES, quick)
    clf, reg = fit_rf(sp.fit_frame(), F.ALL_FEATURES, params)     # deployed models: train + validation windows only
    version = "v1-" + time.strftime("%Y%m%d%H%M%S", time.gmtime())
    bundle = M.make_bundle(clf, reg, version, meta={"horizon": h, "params": params, "split": sp.bounds, "sim": cfg.__dict__ | {"episode_len": list(cfg.episode_len)}})
    import joblib
    joblib.dump(bundle, out_dir / "model_bundle.joblib", compress=3)
    persist.dump_tkg(tkg, out_dir / "enterprise.json.gz")
    engine = M.TrustEngine(bundle)

    test = sp.test
    Xte, yte, rte = D.xy(test)
    P = M.full_probs(clf, Xte)
    ph = P[:, 2] + P[:, 3]
    pred_ts = np.clip(reg.predict(Xte), 0, 100)
    main = clf_metrics(yte, P)
    main["regression"] = {"future_ts_mae": mean_absolute_error(rte, pred_ts), "future_ts_r2": r2_score(rte, pred_ts),
                          "persistence_mae": mean_absolute_error(rte, test["ts"]), "persistence_r2": r2_score(rte, test["ts"]),
                          "decay_mae_model": mean_absolute_error(test["decay"], trust.decay(test["ts"], pred_ts)),
                          "decay_mae_predict_zero": mean_absolute_error(test["decay"], np.zeros(len(test)))}

    # ---- baselines
    Xfit, yfit, _ = D.xy(sp.fit_frame())
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xfit, yfit)
    Plr = np.zeros((len(Xte), 4)); Plr[:, lr.classes_] = lr.predict_proba(Xte)
    trend_score = -test["trend3"].to_numpy()
    baselines = {
        "majority_class": {"accuracy": main["majority_class_accuracy"], "f1_macro": float(f1_score(yte, np.zeros(len(yte)), average="macro", labels=[0, 1, 2, 3])), "auc_high_or_critical": 0.5},
        "persistence (no decay predicted)": {"accuracy": float((yte == 0).mean()), "f1_macro": float(f1_score(yte, np.zeros(len(yte)), average="macro", labels=[0, 1, 2, 3])), "auc_high_or_critical": 0.5},
        "logistic_regression": {k: v for k, v in clf_metrics(yte, Plr).items() if k in ("accuracy", "f1_macro", "f1_weighted", "auc_high_or_critical")},
        "recent_trend (continue last 3-window change)": {"auc_high_or_critical": auc_high(yte, trend_score)},
    }

    # ---- audit prioritisation effectiveness
    rankers = {
        "random (expected)": None,
        "lowest current trust score": -test["ts"].to_numpy(),
        "steepest recent decline": trend_score,
        "most open vulnerabilities": test["sev_sum_open"].to_numpy(),
        "model: P(High or Critical)": ph,
        "model: rank score (with criticality, audit age)": np.array([prioritize.rank_score(p, e, c, a) for p, e, c, a in
                                                                       zip(ph, trust.expected_decay(P), test["criticality"], test["windows_since_audit"])]),
    }
    pri_all = prioritize.prioritise(test, engine.predict(test))
    level = pri_all["priority"].map(prioritize.PRIORITIES.index)
    rankers["dashboard display order (priority, then rank score)"] = (level * 1000 + pri_all["rank_score"]).reindex(test.index).to_numpy()
    base_rate = float((yte >= 2).mean())
    ranking = {}
    for name, s in rankers.items():
        if s is None:
            ranking[name] = {"precision_at_k": base_rate, "recall_at_k": K_FRAC, "decay_captured": K_FRAC}
            continue
        m = topk_effectiveness(test, s)
        ranking[name] = {**m, "precision_at_k_ci95": boot_ci(test, s, "precision_at_k", 200 if quick else 600)}
    flagged = pri_all[pri_all["priority"].isin(["Immediate", "High"])]
    truth = test.loc[flagged.index, "y"]
    false_priority = float((truth == 0).mean()) if len(flagged) else None
    ranking["_meta"] = {"k_fraction": K_FRAC, "k_per_window": ranking["model: P(High or Critical)"]["k_per_window"], "high_or_critical_base_rate": base_rate,
                        "false_priority_rate_immediate_or_high": false_priority, "n_flagged": int(len(flagged))}

    # ---- ablation, new-asset holdout, rolling-origin CV, graph-signal control
    abl = ablation(base, h, params)
    rng = np.random.default_rng(seed)
    assets = np.array(sorted(tkg.assets))
    unseen = set(rng.choice(assets, size=len(assets) // 4, replace=False))
    fit_seen = sp.fit_frame()[~sp.fit_frame()["asset"].isin(unseen)]
    c2, _ = fit_rf(fit_seen, F.ALL_FEATURES, params)
    te_seen, te_new = test[~test["asset"].isin(unseen)], test[test["asset"].isin(unseen)]
    Ps, Pn = (M.full_probs(c2, x[F.ALL_FEATURES]) for x in (te_seen, te_new))
    new_asset = {"n_unseen_assets": len(unseen),
                 "seen_assets": {"f1_macro": float(f1_score(te_seen["y"], Ps.argmax(1), average="macro")), "auc_high_or_critical": auc_high(te_seen["y"], Ps[:, 2] + Ps[:, 3])},
                 "unseen_assets": {"f1_macro": float(f1_score(te_new["y"], Pn.argmax(1), average="macro")), "auc_high_or_critical": auc_high(te_new["y"], Pn[:, 2] + Pn[:, 3])}}
    folds = []
    for tr, te in D.expanding_folds(ds, purge=h):
        c3, _ = fit_rf(tr, F.ALL_FEATURES, params)
        Pf = M.full_probs(c3, te[F.ALL_FEATURES])
        folds.append({"f1_macro": float(f1_score(te["y"], Pf.argmax(1), average="macro")), "auc_high_or_critical": auc_high(te["y"], Pf[:, 2] + Pf[:, 3]),
                      "accuracy": float(accuracy_score(te["y"], Pf.argmax(1))), "majority_accuracy": float(np.bincount(te["y"].astype(int), minlength=4).max() / len(te))})
    cv = {"folds": folds, **{f"mean_{k}": float(np.mean([f[k] for f in folds])) for k in folds[0]}, **{f"sd_{k}": float(np.std([f[k] for f in folds])) for k in ("f1_macro", "auc_high_or_critical")}}
    nonpred = non_predictive_control(base, params)
    control = graph_signal_control(seed, h, params, n_seeds=2 if quick else 3, cfg=cfg)

    metrics = {
        "version": version, "trained_at": bundle["trained_at"], "horizon_windows": h, "window_hours": 24,
        "data": {"kind": "SIMULATED enterprise audit environment", "assets": len(tkg.assets), "windows": tkg.n_windows, "events": len(tkg.events),
                 "labelled_rows": int(len(ds)), "class_mix": {c: float((ds["y"] == i).mean()) for i, c in enumerate(CATEGORIES)}, "sim_config": cfg.__dict__ | {"episode_len": list(cfg.episode_len)}},
        "split": {**sp.bounds, "train_rows": len(sp.train), "val_rows": len(sp.val), "test_rows": len(sp.test), "note": "chronological by window; deployed model is fitted on train+val only"},
        "horizon_sweep": {"rows": sweep, "chosen": h, "criterion": "validation macro-F1 (test columns are informational, not used)"},
        "chosen_params": params, "validation_f1_macro": val_f1,
        "test": main, "baselines": baselines, "ranking": ranking, "ablation": abl, "new_asset_holdout": new_asset,
        "rolling_origin_cv": cv, "graph_signal_control": control, "non_predictive_control": nonpred,
        "feature_importance": engine.feature_importance(),
        "disclosure_comparison": {"disclosure_reported_accuracy": 0.8095, "disclosure_test_records": 168,
                                  "note": "The disclosure's 80.95% comes from its own simulated dataset and labelling. It is not comparable to the numbers here, "
                                          "which come from a different simulator with a strictly forward-looking label, and neither is evidence of real-world performance."},
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    if make_figs:
        from .figures import make_all
        make_all(metrics, tkg, base, sp, engine, out_dir / "figures")
    return metrics
