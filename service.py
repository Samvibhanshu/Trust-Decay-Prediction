"""Ties the modules together: TKG + feature extraction + engine + explanation + prioritisation + repository."""
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import features as F
from . import explain, prioritize, trust
from .config import CATEGORIES
from .models import TrustEngine
from .persist import load_tkg
from .schema import EventError, validate_event
from .store import Store
from .tkg import TemporalKG

MAX_EVENTS_PER_REQUEST = 5000
MAX_FUTURE_WINDOWS = 30


class TrustService:
    def __init__(self, tkg: TemporalKG, engine: TrustEngine, store: Store | None = None, now_fn=lambda: datetime.now(timezone.utc)):
        self.tkg, self.engine, self.store = tkg, engine, store or Store()
        self.h = int(engine.bundle["meta"].get("horizon", 1))
        self.split = engine.bundle["meta"].get("split", {})
        self.now_fn = now_fn
        self.lock = threading.RLock()
        self._feats: pd.DataFrame | None = None
        self._pred: pd.DataFrame | None = None

    # ------------------------------------------------------------------ cache
    def features(self) -> pd.DataFrame:
        with self.lock:
            if self._feats is None:
                self._feats = F.extract(self.tkg)
                self._pred = None
            return self._feats

    def predictions(self) -> pd.DataFrame:
        with self.lock:
            f = self.features()
            if self._pred is None:
                self._pred = self.engine.predict(f)
            return self._pred

    def invalidate(self):
        with self.lock:
            self._feats, self._pred = None, None

    @property
    def latest_window(self) -> int:
        return self.tkg.n_windows - 1

    def window_split(self, w: int) -> str:
        b = self.split
        if not b:
            return "unknown"
        if w + self.h > self.latest_window:
            return "live"           # the outcome for this window has not been observed yet
        if b["train"][0] <= w <= b["train"][1] + self.h:
            return "train"          # a training row at window w used labels up to w+h
        if b["val"][0] <= w <= b["val"][1] + self.h:
            return "validation"
        return "test" if w >= b["test"][0] else "gap"

    # ------------------------------------------------------------------ assessment
    def assess(self, window: int | None = None, with_factors: bool = True, log: bool = True) -> dict:
        """Prioritised audit list for one window (default: the latest)."""
        with self.lock:
            w = self.latest_window if window is None else window
            if not 0 <= w <= self.latest_window:
                raise ValueError(f"window must be between 0 and {self.latest_window}")
            f, p = self.features(), self.predictions()
            rows = f.index[f["window"] == w]
            Fw, Pw = f.loc[rows], p.loc[rows]
            table = prioritize.prioritise(Fw, Pw)
            factors = explain.explain(self.engine, Fw) if with_factors else [[] for _ in range(len(Fw))]
            fmap = dict(zip(Fw["asset"], factors))
            table["factors"] = [fmap[a] for a in table["asset"]]
            obs = {a: explain.observed_penalties(r) for a, r in zip(Fw["asset"], (r for _, r in Fw.iterrows()))}
            table["observed_factors"] = [obs[a][:3] for a in table["asset"]]
            table["primary_factor"] = [o[0]["factor"] if o and o[0]["points"] >= 2 else "No significant findings" for o in table["observed_factors"]]
            last_audit = {a: w - int(x) for a, x in zip(Fw["asset"], Fw["windows_since_audit"])}
            table["last_assessment"] = self.tkg.window_start(w).strftime("%Y-%m-%dT%H:%M:%SZ")
            recs = table.to_dict("records")
            snap = self.tkg.snapshot(w)
            run_id = None
            if log:
                self.store.log_snapshot(snap)
                run_id = self.store.log_run(w, self.h, self.engine.version, snap["sha256"],
                                            [(a, {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in r.items() if k in F.ALL_FEATURES})
                                             for a, r in zip(Fw["asset"], Fw.to_dict("records"))], recs)
            return {"window": w, "window_start": recs[0]["last_assessment"] if recs else None, "split": self.window_split(w), "horizon_windows": self.h,
                    "run_id": run_id, "graph_snapshot": snap, "assets": recs}

    def asset_detail(self, asset: str) -> dict:
        with self.lock:
            if asset not in self.tkg.assets:
                raise KeyError(asset)
            f, p = self.features(), self.predictions()
            hist = f[f["asset"] == asset].sort_values("window")
            ph = p.loc[hist.index]
            w = self.latest_window
            last = hist.iloc[[-1]]
            table = prioritize.prioritise(last, ph.loc[last.index])
            factors = explain.explain(self.engine, last)[0]
            observed = explain.observed_penalties(last.iloc[0])
            hist_rows = []
            for (_, r), (_, q) in zip(hist.iterrows(), ph.iterrows()):
                obs_w = int(r["window"]) + self.h
                obs = float(hist.loc[hist["window"] == obs_w, "ts"].iloc[0]) if obs_w <= w else None
                hist_rows.append({"window": int(r["window"]), "ts": round(float(r["ts"]), 1), "pred_future_ts": round(float(q["pred_future_ts"]), 1), "target_window": obs_w,
                                  "observed_future_ts": None if obs is None else round(obs, 1), "category": q["category"], "split": self.window_split(int(r["window"]))})
            counts = {k: float(last.iloc[0][k]) for k in F.NODE_RAW}
            return {"asset": asset, **self.tkg.assets[asset], "window": w, "current": table.iloc[0].to_dict(),
                    "factors": factors, "observed": observed, "history": hist_rows, "state": counts, "neighbors": sorted(self.tkg.neighbors(asset, w)),
                    "recent_events": [e.to_dict() for e in self.tkg.asset_events(asset)[-15:] if e.r != "accessed"][::-1]}

    def reconcile(self, min_windows_back: int = 0) -> dict:
        """Predicted vs observed, for every (asset, window) whose target window has arrived: the comparison claim 7 describes."""
        with self.lock:
            f, p = self.features(), self.predictions()
            d = f[["asset", "window", "ts"]].copy()
            d["pred_future_ts"] = p["pred_future_ts"].values
            d["category"] = p["category"].values
            fut = f[["asset", "window", "ts"]].rename(columns={"window": "target", "ts": "obs_ts"})
            d["target"] = d["window"] + self.h
            d = d.merge(fut, on=["asset", "target"])
            d["obs_decay"] = trust.decay(d["ts"], d["obs_ts"])
            d["obs_cat"] = [CATEGORIES[i] for i in trust.category_index(d["obs_decay"])]
            by = {}
            for sp in ("train", "validation", "test"):
                sub = d[[self.window_split(int(w)) == sp for w in d["window"]]]
                if len(sub):
                    by[sp] = {"n": int(len(sub)), "future_ts_mae": round(float(np.abs(sub["pred_future_ts"] - sub["obs_ts"]).mean()), 2),
                              "category_accuracy": round(float((sub["category"] == sub["obs_cat"]).mean()), 3)}
            return {"pairs": int(len(d)), "by_split": by, "note": "train/validation numbers are in-sample and optimistic; only 'test' is out-of-sample"}

    # ------------------------------------------------------------------ continuous graph updates (claim 2)
    def ingest(self, raw_events: list, source: str = "api") -> dict:
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError("send a non-empty list of events")
        if len(raw_events) > MAX_EVENTS_PER_REQUEST:
            raise ValueError(f"at most {MAX_EVENTS_PER_REQUEST} events per request")
        with self.lock:
            now, known = self.now_fn(), set(self.tkg.assets)
            accepted, rejected, dups = [], [], 0
            for i, raw in enumerate(raw_events):
                try:
                    e = validate_event(raw, i, known, self.tkg.window_of, self.latest_window + MAX_FUTURE_WINDOWS, now)
                except EventError as err:
                    rejected.append({"index": i, "errors": err.errors})
                    continue
                if self.tkg.add_event(e):
                    accepted.append(e)
                else:
                    dups += 1
            if accepted:
                self.store.log_events(accepted, source)
                self.invalidate()
            return {"received": len(raw_events), "accepted": len(accepted), "duplicates_ignored": dups, "rejected": rejected,
                    "latest_window": self.latest_window, "windows_touched": sorted({e.w for e in accepted})}
