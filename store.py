"""Logging and Historical Repository (disclosure step 7, claim 7): audit events, graph snapshots, trust feature vectors,
predictions, explanations, recommendations, and the predicted-vs-observed comparison."""
import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, s TEXT, r TEXT, o TEXT, t TEXT, window INTEGER, attrs TEXT, source TEXT, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS snapshots(window INTEGER, sha256 TEXT, nodes INTEGER, event_edges INTEGER, structural_edges INTEGER, taken_at TEXT, PRIMARY KEY(window, sha256));
CREATE TABLE IF NOT EXISTS features(run_id INTEGER, asset TEXT, window INTEGER, vector TEXT, PRIMARY KEY(run_id, asset));
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, window INTEGER, horizon INTEGER, model_version TEXT, graph_sha TEXT, created_at TEXT, n_assets INTEGER);
CREATE TABLE IF NOT EXISTS predictions(run_id INTEGER, asset TEXT, window INTEGER, target_window INTEGER, current_ts REAL, pred_future_ts REAL,
  pred_decay REAL, expected_decay REAL, category TEXT, p_high REAL, priority TEXT, rank_score REAL, trend TEXT, factors TEXT, PRIMARY KEY(run_id, asset));
CREATE INDEX IF NOT EXISTS idx_pred_asset ON predictions(asset, window);
"""


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Store:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)

    def _q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def log_events(self, events, source):
        with self.lock:
            self.db.executemany("INSERT INTO events(s,r,o,t,window,attrs,source,ingested_at) VALUES(?,?,?,?,?,?,?,?)",
                                [(e.s, e.r, e.o, e.t.isoformat(), e.w, json.dumps(e.attrs), source, now_iso()) for e in events])
            self.db.commit()

    def log_snapshot(self, snap: dict):
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?,?)",
                            (snap["window"], snap["sha256"], snap["nodes"], snap["event_edges"], snap["structural_edges"], now_iso()))
            self.db.commit()

    def log_run(self, window, horizon, version, graph_sha, feats_rows, pred_rows) -> int:
        """Stores one assessment run: the feature vectors used and every prediction made, atomically."""
        with self.lock:
            cur = self.db.execute("INSERT INTO runs(window,horizon,model_version,graph_sha,created_at,n_assets) VALUES(?,?,?,?,?,?)",
                                  (window, horizon, version, graph_sha, now_iso(), len(pred_rows)))
            rid = cur.lastrowid
            self.db.executemany("INSERT INTO features VALUES(?,?,?,?)", [(rid, a, window, json.dumps(v)) for a, v in feats_rows])
            self.db.executemany("INSERT INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                [(rid, p["asset"], window, window + horizon, p["current_ts"], p["pred_future_ts"], p["pred_decay"], p["expected_decay"],
                                  p["category"], p["p_high_or_critical"], p["priority"], p["rank_score"], p["trend"], json.dumps(p.get("factors", []))) for p in pred_rows])
            self.db.commit()
            return rid

    def runs(self, limit=50):
        return self._q("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (int(limit),))

    def predictions(self, run_id=None, asset=None, limit=500):
        sql, args = "SELECT * FROM predictions", []
        cond = []
        if run_id is not None:
            cond.append("run_id=?"); args.append(run_id)
        if asset:
            cond.append("asset=?"); args.append(asset)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        rows = self._q(sql + " ORDER BY run_id DESC, rank_score DESC LIMIT ?", (*args, int(limit)))
        for r in rows:
            r["factors"] = json.loads(r["factors"] or "[]")
        return rows

    def snapshots(self, limit=100):
        return self._q("SELECT * FROM snapshots ORDER BY window DESC LIMIT ?", (int(limit),))

    def counts(self):
        return {t: self._q(f"SELECT COUNT(*) c FROM {t}")[0]["c"] for t in ("events", "snapshots", "runs", "predictions", "features")}
