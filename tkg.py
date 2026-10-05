"""Temporal Knowledge Graph (disclosure step 3, claim 2).

Nodes are enterprise entities. Edges are time-stamped (S, R, O, T) relationships:
  * structural edges (runs_on, hosts, connects_to, depends_on) carry a validity interval [w_start, w_end]
  * event edges (detected_on, failed_login, backup_result, ...) carry the window they occurred in
The graph is updated whenever an event arrives and keeps the full history, so any past window can be reconstructed.
"""
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx

from .config import ASSET_TYPES, T0, WINDOW_HOURS
from .schema import ASSET_ID, STRUCTURAL_RELATIONS, Event, prefix


class TemporalKG:
    def __init__(self, t0=T0, window_hours=WINDOW_HOURS):
        self.t0, self.window_hours = t0, window_hours
        self.g = nx.MultiDiGraph()
        self.assets: dict[str, dict] = {}                  # asset_id -> {type, criticality}
        self.events: list[Event] = []
        self._by_asset: dict[str, list[Event]] = defaultdict(list)
        self._seen: set = set()
        self._max_w = -1

    # ------------------------------------------------------------------ time
    def window_of(self, t: datetime) -> int:
        return int((t - self.t0) // timedelta(hours=self.window_hours)) if t >= self.t0 else -1

    def window_start(self, w: int) -> datetime:
        return self.t0 + timedelta(hours=self.window_hours * w)

    @property
    def n_windows(self) -> int:
        return self._max_w + 1

    # ------------------------------------------------------------------ construction
    def add_entity(self, eid: str, etype: str, **attrs):
        if eid not in self.g:
            self.g.add_node(eid, etype=etype, **attrs)

    def add_asset(self, aid: str, atype: str, criticality: int):
        if atype not in ASSET_TYPES or not ASSET_ID.match(aid) or not 1 <= criticality <= 5:
            raise ValueError(f"bad asset {aid} {atype} {criticality}")
        self.assets[aid] = {"type": atype, "criticality": criticality}
        self.add_entity(aid, atype, criticality=criticality)

    def add_structural(self, s: str, r: str, o: str, w_start: int = 0, w_end: int | None = None):
        if r not in STRUCTURAL_RELATIONS:
            raise ValueError(r)
        for e in (s, o):
            if e not in self.assets:
                raise ValueError(f"structural edges connect assets; unknown {e}")
        self.g.add_edge(s, o, key=f"{r}:{w_start}", relation=r, w_start=w_start, w_end=w_end)

    @staticmethod
    def _key(e: Event):
        return (e.s, e.r, e.o, e.t, tuple(sorted(e.attrs.items())))

    def add_event(self, e: Event) -> bool:
        """Adds one audit event. Returns False if this exact event is already present (duplicate removal)."""
        k = self._key(e)
        if k in self._seen:
            return False
        self._seen.add(k)
        self.add_entity(e.s, prefix(e.s).lower())
        self.g.add_edge(e.s, e.o, key=f"{e.r}:{len(self.events)}", relation=e.r, w=e.w, t=e.t, **e.attrs)
        self.events.append(e)
        lst = self._by_asset[e.o]
        lst.append(e)
        if len(lst) > 1 and (lst[-2].w, lst[-2].t) > (e.w, e.t):
            lst.sort(key=lambda x: (x.w, x.t))               # late arrival: keep temporal order
        self._max_w = max(self._max_w, e.w)
        return True

    # ------------------------------------------------------------------ queries
    def asset_events(self, aid: str) -> list[Event]:
        return self._by_asset.get(aid, [])

    def _struct_valid(self, d, w):
        return d.get("relation") in STRUCTURAL_RELATIONS and d["w_start"] <= w and (d["w_end"] is None or w <= d["w_end"])

    def neighbors(self, aid: str, w: int) -> set:
        """Assets structurally connected to `aid` in window w (direction ignored)."""
        out = set()
        for u, v, d in list(self.g.out_edges(aid, data=True)) + list(self.g.in_edges(aid, data=True)):
            if self._struct_valid(d, w):
                out.add(v if u == aid else u)
        return out

    def structural_edges(self, w: int):
        for u, v, d in self.g.edges(data=True):
            if self._struct_valid(d, w):
                yield u, d["relation"], v

    def snapshot(self, w: int) -> dict:
        """Compact, verifiable snapshot of the graph as it stood at the END of window w (claim 7)."""
        ev = [e for e in self.events if e.w <= w]
        struct = sorted(self.structural_edges(w))
        h = hashlib.sha256()
        for e in sorted(ev, key=lambda x: (x.w, x.t, x.s, x.r, x.o)):
            h.update(f"{e.s}|{e.r}|{e.o}|{e.t.isoformat()}|{sorted(e.attrs.items())}\n".encode())
        for s, r, o in struct:
            h.update(f"{s}|{r}|{o}\n".encode())
        nodes = {e.s for e in ev} | {e.o for e in ev} | set(self.assets)
        return {"window": w, "nodes": len(nodes), "event_edges": len(ev), "structural_edges": len(struct), "sha256": h.hexdigest()[:16]}

    def ego(self, aid: str, w: int, back: int = 3, max_per_type: int = 6) -> dict:
        """Neighbourhood of an asset over windows [w-back+1, w]: structural neighbours plus the audit entities
        (users, vulnerabilities, backups...) that touched it. Returned as plain nodes/edges for the UI."""
        lo = max(0, w - back + 1)
        nodes = {aid: {"id": aid, "type": self.assets[aid]["type"], "kind": "asset", "focus": True}}
        edges = []
        for n in sorted(self.neighbors(aid, w)):
            nodes[n] = {"id": n, "type": self.assets[n]["type"], "kind": "asset"}
        for u, v, d in self.g.edges(data=True):
            if self._struct_valid(d, w) and aid in (u, v) and u in nodes and v in nodes:
                edges.append({"s": u, "r": d["relation"], "o": v, "w": None})
        per_type = defaultdict(int)
        evs = [e for e in self.asset_events(aid) if lo <= e.w <= w and e.r != "accessed"]
        for e in sorted(evs, key=lambda x: (-x.w, x.t)):
            if e.s not in nodes:
                if per_type[prefix(e.s)] >= max_per_type:
                    continue
                per_type[prefix(e.s)] += 1
                nodes[e.s] = {"id": e.s, "type": prefix(e.s).lower(), "kind": "entity"}
            edges.append({"s": e.s, "r": e.r, "o": aid, "w": e.w, "attrs": e.attrs})
        return {"asset": aid, "window": w, "from_window": lo, "nodes": list(nodes.values()), "edges": edges}
