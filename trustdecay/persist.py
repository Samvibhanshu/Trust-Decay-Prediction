"""Save / load an enterprise (assets, structural edges, audit events) so the app shows exactly what the model was evaluated on."""
import gzip
import json
from pathlib import Path

from .schema import Event, parse_time
from .tkg import TemporalKG


def dump_tkg(tkg: TemporalKG, path: Path):
    struct = [{"s": u, "r": d["relation"], "o": v, "w_start": d["w_start"], "w_end": d["w_end"]}
              for u, v, d in tkg.g.edges(data=True) if "w_start" in d]
    payload = {"assets": tkg.assets, "structural": struct, "events": [e.to_dict() for e in tkg.events]}
    with gzip.open(path, "wt", encoding="utf8") as f:
        json.dump(payload, f, separators=(",", ":"))


def load_tkg(path: Path) -> TemporalKG:
    with gzip.open(path, "rt", encoding="utf8") as f:
        p = json.load(f)
    tkg = TemporalKG()
    for aid, a in p["assets"].items():
        tkg.add_asset(aid, a["type"], a["criticality"])
    for s in p["structural"]:
        tkg.add_structural(s["s"], s["r"], s["o"], s["w_start"], s["w_end"])
    for e in p["events"]:
        t = parse_time(e["t"])
        tkg.add_event(Event(e["s"], e["r"], e["o"], t, tkg.window_of(t), e.get("attrs", {})))
    return tkg
