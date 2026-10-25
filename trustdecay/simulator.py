"""Simulated enterprise audit environment (disclosure section 8.1: 'simulated enterprise audit dataset').

THIS IS SYNTHETIC. Every rate below is an assumption chosen to look like a plausible enterprise, not a measurement.
Anything the models 'learn' about trust decay is a property of these assumptions.

Mechanics
  * each asset has a latent hygiene level and drifts into multi-day 'degradation episodes' (patching lapses, flaky
    backups, noisy access) that raise its event rates and then end, sometimes after an audit
  * `contagion` lets a degraded neighbour raise the event rates of connected assets on the next day; set it to 0
    to build a control world in which relationships carry no predictive information
  * events are emitted as (S, R, O, T) audit records and loaded into the Temporal Knowledge Graph; nothing downstream
    reads the latent variables, only the events
"""
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from .config import ASSET_PREFIX, T0
from .schema import Event
from .tkg import TemporalKG

SEVERITIES = np.arange(2, 11)
SEV_P = np.array([.08, .12, .16, .18, .16, .12, .09, .06, .03])
TYPE_MIX = {"server": .28, "database": .16, "application": .28, "firewall": .08, "network_device": .20}
CRIT_CHOICES = {"firewall": [5, 5, 4], "database": [5, 4, 4], "server": [3, 4, 4, 3], "application": [2, 3, 4, 3], "network_device": [2, 3, 3]}


@dataclass
class SimConfig:
    n_assets: int = 48
    n_users: int = 30
    n_admins: int = 4
    n_windows: int = 64
    seed: int = 7
    contagion: float = 0.6            # 0 = relationships are uninformative (control world)
    episode_start_p: float = 0.05
    episode_len: tuple = (3, 9)


def _counts(n):
    raw = {k: v * n for k, v in TYPE_MIX.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def simulate(cfg: SimConfig | None = None) -> TemporalKG:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    tkg = TemporalKG()

    # ---------------- assets and topology
    by_type = {}
    for atype, n in _counts(cfg.n_assets).items():
        ids = [f"{ASSET_PREFIX[atype]}-{i + 1:02d}" for i in range(n)]
        by_type[atype] = ids
        for aid in ids:
            tkg.add_asset(aid, atype, int(rng.choice(CRIT_CHOICES[atype])))
    srv, db, app, fw, nw = (by_type[k] for k in ("server", "database", "application", "firewall", "network_device"))
    for d in db:
        tkg.add_structural(str(rng.choice(srv)), "hosts", d)
    for a in app:
        tkg.add_structural(a, "runs_on", str(rng.choice(srv)))
        if rng.random() < .7:
            tkg.add_structural(a, "depends_on", str(rng.choice(db)))
    for s in srv:
        for n in rng.choice(nw, size=min(len(nw), int(rng.integers(1, 3))), replace=False):
            tkg.add_structural(s, "connects_to", str(n))
    for n in nw:
        tkg.add_structural(n, "connects_to", str(rng.choice(fw)))
    assets = list(tkg.assets)
    nbrs = {a: sorted(tkg.neighbors(a, 0)) for a in assets}

    users = [f"USER-{i + 1:02d}" for i in range(cfg.n_users)]
    admins = [f"ADMIN-{i + 1:02d}" for i in range(cfg.n_admins)]

    # ---------------- latent state
    hygiene = {a: float(np.clip(rng.beta(2, 5), .03, .6)) for a in assets}
    period = {a: int(rng.integers(10, 15)) for a in assets}
    phase = {a: int(rng.integers(0, 10)) for a in assets}
    ep_left = {a: 0 for a in assets}
    ep_int = {a: 0.0 for a in assets}
    prev_ep = {a: 0.0 for a in assets}                 # episode intensity in the previous window
    open_v: dict[str, dict] = {a: {} for a in assets}  # asset -> {vuln_id: severity}
    vid, bid, cid = 0, 0, 0
    events: list[Event] = []

    def emit(s, r, o, w, attrs=None):
        t = T0 + timedelta(days=w, hours=float(rng.uniform(0, 23.9)))
        events.append(Event(s, r, o, t, w, attrs or {}))

    for w in range(cfg.n_windows):
        cur_ep = {}
        for a in assets:
            u = hygiene[a]
            if ep_left[a] > 0:
                ep_left[a] -= 1
            elif rng.random() < cfg.episode_start_p * (1 + 2 * u):
                ep_left[a] = int(rng.integers(cfg.episode_len[0], cfg.episode_len[1] + 1))
                ep_int[a] = float(rng.uniform(.8, 1.6))
            cur_ep[a] = ep_int[a] if ep_left[a] > 0 else 0.0
        for a in assets:
            u = hygiene[a]
            nb = nbrs[a]
            nb_e = float(np.mean([prev_ep[n] for n in nb])) if nb else 0.0
            E = cur_ep[a] + cfg.contagion * nb_e
            audit_now = (w + phase[a]) % period[a] == 0 and w > 0
            # vulnerabilities
            for _ in range(int(rng.poisson(.10 + .25 * u + .8 * E))):
                vid += 1
                sev = int(rng.choice(SEVERITIES, p=SEV_P))
                open_v[a][f"VULN-{vid}"] = sev
                emit(f"VULN-{vid}", "detected_on", a, w, {"severity": sev})
            for v, sev in list(open_v[a].items()):
                p = float(np.clip((.20 - .10 * u) * (1 - .85 * min(E, 1.2)) * (1 + .1 * (sev - 5)), .01, .6))
                if rng.random() < p or (audit_now and rng.random() < .6):
                    del open_v[a][v]
                    emit(v, "patched_on", a, w)
            # authentication
            n = int(rng.poisson(.6 + 1.0 * u + 3.5 * E))
            if n:
                emit(str(rng.choice(users)), "failed_login", a, w, {"count": n})
            n = int(rng.poisson(.05 + .15 * u + .9 * E))
            if n:
                emit(str(rng.choice(users)), "suspicious_access", a, w, {"count": n})
            if rng.random() < .6:
                emit(str(rng.choice(users)), "accessed", a, w)
            # configuration changes
            for _ in range(int(rng.poisson(.5))):
                ok = bool(rng.random() > float(np.clip(.05 + .2 * u + .35 * E, 0, .9)))
                emit(str(rng.choice(admins)), "modified", a, w, {"authorized": ok})
            for _ in range(int(rng.poisson(.01 + .03 * u + .25 * E))):
                emit(str(rng.choice(users + admins)), "privilege_escalation", a, w)
            # backups, compliance, alerts
            bid += 1
            emit(f"BKP-{bid}", "backup_result", a, w, {"success": bool(rng.random() > float(np.clip(.03 + .10 * u + .30 * E, 0, .9)))})
            for _ in range(int(rng.poisson(.02 + .06 * u + .35 * E))):
                cid += 1
                emit(f"CTL-{cid % 40 + 1}", "compliance_violation", a, w)
            for _ in range(int(rng.poisson(.15 + .2 * u + 2.0 * E))):
                emit("SYSTEM", "security_alert", a, w, {"severity": int(rng.integers(1, 6))})
            if audit_now:
                emit(f"AUD-{w}", "audit_completed", a, w)
                if rng.random() < .5:
                    ep_left[a] = 0
        prev_ep = cur_ep

    for e in events:
        tkg.add_event(e)
    return tkg
