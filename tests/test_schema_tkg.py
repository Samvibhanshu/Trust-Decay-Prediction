from datetime import datetime, timedelta, timezone

import pytest

from trustdecay.schema import Event, EventError, validate_event
from trustdecay.tkg import TemporalKG

NOW = datetime(2026, 9, 28, tzinfo=timezone.utc)
KNOWN = {"SRV-01", "DB-01"}
wo = lambda t: int((t - datetime(2026, 1, 1, tzinfo=timezone.utc)).days) if t >= datetime(2026, 1, 1, tzinfo=timezone.utc) else -1


def ok(**over):
    e = {"s": "VULN-9", "r": "detected_on", "o": "SRV-01", "t": "2026-02-01T10:00:00Z", "attrs": {"severity": 7}}
    e.update(over)
    return e


def errs(raw):
    with pytest.raises(EventError) as e:
        validate_event(raw, 0, KNOWN, wo, 100, NOW)
    return " | ".join(e.value.errors)


def test_valid_event_is_normalised_to_utc_and_windowed():
    e = validate_event(ok(t="2026-02-01T15:30:00+05:30"), 0, KNOWN, wo, 100, NOW)
    assert e.t == datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc) and e.w == 31


@pytest.mark.parametrize("raw,fragment", [
    (ok(r="exploded_on"), "unknown relation"), (ok(s="USER-1"), "requires subject type"), (ok(s="bogus"), "not a valid entity id"),
    (ok(o="SRV-99"), "not a known asset"), (ok(o="USER-01"), "not a known asset"), (ok(attrs={}), "severity"), (ok(attrs={"severity": 11}), "severity"),
    (ok(attrs={"severity": True}), "severity"), (ok(t="yesterday"), "ISO-8601"), (ok(t="2026-02-01T10:00:00"), "timezone"),
    (ok(t="2025-12-31T23:00:00Z"), "before the start"), (ok(t="2026-12-01T00:00:00Z"), "future"), (ok(t="2026-09-20T00:00:00Z"), "beyond the accepted range"),
    (ok(attrs="x"), "attrs must be an object"), ({}, "required"), ("nope", "must be an object"),
    ({"s": "USER-1", "r": "failed_login", "o": "SRV-01", "t": "2026-02-01T10:00:00Z", "attrs": {"count": 0}}, "count"),
    ({"s": "BKP-1", "r": "backup_result", "o": "SRV-01", "t": "2026-02-01T10:00:00Z", "attrs": {}}, "success"),
])
def test_bad_events_are_rejected_with_a_reason(raw, fragment):
    assert fragment in errs(raw)


def small():
    g = TemporalKG()
    for a, t in (("SRV-01", "server"), ("DB-01", "database"), ("APP-01", "application")):
        g.add_asset(a, t, 3)
    g.add_structural("SRV-01", "hosts", "DB-01")
    g.add_structural("APP-01", "runs_on", "SRV-01", w_start=2, w_end=5)
    return g


def ev(g, s, r, o, w, **attrs):
    return Event(s, r, o, g.window_start(w) + timedelta(hours=3), w, attrs)


def test_structural_edges_have_validity_intervals():
    g = small()
    assert g.neighbors("SRV-01", 0) == {"DB-01"}
    assert g.neighbors("SRV-01", 3) == {"DB-01", "APP-01"} and g.neighbors("APP-01", 3) == {"SRV-01"}
    assert g.neighbors("SRV-01", 6) == {"DB-01"} and g.neighbors("APP-01", 6) == set()
    with pytest.raises(ValueError):
        g.add_structural("SRV-01", "hosts", "DB-99")


def test_duplicate_events_are_ignored_and_late_arrivals_stay_ordered():
    g = small()
    e = ev(g, "VULN-1", "detected_on", "SRV-01", 4, severity=5)
    assert g.add_event(e) is True and g.add_event(e) is False and len(g.events) == 1
    g.add_event(ev(g, "VULN-2", "detected_on", "SRV-01", 1, severity=3))          # arrives late, older window
    assert [x.w for x in g.asset_events("SRV-01")] == [1, 4]
    assert g.n_windows == 5


def test_snapshot_is_deterministic_sensitive_and_reconstructs_the_past():
    g = small()
    g.add_event(ev(g, "VULN-1", "detected_on", "SRV-01", 1, severity=5))
    s1 = g.snapshot(1)
    g.add_event(ev(g, "VULN-2", "detected_on", "SRV-01", 3, severity=8))
    assert g.snapshot(1) == s1                                                   # later events do not alter the past snapshot
    assert g.snapshot(3)["sha256"] != s1["sha256"] and g.snapshot(3)["event_edges"] == 2
    h = small(); h.add_event(ev(h, "VULN-1", "detected_on", "SRV-01", 1, severity=5))
    assert h.snapshot(1) == s1


def test_ego_graph_contains_neighbours_and_recent_audit_entities_only():
    g = small()
    g.add_event(ev(g, "VULN-1", "detected_on", "SRV-01", 5, severity=5))
    g.add_event(ev(g, "VULN-0", "detected_on", "SRV-01", 1, severity=5))         # too old for a 3-day look-back at w=5
    g.add_event(ev(g, "USER-01", "accessed", "SRV-01", 5))                        # context only, excluded
    e = g.ego("SRV-01", 5, back=3)
    ids = {n["id"] for n in e["nodes"]}
    assert ids == {"SRV-01", "DB-01", "APP-01", "VULN-1"} and e["from_window"] == 3
    assert any(x["r"] == "detected_on" and x["w"] == 5 for x in e["edges"]) and any(x["r"] == "hosts" and x["w"] is None for x in e["edges"])
