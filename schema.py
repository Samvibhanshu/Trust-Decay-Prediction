"""Audit event schema: the (S, R, O, T) temporal triples of disclosure section 8.3, plus validation."""
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

# relation -> (allowed subject prefixes, meaning)
EVENT_RELATIONS = {
    "detected_on":          ({"VULN"}, "vulnerability detected on asset (attrs.severity 1-10)"),
    "patched_on":           ({"VULN"}, "vulnerability remediated on asset"),
    "failed_login":         ({"USER", "ADMIN"}, "failed authentication attempts (attrs.count)"),
    "suspicious_access":    ({"USER", "ADMIN"}, "suspicious access attempts (attrs.count)"),
    "modified":             ({"USER", "ADMIN"}, "configuration change (attrs.authorized true/false)"),
    "privilege_escalation": ({"USER", "ADMIN"}, "privilege escalation event"),
    "backup_result":        ({"BKP"}, "backup run (attrs.success true/false)"),
    "compliance_violation": ({"CTL"}, "compliance control violated"),
    "security_alert":       ({"SYSTEM"}, "security alert raised (attrs.severity)"),
    "audit_completed":      ({"AUD"}, "formal audit completed on asset"),
    "accessed":             ({"USER", "ADMIN"}, "normal access (context only)"),
}
STRUCTURAL_RELATIONS = {"runs_on", "hosts", "connects_to", "depends_on"}
ENTITY_ID = re.compile(r"^(SRV|DB|APP|FW|NW|USER|ADMIN|VULN|BKP|CTL|AUD)-[A-Za-z0-9]{1,10}$|^SYSTEM$")
ASSET_ID = re.compile(r"^(SRV|DB|APP|FW|NW)-[A-Za-z0-9]{1,10}$")


@dataclass(frozen=True)
class Event:
    s: str
    r: str
    o: str
    t: datetime
    w: int
    attrs: dict = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self):
        return {"s": self.s, "r": self.r, "o": self.o, "t": self.t.isoformat().replace("+00:00", "Z"), "attrs": self.attrs}


def prefix(entity_id: str) -> str:
    return entity_id.split("-")[0]


class EventError(Exception):
    def __init__(self, index, errors):
        super().__init__(f"event {index}: " + "; ".join(errors))
        self.index, self.errors = index, errors


def parse_time(v) -> datetime:
    if isinstance(v, datetime):
        dt = v
    else:
        try:
            dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("t must be ISO-8601, e.g. 2026-02-03T10:15:00Z")
    if dt.tzinfo is None:
        raise ValueError("t must include a timezone (or Z)")
    return dt.astimezone(timezone.utc)


def validate_event(raw: dict, index: int, known_assets: set, window_of, max_window: int, now: datetime) -> Event:
    """Returns a normalised Event or raises EventError listing every problem. Duplicates are the caller's concern."""
    errs = []
    if not isinstance(raw, dict):
        raise EventError(index, ["must be an object with s, r, o, t"])
    s, r, o = raw.get("s"), raw.get("r"), raw.get("o")
    for name, v in (("s", s), ("r", r), ("o", o)):
        if not isinstance(v, str) or not v:
            errs.append(f"{name} is required")
    t = w = None
    try:
        t = parse_time(raw.get("t"))
        if (t - now).total_seconds() > 300:
            errs.append("t is in the future")
        w = window_of(t)
        if w < 0:
            errs.append("t is before the start of the audit timeline")
        elif w > max_window:
            errs.append(f"t is beyond the accepted range (latest allowed window is {max_window})")
    except ValueError as e:
        errs.append(str(e))
    attrs = raw.get("attrs", {})
    if not isinstance(attrs, dict):
        errs.append("attrs must be an object")
        attrs = {}
    if not errs:
        if r not in EVENT_RELATIONS:
            errs.append(f"unknown relation {r!r}; allowed: {', '.join(sorted(EVENT_RELATIONS))}")
        else:
            if not ENTITY_ID.match(s):
                errs.append(f"s {s!r} is not a valid entity id")
            elif prefix(s) not in EVENT_RELATIONS[r][0]:
                errs.append(f"relation {r} requires subject type {'/'.join(sorted(EVENT_RELATIONS[r][0]))}")
            if not ASSET_ID.match(o) or o not in known_assets:
                errs.append(f"o {o!r} is not a known asset")
            if r == "detected_on":
                sev = attrs.get("severity")
                if isinstance(sev, bool) or not isinstance(sev, (int, float)) or not 1 <= sev <= 10:
                    errs.append("detected_on needs attrs.severity between 1 and 10")
            if r in ("failed_login", "suspicious_access"):
                c = attrs.get("count", 1)
                if isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= 10000:
                    errs.append("attrs.count must be a whole number 1-10000")
            if r == "backup_result" and not isinstance(attrs.get("success"), bool):
                errs.append("backup_result needs attrs.success true/false")
            if r == "modified" and not isinstance(attrs.get("authorized", True), bool):
                errs.append("attrs.authorized must be true/false")
    if errs:
        raise EventError(index, errs)
    return Event(s, r, o, t, w, attrs)
