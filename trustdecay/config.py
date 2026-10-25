"""Design parameters. Everything the disclosure calls 'configurable' lives here."""
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)      # start of window 0
WINDOW_HOURS = 24                                    # disclosure 8.3: T1 = 0-24h, T2 = 24-48h, ...
HORIZON = 1                                          # prediction horizon h, in windows

# Trust Decay Score -> risk category (disclosure 8.4). Configurable per organisation.
CATEGORIES = ["Low", "Moderate", "High", "Critical"]
TDS_CUTOFFS = (5.0, 15.0, 30.0)                      # (0,5] Low, (5,15] Moderate, (15,30] High, >30 Critical
CATEGORY_MIDPOINT = {"Low": 2.0, "Moderate": 10.0, "High": 22.5, "Critical": 38.0}   # for expected decay

HIGH_RISK_TS = 60.0                                  # a neighbour with node trust below this is a "high-risk asset"
CRITICAL_SEVERITY = 7

ASSET_TYPES = ["server", "database", "application", "firewall", "network_device"]
ASSET_PREFIX = {"server": "SRV", "database": "DB", "application": "APP", "firewall": "FW", "network_device": "NW"}
PREFIX_TYPE = {v: k for k, v in ASSET_PREFIX.items()}
