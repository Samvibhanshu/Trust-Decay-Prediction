import os
import shutil
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trustdecay import evaluation                       # noqa: E402
from trustdecay.models import TrustEngine               # noqa: E402
from trustdecay.persist import load_tkg                 # noqa: E402
from trustdecay.service import TrustService             # noqa: E402
from trustdecay.simulator import SimConfig              # noqa: E402
from trustdecay.store import Store                      # noqa: E402
from trustdecay.webapp import create_app                # noqa: E402

FIXED_NOW = datetime(2026, 9, 28, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """A real (light) training run on a small simulated enterprise, shared by all tests."""
    out = tmp_path_factory.mktemp("artifacts")
    evaluation.run(quick=True, out_dir=out, make_figs=False, sim_cfg=SimConfig(n_assets=20, n_windows=44, seed=3), horizons=(1, 3))
    return out


@pytest.fixture
def svc(artifacts):
    return TrustService(load_tkg(artifacts / "enterprise.json.gz"), TrustEngine.load(artifacts / "model_bundle.joblib"), Store(), now_fn=lambda: FIXED_NOW)


@pytest.fixture
def client(artifacts, tmp_path):
    art = tmp_path / "art"
    shutil.copytree(artifacts, art)
    (art / "figures").mkdir(exist_ok=True)
    (art / "figures" / "fig12_confusion_matrix.png").write_bytes(b"\x89PNG fake")
    return create_app(artifacts_dir=art, now_fn=lambda: FIXED_NOW).test_client()
