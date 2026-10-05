"""AI-Based Trust Prediction Engine (disclosure step 5). Random Forest classifier for the risk category (as in the
disclosure) plus a Random Forest regressor for the future trust score, so the dashboard can show 'current trust',
'predicted future trust' and 'predicted decay' together."""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from . import trust
from .config import ARTIFACTS, CATEGORIES
from .features import ALL_FEATURES

GRID = [{"n_estimators": n, "min_samples_leaf": l, "class_weight": cw}
        for n in (200, 400) for l in (1, 3, 6) for cw in (None, "balanced_subsample")]


def build_classifier(params=None, seed=42):
    p = {"n_estimators": 300, "min_samples_leaf": 3, "class_weight": None, **(params or {})}
    return RandomForestClassifier(random_state=seed, n_jobs=-1, **p)


def build_regressor(params=None, seed=42):
    p = {"n_estimators": 300, "min_samples_leaf": 3, **{k: v for k, v in (params or {}).items() if k != "class_weight"}}
    return RandomForestRegressor(random_state=seed, n_jobs=-1, **p)


def full_probs(clf, X) -> np.ndarray:
    """Class probabilities as an (n, 4) array even if a class is absent from the training data."""
    P = np.zeros((len(X), len(CATEGORIES)))
    P[:, clf.classes_.astype(int)] = clf.predict_proba(X)
    return P


class TrustEngine:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.clf, self.reg = bundle["clf"], bundle["reg"]
        self.features = bundle["features"]
        self.version = bundle["version"]

    @classmethod
    def load(cls, path: Path | str | None = None) -> "TrustEngine":
        path = Path(path or ARTIFACTS / "model_bundle.joblib")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python train.py` first.")
        return cls(joblib.load(path))

    def predict(self, F: pd.DataFrame) -> pd.DataFrame:
        """F: Trust Feature Vectors. Returns predictions aligned to F's index."""
        X = F[self.features]
        P = full_probs(self.clf, X)
        pred_ts = np.clip(self.reg.predict(X), 0, 100)
        cur = F["ts"].to_numpy(float)
        out = pd.DataFrame({
            "pred_future_ts": pred_ts,
            "pred_decay": trust.decay(cur, pred_ts),
            "category": [CATEGORIES[i] for i in P.argmax(1)],
            "category_index": P.argmax(1),
            "expected_decay": trust.expected_decay(P),
            "p_high_or_critical": P[:, 2] + P[:, 3],
            **{f"p_{c.lower()}": P[:, i] for i, c in enumerate(CATEGORIES)},
        }, index=F.index)
        return out

    def feature_importance(self) -> dict:
        return dict(sorted(zip(self.features, map(float, self.clf.feature_importances_)), key=lambda kv: -kv[1]))


def make_bundle(clf, reg, version, meta=None, features=ALL_FEATURES):
    return {"clf": clf, "reg": reg, "features": list(features), "version": version,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "meta": meta or {}}
