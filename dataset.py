"""Supervised dataset and chronological splitting (disclosure 8.4: 'chronological division ... preferred over a random split')."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HORIZON
from .features import ALL_FEATURES, add_targets, extract
from .tkg import TemporalKG


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    bounds: dict            # window ranges actually used, for the report

    def fit_frame(self):
        return pd.concat([self.train, self.val])


def build(tkg: TemporalKG, horizon: int = HORIZON) -> pd.DataFrame:
    return add_targets(extract(tkg), horizon)


def chronological_split(ds: pd.DataFrame, train_frac=0.60, val_frac=0.20, purge: int = HORIZON) -> Split:
    """Split by WINDOW, never by row. A sample at window w has its label taken from window w+h, so the last `purge`
    windows before each boundary are dropped from the earlier part: no training label is computed from a window that a
    later split uses as an input."""
    ws = np.sort(ds["window"].unique())
    n = len(ws)
    i_val, i_test = int(n * train_frac), int(n * (train_frac + val_frac))
    tr_w, va_w, te_w = ws[:i_val - purge], ws[i_val:i_test - purge], ws[i_test:]
    part = lambda w: ds[ds["window"].isin(w)].copy()
    bounds = {"train": [int(tr_w[0]), int(tr_w[-1])], "val": [int(va_w[0]), int(va_w[-1])], "test": [int(te_w[0]), int(te_w[-1])],
              "purged_windows": purge}
    return Split(part(tr_w), part(va_w), part(te_w), bounds)


def expanding_folds(ds: pd.DataFrame, n_folds=4, first_train_frac=0.40, purge: int = HORIZON):
    """Rolling-origin evaluation: train on windows [0, k), test on the next block, for successive k."""
    ws = np.sort(ds["window"].unique())
    n = len(ws)
    start = int(n * first_train_frac)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        yield ds[ds["window"].isin(ws[: a - purge])].copy(), ds[ds["window"].isin(ws[a:b])].copy()


def xy(df: pd.DataFrame, cols=ALL_FEATURES):
    return df[cols], df["y"].astype(int), df["ts_future"].astype(float)
