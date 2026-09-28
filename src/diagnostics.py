"""Diagnostics that judge an estimator without knowing the true effect."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimators import TREAT


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size: how many equally weighted users the weights are worth."""
    return float(w.sum() ** 2 / np.sum(w**2))


def weight_summary(d: np.ndarray, p: np.ndarray) -> dict:
    """Overlap and weight-concentration diagnostics for ATT weights on the control group."""
    w = p[d == 0] / (1 - p[d == 0])
    top = np.sort(w)[::-1]
    return {
        "max propensity": float(p.max()),
        "control ESS": effective_sample_size(w),
        "control n": int((d == 0).sum()),
        "share of control weight on top 10 users": float(top[:10].sum() / w.sum()),
    }


def balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Covariates as modeled, plus threshold/interaction features the models never see.

    Checking balance only on the modeled columns can hide a misspecified propensity
    model; the derived features are where that shows up.
    """
    heavy = df["l28d_app_sessions"] >= 25
    out = pd.DataFrame({
        "log_app_sessions": np.log1p(df["l28d_app_sessions"]),
        "num_orders": df["l28d_3p_num_orders"],
        "gamer": df["l28d_3p_gamer"],
        "new_buyer": df["l28d_new_buyer"],
        "spend_categories": df["l28d_3p_spend_categories"],
        "premium_device": df["premium_device"],
        "log_spend_pre2": np.log1p(df["3p_spend_pre2"]),
        "log_spend_pre": np.log1p(df["3p_spend_pre"]),
        # derived: never passed to any model directly
        "heavy user (25+ sessions)": heavy.astype(float),
        "heavy user & premium device": (heavy & (df["premium_device"] == 1)).astype(float),
        "3+ categories": (df["l28d_3p_spend_categories"] >= 3).astype(float),
    })
    return out


DERIVED = ["heavy user (25+ sessions)", "heavy user & premium device", "3+ categories"]


def smd(df: pd.DataFrame, w: np.ndarray | None = None) -> pd.Series:
    """Absolute standardized mean difference, treated vs (optionally weighted) control."""
    feats = balance_features(df)
    d = df[TREAT].to_numpy() == 1
    w = np.ones(len(df)) if w is None else w
    out = {}
    for c in feats:
        x = feats[c].to_numpy(dtype=float)
        pooled_sd = np.sqrt((x[d].var() + x[~d].var()) / 2)
        out[c] = abs(x[d].mean() - np.average(x[~d], weights=w[~d])) / pooled_sd
    return pd.Series(out)


def control_change_by(df: pd.DataFrame, col: str, q: int = 10) -> pd.DataFrame:
    """Mean spend change among NON-subscribers, by quantile bin of `col`.

    If untreated users' spend change varies with a covariate on which subscribers
    differ, unconditional parallel trends is exposed in the post period even when
    the pre-trend test passes. A flat profile is evidence DiD's comparison is safe.
    """
    c = df[df[TREAT] == 0]
    bins = pd.qcut(c[col], q, duplicates="drop")
    return c.groupby(bins, observed=True)["3p_spend_delta"].agg(["mean", "sem", "size"])


def selection_auc(df: pd.DataFrame, feature_sets: dict[str, list[str]], seed: int = 0) -> pd.Series:
    """Out-of-fold AUC for predicting subscription from each feature set (boosted trees).

    Used to ask whether pre-period spend predicts subscription beyond engagement and
    device. If it adds a lot, selection may run through spending traits the analyst
    only sees noisily, which is the case where conditioning on lagged spend misleads.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    d = df[TREAT].to_numpy()
    out = {}
    for name, cols in feature_sets.items():
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                             max_leaf_nodes=15, random_state=seed)
        p = cross_val_predict(clf, df[cols].to_numpy(dtype=float), d, cv=3,
                              method="predict_proba")[:, 1]
        out[name] = roc_auc_score(d, p)
    return pd.Series(out)
