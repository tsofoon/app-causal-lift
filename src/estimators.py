"""Thin wrappers around statsmodels, scikit-learn and DoubleML.

The estimation itself is done by the packages; these functions only reshape the data
and return a common result type so the notebook can compare methods side by side.
Every estimator targets the ATT: the average post-period lift in 3P spend among users
who started the 1P subscription.
"""

from __future__ import annotations

from dataclasses import dataclass

import doubleml as dml
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TREAT = "started_1p_sub"
OUTCOME = "3p_spend_delta"
SPEND_COLS = ["3p_spend_pre2", "3p_spend_pre"]
COVARIATES = [
    "l28d_app_sessions",
    "l28d_3p_num_orders",
    "l28d_3p_gamer",
    "l28d_new_buyer",
    "l28d_3p_spend_categories",
    "premium_device",
    "3p_spend_pre2",
    "3p_spend_pre",
]

GBM_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15)


@dataclass
class Estimate:
    method: str
    att: float
    se: float

    @property
    def ci(self) -> tuple[float, float]:
        return self.att - 1.96 * self.se, self.att + 1.96 * self.se


# ---------------------------------------------------------------------------------
# Difference-in-differences (statsmodels)
# ---------------------------------------------------------------------------------
def to_long(df: pd.DataFrame, periods=("pre2", "pre", "post")) -> pd.DataFrame:
    """User x period panel with columns user_id, treated, period, spend."""
    cols = {f"3p_spend_{p}": p for p in periods}
    long = df[["user_id", TREAT, *cols]].melt(
        id_vars=["user_id", TREAT], var_name="period", value_name="spend"
    )
    long["period"] = long["period"].map(cols)
    return long.rename(columns={TREAT: "treated"})


def did(df: pd.DataFrame) -> Estimate:
    """Canonical 2x2 DiD (pre vs post): spend ~ treated * post, SEs clustered by user."""
    long = to_long(df, periods=("pre", "post"))
    long["post"] = (long.period == "post").astype(int)
    fit = smf.ols("spend ~ treated * post", long).fit(
        cov_type="cluster", cov_kwds={"groups": long.user_id}
    )
    return Estimate("DiD", fit.params["treated:post"], fit.bse["treated:post"])


def event_study(df: pd.DataFrame) -> pd.DataFrame:
    """Treated x period coefficients relative to `pre` (the period before launch).

    The pre2 coefficient is the pre-trend test: under parallel trends it is zero.
    """
    long = to_long(df)
    fit = smf.ols(
        "spend ~ C(period, Treatment('pre')) * treated", long
    ).fit(cov_type="cluster", cov_kwds={"groups": long.user_id})
    rows = [{"period": "pre", "estimate": 0.0, "se": 0.0}]
    for p in ("pre2", "post"):
        name = f"C(period, Treatment('pre'))[T.{p}]:treated"
        rows.append({"period": p, "estimate": fit.params[name], "se": fit.bse[name]})
    return pd.DataFrame(rows).set_index("period").loc[["pre2", "pre", "post"]]


# ---------------------------------------------------------------------------------
# Propensity scores (scikit-learn) and IPSW (statsmodels WLS)
# ---------------------------------------------------------------------------------
SKEWED_COLS = ["l28d_app_sessions", *SPEND_COLS]


def design_matrix(df: pd.DataFrame, log_skewed: bool) -> np.ndarray:
    """Covariate matrix; optionally log1p the heavy-tailed columns (sessions, spend)."""
    X = df[COVARIATES].astype(float).copy()
    if log_skewed:
        X[SKEWED_COLS] = np.log1p(X[SKEWED_COLS])
    return X.to_numpy()


def fit_propensity(df: pd.DataFrame, model: str, log_skewed: bool = True) -> np.ndarray:
    """In-sample P(treated | X). model: 'logit' or 'gbm'."""
    X, d = design_matrix(df, log_skewed), df[TREAT].to_numpy()
    if model == "logit":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    elif model == "gbm":
        clf = HistGradientBoostingClassifier(**GBM_PARAMS, random_state=0)
    else:
        raise ValueError(model)
    return clf.fit(X, d).predict_proba(X)[:, 1]


def att_weights(d: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Treated users keep weight 1; controls get p/(1-p) so they resemble the treated."""
    return np.where(d == 1, 1.0, p / (1 - p))


def ipsw(df: pd.DataFrame, p: np.ndarray, label: str) -> Estimate:
    """Weighted regression of the spend change on treatment (robust SEs).

    SEs treat the weights as fixed, so they ignore propensity-estimation error.
    """
    d = df[TREAT].to_numpy()
    fit = sm.WLS(df[OUTCOME].to_numpy(), sm.add_constant(d), weights=att_weights(d, p)).fit(
        cov_type="HC1"
    )
    return Estimate(label, fit.params[1], fit.bse[1])


# ---------------------------------------------------------------------------------
# DoubleML: doubly robust DiD (Sant'Anna & Zhao 2020) with ML nuisance models
# ---------------------------------------------------------------------------------
def dml_did(
    df: pd.DataFrame,
    outcome: str = OUTCOME,
    ml_g=None,
    ml_m=None,
    n_folds: int = 5,
    seed: int = 0,
    label: str = "DoubleML DiD",
):
    """Fit doubleml.DoubleMLDID on the two-period panel (outcome = change in spend).

    ml_g: outcome regression E[delta | X, D=0];  ml_m: propensity P(D=1 | X).
    Returns (Estimate, fitted DoubleMLDID object).
    """
    ml_g = ml_g or HistGradientBoostingRegressor(**GBM_PARAMS, random_state=seed)
    ml_m = ml_m or HistGradientBoostingClassifier(**GBM_PARAMS, random_state=seed)
    np.random.seed(seed)  # DoubleML draws its sample splits from numpy's global RNG
    data = dml.DoubleMLDIDData(df, y_col=outcome, d_cols=TREAT, x_cols=COVARIATES)
    model = dml.DoubleMLDID(data, ml_g=ml_g, ml_m=ml_m, n_folds=n_folds, clipping_threshold=0.01)
    model.fit()
    return Estimate(label, float(model.coef[0]), float(model.se[0])), model


def linear_learners():
    """Linear nuisance models (log sessions/spend, standardized) for a learner-stability check."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import FunctionTransformer

    skewed_idx = [COVARIATES.index(c) for c in SKEWED_COLS]
    prep = ColumnTransformer(
        [("log_skewed", FunctionTransformer(np.log1p), skewed_idx)], remainder="passthrough"
    )
    ml_g = make_pipeline(prep, StandardScaler(), LinearRegression())
    ml_m = make_pipeline(prep, StandardScaler(), LogisticRegression(max_iter=2000))
    return ml_g, ml_m


def winsorize_outcome(df: pd.DataFrame, q: float = 0.999) -> tuple[pd.Series, float]:
    """Change in spend after capping pre and post spend at the q-quantile of pre spend."""
    cap = float(df["3p_spend_pre"].quantile(q))
    delta = np.minimum(df["3p_spend_post"], cap) - np.minimum(df["3p_spend_pre"], cap)
    return delta, cap
