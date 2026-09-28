"""Simulated users for: does starting a 1P subscription increase 3P spend?

Design:
- Treatment: started_1p_sub, binary, adoption during a fixed window starting day 0
- Periods: pre2 = days -56..-29, pre = days -28..-1, post = days 0..27
- l28d_* covariates are computed from the `pre` period, as a real pipeline would
- l28d_3p_gamer is spend-based: any game-category order in the pre period

Users are simulated as a latent purchase process; observed columns are derived from it.

Three scenarios share the same spend process and differ only in how users select into
the subscription and what happens in the post period:

- "main":             selection on observed engagement and device; no calendar shocks.
                      Parallel trends holds, and holds conditional on covariates too.
- "seasonal_shock":   same selection, plus a post-period seasonal lift that scales with
                      each user's spend level (e.g. holidays). The pre-trend test still
                      passes, but unconditional parallel trends fails in the post period.
- "latent_selection": selection on unobserved persistent traits (true purchase rate,
                      game affinity). Parallel trends holds unconditionally, but not
                      conditional on noisy pre-period spend (regression to the mean).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_CATEGORIES = 10  # category 0 = games
SCENARIOS = ("main", "seasonal_shock", "latent_selection")


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(
    n_users: int = 1_000_000,
    seed: int = 0,
    scenario: str = "main",
    trend_sd: float = 0.1,
    lam_shape: float = 0.45,
    aov_sigma: float = 1.3,
    convert_rate: float = 0.02,
    whale_rate: float = 0.02,
    return_latent: bool = False,
):
    """Simulate one dataset.

    Returns (obs, truth): `obs` holds only the columns an analyst would see; `truth`
    holds the true ATT in dollars and as a share of treated users' counterfactual
    post-period spend. Use `load_observed` when the truth must stay out of sight.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    rng = np.random.default_rng(seed)
    n = n_users

    # ---------------- latent user traits (never shown to estimators) ----------------
    is_payer = rng.random(n) < 0.45
    lam = np.where(is_payer, rng.gamma(shape=lam_shape, scale=1.5 / lam_shape, size=n), 0.0)
    # Whales: a small share of payers with much higher order rate and order value
    is_whale = is_payer & (rng.random(n) < whale_rate)
    lam = np.where(is_whale, lam * 4.0, lam)  # orders per 28 days
    wealth = rng.normal(size=n)
    # Latent engagement, correlated with purchase rate; observed through app sessions
    log_lam = np.log1p(lam)
    engagement = 0.7 * (log_lam - log_lam.mean()) / log_lam.std() + np.sqrt(1 - 0.49) * rng.normal(size=n)
    sessions_pre = rng.poisson(10 * np.exp(0.5 * engagement))
    game_affinity = rng.beta(0.6, 1.4, size=n)  # share of a user's orders that are games (mean 0.3)
    premium_device = (rng.random(n) < _sigmoid(-0.6 + 1.2 * wealth)).astype(int)

    # Average order value: lognormal, median ~$3, higher for game-heavy and wealthy users
    aov_median = 3.0 * np.exp(0.4 * game_affinity + 0.5 * wealth) * np.where(is_whale, 4.0, 1.0)

    # Small idiosyncratic drift in each user's log order rate (mean zero)
    trend = rng.normal(0, trend_sd, n)

    # Recent converts: first-ever 3P purchase happens during the `pre` window
    recent_convert = is_payer & (rng.random(n) < convert_rate)

    # Calendar shocks on the log order rate, shared by all users. Multiplicative, so in
    # dollars they are larger for heavier spenders.
    season = {"pre2": 0.0, "pre": 0.0, "post": 0.0}
    if scenario == "seasonal_shock":
        season["post"] = 0.15
    t_idx = {"pre2": -1, "pre": 0, "post": 1}

    # Per-user category preferences over the 9 non-game categories
    cat_pref = rng.dirichlet(np.full(N_CATEGORIES - 1, 0.5), size=n)

    def draw_period(rate):
        """Draw orders for one 28-day period; return spend, #orders, #game orders, #distinct cats."""
        orders = rng.poisson(rate)
        uid = np.repeat(np.arange(n), orders)
        is_game = rng.random(len(uid)) < game_affinity[uid]
        cdf = np.cumsum(cat_pref[uid], axis=1)
        other = 1 + (rng.random(len(uid))[:, None] > cdf).sum(axis=1)
        cat = np.where(is_game, 0, np.minimum(other, N_CATEGORIES - 1))
        value = aov_median[uid] * np.exp(rng.normal(0, aov_sigma, len(uid)))

        spend = np.bincount(uid, weights=value, minlength=n)
        n_game = np.bincount(uid, weights=is_game, minlength=n).astype(int)
        distinct = np.zeros(n, dtype=int)
        if len(uid):
            pairs = np.unique(uid * N_CATEGORIES + cat)
            distinct = np.bincount(pairs // N_CATEGORIES, minlength=n)
        return spend, orders, n_game, distinct

    def rate(period, active):
        return np.where(active, lam * np.exp(trend * t_idx[period] + season[period]), 0.0)

    # ---------------- pre periods ----------------
    spend_pre2, *_ = draw_period(rate("pre2", is_payer & ~recent_convert))
    spend_pre, orders_pre, game_orders_pre, cats_pre = draw_period(rate("pre", is_payer))

    obs = pd.DataFrame({
        "user_id": np.arange(n),
        "l28d_app_sessions": sessions_pre,
        "l28d_3p_num_orders": orders_pre,
        "l28d_3p_gamer": (game_orders_pre > 0).astype(int),
        "l28d_new_buyer": (recent_convert & (orders_pre > 0)).astype(int),
        "l28d_3p_spend_categories": cats_pre,
        "premium_device": premium_device,
        "3p_spend_pre2": spend_pre2,
        "3p_spend_pre": spend_pre,
    })

    # ---------------- treatment ----------------
    if scenario == "latent_selection":
        # Selection on who users are (true purchase rate, game affinity), which the
        # analyst only sees through noisy pre-period spend
        logit = -4.9 + 0.8 * premium_device + 0.9 * log_lam + 2.0 * game_affinity * is_payer
    else:
        # Selection on observed engagement and device, not on spend history.
        # Threshold and interaction terms make a main-effects logit misspecified.
        heavy = (sessions_pre >= 25).astype(float)
        logit = (-4.6 + 0.6 * premium_device + 0.5 * np.log1p(sessions_pre)
                 + 1.0 * heavy + 0.8 * heavy * premium_device)
    p_true = _sigmoid(logit)
    treated = (rng.random(n) < p_true).astype(int)

    # ---------------- post period with heterogeneous effect ----------------
    # Multiplicative lift on order rate: ~+5% for non-gamers up to ~+20% for game-heavy users
    tau = 0.05 + 0.15 * game_affinity
    rate_post0 = rate("post", is_payer)
    spend_post, *_ = draw_period(rate_post0 * np.exp(tau * treated))

    obs["started_1p_sub"] = treated
    obs["3p_spend_post"] = spend_post
    obs["3p_spend_delta"] = obs["3p_spend_post"] - obs["3p_spend_pre"]

    # True ATT in expectation: extra orders x mean order value, averaged over treated users
    mean_order_value = aov_median * np.exp(aov_sigma**2 / 2)
    indiv_effect = rate_post0 * (np.exp(tau) - 1) * mean_order_value
    cf_post = rate_post0 * mean_order_value
    t = treated == 1
    truth = {
        "att_dollars": float(indiv_effect[t].mean()),
        "att_pct_of_counterfactual": float(indiv_effect[t].sum() / cf_post[t].sum()),
    }

    if return_latent:
        latent = pd.DataFrame({"is_payer": is_payer, "lam": lam, "trend": trend,
                               "game_affinity": game_affinity, "p_true": p_true,
                               "tau": tau, "indiv_effect": indiv_effect})
        return obs, truth, latent
    return obs, truth


def load_observed(n_users: int = 1_000_000, seed: int = 0) -> pd.DataFrame:
    """The main scenario's data exactly as an analyst would receive it: no ground truth."""
    obs, _ = simulate(n_users=n_users, seed=seed, scenario="main")
    return obs
