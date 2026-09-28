"""Re-run the estimators on fresh simulated datasets, for each scenario, and save results.

Takes ~25 minutes at 1M users x 8 datasets x 3 scenarios, so the validation notebook
loads the saved CSV. Run from the project root:  python -m src.monte_carlo
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

from . import estimators as E
from .simulate import SCENARIOS, simulate

OUT = Path(__file__).resolve().parents[1] / "data" / "monte_carlo.csv"


def run(n_reps: int = 8, n_users: int = 1_000_000, first_seed: int = 100) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    rows = []
    for scenario in SCENARIOS:
        for rep in range(n_reps):
            t0 = time.time()
            seed = first_seed + rep
            obs, truth = simulate(n_users=n_users, seed=seed, scenario=scenario)
            pre_trend = E.event_study(obs).loc["pre2"]
            ests = [
                E.did(obs),
                E.ipsw(obs, E.fit_propensity(obs, "gbm"), "IPSW (boosted)"),
                E.dml_did(obs, seed=seed)[0],
            ]
            for e in ests:
                lo, hi = e.ci
                rows.append({
                    "scenario": scenario, "seed": seed, "method": e.method,
                    "estimate": e.att, "se": e.se, "true_att": truth["att_dollars"],
                    "covered": lo <= truth["att_dollars"] <= hi,
                    "pre_trend": pre_trend["estimate"], "pre_trend_se": pre_trend["se"],
                })
            print(f"{scenario} rep {rep + 1}/{n_reps} done in {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    df = run(n_reps=n)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {OUT}")
