# Does starting a 1P subscription increase 3P spend?

A simulated reconstruction of an observational causal analysis I did on proprietary
app-store data: estimating how much starting a first-party (1p) subscription *caused*
third-party (3p, commission-generating) spend to rise, when users chose whether to subscribe.

The real data can't be shared, so `src/simulate.py` generates 1M users with the same
structure: self-selection on engagement, heavy-tailed spend, heterogeneous effects.

## Two notebooks

**[`01_analysis.ipynb`](01_analysis.ipynb): the analysis, with no ground truth.** It
loads only the columns an analyst would see and picks a method from diagnostics alone.

| Method | Package | Verdict from diagnostics |
|---|---|---|
| Difference-in-differences | `statsmodels` | Credible: pre-trend test passes; no sign of spend-scaled shocks |
| IPSW, logistic on raw columns | `scikit-learn` | Rejected: propensities near 1 |
| IPSW, logistic on log columns | `scikit-learn` | Suspect: threshold/interaction features unbalanced |
| IPSW, boosted | `scikit-learn` | Credible |
| **Doubly robust DiD** | `doubleml` (`DoubleMLDID`) | **Credible; reported as the headline** |

The core argument (Section 5): parallel trends holds, so DoubleML isn't needed for bias,
and it doesn't buy precision here either (its CI is wider). It's reported because DiD
and DoubleML fail under *different* threats. A post-period shock that scales with spend
breaks DiD; selection on noisily observed traits breaks DoubleML. Their agreement is
what makes the estimate credible, and the notebook checks each threat with a
truth-free diagnostic.

**[`02_validation.ipynb`](02_validation.ipynb): checking the argument against the truth.**
It reveals the true effect and runs a Monte Carlo (8 datasets × 3 scenarios):

| Scenario | DiD bias | DoubleML bias |
|---|---|---|
| Parallel trends holds | +7% | +13% |
| Spend-scaled seasonal shock (pre-trend test still passes) | +100% | +7% |
| Selection on hidden traits | +4% | +56% |

## Layout

```
01_analysis.ipynb        the analysis (no ground truth)
02_validation.ipynb      ground truth, Monte Carlo, when each method wins
src/simulate.py          data-generating process, three scenarios
src/estimators.py        thin wrappers around statsmodels, scikit-learn, doubleml
src/diagnostics.py       overlap, ESS, balance, untreated-trend and selection checks
src/monte_carlo.py       Monte Carlo behind 02_validation
data/monte_carlo.csv     saved Monte Carlo results
```

## Run it

```bash
pip install -r requirements.txt
jupyter notebook 01_analysis.ipynb        # ~3 minutes
jupyter notebook 02_validation.ipynb      # ~2 minutes (loads saved Monte Carlo)
python -m src.monte_carlo 8               # optional: regenerate the Monte Carlo (~20 minutes)
```
