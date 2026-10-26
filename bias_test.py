"""Should the forecast assume polls overstate Democrats again?

Late Senate forecasts overestimated Democrats in 6 of 7 cycles (2012-2024). Before building
that into the call, test it the way it would have been used: in each cycle, estimate the bias
only from cycles that came before, shift every race by it, and score the result.

Uses FiftyPlusOne's out-of-sample Senate forecasts 15 days out. Win probabilities are rebuilt
from each forecast margin with a normal error whose width is backed out from the published
probabilities, so the only thing that changes between methods is the assumed bias.

Output: data/processed/bias_test.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
TAU = 1.0   # prior sd of a persistent bias, points; shrinks a 7-cycle average toward zero


def shrunk_mean(vals, tau=TAU):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return 0.0
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    return float(vals.mean() * tau**2 / (tau**2 + se**2))


def main():
    d = pd.read_csv(ROOT / "data/raw/fpo_backtests_2012_2024.csv")
    r = d[(d.level == "race") & (d.chamber == "Senate") & (d.D == 15)].copy()
    r["err"] = r.actual - r.est
    mid = r[(r.p_dem > 0.02) & (r.p_dem < 0.98)]
    k = ((mid.hi - mid.lo) / (mid.est / norm.ppf(mid.p_dem))).median()  # interval width in sds
    r["sd"] = (r.hi - r.lo) / k
    nat = r[r.est.abs() < 25].groupby("cycle").err.mean()

    methods = {
        "No correction": lambda c: 0.0,
        "Shrunk average of earlier cycles": lambda c: shrunk_mean([nat[k] for k in nat.index if k < c]),
        "Full average of all other cycles (hindsight)": lambda c: float(np.mean([nat[k] for k in nat.index if k != c])),
    }
    rows = []
    for c in sorted(r.cycle.unique()):
        rc = r[r.cycle == c]
        for name, f in methods.items():
            b = f(c)
            p = norm.cdf((rc.est + b) / rc.sd)
            rows.append({"cycle": int(c), "method": name, "bias": round(b, 2),
                         "brier": float(((p - rc.dem_won) ** 2).mean()),
                         "correct": float(((p > 0.5) == rc.dem_won).mean())})
    t = pd.DataFrame(rows)
    summary = t.groupby("method", sort=False)[["brier", "correct"]].mean().round(4).reset_index()
    out = {"n_races": int(len(r)), "tau": TAU,
           "summary": summary.to_dict("records"),
           "by_cycle": t.round(4).to_dict("records")}
    (ROOT / "data/processed/bias_test.json").write_text(json.dumps(out, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
