"""Calibrate forecast error sizes from FiftyPlusOne's published out-of-sample backtests.

Input:  data/raw/fpo_backtests_2012_2024.csv  (github.com/FiftyPlusOne-polls/data)
        Each race row has the forecast margin (est, D minus R) issued D days before
        the election, and the actual margin. Cycles 2012-2024, House and Senate.
Output: data/processed/error_calibration.json

We split each forecast error (actual - est) into:
  - a national error shared by every race in that cycle (the cycle mean), and
  - a race-level residual.
Only competitive-ish races (|est| < 25) are used so blowouts and uncontested seats
don't dominate. With 7 cycles, the national term is estimated from 7 numbers, which is
why the simulator uses a fat-tailed t distribution for it.

Caveat: these are the errors of a much more elaborate model than ours, so they are a
LOWER BOUND on our errors. senate.py inflates the race-level term to account for that.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/fpo_backtests_2012_2024.csv"
OUT = ROOT / "data/processed/error_calibration.json"


def calibrate(days_out: int = 15) -> dict:
    d = pd.read_csv(RAW)
    r = d[d.level == "race"].copy()
    r["err"] = r.actual - r.est
    out = {"source": str(RAW.name), "days_out": days_out, "chambers": {}}
    for chamber in ["Senate", "House"]:
        x = r[(r.chamber == chamber) & (r.D == days_out) & (r.est.abs() < 25)]
        nat = x.groupby("cycle").err.mean()
        resid = x.err - x.cycle.map(nat)
        full = r[(r.chamber == chamber) & (r.D == days_out)]
        brier = float(((full.p_dem - full.dem_won) ** 2).mean())
        out["chambers"][chamber] = {
            "n_races": int(len(x)),
            "national_error_by_cycle": {int(k): round(float(v), 2) for k, v in nat.items()},
            "national_error_mean": round(float(nat.mean()), 2),
            "national_error_rms": round(float(np.sqrt((nat**2).mean())), 2),
            "race_residual_sd": round(float(resid.std()), 2),
            "benchmark_brier": round(brier, 4),
            "benchmark_accuracy": round(float(((full.p_dem > 0.5) == full.dem_won).mean()), 3),
        }
    return out


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = calibrate()
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
