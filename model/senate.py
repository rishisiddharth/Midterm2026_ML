"""2026 Senate forecast, v0.

Pipeline
  1. Fundamentals: state partisan lean (Cook PVI) + national environment (generic ballot)
     + incumbency. Gives an expected margin for every race, polled or not.
  2. Polls: recency-weighted average of individual public polls for each race.
  3. Blend: precision (inverse-variance) weighting of the two. Many polls -> polls dominate;
     no polls -> fundamentals only.
  4. Simulate 40,000 elections with correlated national/regional/race errors whose sizes
     come from out-of-sample backtests (calibrate.py).
  5. Count seats. Republicans keep the majority with 50+ seats (the VP breaks ties).

All margins are "challenger minus Republican" in points. The challenger is the Democrat,
except where an independent is the main opponent (Nebraska; Montana if polls show it).

Parameters marked PRIOR are judgment calls taken from the literature, not fitted here.
Fitting them needs historical state-level inputs we have not assembled yet (see README).
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"

ELECTION = date(2026, 11, 3)
REGIONS = ["Northeast", "Midwest", "South", "West"]

CFG = {
    "incumbent_elected": 3.0,     # PRIOR: margin bonus for an elected incumbent's party
    "incumbent_appointed": 1.0,   # PRIOR: appointed incumbents haven't faced voters
    "fund_sd": 8.0,               # PRIOR: fundamentals-only race error (Senate, late cycle)
    "poll_floor_sd": 3.5,         # PRIOR: error of a perfect poll average (house/mode effects)
    "poll_sampling_var": 30.0,    # PRIOR: per-poll noise; average variance shrinks as 30/n
    "poll_half_life_days": 14,    # recency weighting of polls
    "partisan_poll_weight": 0.5,  # campaign/party-sponsored polls count half
    "partisan_poll_shift": 2.0,   # and are shifted 2 pts against the sponsor
    "race_sd_inflation": 1.25,    # our model is simpler than the benchmark -> wider errors
    "regional_sd": 1.5,           # PRIOR
    "n_sims": 40_000,
    "r_seats_not_up": 31,
    "nonr_seats_not_up": 34,      # 32 D + 2 independents (King, Sanders)
}


def poll_average(polls: pd.DataFrame, today: date) -> pd.DataFrame:
    """Recency-weighted average per state. Returns state, avg margin, effective n."""
    if polls.empty:
        return pd.DataFrame(columns=["state", "poll_avg", "n_eff", "n_polls", "last_poll"])
    p = polls.copy()
    p["end"] = pd.to_datetime(p.end_date)
    age = (pd.Timestamp(today) - p.end).dt.days.clip(lower=0)
    w = 0.5 ** (age / CFG["poll_half_life_days"])
    w *= p.lv_or_rv.str.upper().map({"LV": 1.0, "RV": 0.8}).fillna(0.9)  # UNK = not reported
    partisan = p.sponsor_partisan.fillna("").str.upper()
    w *= np.where(partisan.isin(["D", "R"]), CFG["partisan_poll_weight"], 1.0)
    shift = np.select([partisan.eq("D"), partisan.eq("R")],
                      [-CFG["partisan_poll_shift"], CFG["partisan_poll_shift"]], 0.0)
    p["m"] = p.challenger_pct - p.rep_pct + shift
    p["w"] = w
    g = p.groupby("state")
    out = g.apply(lambda x: pd.Series({
        "poll_avg": np.average(x.m, weights=x.w),
        # effective number of polls: a fresh likely-voter poll counts 1, stale/RV/partisan less
        "n_eff": x.w.sum(),
        "n_polls": len(x),
        "last_poll": x.end.max().date().isoformat(),
    }), include_groups=False).reset_index()
    return out


def build(today: date = date(2026, 10, 22), bias: float = 0.0):
    races = pd.read_csv(P / "senate_races_2026.csv")
    polls = pd.read_csv(P / "senate_polls_2026.csv")
    nat = json.loads((P / "national_2026.json").read_text())
    cal = json.loads((P / "error_calibration.json").read_text())["chambers"]["Senate"]

    G = nat["generic_ballot_margin"]["value"]           # D minus R

    # 1. fundamentals -------------------------------------------------------------
    lean = 2 * races.pvi_d                                # PVI is two-party share -> margin
    inc = np.select(
        [races.incumbent_status.eq("elected"), races.incumbent_status.eq("appointed")],
        [CFG["incumbent_elected"], CFG["incumbent_appointed"]], 0.0)
    inc_sign = np.where(races.incumbent_party.eq("R"), -1.0, 1.0)
    races["fund"] = lean + G + inc * inc_sign + races.challenger_adj.fillna(0)

    # 2. polls ----------------------------------------------------------------------
    pa = poll_average(polls, today)
    races = races.merge(pa, on="state", how="left")
    races["n_eff"] = races.n_eff.fillna(0)

    # 3. blend ----------------------------------------------------------------------
    var_f = CFG["fund_sd"] ** 2
    var_p = np.where(races.n_eff > 0,
                     CFG["poll_floor_sd"] ** 2 + CFG["poll_sampling_var"] / races.n_eff.clip(lower=1e-9),
                     np.inf)
    w_poll = np.where(np.isfinite(var_p), (1 / var_p) / (1 / var_p + 1 / var_f), 0.0)
    races["w_poll"] = w_poll
    races["mu"] = np.where(w_poll > 0, w_poll * races.poll_avg.fillna(0) + (1 - w_poll) * races.fund,
                           races.fund)
    blend_sd = np.sqrt(1 / (1 / var_f + np.where(np.isfinite(var_p), 1 / var_p, 0)))
    floor = cal["race_residual_sd"] * CFG["race_sd_inflation"]
    races["sig_race"] = np.maximum(blend_sd, floor)

    # 4. simulate -------------------------------------------------------------------
    region_idx = races.region.map({r: i for i, r in enumerate(REGIONS)}).to_numpy()
    sims = simulate(races.mu, races.sig_race, region_idx, len(REGIONS),
                    sig_nat=cal["national_error_rms"], sig_reg=CFG["regional_sd"],
                    bias=bias, n_sims=CFG["n_sims"])
    nonr_win = sims > 0

    # 5. aggregate ------------------------------------------------------------------
    races["p_challenger"] = nonr_win.mean(0)
    races["lo80"], races["med"], races["hi80"] = np.percentile(sims, [10, 50, 90], axis=0)
    r_seats = CFG["r_seats_not_up"] + (~nonr_win).sum(1)

    # tipping point: the race that delivers the 51st non-Republican seat in each simulation
    need = 51 - CFG["nonr_seats_not_up"]
    order = np.argsort(-sims, axis=1)[:, need - 1]
    tip = np.bincount(order, minlength=len(races)) / len(sims)
    races["p_tipping"] = tip
    races["call"] = np.where(races.p_challenger > 0.5, races.challenger, races.rep)
    races["flip"] = ((races.p_challenger > 0.5) & races.incumbent_party.eq("R")) | \
                    ((races.p_challenger < 0.5) & ~races.incumbent_party.eq("R"))
    called_r = int(CFG["r_seats_not_up"] + (races.p_challenger < 0.5).sum())
    out = {
        "r_seats_median": float(np.median(r_seats)),
        "called_r_seats": called_r,
        "flips": races.loc[races.flip, "state"].tolist(),
        "as_of": today.isoformat(),
        "days_to_election": (ELECTION - today).days,
        "n_sims": CFG["n_sims"],
        "bias": bias,
        "p_r_majority": float((r_seats >= 50).mean()),
        "r_seats_mean": float(r_seats.mean()),
        "r_seats_80": [int(x) for x in np.percentile(r_seats, [10, 90])],
        "seat_hist": {int(k): int(v) for k, v in zip(*np.unique(r_seats, return_counts=True))},
        "generic_ballot": G,
        "params": CFG,
        "calibration": cal,
    }
    return races, out


if __name__ == "__main__":
    races, out = build()
    cols = ["state", "challenger", "rep", "fund", "poll_avg", "n_polls", "w_poll",
            "mu", "sig_race", "p_challenger"]
    print(races[cols].sort_values("p_challenger").round(2).to_string(index=False))
    print({k: v for k, v in out.items() if k not in ("params", "calibration", "seat_hist")})
    races.to_csv(P / "senate_forecast_2026.csv", index=False)
    (P / "senate_summary_2026.json").write_text(json.dumps(out, indent=2))
