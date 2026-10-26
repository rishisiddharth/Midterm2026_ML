"""Correlated-error Monte Carlo simulator.

Every simulated election draws:
  national  : one shock shared by all races (Student-t, fat tails, 7 cycles of evidence)
  regional  : one shock per census region (Northeast, Midwest, South, West)
  race      : one independent shock per race (larger where polls are scarce)
and adds them to each race's expected margin (challenger minus Republican, points).

`bias` shifts the national error's centre: bias=-2 means "polls/fundamentals overstate
the challengers by 2 points, like the 2014-2024 average". Default 0: we do not assume
the direction of the next polling miss.
"""
import numpy as np


def simulate(mu, sig_race, region_idx, n_regions, sig_nat, sig_reg,
             nat_df=7, bias=0.0, n_sims=40_000, seed=2026):
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    sig_race = np.asarray(sig_race, float)
    # scale a t(df) draw so its standard deviation equals sig_nat
    t_scale = sig_nat / np.sqrt(nat_df / (nat_df - 2))
    nat = bias + t_scale * rng.standard_t(nat_df, size=(n_sims, 1))
    reg = rng.normal(0, sig_reg, size=(n_sims, n_regions))[:, region_idx]
    race = rng.normal(0, 1, size=(n_sims, len(mu))) * sig_race
    return mu + nat + reg + race          # (n_sims, n_races) simulated margins
