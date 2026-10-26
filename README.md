# Senate 2026 Model

My prediction for the 35 U.S. Senate races on November 3, 2026: **Democrats win control 51–49
(64% chance)**, flipping Maine, North Carolina, Ohio and Texas. Locked October 22, 2026, to be
scored against the public models and prediction markets after the election.

Interactive page: `docs/index.html` (prediction, race calls, rationale, model explorer, data guide,
methodology). Enable GitHub Pages on the `docs/` folder to serve it.

## Run it

```bash
pip install -r requirements.txt
python model/fetch_data.py    # FiftyPlusOne backtests (GPL-2.0), not committed here
python model/calibrate.py     # error sizes from out-of-sample backtests -> data/processed/error_calibration.json
python model/senate.py        # forecast -> data/processed/senate_forecast_2026.csv + senate_summary_2026.json
python model/bias_test.py     # does correcting for past polling misses help? -> data/processed/bias_test.json
python web/build.py           # interactive page -> web/senate_2026.html (copy to docs/index.html)
```

## Layout

```
data/raw/        downloaded by model/fetch_data.py (not committed)
data/processed/  hand-assembled 2026 inputs (each row carries a source URL) + model outputs
model/           fetch_data.py, calibrate.py, bias_test.py, simulate.py, senate.py
web/             build.py, page template, state map paths
docs/            the published page (GitHub Pages)
```

## What the model trains on, and what it doesn't

| Piece | Data | Fitted or set? |
|---|---|---|
| Error sizes (national, race) | FiftyPlusOne out-of-sample forecasts vs. results, 2012–2024, 15 days out | Fitted |
| Poll average | Public polls since mid-September 2026 | Formula (recency, LV/RV, partisan sponsor) |
| Fundamentals | Cook PVI, generic ballot, incumbency | Coefficients are priors, not fitted (v1 work) |
| Blend weights | Inverse variance of the two components | Derived from the error priors |

## Known gaps (v1 list)

1. **House.** Needs presidential results on the 2026 district lines (nine states redrew). The data
   exists (Cook 2026 PVI, The Downballot) but is not reachable from the build environment.
2. **Fit the fundamentals.** Assemble state-level PVI, incumbency and results for 2012–2024 Senate
   races and fit the coefficients with leave-one-cycle-out validation instead of using priors.
3. **House effects.** v0 discounts partisan-sponsored polls but does not estimate pollster leans.
4. **Governors.** Same pipeline, different inputs.

## Why inflation, gas prices and immigration are context, not features

These are national numbers: in a given year every race sees the same value, so they can only move
the national environment. We have about 7 comparable cycles to learn that from, which cannot support
several separate coefficients without overfitting. Their effect already reaches the model through
presidential approval and the generic ballot, which voters answer with prices and the border in mind.
The page shows them as context and the honest test (one composite economic index, leave-one-cycle-out)
is on the v1 list.

## Credits

Forecast-error backtests: FiftyPlusOne (github.com/FiftyPlusOne-polls/data, GPL-2.0), downloaded at
build time. Poll numbers were collected from RealClearPolling, 270toWin, Wikipedia and pollster
releases; each row in `data/processed/senate_polls_2026.csv` links to its source. State lean: Cook
Partisan Voting Index. Map geometry: us-atlas (ISC).
