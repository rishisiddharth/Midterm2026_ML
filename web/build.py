"""Assemble the interactive page: model inputs + outputs + map paths -> web/senate_2026.html.

The page re-runs the same simulator in the browser (seeded) so the polling-error slider can
recompute everything live. At bias 0 it should match model/senate.py within Monte Carlo noise;
`python web/build.py` prints both so you can check.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from senate import CFG, REGIONS, build  # noqa: E402

P = ROOT / "data/processed"
WEB = ROOT / "web"


def clean(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def main():
    races, summary = build()
    polls = pd.read_csv(P / "senate_polls_2026.csv")
    nat = json.loads((P / "national_2026.json").read_text())
    cal = json.loads((P / "error_calibration.json").read_text())
    paths = json.loads((WEB / "states_paths.json").read_text())

    race_rows = []
    for _, r in races.iterrows():
        rp = polls[polls.state == r.state].sort_values("end_date", ascending=False)
        race_rows.append({
            **{k: clean(r[k]) for k in ["state", "abbr", "region", "pvi_d", "incumbent_party",
                                        "incumbent_status", "challenger", "challenger_party", "rep",
                                        "fund", "poll_avg", "n_polls", "n_eff", "w_poll", "mu",
                                        "sig_race", "p_challenger", "lo80", "med", "hi80",
                                        "fpo_avg", "ddhq_p", "kalshi_p", "note"]},
            "region_idx": REGIONS.index(r.region),
            "polls": [{k: clean(v) for k, v in p.items()} for p in
                      rp[["pollster", "end_date", "lv_or_rv", "sponsor_partisan",
                          "challenger_pct", "rep_pct", "source"]].to_dict("records")],
        })

    # how the call moves under an assumed national polling miss
    sensitivity = []
    for b in [-3, -2, -1.5, -1, 0, 1, 2]:
        _, s = build(bias=b)
        sensitivity.append({"bias": b, "p_r_majority": round(s["p_r_majority"], 3),
                            "r_seats_mean": round(s["r_seats_mean"], 1)})
    bias_test = json.loads((P / "bias_test.json").read_text())
    for col in ["p_tipping", "call"]:
        for row, v in zip(race_rows, races[col]):
            row[col] = clean(v)
    for row, v in zip(race_rows, races["flip"]):
        row["flip"] = bool(v)

    data = {
        "sensitivity": sensitivity,
        "bias_test": bias_test,
        "summary": {k: v for k, v in summary.items() if k not in ("params",)},
        "params": CFG,
        "regions": REGIONS,
        "races": race_rows,
        "national": nat,
        "calibration": cal,
        "paths": paths,
    }
    html = (WEB / "template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = WEB / "senate_2026.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
    print("python: P(R majority)=%.3f  mean R seats=%.2f" % (summary["p_r_majority"], summary["r_seats_mean"]))


if __name__ == "__main__":
    main()
