"""Download FiftyPlusOne's published backtests (GPL-2.0, github.com/FiftyPlusOne-polls/data).

They are not committed here; run this once before calibrate.py and bias_test.py.
"""
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data/raw"
FILES = {
    "fpo_backtests_2012_2024.csv":
        "https://raw.githubusercontent.com/FiftyPlusOne-polls/data/main/forecasts/2026/timeseries_unified_ens.csv",
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        urllib.request.urlretrieve(url, RAW / name)
        print("saved", RAW / name)
