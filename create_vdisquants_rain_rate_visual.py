from pathlib import Path

import create_rain_rate_difference_visual as base


base.OLD = Path("bnfvdisquantsM1.c1-old")
base.NEW = Path("bnfvdisquantsM1.c1")
base.DATES = ["20251004"]
base.OUTPUT = Path("vdisquants-rain-rate-difference-time-series.html")

if __name__ == "__main__":
    base.main()
