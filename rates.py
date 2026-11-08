"""
Risk-free rate curve
====================
US Treasury constant-maturity yields from FRED (keyless CSV download),
interpolated in maturity and converted to continuous compounding.
"""

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

FRED_CSV      = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TENORS        = {"DGS1MO": 1/12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0}
FALLBACK_RATE = 0.04   # used only if FRED is unreachable


class RateCurve:
    def __init__(self, tenors, yields, asof, source):
        self.tenors = np.asarray(tenors, dtype=float)
        self.yields = np.asarray(yields, dtype=float)   # bond-equivalent, decimal
        self.asof   = asof
        self.source = source

    def rate(self, T):
        """Continuously-compounded zero rate for maturity T (years)."""
        y = float(np.interp(T, self.tenors, self.yields))   # flat beyond the ends
        return 2.0 * np.log1p(y / 2.0)

    def discount(self, T):
        return float(np.exp(-self.rate(T) * T))


def fetch_treasury_curve(asof: date, timeout=15) -> RateCurve:
    """Latest FRED Treasury curve on or before `asof`; flat fallback on failure."""
    start = (asof - timedelta(days=14)).isoformat()
    url   = f"{FRED_CSV}?id={','.join(TENORS)}&cosd={start}&coed={asof.isoformat()}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df = df.set_index(df.columns[0])[list(TENORS)]
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            raise ValueError("no complete rows")
        row = df.iloc[-1]
        return RateCurve(list(TENORS.values()), row.values / 100.0,
                         row.name, "FRED Treasury CMT")
    except Exception as e:
        print(f"  ! FRED rate fetch failed ({e}); using flat {FALLBACK_RATE:.2%}")
        return RateCurve([0.0, 2.0], [FALLBACK_RATE] * 2, asof.isoformat(),
                         f"flat {FALLBACK_RATE:.2%} fallback")
