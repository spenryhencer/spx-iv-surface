import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricing import black76_price, implied_vol, implied_forward, black76_delta
from providers import parse_occ, pick_expiries, prefer_pm_root
from rates import RateCurve


@pytest.mark.parametrize("K,is_call", [(6000, False), (7000, False), (7400, True), (8200, True)])
@pytest.mark.parametrize("sigma,T", [(0.12, 0.05), (0.25, 0.5), (0.40, 1.5)])
def test_implied_vol_round_trip(K, is_call, sigma, T):
    F, DF = 7400.0, np.exp(-0.04 * T)
    price = black76_price(F, K, T, DF, sigma, is_call)
    assert implied_vol(price, F, K, T, DF, is_call) == pytest.approx(sigma, abs=1e-6)


def test_put_call_parity_holds():
    F, K, T, DF, s = 7400.0, 7200.0, 0.3, 0.99, 0.2
    c = black76_price(F, K, T, DF, s, True)
    p = black76_price(F, K, T, DF, s, False)
    assert c - p == pytest.approx(DF * (F - K), abs=1e-8)


def test_implied_vol_rejects_arbitrage_prices():
    F, K, T, DF = 7400.0, 7000.0, 0.25, 0.99
    assert np.isnan(implied_vol(DF * (F - K) - 1, F, K, T, DF, True))   # below intrinsic
    assert np.isnan(implied_vol(DF * F + 1, F, K, T, DF, True))         # above cap


def test_implied_forward_recovers_forward():
    F, T, DF = 7432.5, 0.4, np.exp(-0.04 * 0.4)
    K = np.arange(7000, 7900, 25.0)
    c = [black76_price(F, k, T, DF, 0.18, True) for k in K]
    p = [black76_price(F, k, T, DF, 0.18, False) for k in K]
    assert implied_forward(K, c, p, DF) == pytest.approx(F, abs=1e-6)


def test_delta_signs():
    assert 0 < black76_delta(7400, 7600, 0.25, 0.2, True) < 0.5
    assert -0.5 < black76_delta(7400, 7200, 0.25, 0.2, False) < 0


def test_parse_occ_symbols():
    p = parse_occ(pd.Series(["SPXW261218C07400000", "SPX271119P06375500"]))
    assert list(p["root"]) == ["SPXW", "SPX"]
    assert list(p["expiry"]) == [date(2026, 12, 18), date(2027, 11, 19)]
    assert list(p["cp"]) == ["C", "P"]
    assert list(p["strike"]) == [7400.0, 6375.5]


def test_prefer_pm_root_drops_am_duplicates():
    q = pd.DataFrame({"root": ["SPX", "SPXW", "SPX"],
                      "expiry": [date(2026, 12, 18), date(2026, 12, 18), date(2027, 8, 20)]})
    assert list(prefer_pm_root(q)["root"]) == ["SPXW", "SPX"]


def test_pick_expiries_skips_near_dated_and_dedupes():
    today = date(2026, 11, 8)
    exps = [date(2026, 11, 10), date(2026, 11, 20), date(2026, 12, 18), date(2027, 11, 19)]
    assert pick_expiries(exps, today, [3, 14, 30, 45, 365]) == exps[1:]


def test_rate_curve_continuous_conversion():
    c = RateCurve([0.25, 1.0], [0.04, 0.04], "x", "test")
    assert c.rate(0.5) == pytest.approx(2 * np.log1p(0.02))
    assert c.discount(1.0) == pytest.approx(np.exp(-c.rate(1.0)))
