"""
Option pricing helpers
======================
Black-76 (Black-Scholes written on the forward), implied-volatility inversion
with Brent's method, and forward extraction from put-call parity.

Working on the forward F means dividends and rates enter only through F and
the discount factor DF = exp(-rT), so no separate dividend-yield input is needed.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

IV_LO, IV_HI = 1e-4, 5.0   # Brent search bracket for σ


def black76_price(F, K, T, DF, sigma, is_call):
    """Discounted Black-76 price of a European call/put on forward F."""
    if sigma <= 0 or T <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return DF * intrinsic
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if is_call:
        return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_delta(F, K, T, sigma, is_call):
    """Forward (undiscounted) delta — used to locate the 25Δ wings."""
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0


def implied_vol(price, F, K, T, DF, is_call):
    """
    Solve black76_price(σ) = price for σ with Brent's method.
    Returns NaN when the price violates no-arbitrage bounds (below intrinsic
    or above the forward/strike cap) or no root exists in [IV_LO, IV_HI].
    """
    intrinsic = DF * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    cap       = DF * (F if is_call else K)
    if not (intrinsic < price < cap) or T <= 0:
        return np.nan
    f = lambda s: black76_price(F, K, T, DF, s, is_call) - price
    if f(IV_LO) > 0 or f(IV_HI) < 0:
        return np.nan
    return brentq(f, IV_LO, IV_HI, xtol=1e-8)


def implied_forward(strikes, call_mid, put_mid, DF, n_strikes=5):
    """
    Put-call parity: C − P = DF·(F − K)  ⇒  F = K + (C − P)/DF.
    Evaluated at the n strikes where |C − P| is smallest (i.e. closest to the
    money, where both quotes are tightest) and the median is returned.
    """
    strikes  = np.asarray(strikes, dtype=float)
    diff     = np.asarray(call_mid, dtype=float) - np.asarray(put_mid, dtype=float)
    if len(strikes) == 0:
        return np.nan
    nearest  = np.argsort(np.abs(diff))[:n_strikes]
    return float(np.median(strikes[nearest] + diff[nearest] / DF))
