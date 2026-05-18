"""
Equity Implied Volatility Surface Model
========================================
Fetches live SPX options chain data via yfinance, computes implied volatility
for each option using Black-Scholes + Brent's method, then visualises:
  1. A 3-D implied volatility surface (strike × maturity × IV)
  2. Volatility smile/skew per expiry (2-D)
  3. Term structure at ATM (2-D)

Usage
-----
    python iv_surface.py

Dependencies: see requirements.txt
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
TICKER          = "^SPX"          # S&P 500 index
RISK_FREE_RATE  = 0.05            # Approximate risk-free rate (annualised)
MIN_VOLUME      = 10              # Drop options with volume below this
MIN_OPEN_INT    = 50              # Drop options with open interest below this
MAX_EXPIRIES    = 12              # Max number of expiries to include
MONEYNESS_RANGE = (0.75, 1.25)    # Keep strikes within ±25 % of spot


# ─────────────────────────────────────────────
# Black-Scholes helpers
# ─────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float,
             sigma: float, option_type: str = "call") -> float:
    """Return Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(market_price: float, S: float, K: float, T: float,
                r: float, option_type: str = "call") -> float:
    """
    Invert Black-Scholes to find implied volatility using Brent's method.
    Returns NaN if no solution is found.
    """
    if T <= 0 or market_price <= 0:
        return np.nan

    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price <= intrinsic:
        return np.nan

    try:
        iv = brentq(
            lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price,
            1e-6, 10.0, xtol=1e-7, maxiter=200
        )
        return iv if 0.001 < iv < 9.999 else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────
# Data fetching & IV calculation
# ─────────────────────────────────────────────

def fetch_options_data(ticker: str = TICKER) -> tuple[float, pd.DataFrame]:
    """
    Download the SPX options chain for all available expiries and compute
    implied volatility for each strike/expiry pair.

    Returns
    -------
    spot : float
        Current index level.
    df : pd.DataFrame
        Columns: strike, expiry, T, IV, option_type, volume, openInterest, moneyness
    """
    print(f"Fetching {ticker} spot price …")
    tkr  = yf.Ticker(ticker)
    hist = tkr.history(period="1d")
    if hist.empty:
        raise RuntimeError(f"Could not retrieve price data for {ticker}.")
    spot = float(hist["Close"].iloc[-1])
    print(f"  Spot = {spot:,.2f}")

    expiries = tkr.options
    if not expiries:
        raise RuntimeError("No options expiry dates returned by yfinance.")

    # Limit to the first MAX_EXPIRIES
    expiries = expiries[:MAX_EXPIRIES]
    today    = date.today()

    records = []
    print(f"Processing {len(expiries)} expiries …")

    for exp_str in expiries:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        T = (exp_date - today).days / 365.0
        if T <= 0:
            continue

        try:
            chain = tkr.option_chain(exp_str)
        except Exception as e:
            print(f"  ✗ {exp_str}: {e}")
            continue

        for opt_type, df_opts in [("call", chain.calls), ("put", chain.puts)]:
            # Liquidity filter
            df_opts = df_opts[
                (df_opts["volume"].fillna(0)       >= MIN_VOLUME) &
                (df_opts["openInterest"].fillna(0) >= MIN_OPEN_INT)
            ].copy()

            # Moneyness filter
            lo, hi = MONEYNESS_RANGE
            df_opts = df_opts[
                (df_opts["strike"] >= spot * lo) &
                (df_opts["strike"] <= spot * hi)
            ]

            if df_opts.empty:
                continue

            # Use mid-price for IV calculation
            df_opts["mid"] = (df_opts["bid"] + df_opts["ask"]) / 2
            df_opts = df_opts[df_opts["mid"] > 0]

            for _, row in df_opts.iterrows():
                iv = implied_vol(row["mid"], spot, row["strike"], T,
                                 RISK_FREE_RATE, opt_type)
                if np.isnan(iv):
                    continue
                records.append({
                    "strike":       row["strike"],
                    "expiry":       exp_str,
                    "T":            T,
                    "IV":           iv,
                    "option_type":  opt_type,
                    "volume":       row["volume"],
                    "openInterest": row["openInterest"],
                    "moneyness":    row["strike"] / spot,
                })

        print(f"  ✓ {exp_str}  (T={T:.3f}y, {len(records)} rows so far)")

    if not records:
        raise RuntimeError("No valid implied volatilities computed. "
                           "Markets may be closed or data unavailable.")

    df = pd.DataFrame(records)
    # Average IV where both call and put exist at the same (strike, expiry)
    df = (df.groupby(["strike", "expiry", "T", "moneyness"], as_index=False)
            .agg(IV=("IV", "mean")))
    df.sort_values(["T", "strike"], inplace=True)
    print(f"\nTotal data points: {len(df)}")
    return spot, df


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_iv_surface(spot: float, df: pd.DataFrame) -> None:
    """
    Render three Plotly figures:
      1. Interactive 3-D IV surface
      2. Volatility smile per expiry
      3. ATM term structure
    """

    # ── 1. 3-D Surface ──────────────────────────────────────────────────────
    expiries = sorted(df["expiry"].unique())
    strikes  = sorted(df["strike"].unique())

    # Build a 2-D grid: rows = expiries, cols = strikes
    pivot = df.pivot_table(index="expiry", columns="strike", values="IV")
    pivot = pivot.reindex(expiries)

    T_vals = df.groupby("expiry")["T"].mean().reindex(expiries).values
    K_vals = np.array(pivot.columns.tolist())
    Z      = pivot.values  # shape: (n_expiries, n_strikes)

    fig3d = go.Figure(data=[
        go.Surface(
            x=K_vals,
            y=T_vals,
            z=Z,
            colorscale="Viridis",
            colorbar=dict(title="IV", tickformat=".0%"),
            hovertemplate=(
                "Strike: %{x:,.0f}<br>"
                "Maturity: %{y:.3f} yr<br>"
                "IV: %{z:.1%}<extra></extra>"
            ),
        )
    ])

    fig3d.update_layout(
        title=dict(
            text="SPX Implied Volatility Surface",
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title="Strike"),
            yaxis=dict(title="Maturity (years)"),
            zaxis=dict(title="Implied Volatility", tickformat=".0%"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=700,
    )

    fig3d.write_html("iv_surface_3d.html")
    print("Saved → iv_surface_3d.html")
    fig3d.show()

    # ── 2. Volatility Smile per Expiry ───────────────────────────────────────
    n_exp   = len(expiries)
    n_cols  = min(3, n_exp)
    n_rows  = (n_exp + n_cols - 1) // n_cols

    fig_smile = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Expiry {e}" for e in expiries],
        shared_yaxes=False,
    )

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    ]

    for idx, exp in enumerate(expiries):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        sub = df[df["expiry"] == exp].sort_values("strike")
        fig_smile.add_trace(
            go.Scatter(
                x=sub["moneyness"],
                y=sub["IV"],
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(color=colors[idx % len(colors)]),
                name=exp,
                showlegend=False,
                hovertemplate="K/S: %{x:.3f}<br>IV: %{y:.1%}<extra></extra>",
            ),
            row=row, col=col,
        )
        # Mark ATM
        atm_row = sub.iloc[(sub["moneyness"] - 1.0).abs().argsort()[:1]]
        fig_smile.add_trace(
            go.Scatter(
                x=atm_row["moneyness"],
                y=atm_row["IV"],
                mode="markers",
                marker=dict(size=10, color="red", symbol="star"),
                name="ATM",
                showlegend=(idx == 0),
                hovertemplate="ATM IV: %{y:.1%}<extra></extra>",
            ),
            row=row, col=col,
        )

    fig_smile.update_xaxes(title_text="Moneyness (K/S)")
    fig_smile.update_yaxes(title_text="IV", tickformat=".0%")
    fig_smile.update_layout(
        title="SPX Volatility Smile / Skew by Expiry",
        height=280 * n_rows,
        showlegend=True,
    )

    fig_smile.write_html("iv_smile_by_expiry.html")
    print("Saved → iv_smile_by_expiry.html")
    fig_smile.show()

    # ── 3. ATM Term Structure ────────────────────────────────────────────────
    atm_rows = []
    for exp in expiries:
        sub = df[df["expiry"] == exp]
        closest = sub.iloc[(sub["moneyness"] - 1.0).abs().argsort()[:1]]
        atm_rows.append({
            "expiry":   exp,
            "T":        closest["T"].values[0],
            "IV_atm":   closest["IV"].values[0],
            "moneyness": closest["moneyness"].values[0],
        })
    atm_df = pd.DataFrame(atm_rows).sort_values("T")

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=atm_df["T"],
        y=atm_df["IV_atm"],
        mode="lines+markers",
        marker=dict(size=8, color="royalblue"),
        line=dict(width=2.5, color="royalblue"),
        text=atm_df["expiry"],
        hovertemplate="Expiry: %{text}<br>T: %{x:.3f}y<br>ATM IV: %{y:.1%}<extra></extra>",
        name="ATM IV",
    ))

    fig_ts.update_layout(
        title="SPX ATM Implied Volatility — Term Structure",
        xaxis=dict(title="Maturity (years)"),
        yaxis=dict(title="ATM Implied Volatility", tickformat=".0%"),
        height=450,
    )

    fig_ts.write_html("iv_term_structure.html")
    print("Saved → iv_term_structure.html")
    fig_ts.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("  SPX Implied Volatility Surface Model")
    print("=" * 55)

    spot, df = fetch_options_data()

    if df.empty:
        print("No data to plot. Exiting.")
        sys.exit(1)

    print("\nBuilding plots …")
    plot_iv_surface(spot, df)

    print("\nDone! Three HTML files saved in the current directory:")
    print("  • iv_surface_3d.html      — interactive 3-D surface")
    print("  • iv_smile_by_expiry.html — volatility smile per expiry")
    print("  • iv_term_structure.html  — ATM term structure")


if __name__ == "__main__":
    main()
