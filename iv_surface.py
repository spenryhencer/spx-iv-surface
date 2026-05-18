"""
Equity Implied Volatility Surface Model
========================================
Fetches live SPX options chain data via yfinance and plots:
  1. Interactive 3-D implied volatility surface (strike x maturity x IV)
  2. Volatility smile / skew per expiry
  3. ATM term structure

Usage:  python3 iv_surface.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from scipy.interpolate import griddata

# ── Config ────────────────────────────────────────────────────────────────────
TICKER          = "^SPX"
MONEYNESS_LO    = 0.85     # keep strikes ≥ 85% of spot
MONEYNESS_HI    = 1.15     # keep strikes ≤ 115% of spot
MIN_IV          = 0.02     # drop IV below 2%
MAX_IV          = 1.50     # drop IV above 150%
MIN_VOLUME      = 5        # require meaningful trading activity

# Target maturities in days — pick the nearest expiry to each target
TARGET_DAYS     = [14, 21, 30, 45, 60, 90, 120, 180, 270, 365]


# ── Data fetch ────────────────────────────────────────────────────────────────
def pick_expiries(all_expiries, today):
    """Pick the closest available expiry to each target day count."""
    day_map = {}
    for exp_str in all_expiries:
        d = (datetime.strptime(exp_str, "%Y-%m-%d").date() - today).days
        day_map[exp_str] = d

    chosen = {}
    for target in TARGET_DAYS:
        best = min(day_map, key=lambda e: abs(day_map[e] - target)
                   if day_map[e] >= 7 else 9999)
        if day_map.get(best, 0) >= 7:
            chosen[best] = day_map[best]

    # deduplicate, keep unique expiry strings sorted by days
    return sorted(set(chosen.keys()), key=lambda e: chosen[e])


def fetch_data():
    print(f"Fetching {TICKER} …")
    tkr  = yf.Ticker(TICKER)
    hist = tkr.history(period="5d")
    if hist.empty:
        raise RuntimeError("Could not get price data — are you online?")
    spot = float(hist["Close"].dropna().iloc[-1])
    print(f"  Spot = {spot:,.2f}")

    all_expiries = tkr.options
    if not all_expiries:
        raise RuntimeError("No options data returned.")

    today    = date.today()
    expiries = pick_expiries(all_expiries, today)
    print(f"Selected {len(expiries)} expiries spread across term structure:")

    records = []
    for exp_str in expiries:
        days = (datetime.strptime(exp_str, "%Y-%m-%d").date() - today).days
        T    = days / 365.0
        try:
            chain = tkr.option_chain(exp_str)
        except Exception as e:
            print(f"  ✗ {exp_str}: {e}")
            continue

        n_before = len(records)
        for df_opts in [chain.calls, chain.puts]:
            df_opts = df_opts.copy()

            # Require actual trading volume — filters stale/phantom quotes
            df_opts = df_opts[df_opts["volume"].fillna(0) >= MIN_VOLUME]

            # Near-the-money only
            df_opts = df_opts[
                (df_opts["strike"] >= spot * MONEYNESS_LO) &
                (df_opts["strike"] <= spot * MONEYNESS_HI)
            ]

            # Valid IV from yfinance
            df_opts = df_opts[df_opts["impliedVolatility"].notna()]
            df_opts = df_opts[
                (df_opts["impliedVolatility"] >= MIN_IV) &
                (df_opts["impliedVolatility"] <= MAX_IV)
            ]

            for _, row in df_opts.iterrows():
                records.append({
                    "strike":    row["strike"],
                    "expiry":    exp_str,
                    "T":         T,
                    "days":      days,
                    "IV":        row["impliedVolatility"],
                    "moneyness": row["strike"] / spot,
                })

        print(f"  ✓ {exp_str}  ({days}d  +{len(records)-n_before} pts)")

    if not records:
        raise RuntimeError(
            "No data found.\n"
            "SPX options data is best during US market hours (9:30am–4pm ET).\n"
            "Outside those hours, volume=0 on most strikes. Try again then,\n"
            "or set MIN_VOLUME=0 in the config to use all available quotes."
        )

    df = pd.DataFrame(records)
    df = (df.groupby(["strike", "expiry", "T", "days", "moneyness"],
                     as_index=False)
            .agg(IV=("IV", "mean")))
    df.sort_values(["T", "strike"], inplace=True)

    # Smooth IV per expiry with a rolling median to remove isolated spikes
    smoothed = []
    for _, grp in df.groupby("expiry"):
        grp = grp.sort_values("moneyness").copy()
        grp["IV"] = grp["IV"].rolling(window=5, center=True, min_periods=1).median()
        smoothed.append(grp)
    df = pd.concat(smoothed).reset_index(drop=True)

    print(f"\nTotal points: {len(df)}  across {df['expiry'].nunique()} expiries")
    return spot, df


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_all(spot, df):
    expiries = sorted(df["expiry"].unique(), key=lambda e: df.loc[df["expiry"]==e,"T"].iloc[0])

    # ── 1. 3-D surface ────────────────────────────────────────────────────────
    # Interpolate onto a regular grid for a clean surface
    K_vals  = np.linspace(df["strike"].min(), df["strike"].max(), 80)
    T_vals  = np.linspace(df["T"].min(),      df["T"].max(),      40)
    KK, TT  = np.meshgrid(K_vals, T_vals)
    ZZ      = griddata(
        (df["strike"].values, df["T"].values),
        df["IV"].values,
        (KK, TT),
        method="linear"
    )

    fig3d = go.Figure(go.Surface(
        x=K_vals, y=T_vals, z=ZZ,
        colorscale="Viridis",
        colorbar=dict(title="IV", tickformat=".0%"),
        hovertemplate="Strike: %{x:,.0f}<br>Maturity: %{y:.3f}y<br>IV: %{z:.1%}<extra></extra>",
    ))
    fig3d.update_layout(
        title="SPX Implied Volatility Surface",
        scene=dict(
            xaxis=dict(title="Strike"),
            yaxis=dict(title="Maturity (years)"),
            zaxis=dict(title="Implied Volatility", tickformat=".0%"),
            camera=dict(eye=dict(x=1.8, y=-1.6, z=0.8)),
        ),
        height=700, margin=dict(l=0, r=0, t=60, b=0),
    )
    fig3d.write_html("iv_surface_3d.html")
    print("Saved → iv_surface_3d.html")
    fig3d.show()

    # ── 2. Smile per expiry ───────────────────────────────────────────────────
    n_cols = min(3, len(expiries))
    n_rows = (len(expiries) + n_cols - 1) // n_cols
    fig_smile = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{e} ({df.loc[df['expiry']==e,'days'].iloc[0]}d)" for e in expiries],
    )
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
              "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78"]

    for idx, exp in enumerate(expiries):
        sub = df[df["expiry"] == exp].sort_values("moneyness")
        r, c = idx // n_cols + 1, idx % n_cols + 1
        fig_smile.add_trace(go.Scatter(
            x=sub["moneyness"], y=sub["IV"],
            mode="lines+markers", marker=dict(size=5),
            line=dict(color=colors[idx % len(colors)]),
            name=exp, showlegend=False,
            hovertemplate="K/S: %{x:.3f}<br>IV: %{y:.1%}<extra></extra>",
        ), row=r, col=c)
        # ATM marker
        atm = sub.iloc[(sub["moneyness"] - 1.0).abs().argsort()[:1]]
        fig_smile.add_trace(go.Scatter(
            x=atm["moneyness"], y=atm["IV"],
            mode="markers", marker=dict(size=10, color="red", symbol="star"),
            showlegend=(idx == 0), name="ATM",
            hovertemplate="ATM IV: %{y:.1%}<extra></extra>",
        ), row=r, col=c)

    fig_smile.update_xaxes(title_text="Moneyness (K/S)")
    fig_smile.update_yaxes(title_text="IV", tickformat=".0%")
    fig_smile.update_layout(
        title="SPX Volatility Smile by Expiry",
        height=300 * n_rows, showlegend=True,
    )
    fig_smile.write_html("iv_smile_by_expiry.html")
    print("Saved → iv_smile_by_expiry.html")
    fig_smile.show()

    # ── 3. ATM term structure ─────────────────────────────────────────────────
    atm_rows = []
    for exp in expiries:
        sub = df[df["expiry"] == exp]
        closest = sub.iloc[(sub["moneyness"] - 1.0).abs().argsort()[:1]]
        atm_rows.append({
            "expiry": exp,
            "days":   int(closest["days"].values[0]),
            "IV_atm": float(closest["IV"].values[0]),
        })
    atm_df = pd.DataFrame(atm_rows).sort_values("days")

    fig_ts = go.Figure(go.Scatter(
        x=atm_df["days"], y=atm_df["IV_atm"],
        mode="lines+markers",
        marker=dict(size=8, color="royalblue"),
        line=dict(width=2.5, color="royalblue"),
        text=atm_df["expiry"],
        hovertemplate="Expiry: %{text}<br>Days: %{x}d<br>ATM IV: %{y:.1%}<extra></extra>",
        name="ATM IV",
    ))
    fig_ts.update_layout(
        title="SPX ATM Implied Volatility — Term Structure",
        xaxis=dict(title="Days to Expiry"),
        yaxis=dict(title="ATM Implied Volatility", tickformat=".0%"),
        height=450,
    )
    fig_ts.write_html("iv_term_structure.html")
    print("Saved → iv_term_structure.html")
    fig_ts.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  SPX Implied Volatility Surface Model")
    print("=" * 55)
    spot, df = fetch_data()
    print("\nBuilding plots …")
    plot_all(spot, df)
    print("\nDone! Open these files in your browser:")
    print("  • iv_surface_3d.html")
    print("  • iv_smile_by_expiry.html")
    print("  • iv_term_structure.html")

if __name__ == "__main__":
    main()
