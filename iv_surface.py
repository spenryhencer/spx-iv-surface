"""
Equity Implied Volatility Surface Model
========================================
Pulls SPX option quotes (Cboe delayed feed or yfinance), cleans them, and
computes implied volatility itself by inverting Black-76 on bid/ask mid prices:

  * risk-free rate   — US Treasury curve from FRED, interpolated per expiry
  * forward / carry  — implied from put-call parity (gives the dividend yield)
  * IV               — Brent root-finding on out-of-the-money mids

Then renders a single-page dashboard:
  1. Interactive 3-D implied volatility surface (strike x maturity x IV)
  2. Volatility smile / skew per expiry
  3. ATM term structure + per-expiry summary table

Usage:  python iv_surface.py [--source cboe|yfinance] [--help for all options]
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pricing import implied_vol, implied_forward, black76_delta
from providers import PROVIDERS, ET, expiry_datetime
from rates import fetch_treasury_curve

# ── Config (defaults; most can be overridden on the command line) ─────────────
MONEYNESS_LO    = 0.85     # keep strikes ≥ 85% of the forward
MONEYNESS_HI    = 1.15     # keep strikes ≤ 115% of the forward
MIN_IV          = 0.02     # drop IV below 2%
MAX_IV          = 1.50     # drop IV above 150%
MAX_REL_SPREAD  = 0.50     # drop quotes with (ask − bid) / mid above this
MIN_MID         = 0.05     # drop options with a mid price below $0.05
MIN_VOLUME      = 0        # optional: require traded volume
MIN_OPEN_INT    = 0        # optional: require open interest

# Target maturities in days — pick the nearest expiry to each target
TARGET_DAYS     = [14, 21, 30, 45, 60, 90, 120, 180, 270, 365]
SECONDS_PER_YEAR = 365.0 * 24 * 3600


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SPX implied volatility surface dashboard")
    p.add_argument("--source", choices=sorted(PROVIDERS), default="cboe",
                   help="options data source (default: cboe)")
    p.add_argument("--moneyness", nargs=2, type=float, metavar=("LO", "HI"),
                   default=[MONEYNESS_LO, MONEYNESS_HI], help="K/F range to keep")
    p.add_argument("--max-spread", type=float, default=MAX_REL_SPREAD,
                   help="max relative bid/ask spread (ask-bid)/mid")
    p.add_argument("--min-mid", type=float, default=MIN_MID, help="min mid price ($)")
    p.add_argument("--min-volume", type=float, default=MIN_VOLUME, help="min traded volume")
    p.add_argument("--min-oi", type=float, default=MIN_OPEN_INT, help="min open interest")
    p.add_argument("--smooth", type=int, default=0,
                   help="rolling-median window across strikes (0 = off)")
    p.add_argument("--out", default="iv_dashboard.html", help="dashboard HTML path")
    p.add_argument("--csv", help="also write the cleaned IV points to this CSV")
    p.add_argument("--png", metavar="DIR", help="also save chart PNGs here (needs kaleido)")
    p.add_argument("--no-open", action="store_true", help="don't open the browser")
    return p.parse_args(argv)


# ── Cleaning + IV computation ─────────────────────────────────────────────────
def build_iv_points(chain, curve, args):
    """Clean the quotes, imply forwards, invert Black-76 → (points, per-expiry summary)."""
    q = chain.quotes
    q = q.copy()
    q["valid"]      = (q["bid"] > 0) & (q["ask"] > q["bid"])   # two-sided, not crossed
    q["mid"]        = 0.5 * (q["bid"] + q["ask"])
    q["rel_spread"] = (q["ask"] - q["bid"]) / q["mid"]

    lo, hi = args.moneyness
    records, summary = [], []
    for (exp, root), g in q.groupby(["expiry", "root"]):
        T = (expiry_datetime(exp, root) - chain.as_of).total_seconds() / SECONDS_PER_YEAR
        if T <= 0:
            continue
        r, DF = curve.rate(T), curve.discount(T)

        # Forward from put-call parity on strikes with both sides quoted
        ok    = g[g["valid"]]
        calls = ok[ok["cp"] == "C"].set_index("strike")["mid"]
        puts  = ok[ok["cp"] == "P"].set_index("strike")["mid"]
        both  = calls.index.intersection(puts.index)
        F     = implied_forward(both, calls[both], puts[both], DF)
        if not np.isfinite(F):
            print(f"  ✗ {exp}: no call/put pairs to imply a forward")
            continue
        div_yield = r - np.log(F / chain.spot) / T

        # Out-of-the-money side only: puts below F, calls at/above F
        otm = g[((g["cp"] == "P") & (g["strike"] < F)) |
                ((g["cp"] == "C") & (g["strike"] >= F))].copy()
        otm["moneyness"] = otm["strike"] / F
        otm = otm[(otm["moneyness"] >= lo) & (otm["moneyness"] <= hi)]
        n_in_range = len(otm)
        otm = otm[otm["valid"] &
                  (otm["rel_spread"] <= args.max_spread) &
                  (otm["mid"] >= args.min_mid) &
                  (otm["volume"] >= args.min_volume) &
                  (otm["open_interest"] >= args.min_oi)]
        n_liquid = len(otm)

        # Failed inversions (price outside no-arbitrage bounds / no root) are
        # NaN; IVs at the edge of the plausible range are flagged too. Both are
        # counted and reported, never plotted.
        otm["IV"] = [implied_vol(m, F, K, T, DF, cp == "C")
                     for m, K, cp in zip(otm["mid"], otm["strike"], otm["cp"])]
        n_failed  = int(otm["IV"].isna().sum())
        otm = otm[otm["IV"].notna()]
        n_bounds  = int((~otm["IV"].between(MIN_IV, MAX_IV)).sum())
        otm = otm[otm["IV"].between(MIN_IV, MAX_IV)]
        if otm.empty:
            print(f"  ✗ {exp}: no quotes survived cleaning")
            continue
        otm["delta"] = [black76_delta(F, K, T, s, cp == "C")
                        for K, s, cp in zip(otm["strike"], otm["IV"], otm["cp"])]

        otm = otm.sort_values("moneyness")
        if args.smooth > 1:
            otm["IV"] = otm["IV"].rolling(args.smooth, center=True, min_periods=1).median()

        days = (exp - chain.as_of.date()).days
        otm["expiry"], otm["T"], otm["days"], otm["F"] = str(exp), T, days, F
        records.append(otm)
        summary.append({
            "expiry": str(exp), "root": root, "days": days, "T": T,
            "rate": r, "forward": F, "div_yield": div_yield,
            "IV_atm": float(np.interp(1.0, otm["moneyness"], otm["IV"])),
            "points": len(otm),
            "dropped_quote": n_in_range - n_liquid,
            "dropped_failed": n_failed,
            "dropped_bounds": n_bounds,
        })
        print(f"  ✓ {exp} {root:<4} ({days:>3}d  F={F:,.1f}  q={div_yield:5.2%}  +{len(otm)} pts, dropped {n_in_range - n_liquid} quote / {n_failed} solver / {n_bounds} bounds)")

    if not records:
        hint = ("\nYahoo only publishes bid/ask during US market hours (9:30am–4pm ET).\n"
                "Re-run then, or use the Cboe feed:  python iv_surface.py --source cboe"
                if "yfinance" in chain.source else
                "\nTry loosening the filters (e.g. --max-spread 1.0 --min-mid 0).")
        raise RuntimeError("No usable option quotes after cleaning." + hint)

    df = pd.concat(records, ignore_index=True)
    cols = ["expiry", "days", "T", "strike", "cp", "bid", "ask", "mid", "rel_spread",
            "volume", "open_interest", "F", "moneyness", "IV", "delta", "vendor_iv"]
    return df[cols], pd.DataFrame(summary).sort_values("T").reset_index(drop=True)


STALE_AFTER_MIN = 30   # quotes older than this are labelled "market closed / stale"


def market_status(as_of, now=None):
    """(is_stale, message) describing how fresh the quotes are."""
    now = now or datetime.now(ET)
    age_min = (now - as_of).total_seconds() / 60
    stamp = as_of.strftime("%a %d %b %Y, %H:%M ET")
    if age_min > STALE_AFTER_MIN:
        return True, (f"Market closed or quotes stale — showing the last available "
                      f"quotes from {stamp}. They will update after the next US session opens "
                      f"(9:30am–4:00pm ET, Mon–Fri).")
    return False, f"Live delayed quotes as of {stamp} (~15 min delay)."


def surface_grid(df, atm_df, n_m=80, n_t=40):
    """
    IV on a regular (K/F, T) grid: each smile is interpolated onto a common
    moneyness grid, then maturities are interpolated linearly in total
    variance w = σ²T (no-calendar-arbitrage friendly). NaN outside the
    strike range quoted for that expiry.
    """
    m_vals = np.linspace(df["moneyness"].min(), df["moneyness"].max(), n_m)
    Ts     = atm_df["T"].to_numpy()
    W      = []
    for exp, T in zip(atm_df["expiry"], Ts):
        sub = df[df["expiry"] == exp].sort_values("moneyness")
        m, iv = sub["moneyness"].to_numpy(), sub["IV"].to_numpy()
        inside = (m_vals >= m[0]) & (m_vals <= m[-1])
        W.append(np.where(inside, np.interp(m_vals, m, iv), np.nan) ** 2 * T)
    W = np.array(W)
    if len(Ts) == 1:
        return m_vals, Ts, np.sqrt(W / Ts[:, None])
    T_vals = np.linspace(Ts[0], Ts[-1], n_t)
    hi     = np.clip(np.searchsorted(Ts, T_vals), 1, len(Ts) - 1)
    wgt    = ((T_vals - Ts[hi - 1]) / (Ts[hi] - Ts[hi - 1]))[:, None]
    Wg     = W[hi - 1] * (1 - wgt) + W[hi] * wgt
    return m_vals, T_vals, np.sqrt(Wg / T_vals[:, None])


def risk_reversal_25d(sub):
    """IV(25Δ put) − IV(25Δ call) for one expiry, interpolated in delta."""
    puts  = sub[sub["cp"] == "P"].sort_values("delta")
    calls = sub[sub["cp"] == "C"].sort_values("delta")
    if puts.empty or calls.empty:
        return np.nan
    if not (puts["delta"].min() <= -0.25 <= puts["delta"].max() and
            calls["delta"].min() <= 0.25 <= calls["delta"].max()):
        return np.nan
    return float(np.interp(-0.25, puts["delta"], puts["IV"]) -
                 np.interp(0.25, calls["delta"], calls["IV"]))


# ── Figures + stats (shared by the HTML dashboard and the Streamlit app) ─────
def fmt_expiry(exp, days):
    return f"{datetime.strptime(exp, '%Y-%m-%d'):%d %b %Y} · {days} days"


def make_figures(df, atm_df, x_axis="moneyness"):
    """
    Return the three Plotly figures: 3-D surface, smile grid, term structure.
    x_axis: "moneyness" (strike as % of forward) or "strike" (SPX index level)
    for the smile panels.
    """
    by_strike = x_axis == "strike"
    axis_font = dict(size=13)
    # ── 1. 3-D surface ────────────────────────────────────────────────────────
    m_vals, T_vals, ZZ = surface_grid(df, atm_df)
    fig3d = go.Figure(go.Surface(
        x=m_vals, y=T_vals * 365, z=ZZ,
        colorscale="Viridis",
        colorbar=dict(title="Implied<br>vol", tickformat=".0%", x=1.0),
        hovertemplate=("Strike: %{x:.1%} of forward<br>Expiry: %{y:.0f} days"
                       "<br>Implied vol: %{z:.1%}<extra></extra>"),
    ))
    fig3d.update_layout(
        title=None,
        scene=dict(
            xaxis=dict(title=dict(text="Strike (% of forward)", font=axis_font), tickformat=".0%"),
            yaxis=dict(title=dict(text="Days to expiry", font=axis_font)),
            zaxis=dict(title=dict(text="Implied volatility", font=axis_font), tickformat=".0%"),
            camera=dict(eye=dict(x=1.8, y=-1.6, z=0.8)),
        ),
        height=620, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # ── 2. Smile grid ─────────────────────────────────────────────────────────
    n_exp      = len(atm_df)
    n_cols     = min(3, n_exp)
    n_rows     = (n_exp + n_cols - 1) // n_cols
    colors     = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78"]
    fig_smile  = make_subplots(
        rows=n_rows, cols=n_cols, vertical_spacing=0.32 / n_rows,
        subplot_titles=[fmt_expiry(e, d) for e, d in zip(atm_df["expiry"], atm_df["days"])],
    )
    for idx, row in atm_df.iterrows():
        sub = df[df["expiry"] == row["expiry"]].sort_values("moneyness")
        r, c = idx // n_cols + 1, idx % n_cols + 1
        side = np.where(sub["cp"] == "C", "call", "put")
        fig_smile.add_trace(go.Scatter(
            x=sub["strike"] if by_strike else sub["moneyness"], y=sub["IV"],
            mode="lines+markers", marker=dict(size=4),
            line=dict(color=colors[idx % len(colors)]),
            name=row["expiry"], showlegend=False,
            customdata=np.c_[sub["strike"], side, sub["bid"], sub["ask"], sub["mid"]],
            text=sub["moneyness"],
            hovertemplate=("Strike %{customdata[0]:,.0f} %{customdata[1]} (%{text:.1%} of forward)"
                           "<br>Bid %{customdata[2]:.2f} · Ask %{customdata[3]:.2f}"
                           " · Mid %{customdata[4]:.2f}"
                           "<br>Implied vol: %{y:.1%}<extra></extra>"),
        ), row=r, col=c)
        fig_smile.add_trace(go.Scatter(
            x=[row["forward"] if by_strike else 1.0], y=[row["IV_atm"]], mode="markers",
            marker=dict(size=9, color="red", symbol="star"),
            showlegend=False,
            hovertemplate=(f"At-the-money (forward {row['forward']:,.0f})"
                           "<br>Implied vol: %{y:.1%}<extra></extra>"),
        ), row=r, col=c)
        fig_smile.add_vline(x=row["forward"] if by_strike else 1.0, line=dict(color="#bbb", dash="dot", width=1), row=r, col=c)
    if by_strike:
        fig_smile.update_xaxes(title_text="Strike (SPX index level)", tickformat=",.0f",
                               title_font=axis_font)
    else:
        fig_smile.update_xaxes(title_text="Strike (% of forward)", tickformat=".0%",
                               title_font=axis_font)
    fig_smile.update_yaxes(title_text="Implied volatility", tickformat=".0%",
                           title_font=axis_font)
    fig_smile.update_layout(height=300 * n_rows, margin=dict(t=40, b=60),
                            paper_bgcolor="rgba(0,0,0,0)")

    # ── 3. Term structure ─────────────────────────────────────────────────────
    fig_ts = go.Figure(go.Scatter(
        x=atm_df["days"], y=atm_df["IV_atm"],
        mode="lines+markers",
        marker=dict(size=8, color="royalblue"),
        line=dict(width=2.5, color="royalblue"),
        text=atm_df["expiry"],
        hovertemplate="Expiry: %{text}<br>%{x} days<br>At-the-money implied vol: %{y:.1%}<extra></extra>",
        name="ATM IV",
    ))
    fig_ts.update_layout(
        xaxis=dict(title=dict(text="Days to expiry", font=axis_font)),
        yaxis=dict(title=dict(text="At-the-money implied volatility", font=axis_font),
                   tickformat=".1%"),
        height=380, margin=dict(t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return {"surface": fig3d, "smiles": fig_smile, "term_structure": fig_ts}


def compute_stats(df, atm_df):
    near_atm_iv  = atm_df.iloc[0]["IV_atm"]
    far_atm_iv   = atm_df.iloc[-1]["IV_atm"]
    rr_row       = atm_df.iloc[(atm_df["days"] - 30).abs().argsort().iloc[0]]
    skew_val     = risk_reversal_25d(df[df["expiry"] == rr_row["expiry"]])
    vend         = df["vendor_iv"].between(MIN_IV, MAX_IV)
    iv_gap       = (df.loc[vend, "IV"] - df.loc[vend, "vendor_iv"]).abs().median()
    return dict(
        near_atm_iv=near_atm_iv, far_atm_iv=far_atm_iv,
        ts_slope="upward (contango)" if far_atm_iv > near_atm_iv else "inverted (backwardation)",
        rr_row=rr_row, skew_val=skew_val,
        skew_txt=f"{skew_val:.1%}" if np.isfinite(skew_val) else "n/a",
        q_row=atm_df.iloc[(atm_df["days"] - 90).abs().argsort().iloc[0]],
        iv_gap_txt=f"{iv_gap * 100:.2f} vol pts" if vend.any() else "n/a",
        n_vendor=int(vend.sum()),
        dropped={k: int(atm_df[f"dropped_{k}"].sum()) for k in ("quote", "failed", "bounds")},
    )


# ── Dashboard builder ─────────────────────────────────────────────────────────
def build_dashboard(chain, curve, df, atm_df, args):
    from plotly.io import to_html
    spot     = chain.spot
    n_exp    = len(atm_df)
    as_of    = chain.as_of.strftime("%Y-%m-%d %H:%M")
    stale, status_msg = market_status(chain.as_of)

    figs       = make_figures(df, atm_df)
    html_3d    = to_html(figs["surface"], full_html=False, include_plotlyjs=True)
    html_smile = to_html(figs["smiles"], full_html=False, include_plotlyjs=False)  # reuses embedded js
    html_ts    = to_html(figs["term_structure"], full_html=False, include_plotlyjs=False)

    if args.png:
        os.makedirs(args.png, exist_ok=True)
        sizes = {"surface": (1100, 700), "smiles": (1200, figs["smiles"].layout.height),
                 "term_structure": (1000, 420)}
        for name, fig in figs.items():
            w, h = sizes[name]
            fig = go.Figure(fig).update_layout(paper_bgcolor="white")
            fig.write_image(os.path.join(args.png, f"{name}.png"), width=w, height=h, scale=2)
        print(f"Saved PNGs → {args.png}/")

    # ── Stats for summary cards ───────────────────────────────────────────────
    st           = compute_stats(df, atm_df)
    near_atm_iv, far_atm_iv, ts_slope = st["near_atm_iv"], st["far_atm_iv"], st["ts_slope"]
    rr_row, skew_val, skew_txt = st["rr_row"], st["skew_val"], st["skew_txt"]
    q_row, iv_gap_txt, dropped = st["q_row"], st["iv_gap_txt"], st["dropped"]
    print(f"Median |our IV − vendor IV| = {iv_gap_txt}  ({st['n_vendor']} points)")

    summary_rows = "\n".join(
        f"<tr><td>{r.expiry}</td><td>{r.root}</td><td>{r.days}</td><td>{r.rate:.2%}</td>"
        f"<td>{r.forward:,.1f}</td><td>{r.div_yield:.2%}</td><td>{r.IV_atm:.1%}</td>"
        f"<td>{r.points}</td><td>{r.dropped_quote} / {r.dropped_failed} / {r.dropped_bounds}</td></tr>"
        for r in atm_df.itertuples())

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SPX Implied Volatility Surface — Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f4f6f9; color: #1a1a2e; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
             color: white; padding: 32px 40px 24px; }}
  .header h1 {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.5px; }}
  .header p  {{ margin-top: 6px; opacity: 0.75; font-size: 0.9rem; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 28px 32px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }}
  .card {{ background: white; border-radius: 12px; padding: 20px 24px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
  .card .label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px;
                  color: #888; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }}
  .card .sub   {{ font-size: 0.8rem; color: #aaa; margin-top: 4px; }}
  .section {{ background: white; border-radius: 12px; padding: 24px 28px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 24px; }}
  .section h2 {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 4px; color: #1a1a2e; }}
  .section .subtitle {{ font-size: 0.85rem; color: #888; margin-bottom: 18px; }}
  .method-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .method-box {{ background: #f8f9fc; border-radius: 8px; padding: 18px 20px; }}
  .method-box h3 {{ font-size: 0.85rem; font-weight: 600; color: #0f3460;
                    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }}
  .method-box p, .method-box li {{ font-size: 0.84rem; line-height: 1.65; color: #444; }}
  .method-box ul {{ padding-left: 18px; }}
  .method-box li {{ margin-bottom: 4px; }}
  code {{ background: #eef2ff; color: #3730a3; padding: 2px 6px;
          border-radius: 4px; font-size: 0.82rem; }}
  .interp-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 4px; }}
  .interp-box {{ background: #f8f9fc; border-radius: 8px; padding: 16px 18px; }}
  .interp-box h3 {{ font-size: 0.82rem; font-weight: 700; color: #0f3460;
                    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
  .interp-box p {{ font-size: 0.83rem; line-height: 1.6; color: #444; }}
  .tag {{ display: inline-block; background: #e0e7ff; color: #3730a3;
          font-size: 0.72rem; font-weight: 600; padding: 2px 8px;
          border-radius: 99px; margin-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
  th, td {{ padding: 7px 10px; text-align: right; border-bottom: 1px solid #eee; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ color: #888; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.5px; }}
  hr {{ border: none; border-top: 1px solid #eee; margin: 0; }}
  .status {{ border-radius: 10px; padding: 12px 18px; margin-bottom: 20px; font-size: 0.88rem; }}
  .status.stale {{ background: #fff4e5; color: #8a4b00; border: 1px solid #ffd8a8; }}
  .status.live  {{ background: #e8f7ee; color: #1d6b3a; border: 1px solid #b7e4c7; }}
  @media (max-width: 900px) {{
    .cards, .interp-grid {{ grid-template-columns: 1fr 1fr; }}
    .method-grid {{ grid-template-columns: 1fr; }}
    .container {{ padding: 16px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>SPX Implied Volatility Surface</h1>
  <p>Source: {chain.source} &nbsp;·&nbsp; Quotes as of {as_of} ET &nbsp;·&nbsp;
     Spot: <strong>{spot:,.2f}</strong> &nbsp;·&nbsp;
     {n_exp} expiries &nbsp;·&nbsp; {len(df):,} data points</p>
</div>

<div class="container">

  <div class="status {"stale" if stale else "live"}"><strong>{"⚠ Market closed / stale data" if stale else "● Live"}</strong> — {status_msg}</div>

  <!-- Summary cards -->
  <div class="cards">
    <div class="card">
      <div class="label">Spot (SPX)</div>
      <div class="value">{spot:,.0f}</div>
      <div class="sub">S&amp;P 500 Index</div>
    </div>
    <div class="card">
      <div class="label">Near-term ATM IV</div>
      <div class="value">{near_atm_iv:.1%}</div>
      <div class="sub">{atm_df.iloc[0]['days']}d expiry</div>
    </div>
    <div class="card">
      <div class="label">Long-dated ATM IV</div>
      <div class="value">{far_atm_iv:.1%}</div>
      <div class="sub">{atm_df.iloc[-1]['days']}d expiry</div>
    </div>
    <div class="card">
      <div class="label">Term Structure</div>
      <div class="value" style="font-size:1.1rem;padding-top:6px">{ts_slope.split()[0].title()}</div>
      <div class="sub">{ts_slope}</div>
    </div>
    <div class="card">
      <div class="label">25Δ Risk Reversal</div>
      <div class="value">{skew_txt}</div>
      <div class="sub">put − call IV, {rr_row['days']}d expiry</div>
    </div>
    <div class="card">
      <div class="label">Risk-free rate</div>
      <div class="value">{curve.rate(q_row['T']):.2%}</div>
      <div class="sub">{q_row['days']}d · {curve.source} ({curve.asof})</div>
    </div>
    <div class="card">
      <div class="label">Implied dividend yield</div>
      <div class="value">{q_row['div_yield']:.2%}</div>
      <div class="sub">{q_row['days']}d · from put–call parity</div>
    </div>
    <div class="card">
      <div class="label">IV check vs vendor</div>
      <div class="value" style="font-size:1.1rem;padding-top:6px">{iv_gap_txt}</div>
      <div class="sub">median |our IV − source's IV|</div>
    </div>
  </div>

  <!-- Methodology -->
  <div class="section">
    <h2>Methodology &amp; Calculations</h2>
    <div class="subtitle">How the implied volatility surface is constructed from raw options data</div>
    <div class="method-grid">
      <div class="method-box">
        <h3>Data Pipeline</h3>
        <ul>
          <li><strong>Source:</strong> {chain.source} (<code>--source {args.source}</code>)</li>
          <li><strong>Expiry selection:</strong> 10 target maturities from 14 to 365 days, picking the nearest available expiry to each; PM-settled SPXW series are used where both SPX and SPXW exist</li>
          <li><strong>Quote filter:</strong> two-sided quotes only (<code>bid &gt; 0</code>, <code>ask &gt; bid</code>), mid ≥ <code>${args.min_mid:.2f}</code>, relative spread ≤ <code>{args.max_spread:.0%}</code>{f", volume ≥ <code>{args.min_volume:g}</code>" if args.min_volume else ""}{f", open interest ≥ <code>{args.min_oi:g}</code>" if args.min_oi else ""}</li>
          <li><strong>Price:</strong> mid = (bid + ask) / 2 — not last trade, which can be hours or days stale; zero-bid and crossed (bid ≥ ask) quotes are dropped</li>
          <li><strong>Failed inversions:</strong> counted and excluded, never plotted or clamped (see the per-expiry table)</li>
          <li><strong>OTM only:</strong> puts for K &lt; F, calls for K ≥ F — the liquid side of each strike</li>
          <li><strong>Strike filter:</strong> moneyness K/F <code>{args.moneyness[0]:.0%} – {args.moneyness[1]:.0%}</code></li>
          <li><strong>Smoothing:</strong> {f"rolling median over {args.smooth} strikes" if args.smooth > 1 else "none — every point is a market quote"}</li>
        </ul>
      </div>
      <div class="method-box">
        <h3>Implied Volatility</h3>
        <p>For each expiry, the <strong>rate</strong> r comes from the FRED Treasury curve interpolated to that maturity, and the <strong>forward</strong> from put–call parity at the strikes nearest the money:</p>
        <p style="text-align:center; font-style:italic; font-size:0.9rem; margin:8px 0">
          F = K + e<sup>rT</sup>(C<sub>mid</sub> − P<sub>mid</sub>), &nbsp; q = r − ln(F/S)/T
        </p>
        <p>The <strong>implied volatility</strong> σ* solves the Black-76 equation on that forward, found with Brent's method:</p>
        <p style="text-align:center; font-style:italic; font-size:0.9rem; margin:8px 0">
          e<sup>−rT</sup>[F·N(d<sub>1</sub>) − K·N(d<sub>2</sub>)] = C<sub>mid</sub>
        </p>
        <p>Using the market-implied forward bakes in dividends without needing a dividend forecast. As a sanity check, our σ* is compared with the source's own IV field (card above).</p>
        <p style="margin-top:8px"><strong>Surface interpolation:</strong> each smile is interpolated onto a common K/F grid, then maturities are interpolated linearly in total variance σ²T — an 80×40 grid with no smoothing of the market points.</p>
      </div>
    </div>
  </div>

  <!-- 3D Surface -->
  <div class="section">
    <h2>Implied Volatility Surface (3-D)</h2>
    <div class="subtitle">Drag to rotate · Scroll to zoom · Hover for exact values</div>
    {html_3d}
    <hr style="margin: 20px 0 16px">
    <div class="interp-grid">
      <div class="interp-box">
        <div class="tag">What you see</div>
        <h3>Negative Skew</h3>
        <p>IV is higher on the left (low strikes = OTM puts) than on the right (OTM calls). This is the classic SPX <em>put skew</em> — investors pay a premium for downside protection, bidding up OTM put prices and therefore their implied volatility.</p>
      </div>
      <div class="interp-box">
        <div class="tag">What you see</div>
        <h3>Term Structure</h3>
        <p>IV generally rises moving towards the back of the surface (longer maturities). This reflects the <em>volatility risk premium</em> — uncertainty compounds over time, so longer-dated options carry more implied volatility.</p>
      </div>
      <div class="interp-box">
        <div class="tag">Current reading</div>
        <h3>Surface Shape</h3>
        <p>Near-term ATM IV is <strong>{near_atm_iv:.1%}</strong>, {"rising" if far_atm_iv > near_atm_iv else "falling"} to <strong>{far_atm_iv:.1%}</strong> at the long end — a {ts_slope} term structure. The {rr_row['days']}d 25Δ risk reversal of <strong>{skew_txt}</strong> (put IV minus call IV) indicates {"elevated hedging demand" if np.isfinite(skew_val) and skew_val > 0.05 else "moderate hedging demand"}.</p>
      </div>
    </div>
  </div>

  <!-- Smile grid -->
  <div class="section">
    <h2>Volatility Smile by Expiry</h2>
    <div class="subtitle">Each panel is one expiry. X-axis: strike as a % of that expiry's forward price (100% = at the money, dotted line). Y-axis: implied volatility from the bid/ask mid. Red star = at-the-money IV. Hover a point for its strike, bid, ask and mid.</div>
    {html_smile}
    <hr style="margin: 20px 0 16px">
    <div class="interp-grid">
      <div class="interp-box">
        <div class="tag">How to read</div>
        <h3>The Skew / Smile</h3>
        <p>A downward-sloping curve (left higher than right) is a <em>skew</em> — typical for equity indices where put demand exceeds call demand. A U-shaped curve is a true <em>smile</em>, more common in FX and short-dated equity options.</p>
      </div>
      <div class="interp-box">
        <div class="tag">How to read</div>
        <h3>ATM vs Wings</h3>
        <p>The red star marks the at-the-money forward. Points to the left are OTM puts (K &lt; F); points to the right are OTM calls (K &gt; F). The steeper the slope from right to left, the more the market fears a sharp sell-off.</p>
      </div>
      <div class="interp-box">
        <div class="tag">How to read</div>
        <h3>Skew across expiries</h3>
        <p>Short-dated smiles are usually steeper in K/F terms because a given % move is a bigger shock over a short horizon; longer-dated smiles look flatter, though the skew per unit of √T is often similar.</p>
      </div>
    </div>
  </div>

  <!-- Term structure -->
  <div class="section">
    <h2>ATM Term Structure</h2>
    <div class="subtitle">ATM-forward implied volatility at each expiry — shows how the market prices uncertainty over different horizons</div>
    {html_ts}
    <hr style="margin: 20px 0 16px">
    <div class="interp-grid">
      <div class="interp-box">
        <div class="tag">How to read</div>
        <h3>Contango (upward)</h3>
        <p>When long-dated IV &gt; short-dated IV the curve is in <em>contango</em>. This is the normal regime — markets expect uncertainty to grow over time and price longer options at a premium.</p>
      </div>
      <div class="interp-box">
        <div class="tag">How to read</div>
        <h3>Backwardation (inverted)</h3>
        <p>When short-dated IV &gt; long-dated IV the curve is <em>inverted</em>. This signals elevated near-term fear — often seen around earnings, Fed meetings, or crisis periods — and usually reverts quickly.</p>
      </div>
      <div class="interp-box">
        <div class="tag">Current reading</div>
        <h3>Today's curve: {ts_slope.split("(")[0].strip().title()}</h3>
        <p>ATM IV moves from <strong>{near_atm_iv:.1%}</strong> ({atm_df.iloc[0]['days']}d) to <strong>{far_atm_iv:.1%}</strong> ({atm_df.iloc[-1]['days']}d). The {ts_slope} shape suggests {"no immediate event risk priced in — the market is calm and forward-looking" if far_atm_iv > near_atm_iv else "elevated near-term event risk — the market is pricing in an imminent catalyst"}.</p>
      </div>
    </div>
  </div>

  <!-- Expiry summary -->
  <div class="section">
    <h2>Per-Expiry Inputs</h2>
    <div class="subtitle">Rate, implied forward and dividend yield used for each expiry, with the resulting ATM IV.
      Dropped points are never plotted: <em>quote</em> = no bid, crossed/wide bid–ask or below the liquidity floors;
      <em>solver</em> = mid price outside no-arbitrage bounds so no implied vol exists;
      <em>bounds</em> = implied vol outside {MIN_IV:.0%}–{MAX_IV:.0%}.
      Totals: {dropped["quote"]} / {dropped["failed"]} / {dropped["bounds"]}.</div>
    <table>
      <tr><th>Expiry</th><th>Root</th><th>Days</th><th>Rate r</th><th>Forward F</th><th>Div. yield q</th><th>ATM IV</th><th>Points</th><th>Dropped: quote / solver / bounds</th></tr>
      {summary_rows}
    </table>
  </div>

</div>
</body>
</html>"""

    with open(args.out, "w") as f:
        f.write(html)
    print(f"Saved → {args.out}")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(args.out))


# ── Main ──────────────────────────────────────────────────────────────────────
def load_surface(args):
    """Fetch quotes + rates and compute the IV points: (chain, curve, df, atm_df)."""
    print(f"Fetching SPX options from {args.source} …")
    chain = PROVIDERS[args.source](TARGET_DAYS)
    print(f"  Spot = {chain.spot:,.2f}   quotes as of {chain.as_of:%Y-%m-%d %H:%M} ET")
    curve = fetch_treasury_curve(chain.as_of.date())
    print(f"  Rates: {curve.source} as of {curve.asof}")
    print("Computing implied vols:")
    df, atm_df = build_iv_points(chain, curve, args)
    return chain, curve, df, atm_df


def main(argv=None):
    args = parse_args(argv)
    print("=" * 55)
    print("  SPX Implied Volatility Surface Model")
    print("=" * 55)
    try:
        chain, curve, df, atm_df = load_surface(args)
    except RuntimeError as e:
        sys.exit(f"\n✗ {e}")
    print(f"\nTotal points: {len(df)}  across {len(atm_df)} expiries")
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved points → {args.csv}")

    print("\nBuilding dashboard …")
    build_dashboard(chain, curve, df, atm_df, args)
    print(f"\nDone! Open {args.out} in your browser.")


if __name__ == "__main__":
    main()
