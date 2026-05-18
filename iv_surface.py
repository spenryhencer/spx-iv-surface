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


# ── Dashboard builder ─────────────────────────────────────────────────────────
def build_dashboard(spot, df):
    from plotly.io import to_html
    expiries = sorted(df["expiry"].unique(), key=lambda e: df.loc[df["expiry"]==e,"T"].iloc[0])
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_exp    = len(expiries)
    atm_ivs  = []

    # ── ATM summary per expiry ────────────────────────────────────────────────
    atm_rows = []
    for exp in expiries:
        sub     = df[df["expiry"] == exp]
        closest = sub.iloc[(sub["moneyness"] - 1.0).abs().argsort()[:1]]
        atm_rows.append({
            "expiry": exp,
            "days":   int(closest["days"].values[0]),
            "IV_atm": float(closest["IV"].values[0]),
        })
    atm_df = pd.DataFrame(atm_rows).sort_values("days")

    # ── 1. 3-D surface ────────────────────────────────────────────────────────
    K_vals = np.linspace(df["strike"].min(), df["strike"].max(), 80)
    T_vals = np.linspace(df["T"].min(),      df["T"].max(),      40)
    KK, TT = np.meshgrid(K_vals, T_vals)
    ZZ     = griddata((df["strike"].values, df["T"].values),
                      df["IV"].values, (KK, TT), method="linear")

    fig3d = go.Figure(go.Surface(
        x=K_vals, y=T_vals, z=ZZ,
        colorscale="Viridis",
        colorbar=dict(title="IV", tickformat=".0%", x=1.0),
        hovertemplate="Strike: %{x:,.0f}<br>Maturity: %{y:.2f}y<br>IV: %{z:.1%}<extra></extra>",
    ))
    fig3d.update_layout(
        title=None,
        scene=dict(
            xaxis=dict(title="Strike"),
            yaxis=dict(title="Maturity (years)"),
            zaxis=dict(title="Implied Volatility", tickformat=".0%"),
            camera=dict(eye=dict(x=1.8, y=-1.6, z=0.8)),
        ),
        height=620, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    plotlyjs = to_html(fig3d, full_html=False, include_plotlyjs=True)
    # Split out just the <script src> tag plotly injects and the div
    html_3d = plotlyjs

    # ── 2. Smile grid ─────────────────────────────────────────────────────────
    n_cols     = min(3, n_exp)
    n_rows     = (n_exp + n_cols - 1) // n_cols
    colors     = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78"]
    fig_smile  = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{e} ({df.loc[df['expiry']==e,'days'].iloc[0]}d)" for e in expiries],
    )
    for idx, exp in enumerate(expiries):
        sub = df[df["expiry"] == exp].sort_values("moneyness")
        r, c = idx // n_cols + 1, idx % n_cols + 1
        fig_smile.add_trace(go.Scatter(
            x=sub["moneyness"], y=sub["IV"],
            mode="lines+markers", marker=dict(size=4),
            line=dict(color=colors[idx % len(colors)]),
            name=exp, showlegend=False,
            hovertemplate="K/S: %{x:.3f}<br>IV: %{y:.1%}<extra></extra>",
        ), row=r, col=c)
        atm = sub.iloc[(sub["moneyness"]-1.0).abs().argsort()[:1]]
        fig_smile.add_trace(go.Scatter(
            x=atm["moneyness"], y=atm["IV"], mode="markers",
            marker=dict(size=9, color="red", symbol="star"),
            showlegend=False,
            hovertemplate="ATM IV: %{y:.1%}<extra></extra>",
        ), row=r, col=c)
    fig_smile.update_xaxes(title_text="Moneyness (K/S)")
    fig_smile.update_yaxes(title_text="IV", tickformat=".0%")
    fig_smile.update_layout(height=290 * n_rows, margin=dict(t=40, b=20),
                            paper_bgcolor="rgba(0,0,0,0)")
    html_smile = to_html(fig_smile, full_html=False, include_plotlyjs=False)  # reuses embedded js

    # ── 3. Term structure ─────────────────────────────────────────────────────
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
        xaxis=dict(title="Days to Expiry"),
        yaxis=dict(title="ATM Implied Volatility", tickformat=".0%"),
        height=380, margin=dict(t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    html_ts = to_html(fig_ts, full_html=False, include_plotlyjs=False)  # reuses embedded js

    # ── Stats for summary card ────────────────────────────────────────────────
    near_atm_iv  = atm_df.iloc[0]["IV_atm"]
    far_atm_iv   = atm_df.iloc[-1]["IV_atm"]
    ts_slope     = "upward (contango)" if far_atm_iv > near_atm_iv else "inverted (backwardation)"
    skew_exp     = expiries[len(expiries)//2]
    skew_sub     = df[df["expiry"]==skew_exp]
    otm_put_iv   = skew_sub[skew_sub["moneyness"] <= 0.95]["IV"].mean()
    otm_call_iv  = skew_sub[skew_sub["moneyness"] >= 1.05]["IV"].mean()
    skew_val     = otm_put_iv - otm_call_iv if not np.isnan(otm_put_iv) and not np.isnan(otm_call_iv) else 0

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
  hr {{ border: none; border-top: 1px solid #eee; margin: 0; }}
</style>
</head>
<body>

<div class="header">
  <h1>SPX Implied Volatility Surface</h1>
  <p>Live options chain analysis &nbsp;·&nbsp; Generated {run_time} ET &nbsp;·&nbsp;
     Spot: <strong>{spot:,.2f}</strong> &nbsp;·&nbsp;
     {n_exp} expiries &nbsp;·&nbsp; {len(df):,} data points</p>
</div>

<div class="container">

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
  </div>

  <!-- Methodology -->
  <div class="section">
    <h2>Methodology &amp; Calculations</h2>
    <div class="subtitle">How the implied volatility surface is constructed from raw options data</div>
    <div class="method-grid">
      <div class="method-box">
        <h3>Data Pipeline</h3>
        <ul>
          <li><strong>Source:</strong> Live SPX options chain via <code>yfinance</code></li>
          <li><strong>Expiry selection:</strong> 10 target maturities spread from 14 to 365 days, picking the nearest available expiry to each target so the term structure spans the full curve</li>
          <li><strong>Liquidity filter:</strong> Only options with <code>volume ≥ {MIN_VOLUME}</code> are used, removing stale/phantom quotes</li>
          <li><strong>Strike filter:</strong> Moneyness <code>{MONEYNESS_LO:.0%} – {MONEYNESS_HI:.0%}</code> of spot (±15%), focusing on the traded range</li>
          <li><strong>Call/put merge:</strong> Where both sides exist at the same strike, IV is averaged — put–call parity means they should be equal in theory</li>
          <li><strong>Smoothing:</strong> Rolling median (window = 5 strikes) per expiry removes isolated bad ticks</li>
        </ul>
      </div>
      <div class="method-box">
        <h3>Implied Volatility</h3>
        <p>The <strong>implied volatility (IV)</strong> σ* is the value that makes the Black–Scholes model price equal to the observed market price:</p>
        <br>
        <p style="text-align:center; font-style:italic; font-size:0.9rem">
          C<sub>BS</sub>(S, K, T, r, σ*) = C<sub>market</sub>
        </p>
        <br>
        <p>Rather than inverting this numerically, <code>yfinance</code> returns pre-computed IV directly from the exchange feed, which is faster and avoids numerical instability near expiry.</p>
        <br>
        <p><strong>Surface interpolation:</strong> Raw IV points (discrete strikes × expiries) are interpolated onto an 80×40 regular grid using <code>scipy.interpolate.griddata</code> (linear method) to produce the smooth 3-D surface.</p>
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
        <p>Near-term ATM IV is <strong>{near_atm_iv:.1%}</strong>, rising to <strong>{far_atm_iv:.1%}</strong> at the long end — a {ts_slope} term structure. The put skew of ~<strong>{skew_val:.1%}</strong> (25Δ put vs call) indicates {"elevated hedging demand" if skew_val > 0.03 else "moderate hedging demand"}.</p>
      </div>
    </div>
  </div>

  <!-- Smile grid -->
  <div class="section">
    <h2>Volatility Smile by Expiry</h2>
    <div class="subtitle">Each panel shows IV across strikes for one expiry · Red star = ATM · X-axis = K/S (1.0 = at-the-money)</div>
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
        <p>The red star marks the ATM strike. Points to the left are OTM puts (K &lt; S); points to the right are OTM calls (K &gt; S). The steeper the slope from right to left, the more the market fears a sharp sell-off.</p>
      </div>
      <div class="interp-box">
        <div class="tag">Current reading</div>
        <h3>Skew across expiries</h3>
        <p>Short-dated expiries show a steeper left-side slope, meaning near-term tail-risk hedging demand is elevated. Longer-dated smiles flatten — the market prices in mean-reversion over time, reducing the premium for downside protection.</p>
      </div>
    </div>
  </div>

  <!-- Term structure -->
  <div class="section">
    <h2>ATM Term Structure</h2>
    <div class="subtitle">ATM implied volatility at each expiry — shows how the market prices uncertainty over different horizons</div>
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

</div>
</body>
</html>"""

    out = "iv_dashboard.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"Saved → {out}")
    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(out))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  SPX Implied Volatility Surface Model")
    print("=" * 55)
    spot, df = fetch_data()
    print("\nBuilding dashboard …")
    build_dashboard(spot, df)
    print("\nDone! Open iv_dashboard.html in your browser.")

if __name__ == "__main__":
    main()
