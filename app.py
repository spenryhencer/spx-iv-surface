"""
Streamlit app — live version of the SPX implied volatility dashboard.

Run locally:   streamlit run app.py      (or double-click run.command on a Mac)
Data refreshes automatically every 5 minutes, or on demand with "Refresh now".
"""

import numpy as np
import pandas as pd
import streamlit as st

from iv_surface import (MAX_IV, MIN_IV, compute_stats, fmt_expiry, load_surface,
                        make_figures, market_status, parse_args)

REFRESH_SECONDS = 300

st.set_page_config(page_title="SPX IV Surface", page_icon="📈", layout="wide")


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner="Fetching SPX option quotes and rates …")
def get_surface(source, moneyness, max_spread, min_mid, min_volume, min_oi):
    args = parse_args(["--source", source,
                       "--moneyness", str(moneyness[0]), str(moneyness[1]),
                       "--max-spread", str(max_spread), "--min-mid", str(min_mid),
                       "--min-volume", str(min_volume), "--min-oi", str(min_oi)])
    return load_surface(args)


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["cboe", "yfinance"],
                      format_func={"cboe": "Cboe delayed quotes",
                                   "yfinance": "yfinance (market hours only)"}.get)
    if st.button("Refresh now", width="stretch"):
        get_surface.clear()
    st.caption(f"Auto-refreshes every {REFRESH_SECONDS // 60} min.")

    st.header("Filters")
    moneyness  = st.slider("Strike range (% of forward)", 0.70, 1.30, (0.85, 1.15), 0.01)
    max_spread = st.slider("Max bid/ask spread (% of mid)", 0.05, 1.00, 0.50, 0.05)
    min_mid    = st.number_input("Min mid price ($)", 0.0, 10.0, 0.05, 0.05)
    min_volume = st.number_input("Min volume", 0, 10_000, 0, 10)
    min_oi     = st.number_input("Min open interest", 0, 100_000, 0, 50)

st.title("SPX Implied Volatility Surface")

try:
    chain, curve, df, atm_df = get_surface(source, moneyness, max_spread, min_mid,
                                           min_volume, min_oi)
except Exception as e:
    st.error(str(e))
    st.stop()

stale, status_msg = market_status(chain.as_of)
(st.warning if stale else st.success)(status_msg)
st.caption(f"Source: {chain.source} · Rates: {curve.source} ({curve.asof}) · "
           f"{len(atm_df)} expiries · {len(df):,} points")

# ── Summary metrics ───────────────────────────────────────────────────────────
s = compute_stats(df, atm_df)
c = st.columns(4)
c[0].metric("SPX spot", f"{chain.spot:,.2f}")
c[1].metric(f"ATM IV · {atm_df.iloc[0]['days']}d", f"{s['near_atm_iv']:.1%}")
c[2].metric(f"ATM IV · {atm_df.iloc[-1]['days']}d", f"{s['far_atm_iv']:.1%}")
c[3].metric("Term structure", s["ts_slope"].split()[0].title(), s["ts_slope"], delta_color="off")
c = st.columns(4)
c[0].metric(f"25Δ risk reversal · {s['rr_row']['days']}d", s["skew_txt"])
c[1].metric(f"Risk-free rate · {s['q_row']['days']}d", f"{curve.rate(s['q_row']['T']):.2%}")
c[2].metric(f"Implied dividend yield · {s['q_row']['days']}d", f"{s['q_row']['div_yield']:.2%}")
c[3].metric("IV check vs source", s["iv_gap_txt"], "median |ours − source|", delta_color="off")

# ── Charts ────────────────────────────────────────────────────────────────────
figs = make_figures(df, atm_df)
tab_surf, tab_smile, tab_ts, tab_inputs, tab_data = st.tabs(
    ["3-D surface", "Smiles by expiry", "Term structure", "Per-expiry inputs", "Data"])

with tab_surf:
    st.plotly_chart(figs["surface"], width="stretch")
with tab_smile:
    st.caption("X-axis: strike as a % of each expiry's forward (100% = at the money). "
               "Y-axis: implied vol from the bid/ask mid. Hover a point for its bid, ask and mid.")
    st.plotly_chart(figs["smiles"], width="stretch")
with tab_ts:
    st.plotly_chart(figs["term_structure"], width="stretch")
with tab_inputs:
    st.caption(f"Dropped points are never plotted. quote = no bid, crossed/wide spread or below "
               f"liquidity floors · solver = price outside no-arbitrage bounds · "
               f"bounds = IV outside {MIN_IV:.0%}–{MAX_IV:.0%}.")
    show = atm_df.assign(expiry=[fmt_expiry(e, d) for e, d in zip(atm_df["expiry"], atm_df["days"])])
    st.dataframe(
        show[["expiry", "root", "rate", "forward", "div_yield", "IV_atm", "points",
              "dropped_quote", "dropped_failed", "dropped_bounds"]].style.format({
            "rate": "{:.2%}", "forward": "{:,.1f}", "div_yield": "{:.2%}", "IV_atm": "{:.1%}"}),
        hide_index=True, width="stretch")
with tab_data:
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False), "iv_points.csv", "text/csv")
