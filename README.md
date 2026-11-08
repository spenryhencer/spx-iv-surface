# SPX Equity Implied Volatility Surface Model

A Python tool that pulls **SPX (S&P 500 index) option quotes**, cleans them, and **computes implied volatility itself** by inverting Black-76 (Black-Scholes on the forward) on bid/ask mid prices. The risk-free rate comes from the US Treasury curve (FRED) and the forward / dividend yield is implied from put-call parity. It renders a single-page interactive dashboard with the full IV surface, per-expiry skew, and the ATM term structure.

![IV surface](docs/surface.png)

---

## How to Open the Dashboard

There are two ways to view it. Both build from SPX option quotes at run time, so nothing is stored in this repo. You generate it locally, and the first time takes about a minute.

### 1. Get the code
```bash
git clone https://github.com/spenryhencer/spx-iv-surface.git
cd spx-iv-surface
```
Or click **Code → Download ZIP** on this page and unzip it.

### 2a. Easiest on a Mac: double-click `run.command`
In Finder, double-click `run.command` in the project folder. The first run creates the Python environment and installs everything. It then opens the **live dashboard** at <http://localhost:8501>. Close the Terminal window to stop it.

### 2b. From Terminal
Install (first time only):
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Then either run the live app:
```bash
streamlit run app.py             # opens http://localhost:8501
```
or build a one-off static page:
```bash
python iv_surface.py             # writes iv_dashboard.html and opens it
```
To reopen the last static dashboard without pulling new data, open `iv_dashboard.html` in the project folder (double-click it, or run `open iv_dashboard.html` on macOS).

The **live app** (`app.py`) re-fetches quotes every 5 minutes (or press *Refresh now*). From the sidebar you can switch data source, switch the smile x-axis between % of forward and SPX strike, and adjust filters. It has tabs for the surface, smiles, term structure, per-expiry inputs, and the raw data with CSV download.

### Useful options (static page)
| Command | What it does |
|---|---|
| `python iv_surface.py --source yfinance` | Use Yahoo Finance instead of the default Cboe delayed quotes |
| `python iv_surface.py --csv iv_points.csv` | Also save the cleaned IV points to a CSV |
| `python iv_surface.py --no-open` | Build the page without opening the browser |
| `python iv_surface.py --help` | List every option, including the quote-filtering thresholds |

> **Note:** quotes only update while US markets are open (Mon–Fri, 9:30am–4pm New York time). Outside those hours the dashboard shows a *"Market closed / stale data — quotes as of …"* banner. The Cboe source (default) then shows the last session's closing quotes; yfinance has no usable quotes until the market reopens.

Run the tests with `python -m pytest`.

---

## Data sources

Choose with `--source`:

| Source | Flag | Cost / key | When it works | Notes |
|---|---|---|---|---|
| **Cboe delayed quotes** (default) | `--source cboe` | Free, no key | Any time — keeps the last session's closing bid/ask after hours and on weekends | ~15 min delayed. Public but undocumented endpoint, so it may change. Also gives Cboe's own IV, used as a cross-check. |
| **yfinance** (Yahoo Finance) | `--source yfinance` | Free, no key | **US market hours only** (9:30am–4pm ET) | Outside market hours Yahoo returns `bid = ask = 0` and placeholder IVs; the script detects this and tells you to switch source. |

Rates: US Treasury constant-maturity yields (1M, 3M, 6M, 1Y, 2Y) from [FRED](https://fred.stlouisfed.org/), no key needed. If FRED is unreachable it falls back to a flat 4% and says so on the dashboard.

---

## Methodology

For each selected expiry (nearest available to 14, 21, 30, 45, 60, 90, 120, 180, 270 and 365 days):

1. **Clean quotes** — keep two-sided quotes only (`bid > 0`, `ask > bid`), mid ≥ $0.05, relative spread `(ask − bid)/mid ≤ 50%`. Optional `--min-volume` / `--min-oi`. Mid prices are used, never last trade (which can be days stale).
2. **Rate** — Treasury yield interpolated to the option's maturity, converted to continuous compounding; `DF = e^(−rT)`.
3. **Forward** — put-call parity at the strikes nearest the money: `F = K + (C_mid − P_mid) / DF`. This gives the implied dividend yield `q = r − ln(F/S)/T` with no dividend forecast needed.
4. **OTM only** — puts for `K < F`, calls for `K ≥ F` (the liquid side of each strike); strikes within 85%–115% of F.
5. **Implied vol** — solve `Black76(F, K, T, DF, σ) = mid` with Brent's method; prices outside no-arbitrage bounds are dropped.
6. **Dropped points are reported, never plotted** — each expiry's table row shows how many quotes were dropped for bad quotes (zero bid, crossed or wide spread, below liquidity floors), failed inversions (price outside no-arbitrage bounds) or implausible IVs (outside 2%–150%).
7. **Surface** — each smile is interpolated onto a common K/F grid, then maturities are interpolated linearly in total variance `σ²T`. No smoothing of the market points (optional `--smooth N`).

Time to expiry is measured in calendar time from the quote timestamp to settlement (4:00pm ET for PM-settled SPXW, 9:30am ET for AM-settled SPX monthlies; SPXW is preferred when both exist).

**Validation:** on the Cboe feed, the median gap between our IVs and Cboe's own published IVs is **0.09 vol points** across ~2,000 quotes (6 Nov 2026 close).

---

## Output

`iv_dashboard.html` — one self-contained page with:

| Section | What it shows |
|---|---|
| Summary cards | Spot, near/long-dated ATM IV, term-structure shape, 30-day 25Δ risk reversal, risk-free rate, implied dividend yield, IV cross-check vs the source |
| 3-D surface | IV across moneyness (K/F) and maturity |
| Smile grid | IV vs K/F for each expiry; hover shows strike, put/call and bid/ask |
| Term structure | ATM-forward IV by days to expiry |
| Per-expiry table | Rate, forward, dividend yield, ATM IV, points used and points dropped (by reason) for each expiry |

Snapshot from the 6 Nov 2026 close (Cboe feed):

![Smiles](docs/smiles.png)
![Term structure](docs/term_structure.png)

---

## Command-line options

```
python iv_surface.py [--source cboe|yfinance] [--moneyness LO HI]
                     [--max-spread X] [--min-mid X] [--min-volume N] [--min-oi N]
                     [--smooth N] [--out FILE] [--csv FILE] [--png DIR] [--no-open]
```

| Option | Default | Description |
|---|---|---|
| `--source` | `cboe` | Data source |
| `--moneyness` | `0.85 1.15` | K/F range kept |
| `--max-spread` | `0.5` | Max relative bid/ask spread |
| `--min-mid` | `0.05` | Min mid price ($) |
| `--min-volume`, `--min-oi` | `0` | Optional liquidity floors |
| `--smooth` | `0` | Rolling-median window across strikes (0 = off) |
| `--out` | `iv_dashboard.html` | Dashboard path |
| `--csv` | — | Also save the cleaned IV points |
| `--png` | — | Also save chart PNGs (needs `kaleido` + Google Chrome) |
| `--no-open` | — | Don't open the browser |

---

## Project structure

```
spx-iv-surface/
├── app.py               # Streamlit live app
├── run.command          # double-click launcher (macOS)
├── iv_surface.py        # CLI, cleaning + IV pipeline, figures, static dashboard
├── providers.py         # Cboe and yfinance data providers
├── pricing.py           # Black-76, IV inversion, put-call-parity forward
├── rates.py             # FRED Treasury curve
├── tests/               # pytest unit tests
├── docs/                # screenshots used in this README
└── requirements.txt
```

---

## Changelog

**v2 (Nov 2026)**
- Added the **Cboe delayed-quotes provider** (now the default) alongside yfinance, selectable with `--source`.
- IV is now **computed in-house** (Black-76 + Brent) from bid/ask mids. v1 plotted Yahoo's pre-computed `impliedVolatility` field, which outside market hours is a placeholder (e.g. 0.00001, 3.1%, 6.25%, 12.5%).
- Added the Treasury rate curve (FRED), put-call-parity forwards and implied dividend yields.
- Bid/ask spread and mid-price filters; OTM-only selection; true 25Δ risk reversal; ATM measured at the forward.
- Surface built per expiry and interpolated in total variance (replaces `griddata` over strike × maturity).
- CLI options, CSV/PNG export, unit tests.
- Streamlit live app (`app.py`) with 5-minute refresh, plus a `run.command` launcher.
- "Market closed / stale data" banner, dropped-point counts by reason, and clearer axis labels (strike as % of forward, days to expiry, bid/ask/mid on hover).

---

## Disclaimer

This project is for **educational purposes only** and is not financial advice. Quotes are delayed and may be inaccurate.
