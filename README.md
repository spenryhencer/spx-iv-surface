# SPX Equity Implied Volatility Surface Model

A Python tool that pulls **live SPX (S&P 500) options chain data** via `yfinance`, inverts the Black-Scholes formula to extract implied volatility at each strike and maturity, and renders interactive charts showing the full IV surface, per-expiry skew dynamics, and the ATM term structure.

---

## What It Does

| Output | Description |
|---|---|
| `iv_surface_3d.html` | Interactive 3-D surface — IV plotted across strike (x) and maturity (y) |
| `iv_smile_by_expiry.html` | Volatility smile / skew for each expiry date |
| `iv_term_structure.html` | ATM implied volatility as a function of time to expiry |

### Key Features
- **Live data** — fetches the real-time SPX options chain at runtime via `yfinance`
- **Black-Scholes inversion** — uses Brent's root-finding method for accurate, fast IV extraction
- **Liquidity filtering** — removes illiquid strikes (low volume / open interest) that distort the surface
- **Call-put averaging** — merges call and put IVs at the same strike/expiry for a cleaner surface
- **Moneyness filter** — restricts the surface to ±25 % of spot (configurable)

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/spx-iv-surface.git
cd spx-iv-surface
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python iv_surface.py
```

The script will:
1. Download the current SPX spot price
2. Iterate over all available option expiries (up to 12)
3. Compute implied volatility for each liquid strike
4. Open three interactive HTML charts in your browser
5. Save those charts as `.html` files in the current directory

> **Note:** Markets must be open (or have recently closed) for `yfinance` to return option data. Running outside market hours may return partial or cached data.

---

## Configuration

All parameters are at the top of `iv_surface.py`:

| Parameter | Default | Description |
|---|---|---|
| `TICKER` | `^SPX` | Underlying ticker |
| `RISK_FREE_RATE` | `0.05` | Annualised risk-free rate used in BS |
| `MIN_VOLUME` | `10` | Minimum option volume to include |
| `MIN_OPEN_INT` | `50` | Minimum open interest to include |
| `MAX_EXPIRIES` | `12` | Maximum number of expiry dates to process |
| `MONEYNESS_RANGE` | `(0.75, 1.25)` | Strike range relative to spot |

---

## Project Structure

```
spx-iv-surface/
├── iv_surface.py        # Main script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Methodology

Implied volatility σ* is the value that solves:

```
C_BS(S, K, T, r, σ*) = C_market
```

where `C_BS` is the Black-Scholes call price. Brent's method is used to find σ* numerically, with bounds [0.0001, 10.0] and tolerance 1e-7.

For each (strike, expiry) pair where both a call and a put exist, the two IVs are averaged to reduce bid-ask and model noise (put-call parity ensures they should be equal in theory).

---

## Requirements

- Python 3.10+
- See `requirements.txt` for package versions

---

## Disclaimer

This project is for **educational purposes only**. It is not financial advice. Options data is sourced from Yahoo Finance and may be delayed or inaccurate.
