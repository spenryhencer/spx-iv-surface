## How to Open the Dashboard

The dashboard is an interactive HTML page (`iv_dashboard.html`) that the script builds from live SPX option quotes. It isn't stored in this repo, so you generate it locally. It takes about a minute.

### 1. Get the code
```bash
git clone https://github.com/spenryhencer/spx-iv-surface.git
cd spx-iv-surface
```
Or click **Code → Download ZIP** on this page and unzip it.

### 2. Install (first time only)
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Build and open it
```bash
python iv_surface.py
```
This pulls the current SPX options chain and writes `iv_dashboard.html`, then opens it in your browser.

To reopen the last dashboard without pulling new data, open `iv_dashboard.html` in the project folder (double-click it, or run `open iv_dashboard.html` on macOS).

### Useful options
| Command | What it does |
|---|---|
| `python iv_surface.py --source yfinance` | Use Yahoo Finance instead of the default CBOE delayed quotes |
| `python iv_surface.py --csv iv_points.csv` | Also save the cleaned IV points to a CSV |
| `python iv_surface.py --no-open` | Build the page without opening the browser |
| `python iv_surface.py --help` | List every option, including the quote-filtering thresholds |

> **Note:** quotes only update while US markets are open (Mon–Fri, 9:30am–4pm New York time). Outside those hours the data is stale, and some expiries may show few or no valid points.
