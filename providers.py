"""
Options data providers
======================
Each provider returns a `Chain`: spot, quote timestamp, and a tidy quotes table

    root | expiry | cp | strike | bid | ask | volume | open_interest | vendor_iv

for the expiries closest to the requested target maturities.

  yfinance — Yahoo Finance. Bid/ask are only populated during US market hours;
             outside them Yahoo returns bid = ask = 0 and placeholder IVs.
  cboe     — Cboe's public delayed-quotes feed (~15 min delay, no API key).
             Keeps the last session's closing bid/ask after hours, so it also
             works on evenings and weekends. Unofficial endpoint: may change.
"""

import json
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

ET          = ZoneInfo("America/New_York")
MIN_DAYS    = 7          # skip expiries closer than this (IV unstable near expiry)
OCC_PATTERN = r"^(?P<root>[A-Z]+)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$"
CBOE_URL    = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
USER_AGENT  = "spx-iv-surface/2.0 (+https://github.com/spenryhencer/spx-iv-surface)"

# SPXW (weeklies/EOM) settle on the close; standard SPX monthlies settle at the open
SETTLE_TIME = {"SPXW": time(16, 0), "SPX": time(9, 30)}


@dataclass
class Chain:
    source: str
    spot:   float
    as_of:  datetime          # tz-aware, America/New_York
    quotes: pd.DataFrame


def expiry_datetime(expiry: date, root: str) -> datetime:
    return datetime.combine(expiry, SETTLE_TIME.get(root, time(16, 0)), tzinfo=ET)


def pick_expiries(expiries, today: date, target_days):
    """Closest available expiry (≥ MIN_DAYS out) to each target day count."""
    days = {e: (e - today).days for e in set(expiries)}
    days = {e: d for e, d in days.items() if d >= MIN_DAYS}
    if not days:
        return []
    chosen = {min(days, key=lambda e: abs(days[e] - t)) for t in target_days}
    return sorted(chosen)


def parse_occ(symbols: pd.Series) -> pd.DataFrame:
    """Split OCC-style symbols (e.g. SPXW261218C07400000) into fields."""
    p = symbols.str.extract(OCC_PATTERN)
    return pd.DataFrame({
        "root":   p["root"],
        "expiry": pd.to_datetime(p["ymd"], format="%y%m%d").dt.date,
        "cp":     p["cp"],
        "strike": p["strike"].astype(float) / 1000.0,
    }, index=symbols.index)


def prefer_pm_root(q: pd.DataFrame) -> pd.DataFrame:
    """Where an expiry has both SPX (AM) and SPXW (PM) series, keep only SPXW."""
    has_w = q.groupby("expiry")["root"].transform(lambda r: (r == "SPXW").any())
    return q[~has_w | (q["root"] == "SPXW")]


# ── yfinance ─────────────────────────────────────────────────────────────────
def load_yfinance(target_days) -> Chain:
    import yfinance as yf

    tkr  = yf.Ticker("^SPX")
    hist = tkr.history(period="5d")
    if hist.empty:
        raise RuntimeError("yfinance returned no SPX price data — are you online?")
    spot = float(hist["Close"].dropna().iloc[-1])

    today    = datetime.now(ET).date()
    expiries = pick_expiries([date.fromisoformat(e) for e in tkr.options], today, target_days)
    frames, last_trades = [], []
    for exp in expiries:
        try:
            oc = tkr.option_chain(exp.isoformat())
        except Exception as e:
            print(f"  ✗ {exp}: {e}")
            continue
        for d in (oc.calls, oc.puts):
            f = parse_occ(d["contractSymbol"])
            f["bid"], f["ask"]       = d["bid"], d["ask"]
            f["volume"]              = d["volume"].fillna(0)
            f["open_interest"]       = d["openInterest"].fillna(0)
            f["vendor_iv"]           = d["impliedVolatility"]
            frames.append(f)
            last_trades.append(pd.to_datetime(d["lastTradeDate"], utc=True).max())

    if not frames:
        raise RuntimeError("yfinance returned no option chains.")
    as_of = max(last_trades).tz_convert(ET).to_pydatetime()
    q = prefer_pm_root(pd.concat(frames, ignore_index=True).dropna(subset=["root"]))
    return Chain("yfinance (Yahoo Finance)", spot, as_of, q.reset_index(drop=True))


# ── Cboe delayed quotes ──────────────────────────────────────────────────────
def load_cboe(target_days, timeout=60) -> Chain:
    req = urllib.request.Request(CBOE_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    data = payload["data"]
    spot = float(data["current_price"])
    stamp = data.get("last_trade_time") or payload["timestamp"]
    as_of = datetime.fromisoformat(stamp.replace(" ", "T")).replace(tzinfo=ET)

    raw = pd.DataFrame(data["options"])
    q   = parse_occ(raw["option"])
    q["bid"], q["ask"]         = raw["bid"], raw["ask"]
    q["volume"]                = raw["volume"].fillna(0)
    q["open_interest"]         = raw["open_interest"].fillna(0)
    q["vendor_iv"]             = raw["iv"]
    q = prefer_pm_root(q.dropna(subset=["root"]))

    chosen = pick_expiries(q["expiry"].unique(), as_of.date(), target_days)
    q = q[q["expiry"].isin(chosen)].reset_index(drop=True)
    return Chain("Cboe delayed quotes", spot, as_of, q)


PROVIDERS = {"cboe": load_cboe, "yfinance": load_yfinance}
