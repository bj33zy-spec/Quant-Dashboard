"""
Daily NQ/QQQ Quant Dashboard -- Interactive HTML Edition v2
=============================================================
Free data sources only:
  - yfinance          -> QQQ options chain, intraday price, VIX, SPY/DXY/BTC
  - CFTC public API   -> Nasdaq-100 (E-mini) leveraged-fund positioning (best-effort)

Produces: latest.html (self-contained interactive page, overwritten every run)

NEW IN v2:
  - Intraday price chart with call wall / put wall / gamma flip / VWAP overlaid
  - ICT killzone session tracker with a LIVE ticking ET clock + countdown (pure
    client-side JS -- keeps updating even while the page stays open, no server needed)
  - Overnight (Asia+London) range as a liquidity-level proxy
  - 0DTE gamma, Max Pain, and Put/Call OI ratio
  - Cross-asset correlation snapshot (QQQ vs SPY / DXY / BTC)

HONESTY NOTES (same spirit as v1):
  - QQQ is a free proxy for NQ/NDX -- not identical, especially overnight, since QQQ's
    pre/post-market liquidity is much thinner than NQ futures' near-24h liquidity. The
    overnight range panel is the weakest proxy in this dashboard for that reason.
  - Killzone times below are a common, standard definition -- adjust the KILLZONES_JS
    list in the template if your specific framework uses different windows.
  - CFTC/0DTE/correlation panels are all best-effort: if a fetch fails, that panel says
    "data unavailable" instead of crashing the rest of the page.
"""

import os
import math
import json
import textwrap
from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest.html")
ET = ZoneInfo("America/New_York")

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_errors.log")
MAX_LOG_LINES = 500


def log_error(message):
    """Writes to dashboard_errors.log with a timestamp, in addition to printing --
    print() alone is invisible once this runs headless under Task Scheduler."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    try:
        lines = []
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
        lines.append(line + "\n")
        lines = lines[-MAX_LOG_LINES:]
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass  # logging itself failing should never crash the actual dashboard build


# ---------------------------------------------------------------------------
# SHARED GAMMA MATH
# ---------------------------------------------------------------------------
def norm_pdf(x):
    """Standard normal PDF, hand-written to avoid needing scipy (some Windows
    setups block its compiled DLLs)."""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def compute_gex_from_chain(calls, puts, spot, T, r=0.05):
    """
    Computes gamma, vanna, and charm exposure per strike in one pass since they
    share the same d1/d2 math. Vanna and charm are mathematically identical for a
    call and put at the same strike/expiry (both derive from Delta_call - Delta_put
    = 1, a constant) -- verified numerically against finite-difference derivatives
    of Black-Scholes delta before shipping this. Same sign convention as GEX:
    calls contribute positively, puts negatively (assumes dealers are long calls
    sold to them, short puts sold to them -- the standard retail approximation,
    not a certainty about actual dealer books).
    """
    def bs_greeks(S, K, T, sigma):
        if sigma <= 0 or T <= 0:
            return 0.0, 0.0, 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        phi_d1 = norm_pdf(d1)
        gamma = phi_d1 / (S * sigma * np.sqrt(T))
        vanna = -phi_d1 * d2 / sigma
        charm = -phi_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        return gamma, vanna, charm

    strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
    call_gex, put_gex, call_iv_list, put_iv_list = [], [], [], []
    call_vex, put_vex, call_chex, put_chex = [], [], [], []
    for k in strikes:
        crow, prow = calls[calls["strike"] == k], puts[puts["strike"] == k]
        c_oi = crow["openInterest"].fillna(0).sum()
        p_oi = prow["openInterest"].fillna(0).sum()
        c_iv = crow["impliedVolatility"].mean() if len(crow) else np.nan
        p_iv = prow["impliedVolatility"].mean() if len(prow) else np.nan
        c_iv_safe = c_iv if c_iv and c_iv > 0 else 0.3
        p_iv_safe = p_iv if p_iv and p_iv > 0 else 0.3
        c_gamma, c_vanna, c_charm = bs_greeks(spot, k, T, c_iv_safe)
        p_gamma, p_vanna, p_charm = bs_greeks(spot, k, T, p_iv_safe)

        call_gex.append(c_gamma * c_oi * 100 * spot ** 2 * 0.01)
        put_gex.append(-p_gamma * p_oi * 100 * spot ** 2 * 0.01)
        # VEX: dollar delta-shift per 1 vol point (0.01 change in IV)
        call_vex.append(c_vanna * c_oi * 100 * spot * 0.01)
        put_vex.append(-p_vanna * p_oi * 100 * spot * 0.01)
        # CHEX: dollar delta-decay per 1 calendar day (charm formula is per year of T, so /365)
        call_chex.append(c_charm * c_oi * 100 * spot / 365.0)
        put_chex.append(-p_charm * p_oi * 100 * spot / 365.0)
        call_iv_list.append(c_iv)
        put_iv_list.append(p_iv)

    df = pd.DataFrame({
        "strike": strikes, "call_gex": call_gex, "put_gex": put_gex,
        "call_vex": call_vex, "put_vex": put_vex,
        "call_chex": call_chex, "put_chex": put_chex,
        "call_iv": call_iv_list, "put_iv": put_iv_list,
    })
    df["net_gex"] = df["call_gex"] + df["put_gex"]
    df["cum_gex"] = df["net_gex"].cumsum()
    df["net_vex"] = df["call_vex"] + df["put_vex"]
    df["net_chex"] = df["call_chex"] + df["put_chex"]
    return df


def compute_iv_metrics(df, spot):
    """ATM IV (nearest strike to spot) and put/call skew (5%-OTM put IV minus
    5%-OTM call IV -- positive skew means the market prices downside moves as
    more likely/severe than equivalent upside moves, the normal equity pattern)."""
    df = df.copy()
    df["dist_to_spot"] = (df["strike"] - spot).abs()
    atm_row = df.loc[df["dist_to_spot"].idxmin()]
    atm_iv = np.nanmean([atm_row["call_iv"], atm_row["put_iv"]])

    target_put_strike = spot * 0.95
    target_call_strike = spot * 1.05
    put_row = df.iloc[(df["strike"] - target_put_strike).abs().argsort()[:1]]
    call_row = df.iloc[(df["strike"] - target_call_strike).abs().argsort()[:1]]
    otm_put_iv = put_row["put_iv"].values[0] if len(put_row) else np.nan
    otm_call_iv = call_row["call_iv"].values[0] if len(call_row) else np.nan
    skew = (otm_put_iv - otm_call_iv) if (not np.isnan(otm_put_iv) and not np.isnan(otm_call_iv)) else np.nan

    return {"atm_iv": atm_iv, "skew": skew,
            "otm_put_strike": float(put_row["strike"].values[0]) if len(put_row) else None,
            "otm_call_strike": float(call_row["strike"].values[0]) if len(call_row) else None}


# ---------------------------------------------------------------------------
# 1. GAMMA EXPOSURE BY STRIKE (weekly-ish nearest expiry, "the gamma rays")
# ---------------------------------------------------------------------------
def get_gex_data():
    tk = yf.Ticker("QQQ")
    spot = tk.history(period="1d")["Close"].iloc[-1]

    today = datetime.now(timezone.utc).date()
    best_exp, best_dte = None, None
    for e in tk.options:
        edate = datetime.strptime(e, "%Y-%m-%d").date()
        dte = (edate - today).days
        if dte >= 1:
            best_exp, best_dte = e, dte
            break
    if best_exp is None:
        raise RuntimeError("No usable QQQ expiry found")

    chain = tk.option_chain(best_exp)
    df = compute_gex_from_chain(chain.calls, chain.puts, spot, best_dte / 365.0)
    iv_metrics = compute_iv_metrics(df, spot)
    net_delta_exposure = compute_net_delta_exposure(chain.calls, chain.puts, spot, best_dte / 365.0)

    sign_changes = np.where(np.diff(np.sign(df["cum_gex"])) != 0)[0]
    gamma_flip = df["strike"].iloc[sign_changes[0]] if len(sign_changes) else np.nan
    call_wall = df.loc[df["call_gex"].idxmax(), "strike"] if df["call_gex"].abs().sum() > 0 else np.nan
    put_wall = df.loc[df["put_gex"].idxmin(), "strike"] if df["put_gex"].abs().sum() > 0 else np.nan

    return {
        "spot": spot, "expiry": best_exp, "dte": best_dte, "df": df,
        "gamma_flip": gamma_flip, "call_wall": call_wall, "put_wall": put_wall,
        "net_gex_total": df["net_gex"].sum(),
        "net_vex_total": df["net_vex"].sum(),
        "net_chex_total": df["net_chex"].sum(),
        "net_delta_exposure": net_delta_exposure,
        "atm_iv": iv_metrics["atm_iv"], "skew": iv_metrics["skew"],
    }


# ---------------------------------------------------------------------------
# 1B. GAMMA EXPOSURE BY EXPIRY BUCKET (nearest / next / ~monthly)
# ---------------------------------------------------------------------------
def get_multi_expiry_gex(spot):
    try:
        tk = yf.Ticker("QQQ")
        today = datetime.now(timezone.utc).date()
        usable = [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days)
                  for e in tk.options]
        usable = [u for u in usable if u[1] >= 1]
        if not usable:
            return None

        nearest = usable[0]
        next_exp = usable[1] if len(usable) > 1 else None
        monthly = min(usable, key=lambda u: abs(u[1] - 30))

        buckets = {"This Week": nearest}
        if next_exp and next_exp[0] != nearest[0]:
            buckets["Next Week"] = next_exp
        if monthly[0] not in (b[0] for b in buckets.values()):
            buckets["Monthly"] = monthly

        results = {}
        for label, (exp, dte) in buckets.items():
            chain = tk.option_chain(exp)
            df = compute_gex_from_chain(chain.calls, chain.puts, spot, dte / 365.0)
            results[label] = round(df["net_gex"].sum() / 1e6, 2)
        return results
    except Exception as e:
        log_error(f"Multi-expiry GEX fetch failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# 2. VOLATILITY REGIME
# ---------------------------------------------------------------------------
def get_vol_data():
    vix_hist = yf.Ticker("^VIX").history(period="5y")["Close"]
    vix_now, vix_avg5y = vix_hist.iloc[-1], vix_hist.mean()
    qqq_hist = yf.Ticker("QQQ").history(period="6mo")["Close"]
    rets = np.log(qqq_hist / qqq_hist.shift(1)).dropna()
    hv20 = rets[-20:].std() * np.sqrt(252) * 100
    hv60 = rets[-60:].std() * np.sqrt(252) * 100
    return {"vix_now": vix_now, "vix_avg5y": vix_avg5y, "hv20": hv20, "hv60": hv60}


# ---------------------------------------------------------------------------
# 3. SEASONALITY -- computed LIVE from full QQQ history
# ---------------------------------------------------------------------------
def get_seasonality():
    hist = yf.Ticker("QQQ").history(period="max")["Close"]
    monthly = hist.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna().to_frame("ret")
    monthly_ret["month"] = monthly_ret.index.month
    stats = monthly_ret.groupby("month")["ret"].agg(["mean", lambda x: (x > 0).mean(), "count"])
    stats.columns = ["avg_return", "win_rate", "n_years"]
    return stats


# ---------------------------------------------------------------------------
# 4. CFTC POSITIONING -- best-effort, free public Socrata API
# ---------------------------------------------------------------------------
def get_cot_data():
    try:
        url = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
        params = {
            "$where": "market_and_exchange_names like '%NASDAQ-100%'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": 1,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        row = data[0]
        long_ = float(row.get("noncomm_positions_long_all", 0))
        short_ = float(row.get("noncomm_positions_short_all", 0))
        oi = float(row.get("open_interest_all", 1)) or 1
        net = long_ - short_
        return {"net": net, "pct_oi": net / oi * 100, "date": row.get("report_date_as_yyyy_mm_dd", "")}
    except Exception as e:
        log_error(f"COT fetch failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# 5. 0DTE GAMMA + MAX PAIN + PUT/CALL RATIO
# ---------------------------------------------------------------------------
def get_0dte_data(spot):
    try:
        tk = yf.Ticker("QQQ")
        today = datetime.now(timezone.utc).date()
        zero_dte_exp = None
        for e in tk.options:
            if datetime.strptime(e, "%Y-%m-%d").date() == today:
                zero_dte_exp = e
                break
        if zero_dte_exp is None:
            return None

        chain = tk.option_chain(zero_dte_exp)
        calls, puts = chain.calls, chain.puts

        now_et = datetime.now(ET)
        close_today = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        hours_remaining = max((close_today - now_et).total_seconds() / 3600, 0.25)
        T = hours_remaining / (24 * 365)

        df = compute_gex_from_chain(calls, puts, spot, T)
        net_gex_0dte = df["net_gex"].sum()

        strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
        pain = []
        for candidate in strikes:
            call_pain = ((candidate - calls["strike"]).clip(lower=0) * calls["openInterest"].fillna(0)).sum()
            put_pain = ((puts["strike"] - candidate).clip(lower=0) * puts["openInterest"].fillna(0)).sum()
            pain.append(call_pain + put_pain)
        max_pain_strike = strikes[int(np.argmin(pain))]

        call_oi = calls["openInterest"].fillna(0).sum()
        put_oi = puts["openInterest"].fillna(0).sum()
        pc_ratio = put_oi / call_oi if call_oi > 0 else np.nan

        return {
            "expiry": zero_dte_exp, "net_gex_total": net_gex_0dte,
            "max_pain": max_pain_strike, "pc_ratio": pc_ratio,
            "hours_remaining": hours_remaining,
        }
    except Exception as e:
        log_error(f"0DTE fetch failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# 6. OVERNIGHT RANGE (Asia + London window, thin-liquidity QQQ proxy)
# ---------------------------------------------------------------------------
def get_overnight_range():
    try:
        hist = yf.Ticker("QQQ").history(period="2d", interval="15m", prepost=True)
        if hist.empty:
            return None
        hist.index = hist.index.tz_convert(ET)
        now_et = datetime.now(ET)
        yesterday = now_et.date() - pd.Timedelta(days=1)
        window_start = datetime.combine(yesterday, dtime(20, 0), tzinfo=ET)
        window_end = datetime.combine(now_et.date(), dtime(7, 0), tzinfo=ET)
        window = hist.loc[(hist.index >= window_start) & (hist.index <= window_end)]
        if window.empty:
            return None
        return {"high": float(window["High"].max()), "low": float(window["Low"].min())}
    except Exception as e:
        log_error(f"Overnight range fetch failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# 7. INTRADAY PRICE + SESSION VWAP (for the overlay chart)
# ---------------------------------------------------------------------------
def get_intraday_data():
    """Single intraday OHLCV fetch, shared by both the price/VWAP series and the
    FVG/order-block detector so their bar indices always refer to the same bars."""
    try:
        hist = yf.Ticker("QQQ").history(period="1d", interval="5m")
        if hist.empty:
            return None
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize(ET)
        else:
            hist.index = hist.index.tz_convert(ET)
        return hist
    except Exception as e:
        log_error(f"Intraday fetch failed (non-fatal): {e}")
        return None


def build_price_series(hist):
    if hist is None or hist.empty:
        return None
    times = [t.strftime("%H:%M") for t in hist.index]
    closes = [round(v, 2) for v in hist["Close"].tolist()]
    typical = (hist["High"] + hist["Low"] + hist["Close"]) / 3
    cum_vol = hist["Volume"].cumsum().replace(0, np.nan)
    vwap = ((typical * hist["Volume"]).cumsum() / cum_vol).round(2)
    vwap = vwap.bfill().ffill().tolist()
    return {"times": times, "closes": closes, "vwap": vwap}


# ---------------------------------------------------------------------------
# 8. CROSS-ASSET CORRELATION SNAPSHOT
# ---------------------------------------------------------------------------
def get_correlation_data():
    try:
        tickers = {"SPY": "SPY", "DXY": "DX-Y.NYB", "BTC": "BTC-USD"}
        closes = {"QQQ": yf.Ticker("QQQ").history(period="3mo")["Close"]}
        for name, sym in tickers.items():
            try:
                closes[name] = yf.Ticker(sym).history(period="3mo")["Close"]
            except Exception:
                pass
        df = pd.DataFrame(closes).dropna()
        if "QQQ" not in df.columns or len(df) < 20:
            return None
        rets = df.pct_change().dropna()
        corr = rets.corr()["QQQ"].drop("QQQ")
        return {k: round(v, 2) for k, v in corr.to_dict().items()}
    except Exception as e:
        log_error(f"Correlation fetch failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# 9. QUANTITATIVE RESEARCH SUMMARY
# ---------------------------------------------------------------------------
def generate_research_summary(gex, vol, season, cot):
    lines = []
    spot, flip = gex["spot"], gex["gamma_flip"]
    if gex["net_gex_total"] > 0:
        gamma_line = ("Dealers are net LONG gamma \u2192 hedging flow should dampen volatility "
                       "(mean-reversion-friendly) while price stays above the flip.")
    else:
        gamma_line = ("Dealers are net SHORT gamma \u2192 hedging flow can amplify moves "
                       "(trend-friendly, higher realized vol likely) below the flip.")
    if not np.isnan(flip):
        dist_pct = (spot - flip) / spot * 100
        if abs(dist_pct) < 1.5:
            gamma_line += f" Spot is only {abs(dist_pct):.1f}% from the flip \u2014 regime could switch on a modest move."
    lines.append(gamma_line)

    vix_regime = ("elevated" if vol["vix_now"] > vol["vix_avg5y"] * 1.15
                  else "low" if vol["vix_now"] < vol["vix_avg5y"] * 0.85 else "roughly average")
    vol_line = (f"Volatility is {vix_regime} relative to its own 5-year history "
                f"(VIX {vol['vix_now']:.1f} vs {vol['vix_avg5y']:.1f} avg).")
    if vol["hv20"] > vol["hv60"] * 1.15:
        vol_line += " Realized vol has been picking up over the last month."
    elif vol["hv20"] < vol["hv60"] * 0.85:
        vol_line += " Realized vol has been cooling off over the last month."
    lines.append(vol_line)

    current_month = datetime.now().month
    if current_month in season.index:
        avg_ret = season.loc[current_month, "avg_return"] * 100
        win_rate = season.loc[current_month, "win_rate"] * 100
        n_years = int(season.loc[current_month, "n_years"])
        direction = "tailwind" if avg_ret > 0 else "headwind"
        lines.append(f"This calendar month has historically been a mild {direction} "
                      f"({avg_ret:+.1f}% avg, positive {win_rate:.0f}% of {n_years} years) "
                      f"\u2014 real but noisy, small-sample caveat applies.")

    if cot:
        pct_oi = cot["pct_oi"]
        if pct_oi < -15:
            lines.append(f"Leveraged funds are crowded SHORT ({pct_oi:.1f}% of OI) \u2014 raises squeeze risk on any upside move.")
        elif pct_oi > 15:
            lines.append(f"Leveraged funds are crowded LONG ({pct_oi:.1f}% of OI) \u2014 raises long-squeeze risk on any downside move.")
        else:
            lines.append(f"Positioning is not at an extreme currently ({pct_oi:+.1f}% of OI).")
    else:
        lines.append("Positioning data unavailable this run.")

    if gex["net_gex_total"] > 0 and vix_regime == "low":
        bottom = "conditions favor range-bound, mean-reverting price action near-term."
    elif gex["net_gex_total"] < 0 and vix_regime == "elevated":
        bottom = "conditions favor larger, trend-extending moves \u2014 size cautiously."
    else:
        bottom = "no single dominant regime \u2014 mixed signals, stay flexible."
    lines.append("BOTTOM LINE: " + bottom)
    return lines

# ---------------------------------------------------------------------------
# 10. REFERENCE LEVELS -- Previous Day High/Low, Weekly & Monthly True Open
# ---------------------------------------------------------------------------
def get_reference_levels():
    try:
        hist = yf.Ticker("QQQ").history(period="3mo", interval="1d")
        if hist.empty:
            return None
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize(ET)
        else:
            hist.index = hist.index.tz_convert(ET)

        today_et = datetime.now(ET).date()
        dates = hist.index.date
        prior = hist[dates < today_et]
        if len(prior) == 0:
            return None
        pdh, pdl = float(prior.iloc[-1]["High"]), float(prior.iloc[-1]["Low"])

        monday = today_et - pd.Timedelta(days=datetime.now(ET).weekday())
        week_rows = hist[dates >= monday]
        weekly_open = float(week_rows.iloc[0]["Open"]) if len(week_rows) else None

        month_start = today_et.replace(day=1)
        month_rows = hist[dates >= month_start]
        monthly_open = float(month_rows.iloc[0]["Open"]) if len(month_rows) else None

        return {"pdh": pdh, "pdl": pdl, "weekly_open": weekly_open, "monthly_open": monthly_open}
    except Exception as e:
        log_error(f"Reference levels fetch failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# 11. ATR-BASED EXPECTED RANGE
# ---------------------------------------------------------------------------
def get_atr_data():
    try:
        hist = yf.Ticker("QQQ").history(period="30d", interval="1d")
        if len(hist) < 15:
            return None
        high, low, close = hist["High"], hist["Low"], hist["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        todays_open = float(hist["Open"].iloc[-1])
        return {"atr14": float(atr14), "todays_open": todays_open,
                "range_high": todays_open + atr14, "range_low": todays_open - atr14}
    except Exception as e:
        log_error(f"ATR fetch failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# 12. HISTORICAL NET GEX LOG -- persists across daily runs in a small JSON file
# ---------------------------------------------------------------------------
HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gex_history.json")


def update_and_load_history(extra_fields):
    """extra_fields is a dict of everything worth tracking day-over-day.
    Backward compatible: old log entries just won't have the newer keys,
    and every consumer below checks with .get() rather than assuming presence."""
    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    history = [h for h in history if h.get("date") != today_str]
    entry = {"date": today_str}
    entry.update(extra_fields)
    history.append(entry)
    history = history[-90:]
    try:
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f)
    except Exception as e:
        log_error(f"History write failed (non-fatal): {e}")
    return history


# ---------------------------------------------------------------------------
# 13. ALERT BANNER LOGIC
# ---------------------------------------------------------------------------
def generate_alerts(gex, vol, cot):
    alerts = []
    vix_regime = ("Elevated" if vol["vix_now"] > vol["vix_avg5y"] * 1.15
                  else "Low" if vol["vix_now"] < vol["vix_avg5y"] * 0.85 else "Normal")
    gex_amplifying = gex["net_gex_total"] < 0

    if vix_regime == "Elevated" and gex_amplifying:
        alerts.append("HIGH VOLATILITY RISK: Elevated VIX + dealers short gamma \u2014 expect amplified, larger moves today.")

    if not np.isnan(gex["gamma_flip"]):
        dist_pct = abs(gex["spot"] - gex["gamma_flip"]) / gex["spot"] * 100
        if dist_pct < 0.5:
            alerts.append(f"REGIME BOUNDARY: Spot is within {dist_pct:.2f}% of the gamma flip \u2014 dealer hedging behavior could flip today.")

    if cot and abs(cot["pct_oi"]) > 20:
        direction = "SHORT" if cot["pct_oi"] < 0 else "LONG"
        alerts.append(f"CROWDED POSITIONING: Leveraged funds at a {direction} extreme ({cot['pct_oi']:.1f}% of OI) \u2014 elevated squeeze risk.")

    return alerts


# ---------------------------------------------------------------------------
# 14. FAIR VALUE GAP + ORDER BLOCK DETECTOR (simple heuristic, not gospel)
# ---------------------------------------------------------------------------
def detect_fvg_and_ob(hist):
    """
    Heuristic pattern-matcher, not a rigorous ICT-certified detector:
      - FVG: classic 3-candle imbalance (candle[i-2].high < candle[i].low, or reverse)
      - Order block: last opposite-colored candle before a move at least 1.5x its size
    Real ICT analysis considers displacement, mitigation, and HTF context this doesn't.
    Treat this as a rough visual aid, not a signal to trade on its own.
    Takes the SAME dataframe used for the price chart, so indices always line up.
    """
    try:
        if hist is None or len(hist) < 3:
            return None

        opens = hist["Open"].tolist()
        highs = hist["High"].tolist()
        lows = hist["Low"].tolist()
        closes = hist["Close"].tolist()
        times_list = [t.strftime("%H:%M") for t in hist.index]

        fvgs, obs = [], []
        for i in range(2, len(hist)):
            if highs[i - 2] < lows[i]:
                fvgs.append({"type": "bullish", "top": lows[i], "bottom": highs[i - 2], "index": i - 2})
            if lows[i - 2] > highs[i]:
                fvgs.append({"type": "bearish", "top": lows[i - 2], "bottom": highs[i], "index": i - 2})

        for i in range(1, len(hist)):
            prev_body = abs(closes[i - 1] - opens[i - 1])
            cur_body = abs(closes[i] - opens[i])
            if closes[i - 1] < opens[i - 1] and closes[i] > opens[i] and cur_body > 1.5 * max(prev_body, 0.001):
                obs.append({"type": "bullish", "top": opens[i - 1], "bottom": closes[i - 1],
                            "time": times_list[i - 1]})
            if closes[i - 1] > opens[i - 1] and closes[i] < opens[i] and cur_body > 1.5 * max(prev_body, 0.001):
                obs.append({"type": "bearish", "top": closes[i - 1], "bottom": opens[i - 1],
                            "time": times_list[i - 1]})

        return {"fvgs": fvgs[-8:], "obs": obs[-6:]}
    except Exception as e:
        log_error(f"FVG/OB detection failed (non-fatal): {e}")
        return None


# ===========================================================================
# HTML TEMPLATE -- dark theme, Chart.js (CDN), animated numbers, live killzone
# clock, position calculator, alert banner, and layered chart overlays
# ===========================================================================

# ---------------------------------------------------------------------------
# 15. HURST EXPONENT -- multi-window R/S regression (NOT single-window; see
# code history/testing notes -- a single-window R/S estimate is mathematically
# biased because it conflates the Hurst exponent with an unknown scaling
# constant. This version fits log(R/S) vs log(window) across multiple window
# sizes via linear regression, verified against synthetic series with known
# persistent/random/mean-reverting properties before shipping.
# ---------------------------------------------------------------------------
def compute_hurst(prices, min_window=10, max_window=80, n_windows=8):
    prices = np.array(prices)
    log_rets = np.diff(np.log(prices))
    n_total = len(log_rets)
    max_w = min(max_window, n_total // 2)
    if max_w < min_window * 2:
        return None
    window_sizes = np.unique(np.logspace(np.log10(min_window), np.log10(max_w), n_windows).astype(int))
    window_sizes = window_sizes[window_sizes >= 10]
    rs_values, valid_windows = [], []
    for w in window_sizes:
        n_chunks = n_total // w
        if n_chunks < 1:
            continue
        chunk_rs = []
        for i in range(n_chunks):
            chunk = log_rets[i * w:(i + 1) * w]
            dev = chunk - chunk.mean()
            cum_dev = np.cumsum(dev)
            R = cum_dev.max() - cum_dev.min()
            S = chunk.std()
            if S > 0:
                chunk_rs.append(R / S)
        if chunk_rs:
            rs_values.append(np.mean(chunk_rs))
            valid_windows.append(w)
    if len(valid_windows) < 3:
        return None
    h, _ = np.polyfit(np.log(valid_windows), np.log(rs_values), 1)
    return max(min(h, 1.0), 0.0)


def get_hurst_data():
    try:
        hist = yf.Ticker("QQQ").history(period="1y")["Close"]
        if len(hist) < 100:
            return None
        h = compute_hurst(hist.tolist())
        if h is None:
            return None
        regime = "Trending" if h > 0.55 else "Mean-Reverting" if h < 0.45 else "Random-ish"
        return {"h": round(h, 3), "regime": regime}
    except Exception as e:
        log_error(f"Hurst fetch failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# 16. FORWARD DELTA EXPOSURE PROJECTION (charm-driven, price/IV held constant)
# ---------------------------------------------------------------------------
def compute_net_delta_exposure(calls, puts, spot, T, r=0.05):
    """Current dollar net delta exposure -- the baseline the charm projection
    extrapolates forward from. Same call/put sign convention as GEX."""
    total = 0.0
    for _, row in calls.iterrows():
        K, oi, iv = row["strike"], row.get("openInterest", 0) or 0, row.get("impliedVolatility", 0.3)
        iv = iv if iv and iv > 0 else 0.3
        if T <= 0 or iv <= 0:
            continue
        d1 = (np.log(spot / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        delta = norm_cdf(d1)
        total += delta * oi * 100 * spot
    for _, row in puts.iterrows():
        K, oi, iv = row["strike"], row.get("openInterest", 0) or 0, row.get("impliedVolatility", 0.3)
        iv = iv if iv and iv > 0 else 0.3
        if T <= 0 or iv <= 0:
            continue
        d1 = (np.log(spot / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        delta = norm_cdf(d1) - 1
        total -= delta * oi * 100 * spot
    return total


def norm_cdf(x):
    """Standard normal CDF via the erf-based closed form -- avoids needing scipy."""
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))


def get_forward_projection(gex):
    try:
        current_delta = gex.get("net_delta_exposure")
        chex_total = gex.get("net_chex_total")
        if current_delta is None or chex_total is None:
            return None
        days = list(range(0, 6))
        projected = [current_delta + chex_total * d for d in days]
        return {"days": days, "values": [round(v / 1e6, 2) for v in projected]}
    except Exception as e:
        log_error(f"Forward projection failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# 17. COMPOSITE CONFLUENCE SCORE (directional lean, -100 to +100)
# ---------------------------------------------------------------------------
def compute_confluence_score(gex, cot, season):
    score = 0.0
    parts = []

    if cot:
        pct_oi = cot["pct_oi"]
        contrarian = max(min(-pct_oi * 2, 40), -40)
        score += contrarian
        parts.append(f"Positioning (contrarian): {contrarian:+.0f}")

    skew = gex.get("skew")
    if skew is not None and not np.isnan(skew):
        skew_component = max(min(skew * 500, 20), -20)
        score += skew_component
        parts.append(f"Skew (contrarian fear read): {skew_component:+.0f}")

    current_month = datetime.now().month
    if current_month in season.index:
        season_component = max(min(season.loc[current_month, "avg_return"] * 1000, 20), -20)
        score += season_component
        parts.append(f"Seasonality: {season_component:+.0f}")

    score = max(min(score, 100), -100)
    label = "Bullish Lean" if score > 15 else "Bearish Lean" if score < -15 else "Neutral"
    return {"score": round(score, 1), "label": label, "parts": parts}


# ---------------------------------------------------------------------------
# 18. REGIME RADAR (6-axis intensity/magnitude, 0-100 each)
# ---------------------------------------------------------------------------
def compute_radar_data(gex, vol, cot, hurst, season):
    def clamp(v, lo=0, hi=100):
        return max(min(v, hi), lo)

    gamma_intensity = clamp(abs(gex["net_gex_total"]) / 5e7 * 100)
    vol_deviation = clamp(abs(vol["vix_now"] - vol["vix_avg5y"]) / vol["vix_avg5y"] * 300)
    positioning_extremity = clamp(abs(cot["pct_oi"]) / 30 * 100) if cot else 0
    hurst_deviation = clamp(abs(hurst["h"] - 0.5) * 200) if hurst else 0
    skew_val = gex.get("skew")
    skew_val = skew_val if skew_val is not None and not np.isnan(skew_val) else 0.0
    skew_magnitude = clamp(abs(skew_val) * 2000)
    current_month = datetime.now().month
    season_strength = clamp(abs(season.loc[current_month, "avg_return"]) * 2000) if current_month in season.index else 0

    return {
        "labels": ["Gamma", "Volatility", "Positioning", "Hurst", "Skew", "Seasonality"],
        "values": [round(gamma_intensity, 1), round(vol_deviation, 1), round(positioning_extremity, 1),
                   round(hurst_deviation, 1), round(skew_magnitude, 1), round(season_strength, 1)],
    }


# ---------------------------------------------------------------------------
# 19. PINE SCRIPT EXPORT (QQQ-derived levels scaled to NQ's current price ratio)
# ---------------------------------------------------------------------------
def get_nq_scale_ratio(qqq_spot):
    try:
        nq_price = yf.Ticker("NQ=F").history(period="1d")["Close"].iloc[-1]
        return float(nq_price) / float(qqq_spot)
    except Exception as e:
        log_error(f"NQ ratio fetch failed (non-fatal): {e}")
        return None


def generate_pine_export(gex, ref_levels, atr, ratio, zero_dte, overnight, confluence, hurst, cot, vol):
    def scaled(val):
        if val is None or (isinstance(val, float) and np.isnan(val)) or ratio is None:
            return None
        return round(val * ratio, 2)

    lines = [
        "//@version=6",
        f"// Auto-generated from Daily NQ/QQQ Quant Dashboard -- {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "// QQQ-derived levels scaled to today's NQ/QQQ price ratio -- NOT independently",
        "// computed from NQ's own options data (no free source for that exists).",
        "indicator(\"Dashboard Levels\", overlay=true, max_labels_count=50)",
        "",
    ]

    level_defs = [
        ("gammaFlip", scaled(gex.get("gamma_flip")), "yellow", "Gamma Flip"),
        ("callWall", scaled(gex.get("call_wall")), "green", "Call Wall"),
        ("putWall", scaled(gex.get("put_wall")), "red", "Put Wall"),
    ]
    if atr:
        level_defs.append(("atrRangeHigh", scaled(atr.get("range_high")), "gray", "ATR Range High"))
        level_defs.append(("atrRangeLow", scaled(atr.get("range_low")), "gray", "ATR Range Low"))
    if ref_levels:
        level_defs.append(("pdh", scaled(ref_levels.get("pdh")), "purple", "PDH"))
        level_defs.append(("pdl", scaled(ref_levels.get("pdl")), "purple", "PDL"))
        level_defs.append(("weeklyOpen", scaled(ref_levels.get("weekly_open")), "aqua", "Weekly Open"))
        level_defs.append(("monthlyOpen", scaled(ref_levels.get("monthly_open")), "fuchsia", "Monthly Open"))
    if zero_dte:
        level_defs.append(("maxPain", scaled(zero_dte.get("max_pain")), "orange", "Max Pain"))
    if overnight:
        level_defs.append(("onHigh", scaled(overnight.get("high")), "teal", "Overnight High"))
        level_defs.append(("onLow", scaled(overnight.get("low")), "teal", "Overnight Low"))

    valid_levels = [ld for ld in level_defs if ld[1] is not None]

    for name, val, color, label in valid_levels:
        lines.append(f"{name} = {val}")
    if valid_levels:
        lines.append("")
        for name, val, color, label in valid_levels:
            lines.append(f'plot({name}, "{label}", color=color.{color}, style=plot.style_linebr, linewidth=1)')
    else:
        # Guarantees the script always has at least one render call, even with
        # zero usable levels today -- an empty script fails Pine's CE10246
        # ("no plot/drawing found") compile check, which is exactly what
        # happened when this branch was missing.
        lines.append("")
        lines.append('plot(close, "Dashboard Levels (unavailable today)", display=display.none)')

    # ---- Info table: everything that isn't a price level goes here instead ----
    stat_rows = []
    net_gex = gex.get("net_gex_total")
    if net_gex is not None:
        stat_rows.append(("Net GEX", f"${net_gex/1e6:.1f}M"))
    vex = gex.get("net_vex_total")
    if vex is not None:
        stat_rows.append(("Net VEX", f"${vex/1e6:.2f}M"))
    chex = gex.get("net_chex_total")
    if chex is not None:
        stat_rows.append(("Net CHEX", f"${chex/1e6:.2f}M"))
    skew = gex.get("skew")
    if skew is not None and not (isinstance(skew, float) and np.isnan(skew)):
        stat_rows.append(("Put/Call Skew", f"{skew*100:+.1f} pts"))
    if vol:
        stat_rows.append(("VIX", f"{vol['vix_now']:.2f}"))
    if hurst:
        stat_rows.append(("Hurst (H)", f"{hurst['h']:.3f} {hurst['regime']}"))
    if cot:
        stat_rows.append(("COT % OI", f"{cot['pct_oi']:+.1f}%"))
    if confluence:
        stat_rows.append(("Confluence", f"{confluence['score']:+.0f} {confluence['label']}"))

    n_rows = len(stat_rows) + 1  # +1 for header row
    lines.append("")
    lines.append(f"var infoTable = table.new(position.top_right, 2, {n_rows}, border_width=1)")
    lines.append("if barstate.islast")
    lines.append(f'    table.cell(infoTable, 0, 0, "Dashboard \u2014 {datetime.now().strftime("%Y-%m-%d")}", text_color=color.white, bgcolor=color.blue)')
    lines.append('    table.cell(infoTable, 1, 0, "", bgcolor=color.blue)')
    for i, (label, value) in enumerate(stat_rows, start=1):
        lines.append(f'    table.cell(infoTable, 0, {i}, "{label}")')
        lines.append(f'    table.cell(infoTable, 1, {i}, "{value}")')

    return "\n".join(lines)



# ---------------------------------------------------------------------------
# 20. EVENT CALENDAR -- FOMC dates verified from federalreserve.gov (official
# source, checked directly); NFP computed via the "first Friday" rule, which
# is a genuine, stable calendar rule, not a guess.
# ---------------------------------------------------------------------------
FOMC_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28",
    "2027-09-15", "2027-10-27", "2027-12-08",
]


def get_first_fridays(start_date, n_months):
    fridays = []
    d = start_date.replace(day=1)
    for _ in range(n_months):
        first = d
        while first.weekday() != 4:
            first += pd.Timedelta(days=1)
        fridays.append(first)
        d = (d + pd.Timedelta(days=32)).replace(day=1)
    return fridays


def get_event_calendar_data():
    try:
        today = datetime.now(ET).date()
        fomc_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in FOMC_DATES]
        upcoming_fomc = [d for d in fomc_dates if d >= today]
        next_fomc = min(upcoming_fomc) if upcoming_fomc else None

        nfp_dates = [f.date() for f in get_first_fridays(pd.Timestamp(today) - pd.Timedelta(days=5), 3)]
        upcoming_nfp = [d for d in nfp_dates if d >= today]
        next_nfp = min(upcoming_nfp) if upcoming_nfp else None

        hist = yf.Ticker("QQQ").history(period="2y")["Close"]
        hist.index = pd.to_datetime(hist.index).date
        daily_ret = hist.pct_change().dropna()

        past_fomc = [d for d in fomc_dates if d < today]
        fomc_moves = [abs(daily_ret.get(d)) for d in past_fomc if d in daily_ret.index and daily_ret.get(d) is not None]
        avg_fomc_move = float(np.mean(fomc_moves) * 100) if fomc_moves else None

        past_nfp = [f.date() for f in get_first_fridays(pd.Timestamp(today) - pd.Timedelta(days=400), 13)]
        nfp_moves = [abs(daily_ret.get(d)) for d in past_nfp if d in daily_ret.index and daily_ret.get(d) is not None]
        avg_nfp_move = float(np.mean(nfp_moves) * 100) if nfp_moves else None

        return {
            "next_fomc": next_fomc.strftime("%Y-%m-%d") if next_fomc else None,
            "next_nfp": next_nfp.strftime("%Y-%m-%d") if next_nfp else None,
            "avg_fomc_move_pct": round(avg_fomc_move, 2) if avg_fomc_move else None,
            "avg_nfp_move_pct": round(avg_nfp_move, 2) if avg_nfp_move else None,
            "n_fomc_sample": len(fomc_moves), "n_nfp_sample": len(nfp_moves),
        }
    except Exception as e:
        log_error(f"Event calendar fetch failed (non-fatal): {e}")
        return None

# ---------------------------------------------------------------------------
# 21. ADAPTIVE Z-SCORE ANOMALY DETECTION (against each metric's own history)
# ---------------------------------------------------------------------------
def compute_zscore(current, history_values, min_samples=8):
    vals = [v for v in history_values if v is not None]
    if len(vals) < min_samples:
        return None
    mean, std = np.mean(vals), np.std(vals)
    if std == 0:
        return 0.0
    return (current - mean) / std


def get_anomaly_data(history, current):
    """Compares today's values against the distribution of everything logged
    so far. Needs ~8+ days of history to say anything meaningful -- returns
    an explicit 'insufficient history' flag rather than a misleadingly
    confident number early on."""
    past = history[:-1] if history else []  # exclude today's just-appended entry
    metrics = {
        "Net GEX ($M)": (current.get("net_gex_m"), [h.get("net_gex_m") for h in past]),
        "VIX": (current.get("vix"), [h.get("vix") for h in past]),
        "ATM IV (%)": (current.get("atm_iv_pct"), [h.get("atm_iv_pct") for h in past]),
        "COT % OI": (current.get("cot_pct_oi"), [h.get("cot_pct_oi") for h in past]),
    }
    results = []
    for name, (cur_val, hist_vals) in metrics.items():
        if cur_val is None:
            continue
        z = compute_zscore(cur_val, hist_vals)
        results.append({"name": name, "z": round(z, 2) if z is not None else None,
                         "n_samples": len([v for v in hist_vals if v is not None])})
    return results


# ---------------------------------------------------------------------------
# 22. "WHAT CHANGED SINCE YESTERDAY" DIFF VIEW
# ---------------------------------------------------------------------------
def get_diff_view(history):
    if len(history) < 2:
        return None
    today, yesterday = history[-1], history[-2]
    fields = [
        ("Net GEX ($M)", "net_gex_m", "{:+.1f}"),
        ("Spot", "spot", "{:+.2f}"),
        ("VIX", "vix", "{:+.2f}"),
        ("ATM IV (%)", "atm_iv_pct", "{:+.2f}"),
        ("COT % OI", "cot_pct_oi", "{:+.1f}"),
        ("Hurst (H)", "hurst_h", "{:+.3f}"),
    ]
    diffs = []
    for label, key, fmt in fields:
        t, y = today.get(key), yesterday.get(key)
        if t is not None and y is not None:
            diffs.append({"label": label, "diff": fmt.format(t - y), "from": y, "to": t})
    regime_fields = [("Vol Regime", "vix_regime"), ("Gamma Regime", "gex_regime"), ("Hurst Regime", "hurst_regime")]
    for label, key in regime_fields:
        t, y = today.get(key), yesterday.get(key)
        if t is not None and y is not None and t != y:
            diffs.append({"label": label, "diff": f"{y} \u2192 {t}", "from": y, "to": t})
    return {"yesterday_date": yesterday["date"], "diffs": diffs}


# ---------------------------------------------------------------------------
# 23. GAMMA VELOCITY (rate of change of dealer positioning)
# ---------------------------------------------------------------------------
def get_gamma_velocity(history):
    vals = [h.get("net_gex_m") for h in history if h.get("net_gex_m") is not None]
    if len(vals) < 4:
        return None
    recent = vals[-4:]
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    velocity = float(np.mean(diffs))
    direction = "Toward Amplifying" if velocity < -1 else "Toward Dampening" if velocity > 1 else "Stable"
    return {"velocity": round(velocity, 2), "direction": direction}


# ---------------------------------------------------------------------------
# 24. MINI VOLATILITY SURFACE (IV across strikes x expiries)
# ---------------------------------------------------------------------------
def get_vol_surface_data(spot):
    try:
        tk = yf.Ticker("QQQ")
        today = datetime.now(timezone.utc).date()
        usable = [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days) for e in tk.options]
        usable = [u for u in usable if u[1] >= 1][:3]
        if not usable:
            return None

        offsets = [-0.10, -0.05, 0.0, 0.05, 0.10]
        rows = []
        for exp, dte in usable:
            chain = tk.option_chain(exp)
            calls, puts = chain.calls, chain.puts
            row_ivs = []
            for off in offsets:
                target = spot * (1 + off)
                if off < 0:
                    src = puts
                elif off > 0:
                    src = calls
                else:
                    src = pd.concat([calls, puts])
                if len(src) == 0:
                    row_ivs.append(None)
                    continue
                nearest = src.iloc[(src["strike"] - target).abs().argsort()[:1]]
                iv = nearest["impliedVolatility"].values[0] if len(nearest) else None
                row_ivs.append(round(iv * 100, 1) if iv else None)
            rows.append({"expiry": f"{dte}d", "ivs": row_ivs})
        return {"offsets": [f"{int(o*100):+d}%" for o in offsets], "rows": rows}
    except Exception as e:
        log_error(f"Vol surface fetch failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# 25. DEVIL'S ADVOCATE COUNTER-NARRATIVE
# ---------------------------------------------------------------------------
def get_devils_advocate(confluence, gex, vol):
    if not confluence or not confluence.get("parts"):
        return "Not enough signals available today to construct a counter-case."

    score = confluence["score"]
    opposing = []
    for part in confluence["parts"]:
        try:
            val_str = part.split(":")[-1].strip()
            val = float(val_str.replace("+", ""))
            if (score > 0 and val < 0) or (score < 0 and val > 0):
                opposing.append(part)
        except (ValueError, IndexError):
            continue

    if opposing:
        return "Counter-evidence: " + "; ".join(opposing) + ". If this component strengthens, the current lean weakens."

    vix_regime = "Elevated" if vol["vix_now"] > vol["vix_avg5y"] * 1.15 else "Low" if vol["vix_now"] < vol["vix_avg5y"] * 0.85 else "Normal"
    if abs(score) < 15:
        return "No component strongly opposes the read, but the score itself is weak/neutral -- there isn't much conviction to argue against or for."
    return (f"No individual component directly opposes this lean today, which is itself worth distrusting a little -- "
            f"unanimous-looking signals can mean genuine alignment, or can mean the underlying drivers aren't actually independent. "
            f"Current backdrop: {vix_regime} volatility -- a regime shift here would likely move every component at once.")


# ---------------------------------------------------------------------------
# 26. LEAD-LAG CROSS-CORRELATION (does yesterday's move predict today's?)
# ---------------------------------------------------------------------------
def get_lead_lag_data():
    try:
        tickers = {"BTC": "BTC-USD", "DXY": "DX-Y.NYB"}
        qqq = yf.Ticker("QQQ").history(period="6mo")["Close"]
        results = {}
        for name, sym in tickers.items():
            try:
                other = yf.Ticker(sym).history(period="6mo")["Close"]
                df = pd.DataFrame({"qqq": qqq, "other": other}).dropna()
                if len(df) < 30:
                    continue
                same_day = df["qqq"].pct_change().corr(df["other"].pct_change())
                lagged = df["qqq"].pct_change().corr(df["other"].pct_change().shift(1))
                results[name] = {"same_day": round(same_day, 2), "lagged": round(lagged, 2)}
            except Exception:
                continue
        return results if results else None
    except Exception as e:
        log_error(f"Lead-lag fetch failed (non-fatal): {e}")
        return None
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NQ/QQQ Quant Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {
    --bg: #0b0e14; --panel: #11151c; --grid: #1f2530; --text: #e6e9ef;
    --muted: #8b95a7; --green: #39ff88; --red: #ff3b5c; --cyan: #22d3ee; --amber: #fbbf24; --purple: #a78bfa;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text); font-family: Consolas, 'Cascadia Code', 'Courier New', monospace;
    margin: 0; padding: 24px; max-width: 1200px; margin-left: auto; margin-right: auto;
  }
  h1 { color: var(--cyan); text-align: center; font-size: 22px; margin-bottom: 4px; }
  .subtitle { text-align: center; color: var(--muted); font-size: 12px; margin-bottom: 20px; }
  .panel {
    background: var(--panel); border: 1px solid var(--grid); border-radius: 10px;
    padding: 18px 20px; margin-bottom: 20px;
  }
  .panel h2 { color: var(--cyan); font-size: 13px; letter-spacing: 1px; margin: 0 0 14px 0; text-transform: uppercase; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px dashed var(--grid); font-size: 14px; }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: var(--muted); }
  .stat-value { color: var(--text); font-weight: bold; }
  .regime-tag { padding: 1px 8px; border-radius: 4px; font-size: 12px; margin-left: 6px; }
  .tag-low, .tag-dampening { background: rgba(57,255,136,0.15); color: var(--green); }
  .tag-elevated, .tag-amplifying { background: rgba(255,59,92,0.15); color: var(--red); }
  .tag-normal { background: rgba(139,149,167,0.2); color: var(--muted); }
  canvas { max-height: 380px; }
  .summary-line { padding: 8px 0; border-bottom: 1px dashed var(--grid); font-size: 13.5px; line-height: 1.5; }
  .summary-line:last-child { border-bottom: none; }
  .bullet { color: var(--cyan); margin-right: 6px; }
  .bottom-line { color: var(--amber); font-weight: bold; }
  .footer { text-align: center; color: var(--muted); font-size: 11px; margin-top: 8px; }
  .clock { font-size: 32px; color: var(--cyan); text-align: center; font-weight: bold; letter-spacing: 2px; }
  .clock-sub { text-align: center; color: var(--muted); font-size: 12px; margin-top: 4px; margin-bottom: 14px; }
  .countdown { text-align: center; font-size: 15px; margin-top: 6px; }
  .corr-bar-bg { background: var(--grid); border-radius: 4px; height: 10px; width: 100%; overflow: hidden; }
  .corr-bar-fill { height: 100%; border-radius: 4px; }
  .alert-banner {
    border-radius: 10px; padding: 14px 20px; margin-bottom: 20px; font-size: 13.5px; font-weight: bold;
  }
  .alert-calm { background: rgba(57,255,136,0.1); border: 1px solid rgba(57,255,136,0.3); color: var(--green); }
  .alert-active { background: rgba(255,59,92,0.12); border: 1px solid rgba(255,59,92,0.4); color: #ffb3c0;
    animation: pulseGlow 2s ease-in-out infinite; }
  @keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 6px rgba(255,59,92,0.15); }
    50% { box-shadow: 0 0 18px rgba(255,59,92,0.45); }
  }
  .alert-item { margin-top: 4px; font-weight: normal; }
  .calc-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; }
  .calc-row label { color: var(--muted); font-size: 13px; }
  .calc-row input {
    background: var(--bg); border: 1px solid var(--grid); color: var(--text); border-radius: 6px;
    padding: 6px 10px; width: 110px; font-family: inherit; font-size: 14px; text-align: right;
  }
  .calc-result { text-align: center; margin-top: 14px; font-size: 20px; color: var(--cyan); font-weight: bold; }
  .legend-row { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: 11px; color: var(--muted); }
  .legend-dot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
  .fvg-item { padding: 5px 0; border-bottom: 1px dashed var(--grid); font-size: 12.5px; }
  .gauge-wrap { text-align: center; }
  .gauge-svg { width: 100%; max-width: 260px; }
  #gaugeArc { transition: stroke-dashoffset 1.2s ease-out, stroke 0.5s; }
  .pine-box {
    background: var(--bg); border: 1px solid var(--grid); border-radius: 6px; padding: 12px;
    font-size: 11.5px; white-space: pre-wrap; color: var(--text); max-height: 220px; overflow-y: auto;
  }
  .copy-btn {
    background: var(--cyan); color: var(--bg); border: none; border-radius: 6px; padding: 8px 16px;
    font-family: inherit; font-weight: bold; cursor: pointer; margin-top: 10px; font-size: 12px;
  }
  .copy-btn:hover { opacity: 0.85; }
  .diff-item { padding: 5px 0; border-bottom: 1px dashed var(--grid); font-size: 12.5px; }
  .diff-item:last-child { border-bottom: none; }
  .zscore-item { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px dashed var(--grid); font-size: 13px; }
  .zscore-item:last-child { border-bottom: none; }
  .zscore-flag { padding: 1px 8px; border-radius: 4px; font-size: 11px; }
  .surface-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }
  .surface-table th, .surface-table td { padding: 8px 6px; text-align: center; border: 1px solid var(--grid); }
  .surface-table th { color: var(--muted); font-weight: normal; }
  .advocate-box { background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.25); border-radius: 6px; padding: 12px; font-size: 12.5px; line-height: 1.6; }
  .fvg-item:last-child { border-bottom: none; }
  .tab-bar { display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--grid); padding-bottom: 0; }
  .tab-btn {
    background: transparent; border: none; color: var(--muted); font-family: inherit;
    font-size: 13px; font-weight: bold; padding: 10px 18px; cursor: pointer;
    border-bottom: 2px solid transparent; letter-spacing: 0.5px;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { color: var(--cyan); border-bottom: 2px solid var(--cyan); }
  .glossary-section { margin-bottom: 22px; }
  .glossary-section h3 { color: var(--cyan); font-size: 14px; margin: 0 0 8px 0; }
  .glossary-section p { color: var(--text); font-size: 13px; line-height: 1.65; margin: 0 0 8px 0; }
  .glossary-section .g-caveat { color: var(--amber); font-size: 12px; }
</style>
</head>
<body>

<div class="tab-bar">
  <button class="tab-btn active" id="tabBtnDashboard" onclick="switchTab('dashboard')">Dashboard</button>
  <button class="tab-btn" id="tabBtnGlossary" onclick="switchTab('glossary')">How to Read This Dashboard</button>
</div>

<div id="dashboardView">

<h1>DAILY NQ / QQQ QUANT DASHBOARD</h1>
<div class="subtitle">__DATE_STR__ &nbsp;|&nbsp; Generated __TIME_STR__ &nbsp;|&nbsp; Free proxy via QQQ options</div>

<div class="alert-banner __ALERT_CLASS__">__ALERT_HTML__</div>

<div class="two-col">
  <div class="panel">
    <h2>Composite Confluence Score</h2>
    <div class="gauge-wrap">
      <svg viewBox="0 0 200 110" class="gauge-svg">
        <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="#1f2530" stroke-width="14"/>
        <path id="gaugeArc" d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="#22d3ee" stroke-width="14"
              stroke-dasharray="283" stroke-dashoffset="283"/>
        <text x="100" y="85" text-anchor="middle" fill="#e6e9ef" font-size="26" font-weight="bold" id="gaugeText">0</text>
      </svg>
      <div class="clock-sub" id="gaugeLabel">__CONFLUENCE_LABEL__</div>
    </div>
    <div class="clock-sub" style="margin-top:10px;">__CONFLUENCE_PARTS__</div>
    <div class="advocate-box" style="margin-top:14px;"><b>Devil's Advocate:</b> __ADVOCATE_TEXT__</div>
  </div>

  <div class="panel">
    <h2>Regime Radar</h2>
    <canvas id="radarChart" style="max-height:280px;"></canvas>
  </div>
</div>

<div class="two-col">
  <div class="panel">
    <h2>What Changed Since Yesterday</h2>
    __DIFF_HTML__
  </div>

  <div class="panel">
    <h2>Adaptive Anomaly Detection (z-scores)</h2>
    __ANOMALY_HTML__
  </div>
</div>

<div class="two-col">
  <div class="panel">
    <h2>Gamma Velocity</h2>
    __VELOCITY_HTML__
  </div>

  <div class="panel">
    <h2>Lead-Lag Cross-Correlation (vs QQQ)</h2>
    __LEADLAG_HTML__
  </div>
</div>

<div class="panel">
  <h2>Mini Volatility Surface (IV %, by strike offset and expiry)</h2>
  __SURFACE_HTML__
</div>

<div class="panel">
  <h2>ICT Killzone Tracker (live)</h2>
  <div class="clock" id="etClock">--:--:-- ET</div>
  <div class="clock-sub" id="sessionStatus">loading session status...</div>
  <div class="countdown" id="nextSession"></div>
  <div class="two-col" style="margin-top:16px;">
    <div class="stat-row"><span class="stat-label">Overnight High (Asia+London)</span>
      <span class="stat-value">__ON_HIGH__</span></div>
    <div class="stat-row"><span class="stat-label">Overnight Low (Asia+London)</span>
      <span class="stat-value">__ON_LOW__</span></div>
  </div>
</div>

<div class="two-col">
  <div class="panel">
    <h2>Reference Levels</h2>
    <div class="stat-row"><span class="stat-label">Prev Day High / Low</span>
      <span class="stat-value">__PDH__ / __PDL__</span></div>
    <div class="stat-row"><span class="stat-label">Weekly True Open</span>
      <span class="stat-value">__WEEKLY_OPEN__</span></div>
    <div class="stat-row"><span class="stat-label">Monthly True Open</span>
      <span class="stat-value">__MONTHLY_OPEN__</span></div>
    <div class="stat-row"><span class="stat-label">14d ATR / Today's Expected Range</span>
      <span class="stat-value">__ATR_VAL__ (__RANGE_LOW__ - __RANGE_HIGH__)</span></div>
  </div>

  <div class="panel">
    <h2>Position Size Calculator</h2>
    <div class="calc-row"><label>Account Risk ($)</label><input type="number" id="calcRisk" value="200"></div>
    <div class="calc-row"><label>Stop Distance (pts)</label><input type="number" id="calcStop" value="__ATR_RAW__"></div>
    <div class="calc-row"><label>$ per point per contract</label><input type="number" id="calcMultiplier" value="2"></div>
    <div class="calc-result">Suggested size: <span id="calcResult">0</span> contracts</div>
  </div>
</div>

<div class="panel">
  <h2>Intraday Price with Levels Overlaid</h2>
  <canvas id="priceChart"></canvas>
  <div class="legend-row">
    <span><span class="legend-dot" style="background:#22d3ee;"></span>Session VWAP</span>
    <span><span class="legend-dot" style="background:#39ff88;"></span>Call Wall</span>
    <span><span class="legend-dot" style="background:#ff3b5c;"></span>Put Wall</span>
    <span><span class="legend-dot" style="background:#fbbf24;"></span>Gamma Flip</span>
    <span><span class="legend-dot" style="background:#a78bfa;"></span>Prev Day High/Low</span>
    <span><span class="legend-dot" style="background:#8b95a7;"></span>ATR Range</span>
    <span><span class="legend-dot" style="background:rgba(57,255,136,0.5);"></span>Bullish FVG zone</span>
    <span><span class="legend-dot" style="background:rgba(255,59,92,0.5);"></span>Bearish FVG zone</span>
  </div>
</div>

<div class="panel">
  <h2>Gamma Exposure by Strike ("the gamma rays") &mdash; expiry __EXPIRY__, __DTE__d</h2>
  <canvas id="gexChart"></canvas>
</div>

<div class="two-col">
  <div class="panel">
    <h2>Volatility &amp; Gamma Regime</h2>
    <div class="stat-row"><span class="stat-label">Spot (QQQ)</span>
      <span class="stat-value animated-number" data-target="__SPOT__" data-decimals="2" data-prefix="$">$0.00</span></div>
    <div class="stat-row"><span class="stat-label">VIX now</span>
      <span class="stat-value"><span class="animated-number" data-target="__VIX_NOW__" data-decimals="2">0.00</span>
      <span class="regime-tag tag-__VIX_REGIME_CLASS__">__VIX_REGIME__</span></span></div>
    <div class="stat-row"><span class="stat-label">QQQ HV20 / HV60</span>
      <span class="stat-value"><span class="animated-number" data-target="__HV20__" data-decimals="1" data-suffix="%">0%</span> /
      <span class="animated-number" data-target="__HV60__" data-decimals="1" data-suffix="%">0%</span></span></div>
    <div class="stat-row"><span class="stat-label">Net GEX</span>
      <span class="stat-value"><span class="animated-number" data-target="__NET_GEX_M__" data-decimals="1" data-prefix="$" data-suffix="M">$0M</span>
      <span class="regime-tag tag-__GEX_REGIME_CLASS__">__GEX_REGIME__</span></span></div>
    <div class="stat-row"><span class="stat-label">Call Wall / Put Wall</span>
      <span class="stat-value">__CALL_WALL__ / __PUT_WALL__</span></div>
  </div>

  <div class="panel">
    <h2>0DTE Gamma, Max Pain &amp; Put/Call Ratio</h2>
    __ZERO_DTE_HTML__
  </div>
</div>

<div class="two-col">
  <div class="panel">
    <h2>CFTC Positioning (Nasdaq-100)</h2>
    __COT_HTML__
  </div>

  <div class="panel">
    <h2>Cross-Asset Correlation (60d, vs QQQ)</h2>
    __CORR_HTML__
  </div>
</div>

<div class="two-col">
  <div class="panel">
    <h2>Vanna &amp; Charm Exposure</h2>
    __VEX_CHEX_HTML__
  </div>

  <div class="panel">
    <h2>Volatility Risk Premium &amp; Skew</h2>
    __VRP_SKEW_HTML__
  </div>
</div>

<div class="panel">
  <h2>Gamma Exposure by Expiry</h2>
  <canvas id="expiryChart"></canvas>
</div>

<div class="two-col">
  <div class="panel">
    <h2>Hurst Exponent Regime</h2>
    __HURST_HTML__
  </div>

  <div class="panel">
    <h2>Forward Delta Projection (charm, price/IV held constant)</h2>
    <canvas id="forwardChart" style="max-height:220px;"></canvas>
  </div>
</div>

<div class="two-col">
  <div class="panel">
    <h2>Event Calendar &amp; Historical Reaction</h2>
    __EVENT_HTML__
  </div>

  <div class="panel">
    <h2>Pine Script Export (NQ-scaled)</h2>
    <div class="pine-box" id="pineBox">__PINE_TEXT__</div>
    <button class="copy-btn" onclick="copyPineScript()">Copy to Clipboard</button>
  </div>
</div>

<div class="panel">
  <h2>Net GEX History vs Price __HISTORY_NOTE__</h2>
  <canvas id="historyChart"></canvas>
</div>

<div class="panel">
  <h2>Fair Value Gaps &amp; Order Blocks (heuristic, intraday 5m)</h2>
  __FVG_OB_HTML__
</div>

<div class="panel">
  <h2>QQQ Seasonality &mdash; Avg Monthly Return (live-computed, __N_YEARS__ years)</h2>
  <canvas id="seasonChart"></canvas>
</div>

<div class="panel">
  <h2>Quantitative Research Summary</h2>
  __SUMMARY_HTML__
</div>

<div class="footer">Free data via yfinance (QQQ proxy) &amp; CFTC public API. Not investment advice. FVG/OB detection is a simple heuristic, not rigorous ICT analysis.</div>

</div>

<div id="glossaryView" style="display:none;">

<h1>HOW TO READ THIS DASHBOARD</h1>
<div class="subtitle">Every panel explained, in the order they appear on the Dashboard tab</div>

<div class="panel">
<div class="glossary-section">
<h3>The Big Picture</h3>
<p>Everything here is built from <b>free</b> data: yfinance (QQQ options, price history, VIX, SPY/DXY/BTC) and the CFTC's public API. QQQ (the Nasdaq-100 ETF) stands in for NQ/NDX throughout, since free options data for the actual futures contract doesn't exist anywhere. It's a close, genuinely correlated proxy -- not an identical one.</p>
<p class="g-caveat">Nothing on this page is investment advice or a signal to trade. It's context to sit alongside your own CRT/HTF framework, not a replacement for it.</p>
</div>

<div class="glossary-section">
<h3>ICT Killzone Tracker</h3>
<p>A live, second-by-second clock in Eastern Time, plus which killzone (Asian/London/NY AM/NY PM) is active right now and a countdown to the next one. This part runs entirely in your browser's JavaScript, so it keeps ticking correctly all day even though the rest of the dashboard's data is a frozen snapshot from whenever the script last ran.</p>
<p>Below the clock: the overnight high/low (Asia+London session), included as a liquidity-level proxy. <span class="g-caveat">QQQ's overnight liquidity is much thinner than NQ futures' near-24-hour liquidity, so treat this range as rough, not exact.</span></p>
</div>

<div class="glossary-section">
<h3>Reference Levels &amp; Position Size Calculator</h3>
<p>Previous Day High/Low, Weekly True Open, and Monthly True Open -- classic reference levels. Weekly/Monthly Open uses the regular 9:30am ET session open, not the ICT-purist Sunday-futures-open definition, since QQQ doesn't trade Sundays.</p>
<p>The calculator is fully interactive: type in your account risk in dollars, your stop distance in points, and the dollar value per point per contract (defaults to MNQ's $2/point) -- it recalculates your suggested position size live as you type.</p>
</div>

<div class="glossary-section">
<h3>Intraday Price Chart</h3>
<p>Today's QQQ price with several layers overlaid: session VWAP, Call Wall / Put Wall / Gamma Flip (dashed lines), Previous Day High/Low (thin purple), the ATR expected range (gray dashed), and shaded green/red zones for detected Fair Value Gaps.</p>
</div>

<div class="glossary-section">
<h3>Gamma Exposure by Strike ("The Gamma Rays")</h3>
<p>For every strike in the nearest QQQ options expiry, this computes how much delta-hedging flow dealers would need to do if price moved there, using the Black-Scholes gamma formula applied to each strike's open interest. Green bars = call gamma, red bars = put gamma. The <b>Gamma Flip</b> (amber dashed line) is where cumulative gamma exposure crosses zero -- above it, dealer hedging tends to dampen volatility; below it, dealer hedging tends to amplify it. <b>Call Wall</b> and <b>Put Wall</b> mark the strikes with the largest gamma concentration on each side, acting as informal magnets/resistance.</p>
<p class="g-caveat">This assumes dealers are net long calls and short puts sold to them -- the standard retail approximation, not a certainty about actual dealer books.</p>
</div>

<div class="glossary-section">
<h3>Volatility &amp; Gamma Regime</h3>
<p>VIX compared to its own 5-year average (Low/Normal/Elevated), 20-day and 60-day realized volatility (HV20/HV60), Net GEX with a Dampening/Amplifying tag, and the Call Wall/Put Wall levels again for quick reference.</p>
</div>

<div class="glossary-section">
<h3>0DTE Gamma, Max Pain &amp; Put/Call Ratio</h3>
<p><b>0DTE</b> = options expiring today specifically, which dominate same-day pinning behavior far more than the weekly view above. <b>Max Pain</b> is the strike where option sellers collectively lose the least money if price settles there at expiry -- a classic "pin" level. <b>Put/Call OI Ratio</b> is total put open interest divided by total call open interest, a rough sentiment gauge.</p>
</div>

<div class="glossary-section">
<h3>CFTC Positioning</h3>
<p>The most recent Commitment of Traders report for E-mini Nasdaq-100 futures: how leveraged/non-commercial funds are positioned, net, as a percentage of total open interest. Deeply negative = crowded short (raises squeeze-up risk); deeply positive = crowded long (raises squeeze-down risk). This is real, official, weekly-updated government data.</p>
</div>

<div class="glossary-section">
<h3>Cross-Asset Correlation</h3>
<p>How QQQ's daily returns have correlated with SPY, DXY (dollar index), and BTC over the trailing 60 days -- a quick read on the risk-on/risk-off backdrop.</p>
</div>

<div class="glossary-section">
<h3>Composite Confluence Score &amp; Devil's Advocate</h3>
<p>A single -100 to +100 gauge combining CFTC positioning (contrarian), put/call skew, and this month's historical seasonality into one directional lean. It's a hand-weighted heuristic, not a fitted or trained model -- the direction is more meaningful than the exact number.</p>
<p>Directly below it, <b>Devil's Advocate</b> automatically checks whether any of the score's own components actually disagree with its overall lean, and calls that out explicitly. If nothing disagrees, it says so too -- unanimous-looking signals aren't automatically more trustworthy, they can just mean the underlying drivers aren't independent.</p>
</div>

<div class="glossary-section">
<h3>Regime Radar</h3>
<p>A six-axis radar chart -- Gamma, Volatility, Positioning, Hurst, Skew, Seasonality -- where every axis represents <i>intensity/magnitude</i> (0-100), not direction. A big, spread-out shape means today is extreme across several dimensions at once; a small tight shape means a quiet, unremarkable day everywhere.</p>
</div>

<div class="glossary-section">
<h3>What Changed Since Yesterday / Anomaly Detection / Gamma Velocity</h3>
<p>These three all read from a small history log (<code>gex_history.json</code>) that gets one new entry appended every time the script runs, so they need several days of accumulated history before they say much.</p>
<p><b>What Changed</b> is a literal diff against yesterday's logged snapshot. <b>Anomaly Detection</b> computes a z-score for key metrics against their own recent distribution (needs 8+ days) and flags genuine statistical outliers (|z| &gt; 2), rather than using fixed thresholds picked once and never revisited. <b>Gamma Velocity</b> is the rate of change of Net GEX over the last few logged days -- is dealer positioning accelerating toward Amplifying or Dampening.</p>
</div>

<div class="glossary-section">
<h3>Lead-Lag Cross-Correlation</h3>
<p>Checks whether BTC's or DXY's move <i>yesterday</i> tends to predict QQQ's move <i>today</i>, separately from same-day correlation. <span class="g-caveat">Correlations measured over a 6-month window can shift a lot -- treat this as one data point, not a rule.</span></p>
</div>

<div class="glossary-section">
<h3>Mini Volatility Surface</h3>
<p>Implied volatility across five strike offsets (-10% to +10% from spot) and up to three expiries (This Week / Next Week / Monthly), shown as a small table. Richer than a single skew number -- shows how the whole curve is shaped, not just one comparison.</p>
</div>

<div class="glossary-section">
<h3>Vanna &amp; Charm Exposure (VEX/CHEX)</h3>
<p>Second-order Greeks, computed from the same option chain as the gamma numbers. <b>Vanna</b> captures how dealer hedging shifts when volatility itself moves, not just price. <b>Charm</b> captures how hedging shifts purely from time passing, even if nothing else changes. Both were numerically verified against finite-difference derivatives of Black-Scholes delta before being trusted here.</p>
<p class="g-caveat">These assume no dividend yield (QQQ does pay a small one) and there's no single universal sign convention across the industry for these -- the ones here match the same "calls positive, puts negative" convention used for gamma, for internal consistency.</p>
</div>

<div class="glossary-section">
<h3>Volatility Risk Premium &amp; Skew</h3>
<p><b>VRP</b> is at-the-money implied volatility minus 20-day realized volatility -- positive means options are pricing in more movement than has actually been happening (options "rich"); negative means the opposite ("cheap"). <b>Put/Call Skew</b> compares IV of a ~5%-OTM put against a ~5%-OTM call -- positive skew (the normal pattern) means the market prices downside moves as more likely/severe than equivalent upside ones.</p>
</div>

<div class="glossary-section">
<h3>Gamma Exposure by Expiry</h3>
<p>Net GEX broken out separately for This Week, Next Week, and the Monthly expiry, instead of only the single nearest one shown in the main gamma chart -- reveals whether dealer positioning is concentrated near-term or spread across time.</p>
</div>

<div class="glossary-section">
<h3>Hurst Exponent Regime</h3>
<p>A statistical measure of whether the market's recent price action behaves as trending, mean-reverting, or a pure random walk -- independent of anything options-related. H &gt; 0.55 = Trending, H &lt; 0.45 = Mean-Reverting, in between = Random-ish. Computed via proper multi-window R/S regression (not a naive single-window estimate, which is mathematically biased and was caught and fixed during development).</p>
</div>

<div class="glossary-section">
<h3>Forward Delta Projection</h3>
<p>Projects net dollar delta exposure forward 1-5 days using the Charm value, under one explicit assumption: price and implied volatility both stay exactly where they are right now. It isolates the pure time-decay effect in isolation -- <span class="g-caveat">it is not a price forecast, since price essentially never actually stays constant.</span></p>
</div>

<div class="glossary-section">
<h3>Event Calendar &amp; Historical Reaction</h3>
<p>Upcoming FOMC dates (sourced directly from federalreserve.gov) and NFP dates (computed via the reliable "first Friday of the month" rule), each paired with the average QQQ move on real past occurrences of that event, computed from actual price history.</p>
<p class="g-caveat">CPI dates are deliberately left out -- unlike FOMC and NFP, there's no verified free source for the exact future release schedule, so it was omitted rather than guessed.</p>
</div>

<div class="glossary-section">
<h3>Pine Script Export</h3>
<p>Auto-generates a ready-to-paste Pine Script v6 indicator with today's key levels (Gamma Flip, Call/Put Wall, ATR range, PDH/PDL, Weekly/Monthly Open, Max Pain, Overnight High/Low) plus a live stats table (Net GEX, VEX, CHEX, Skew, VIX, Hurst, COT, Confluence Score) for TradingView. Since these levels are computed from QQQ, they're scaled by today's actual NQ/QQQ price ratio (fetched fresh each run) before export, so they land in NQ's real price range instead of QQQ's.</p>
</div>

<div class="glossary-section">
<h3>Net GEX History vs Price</h3>
<p>A running log, one point per day, of Net GEX plotted against QQQ's price on a second axis -- shows whether dealer positioning has been trending toward long or short gamma over time, not just today's snapshot.</p>
</div>

<div class="glossary-section">
<h3>Fair Value Gaps &amp; Order Blocks</h3>
<p>A simple heuristic pattern-matcher scanning intraday 5-minute candles: FVGs are the classic 3-candle imbalance pattern; Order Blocks flag the last opposite-colored candle before a move at least 1.5x its size. <span class="g-caveat">This is a basic rule-based scanner, not rigorous ICT analysis -- it doesn't weigh displacement, mitigation status, or higher-timeframe context the way a trained eye would. Treat it as a rough visual aid.</span></p>
</div>

<div class="glossary-section">
<h3>Seasonality</h3>
<p>Average monthly return for QQQ, computed live from its full available price history (not a fixed/hardcoded lookup) -- the current month is highlighted. A real but genuinely noisy signal; the exact numbers can shift meaningfully depending on how many years you include.</p>
</div>

<div class="glossary-section">
<h3>Quantitative Research Summary &amp; Alert Banner</h3>
<p>The Research Summary is a plain-English synthesis of the gamma regime, volatility regime, seasonality, and positioning reads, generated fresh from that day's actual numbers, ending in one bottom-line sentence. The Alert Banner at the very top of the Dashboard tab turns red and pulses only when specific conditions stack up (e.g. elevated vol + amplifying gamma together, or crowded positioning past a threshold) -- a calm green banner means nothing unusual tripped today, not that nothing was checked.</p>
</div>

<div class="glossary-section">
<h3>Behind the Scenes</h3>
<p>This whole dashboard is generated once per run by a Python script, scheduled to run automatically. Every external data fetch is wrapped so a single failure (a missing 0DTE expiry, a rate-limited request) degrades that one panel gracefully to "data unavailable" instead of breaking the page. Every run, successful or not, gets a timestamped line in <code>dashboard_errors.log</code>, and a full dated copy of that day's page is saved to the <code>archive/</code> folder alongside <code>latest.html</code>.</p>
</div>
</div>

</div>

<script>
function switchTab(tab) {
  document.getElementById('dashboardView').style.display = tab === 'dashboard' ? 'block' : 'none';
  document.getElementById('glossaryView').style.display = tab === 'glossary' ? 'block' : 'none';
  document.getElementById('tabBtnDashboard').classList.toggle('active', tab === 'dashboard');
  document.getElementById('tabBtnGlossary').classList.toggle('active', tab === 'glossary');
}
// ---------- animated count-up numbers ----------
function animateNumber(el, target, decimals, duration, prefix, suffix) {
  const startTime = performance.now();
  function tick(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = target * eased;
    el.textContent = prefix + value.toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
document.querySelectorAll('.animated-number').forEach(el => {
  const target = parseFloat(el.dataset.target);
  const decimals = parseInt(el.dataset.decimals || '1');
  animateNumber(el, isNaN(target) ? 0 : target, decimals, 1100, el.dataset.prefix || '', el.dataset.suffix || '');
});

// ---------- position size calculator (live, client-side) ----------
function updateCalc() {
  const risk = parseFloat(document.getElementById('calcRisk').value) || 0;
  const stop = parseFloat(document.getElementById('calcStop').value) || 0;
  const mult = parseFloat(document.getElementById('calcMultiplier').value) || 1;
  const size = (stop > 0 && mult > 0) ? risk / (stop * mult) : 0;
  document.getElementById('calcResult').textContent = size.toFixed(2);
}
['calcRisk', 'calcStop', 'calcMultiplier'].forEach(id => {
  document.getElementById(id).addEventListener('input', updateCalc);
});
updateCalc();

// ---------- LIVE killzone clock + countdown ----------
const KILLZONES_JS = [
  {name: "Asian Killzone",  startH: 20, startM: 0,  endH: 0,  endM: 0},
  {name: "London Killzone", startH: 2,  startM: 0,  endH: 5,  endM: 0},
  {name: "NY AM Killzone",  startH: 7,  startM: 0,  endH: 10, endM: 0},
  {name: "NY PM Killzone",  startH: 13, startM: 30, endH: 16, endM: 0},
];
function getETParts() {
  const fmt = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York', hour12: false,
    hour: 'numeric', minute: 'numeric', second: 'numeric'
  });
  const parts = fmt.formatToParts(new Date());
  const get = (t) => parseInt(parts.find(p => p.type === t).value);
  return { h: get('hour') % 24, m: get('minute'), s: get('second') };
}
function updateKillzoneClock() {
  const { h, m, s } = getETParts();
  const nowMin = h * 60 + m;
  document.getElementById('etClock').textContent =
    String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0') + ' ET';
  let active = null;
  for (const kz of KILLZONES_JS) {
    const start = kz.startH * 60 + kz.startM, end = kz.endH * 60 + kz.endM;
    if (start > end) { if (nowMin >= start || nowMin < end) active = kz.name; }
    else { if (nowMin >= start && nowMin < end) active = kz.name; }
  }
  let best = null, bestDiff = Infinity;
  for (const kz of KILLZONES_JS) {
    const start = kz.startH * 60 + kz.startM;
    let diff = (start - nowMin + 1440) % 1440;
    if (diff === 0) diff = 1440;
    if (diff < bestDiff) { bestDiff = diff; best = kz.name; }
  }
  const nh = Math.floor(bestDiff / 60), nm = bestDiff % 60;
  document.getElementById('sessionStatus').innerHTML = active
    ? `<span class="regime-tag tag-dampening">ACTIVE: ${active}</span>`
    : `<span class="regime-tag tag-normal">No active killzone right now</span>`;
  document.getElementById('nextSession').textContent = `Next: ${best} in ${nh}h ${nm}m`;
}
updateKillzoneClock();
setInterval(updateKillzoneClock, 1000);

// ---------- data blobs ----------
const priceData = __PRICE_JSON__;
const callWallVal = __CALL_WALL_JS__;
const putWallVal = __PUT_WALL_JS__;
const gammaFlipVal2 = __GAMMA_FLIP_JS__;
const pdhVal = __PDH_JS__;
const pdlVal = __PDL_JS__;
const atrHighVal = __RANGE_HIGH_JS__;
const atrLowVal = __RANGE_LOW_JS__;
const fvgObData = __FVG_OB_JSON__;

// ---------- Confluence gauge animation ----------
const confluenceScore = __CONFLUENCE_SCORE__;  // -100 to +100
(function animateGauge() {
  const circumference = 283;
  const pct = (confluenceScore + 100) / 200;  // 0 to 1
  const offset = circumference * (1 - pct);
  const arc = document.getElementById('gaugeArc');
  const color = confluenceScore > 15 ? '#39ff88' : confluenceScore < -15 ? '#ff3b5c' : '#8b95a7';
  setTimeout(() => {
    arc.style.strokeDashoffset = offset;
    arc.style.stroke = color;
  }, 100);
  let start = null;
  function tick(ts) {
    if (!start) start = ts;
    const progress = Math.min((ts - start) / 1200, 1);
    document.getElementById('gaugeText').textContent = Math.round(confluenceScore * progress);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();

// ---------- Regime radar chart ----------
const radarData = __RADAR_JSON__;
if (radarData) {
  new Chart(document.getElementById('radarChart'), {
    type: 'radar',
    data: {
      labels: radarData.labels,
      datasets: [{
        label: "Today's Intensity", data: radarData.values,
        borderColor: '#22d3ee', backgroundColor: 'rgba(34,211,238,0.2)',
        pointBackgroundColor: '#22d3ee'
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#e6e9ef' } } },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { color: '#8b95a7', backdropColor: 'transparent' },
          grid: { color: '#1f2530' }, angleLines: { color: '#1f2530' },
          pointLabels: { color: '#e6e9ef', font: { size: 11 } }
        }
      }
    }
  });
}

// ---------- Forward delta projection chart ----------
const forwardData = __FORWARD_JSON__;
if (forwardData) {
  new Chart(document.getElementById('forwardChart'), {
    type: 'line',
    data: {
      labels: forwardData.days.map(d => d === 0 ? 'Today' : `+${d}d`),
      datasets: [{
        label: 'Projected Net Delta Exposure ($M)', data: forwardData.values,
        borderColor: '#a78bfa', backgroundColor: 'rgba(167,139,250,0.1)', fill: true, tension: 0.2
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8b95a7' }, grid: { display: false } },
        y: { ticks: { color: '#8b95a7' }, grid: { color: '#1f2530' } }
      }
    }
  });
} else {
  document.getElementById('forwardChart').replaceWith(document.createTextNode('Projection unavailable this run.'));
}

// ---------- Pine Script copy button ----------
function copyPineScript() {
  const text = document.getElementById('pineBox').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    const original = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = original; }, 1500);
  });
}

// ---------- price chart plugins: reference lines + FVG zones ----------
const horizontalLinePlugin = {
  id: 'horizontalLines',
  afterDatasetsDraw(chart) {
    if (chart.canvas.id !== 'priceChart') return;
    const { ctx, chartArea: { left, right }, scales: { y } } = chart;
    ctx.save();
    function drawLine(value, color, dash, label) {
      if (value === null || value === undefined || isNaN(value)) return;
      const yPix = y.getPixelForValue(value);
      ctx.beginPath();
      ctx.moveTo(left, yPix);
      ctx.lineTo(right, yPix);
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = color;
      ctx.setLineDash(dash);
      ctx.stroke();
      if (label) {
        ctx.fillStyle = color;
        ctx.font = '10px monospace';
        ctx.fillText(label, left + 6, yPix - 3);
      }
    }
    drawLine(callWallVal, '#39ff88', [5,4], 'Call Wall');
    drawLine(putWallVal, '#ff3b5c', [5,4], 'Put Wall');
    drawLine(gammaFlipVal2, '#fbbf24', [2,2], 'Gamma Flip');
    drawLine(pdhVal, '#a78bfa', [1,3], null);
    drawLine(pdlVal, '#a78bfa', [1,3], null);
    drawLine(atrHighVal, '#8b95a7', [6,6], null);
    drawLine(atrLowVal, '#8b95a7', [6,6], null);
    ctx.restore();
  }
};
Chart.register(horizontalLinePlugin);

const fvgZonePlugin = {
  id: 'fvgZones',
  beforeDatasetsDraw(chart) {
    if (chart.canvas.id !== 'priceChart' || !fvgObData) return;
    const { ctx, chartArea: { right }, scales: { x, y } } = chart;
    ctx.save();
    (fvgObData.fvgs || []).forEach(z => {
      const xPix = x.getPixelForValue(z.index);
      const yTop = y.getPixelForValue(z.top);
      const yBot = y.getPixelForValue(z.bottom);
      ctx.fillStyle = z.type === 'bullish' ? 'rgba(57,255,136,0.15)' : 'rgba(255,59,92,0.15)';
      ctx.fillRect(xPix, Math.min(yTop, yBot), right - xPix, Math.abs(yBot - yTop));
    });
    ctx.restore();
  }
};
Chart.register(fvgZonePlugin);

if (priceData) {
  new Chart(document.getElementById('priceChart'), {
    type: 'line',
    data: {
      labels: priceData.times,
      datasets: [
        { label: 'Price', data: priceData.closes, borderColor: '#e6e9ef', borderWidth: 2, pointRadius: 0, tension: 0.1 },
        { label: 'Session VWAP', data: priceData.vwap, borderColor: '#22d3ee', borderWidth: 1.5, borderDash: [3,3], pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { ticks: { color: '#8b95a7', maxTicksLimit: 12 }, grid: { display: false } },
        y: { ticks: { color: '#8b95a7' }, grid: { color: '#1f2530' } }
      },
      plugins: { legend: { labels: { color: '#e6e9ef' } } }
    }
  });
} else {
  document.getElementById('priceChart').replaceWith(document.createTextNode('Intraday price data unavailable this run.'));
}

// ---------- gamma exposure chart ----------
const gexData = __GEX_JSON__;
const spotPrice = __SPOT__;
const verticalLinePlugin = {
  id: 'verticalLines',
  afterDatasetsDraw(chart) {
    if (chart.canvas.id !== 'gexChart') return;
    const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
    ctx.save();
    function drawLine(value, color, dash) {
      if (value === null || value === undefined || isNaN(value)) return;
      const xPix = x.getPixelForValue(value);
      ctx.beginPath(); ctx.moveTo(xPix, top); ctx.lineTo(xPix, bottom);
      ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.setLineDash(dash || []); ctx.stroke();
    }
    drawLine(spotPrice, '#22d3ee', null);
    drawLine(gammaFlipVal2, '#fbbf24', [6, 4]);
    ctx.restore();
  }
};
Chart.register(verticalLinePlugin);
new Chart(document.getElementById('gexChart'), {
  type: 'bar',
  data: {
    datasets: [
      { label: 'Call Gamma', data: gexData.strikes.map((s, i) => ({x: s, y: gexData.call[i]})),
        backgroundColor: '#39ff88', barThickness: 6 },
      { label: 'Put Gamma', data: gexData.strikes.map((s, i) => ({x: s, y: gexData.put[i]})),
        backgroundColor: '#ff3b5c', barThickness: 6 }
    ]
  },
  options: {
    responsive: true, interaction: { mode: 'index', intersect: false },
    scales: {
      x: { type: 'linear', title: { display: true, text: 'Strike', color: '#8b95a7' },
           ticks: { color: '#8b95a7' }, grid: { color: '#1f2530' } },
      y: { title: { display: true, text: 'Gamma Exposure ($)', color: '#8b95a7' },
           ticks: { color: '#8b95a7' }, grid: { color: '#1f2530' } }
    },
    plugins: {
      legend: { labels: { color: '#e6e9ef' } },
      tooltip: { callbacks: {
        title: (items) => `Strike $${items[0].parsed.x}`,
        label: (item) => `${item.dataset.label}: $${item.parsed.y.toLocaleString(undefined, {maximumFractionDigits: 0})}`
      }}
    }
  }
});

// ---------- seasonality chart ----------
const seasonData = __SEASON_JSON__;
new Chart(document.getElementById('seasonChart'), {
  type: 'bar',
  data: {
    labels: seasonData.labels,
    datasets: [{ data: seasonData.values,
      backgroundColor: seasonData.values.map((v, i) =>
        i === seasonData.current_index ? '#22d3ee' : (v >= 0 ? '#39ff88' : '#ff3b5c')) }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false },
      tooltip: { callbacks: { label: (item) => `Avg return: ${item.parsed.y.toFixed(2)}%` } } },
    scales: {
      x: { ticks: { color: '#8b95a7' }, grid: { display: false } },
      y: { title: { display: true, text: 'Avg return (%)', color: '#8b95a7' },
           ticks: { color: '#8b95a7' }, grid: { color: '#1f2530' } }
    }
  }
});

// ---------- Net GEX history chart (dual-axis: GEX + price) ----------
const historyData = __HISTORY_JSON__;
new Chart(document.getElementById('historyChart'), {
  type: 'line',
  data: {
    labels: historyData.dates,
    datasets: [
      {
        label: 'Net GEX ($M)', data: historyData.values, borderColor: '#22d3ee',
        backgroundColor: 'rgba(34,211,238,0.1)', fill: true, tension: 0.2, pointRadius: 3,
        pointBackgroundColor: historyData.values.map(v => v >= 0 ? '#39ff88' : '#ff3b5c'),
        yAxisID: 'yGex'
      },
      {
        label: 'Spot Price', data: historyData.spots, borderColor: '#fbbf24',
        borderDash: [4,3], borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.2,
        yAxisID: 'yPrice'
      }
    ]
  },
  options: {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { color: '#e6e9ef' } } },
    scales: {
      x: { ticks: { color: '#8b95a7' }, grid: { display: false } },
      yGex: { position: 'left', title: { display: true, text: 'Net GEX ($M)', color: '#8b95a7' },
              ticks: { color: '#8b95a7' }, grid: { color: '#1f2530' } },
      yPrice: { position: 'right', title: { display: true, text: 'Price', color: '#8b95a7' },
                ticks: { color: '#8b95a7' }, grid: { display: false } }
    }
  }
});

// ---------- Gamma exposure by expiry chart ----------
const expiryData = __EXPIRY_JSON__;
if (expiryData) {
  new Chart(document.getElementById('expiryChart'), {
    type: 'bar',
    data: {
      labels: expiryData.labels,
      datasets: [{
        data: expiryData.values,
        backgroundColor: expiryData.values.map(v => v >= 0 ? '#39ff88' : '#ff3b5c')
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (item) => `Net GEX: $${item.parsed.y.toFixed(1)}M` } } },
      scales: {
        x: { ticks: { color: '#8b95a7' }, grid: { display: false } },
        y: { title: { display: true, text: 'Net GEX ($M)', color: '#8b95a7' },
             ticks: { color: '#8b95a7' }, grid: { color: '#1f2530' } }
      }
    }
  });
} else {
  document.getElementById('expiryChart').replaceWith(document.createTextNode('Multi-expiry data unavailable this run.'));
}
</script>
</body>
</html>
"""


def render_html(gex, vol, season, cot, zero_dte, overnight, intraday, corr,
                 ref_levels, atr, history, alerts, fvg_ob, expiry_gex,
                 hurst, forward_proj, confluence, radar, pine_export, event_cal,
                 anomalies, diff_view, velocity, vol_surface, lead_lag):
    df = gex["df"]
    gex_json = json.dumps({
        "strikes": [round(s, 2) for s in df["strike"].tolist()],
        "call": [round(v, 2) for v in df["call_gex"].tolist()],
        "put": [round(v, 2) for v in df["put_gex"].tolist()],
    })

    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    months_present = [m for m in range(1, 13) if m in season.index]
    current_month = datetime.now().month
    season_json = json.dumps({
        "labels": [month_names[m - 1] for m in months_present],
        "values": [round(season.loc[m, "avg_return"] * 100, 3) for m in months_present],
        "current_index": months_present.index(current_month) if current_month in months_present else -1,
    })

    price_json = "null" if intraday is None else json.dumps(intraday)

    vix_regime = ("Elevated" if vol["vix_now"] > vol["vix_avg5y"] * 1.15
                  else "Low" if vol["vix_now"] < vol["vix_avg5y"] * 0.85 else "Normal")
    gex_regime = "Dampening" if gex["net_gex_total"] > 0 else "Amplifying"

    if cot:
        cot_html = (
            f'<div class="stat-row"><span class="stat-label">Report date</span>'
            f'<span class="stat-value">{cot["date"][:10]}</span></div>'
            f'<div class="stat-row"><span class="stat-label">Non-commercial net</span>'
            f'<span class="stat-value animated-number" data-target="{cot["net"]}" data-decimals="0"> contracts</span></div>'
            f'<div class="stat-row"><span class="stat-label">Net as % of OI</span>'
            f'<span class="stat-value animated-number" data-target="{cot["pct_oi"]}" data-decimals="1" data-suffix="%">0%</span></div>'
        )
    else:
        cot_html = '<div class="stat-row"><span class="stat-label">Data unavailable this run</span></div>'

    if zero_dte:
        zdte_html = (
            f'<div class="stat-row"><span class="stat-label">0DTE Expiry</span>'
            f'<span class="stat-value">{zero_dte["expiry"]} ({zero_dte["hours_remaining"]:.1f}h left)</span></div>'
            f'<div class="stat-row"><span class="stat-label">0DTE Net GEX</span>'
            f'<span class="stat-value animated-number" data-target="{zero_dte["net_gex_total"]/1e6:.2f}" data-decimals="1" data-prefix="$" data-suffix="M">$0M</span></div>'
            f'<div class="stat-row"><span class="stat-label">Max Pain</span>'
            f'<span class="stat-value">${zero_dte["max_pain"]:.0f}</span></div>'
            f'<div class="stat-row"><span class="stat-label">Put/Call OI Ratio</span>'
            f'<span class="stat-value animated-number" data-target="{zero_dte["pc_ratio"]:.3f}" data-decimals="2">0.00</span></div>'
        )
    else:
        zdte_html = '<div class="stat-row"><span class="stat-label">No 0DTE expiry available today</span></div>'

    if corr:
        corr_rows = ""
        for name, val in corr.items():
            pct_width = min(abs(val) * 100, 100)
            color = "#39ff88" if val >= 0 else "#ff3b5c"
            corr_rows += (
                f'<div style="margin-bottom:10px;">'
                f'<div class="stat-row" style="border-bottom:none;padding-bottom:2px;">'
                f'<span class="stat-label">QQQ vs {name}</span><span class="stat-value">{val:+.2f}</span></div>'
                f'<div class="corr-bar-bg"><div class="corr-bar-fill" style="width:{pct_width}%;background:{color};"></div></div>'
                f'</div>'
            )
        corr_html = corr_rows
    else:
        corr_html = '<div class="stat-row"><span class="stat-label">Data unavailable this run</span></div>'

    on_high = f"${overnight['high']:.2f}" if overnight else "n/a"
    on_low = f"${overnight['low']:.2f}" if overnight else "n/a"

    pdh = f"${ref_levels['pdh']:.2f}" if ref_levels and ref_levels.get("pdh") else "n/a"
    pdl = f"${ref_levels['pdl']:.2f}" if ref_levels and ref_levels.get("pdl") else "n/a"
    weekly_open = f"${ref_levels['weekly_open']:.2f}" if ref_levels and ref_levels.get("weekly_open") else "n/a"
    monthly_open = f"${ref_levels['monthly_open']:.2f}" if ref_levels and ref_levels.get("monthly_open") else "n/a"
    pdh_js = "null" if not (ref_levels and ref_levels.get("pdh")) else f"{ref_levels['pdh']:.2f}"
    pdl_js = "null" if not (ref_levels and ref_levels.get("pdl")) else f"{ref_levels['pdl']:.2f}"

    atr_val = f"{atr['atr14']:.2f} pts" if atr else "n/a"
    atr_raw = f"{atr['atr14']:.1f}" if atr else "20"
    range_low = f"${atr['range_low']:.2f}" if atr else "n/a"
    range_high = f"${atr['range_high']:.2f}" if atr else "n/a"
    range_high_js = "null" if not atr else f"{atr['range_high']:.2f}"
    range_low_js = "null" if not atr else f"{atr['range_low']:.2f}"

    if alerts:
        alert_class = "alert-active"
        alert_html = "\u26a0 " + str(len(alerts)) + " ALERT(S) TODAY" + "".join(
            f'<div class="alert-item">\u2022 {a}</div>' for a in alerts)
    else:
        alert_class = "alert-calm"
        alert_html = "\u2713 No major alerts \u2014 conditions look orderly"

    if history and len(history) >= 1:
        history_json = json.dumps({
            "dates": [h["date"][5:] for h in history],
            "values": [h["net_gex_m"] for h in history],
            "spots": [h.get("spot") for h in history],
        })
        history_note = "" if len(history) > 1 else "(first day logged \u2014 trend builds over time)"
    else:
        history_json = json.dumps({"dates": [], "values": [], "spots": []})
        history_note = "(no history yet)"

    if hurst:
        hurst_html = (
            f'<div class="stat-row"><span class="stat-label">Hurst Exponent (H)</span>'
            f'<span class="stat-value animated-number" data-target="{hurst["h"]:.3f}" data-decimals="3">0.000</span></div>'
            f'<div class="stat-row"><span class="stat-label">Regime</span><span class="stat-value">{hurst["regime"]}</span></div>'
        )
    else:
        hurst_html = '<div class="stat-row"><span class="stat-label">Data unavailable this run</span></div>'

    if confluence:
        parts_text = " &nbsp;|&nbsp; ".join(confluence["parts"]) if confluence["parts"] else "No components available"
    else:
        confluence = {"score": 0, "label": "Unavailable"}
        parts_text = "Data unavailable this run"

    if radar:
        radar_json = json.dumps(radar)
    else:
        radar_json = "null"

    if forward_proj:
        forward_json = json.dumps(forward_proj)
    else:
        forward_json = "null"

    if event_cal:
        fomc_line = f'{event_cal["next_fomc"]}' if event_cal.get("next_fomc") else "n/a"
        nfp_line = f'{event_cal["next_nfp"]}' if event_cal.get("next_nfp") else "n/a"
        fomc_hist = (f'{event_cal["avg_fomc_move_pct"]:.2f}% avg move (n={event_cal["n_fomc_sample"]})'
                     if event_cal.get("avg_fomc_move_pct") is not None else "n/a")
        nfp_hist = (f'{event_cal["avg_nfp_move_pct"]:.2f}% avg move (n={event_cal["n_nfp_sample"]})'
                    if event_cal.get("avg_nfp_move_pct") is not None else "n/a")
        event_html = (
            f'<div class="stat-row"><span class="stat-label">Next FOMC</span><span class="stat-value">{fomc_line}</span></div>'
            f'<div class="stat-row"><span class="stat-label">Historical FOMC reaction</span><span class="stat-value">{fomc_hist}</span></div>'
            f'<div class="stat-row"><span class="stat-label">Next NFP (first Friday)</span><span class="stat-value">{nfp_line}</span></div>'
            f'<div class="stat-row"><span class="stat-label">Historical NFP reaction</span><span class="stat-value">{nfp_hist}</span></div>'
        )
    else:
        event_html = '<div class="stat-row"><span class="stat-label">Data unavailable this run</span></div>'

    advocate_text = get_devils_advocate(confluence, gex, vol)

    if diff_view and diff_view.get("diffs"):
        diff_html = f'<div class="clock-sub" style="text-align:left;margin-bottom:8px;">vs {diff_view["yesterday_date"]}</div>'
        for d in diff_view["diffs"]:
            diff_html += f'<div class="diff-item">{d["label"]}: <b>{d["diff"]}</b> ({d["from"]} \u2192 {d["to"]})</div>'
    elif diff_view is None:
        diff_html = '<div class="stat-row"><span class="stat-label">Need at least 2 days of history \u2014 check back tomorrow</span></div>'
    else:
        diff_html = '<div class="stat-row"><span class="stat-label">No comparable metrics changed</span></div>'

    if anomalies:
        anomaly_html = ""
        for a in anomalies:
            if a["z"] is None:
                anomaly_html += f'<div class="zscore-item"><span>{a["name"]}</span><span class="stat-label">Building history ({a["n_samples"]}/8 days)</span></div>'
            else:
                flag = "amplifying" if abs(a["z"]) > 2 else "normal"
                flag_text = f'{a["z"]:+.2f}\u03c3' + (" \u26a0 OUTLIER" if abs(a["z"]) > 2 else "")
                anomaly_html += f'<div class="zscore-item"><span>{a["name"]}</span><span class="regime-tag tag-{flag}">{flag_text}</span></div>'
    else:
        anomaly_html = '<div class="stat-row"><span class="stat-label">Data unavailable this run</span></div>'

    if velocity:
        vel_class = "amplifying" if velocity["velocity"] < -1 else "dampening" if velocity["velocity"] > 1 else "normal"
        velocity_html = (
            f'<div class="stat-row"><span class="stat-label">Net GEX velocity</span>'
            f'<span class="stat-value">{velocity["velocity"]:+.2f} $M/day</span></div>'
            f'<div class="stat-row"><span class="stat-label">Direction</span>'
            f'<span class="regime-tag tag-{vel_class}">{velocity["direction"]}</span></div>'
        )
    else:
        velocity_html = '<div class="stat-row"><span class="stat-label">Needs 4+ days of history \u2014 still building</span></div>'

    if lead_lag:
        leadlag_html = ""
        for name, vals in lead_lag.items():
            leadlag_html += (f'<div class="stat-row"><span class="stat-label">QQQ vs {name}</span>'
                              f'<span class="stat-value">Same-day {vals["same_day"]:+.2f} | Lagged {vals["lagged"]:+.2f}</span></div>')
    else:
        leadlag_html = '<div class="stat-row"><span class="stat-label">Data unavailable this run</span></div>'

    if vol_surface:
        surface_html = '<table class="surface-table"><tr><th>Expiry</th>' + "".join(f"<th>{o}</th>" for o in vol_surface["offsets"]) + "</tr>"
        for row in vol_surface["rows"]:
            surface_html += f'<tr><td>{row["expiry"]}</td>' + "".join(
                f"<td>{v if v is not None else 'n/a'}</td>" for v in row["ivs"]) + "</tr>"
        surface_html += "</table>"
    else:
        surface_html = '<div class="stat-row"><span class="stat-label">Data unavailable this run</span></div>'

    pine_text_raw = pine_export if pine_export else "// Pine export unavailable this run (NQ price or level data missing)."
    pine_text = pine_text_raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    expiry_json = "null" if not expiry_gex else json.dumps({
        "labels": list(expiry_gex.keys()), "values": list(expiry_gex.values())
    })

    vex_total = gex.get("net_vex_total")
    chex_total = gex.get("net_chex_total")
    if vex_total is not None and chex_total is not None:
        vex_read = "Bullish on rising vol" if vex_total > 0 else "Bearish on rising vol"
        chex_read = "Bullish decay (time helps longs)" if chex_total > 0 else "Bearish decay (time hurts longs)"
        vex_chex_html = (
            f'<div class="stat-row"><span class="stat-label">Net VEX ($/1 vol pt)</span>'
            f'<span class="stat-value animated-number" data-target="{vex_total/1e6:.2f}" data-decimals="2" data-prefix="$" data-suffix="M">$0M</span></div>'
            f'<div class="stat-row"><span class="stat-label">Vanna read</span><span class="stat-value">{vex_read}</span></div>'
            f'<div class="stat-row"><span class="stat-label">Net CHEX ($/day)</span>'
            f'<span class="stat-value animated-number" data-target="{chex_total/1e6:.2f}" data-decimals="2" data-prefix="$" data-suffix="M">$0M</span></div>'
            f'<div class="stat-row"><span class="stat-label">Charm read</span><span class="stat-value">{chex_read}</span></div>'
        )
    else:
        vex_chex_html = '<div class="stat-row"><span class="stat-label">Data unavailable this run</span></div>'

    atm_iv = gex.get("atm_iv")
    skew = gex.get("skew")
    if atm_iv is not None and not np.isnan(atm_iv):
        vrp = atm_iv * 100 - vol["hv20"]
        vrp_tag = "Vol Rich" if vrp > 2 else "Vol Cheap" if vrp < -2 else "Fair Value"
        vrp_class = "amplifying" if vrp > 2 else "dampening" if vrp < -2 else "normal"
        skew_line = ""
        if skew is not None and not np.isnan(skew):
            skew_read = "Puts pricier (normal crash-risk skew)" if skew > 0.01 else "Calls pricier (unusual)" if skew < -0.01 else "Roughly flat"
            skew_line = (f'<div class="stat-row"><span class="stat-label">Put/Call Skew (5% OTM)</span>'
                         f'<span class="stat-value">{skew*100:+.1f} vol pts</span></div>'
                         f'<div class="stat-row"><span class="stat-label">Skew read</span><span class="stat-value">{skew_read}</span></div>')
        vrp_skew_html = (
            f'<div class="stat-row"><span class="stat-label">ATM IV</span>'
            f'<span class="stat-value animated-number" data-target="{atm_iv*100:.2f}" data-decimals="1" data-suffix="%">0%</span></div>'
            f'<div class="stat-row"><span class="stat-label">HV20</span>'
            f'<span class="stat-value animated-number" data-target="{vol["hv20"]:.2f}" data-decimals="1" data-suffix="%">0%</span></div>'
            f'<div class="stat-row"><span class="stat-label">VRP (IV - HV)</span>'
            f'<span class="stat-value">{vrp:+.1f} pts <span class="regime-tag tag-{vrp_class}">{vrp_tag}</span></span></div>'
            f'{skew_line}'
        )
    else:
        vrp_skew_html = '<div class="stat-row"><span class="stat-label">Data unavailable this run</span></div>'

    fvg_ob_json = json.dumps(fvg_ob) if fvg_ob else json.dumps({"fvgs": [], "obs": []})
    if fvg_ob and (fvg_ob.get("fvgs") or fvg_ob.get("obs")):
        rows = ""
        for f in fvg_ob.get("fvgs", [])[-4:]:
            color = "#39ff88" if f["type"] == "bullish" else "#ff3b5c"
            rows += (f'<div class="fvg-item"><span style="color:{color};">\u25a0 FVG '
                      f'{f["type"]}</span> &nbsp; ${f["bottom"]:.2f} - ${f["top"]:.2f}</div>')
        for o in fvg_ob.get("obs", [])[-4:]:
            color = "#39ff88" if o["type"] == "bullish" else "#ff3b5c"
            rows += (f'<div class="fvg-item"><span style="color:{color};">\u25a0 Order Block '
                      f'{o["type"]}</span> &nbsp; ${o["bottom"]:.2f} - ${o["top"]:.2f} @ {o["time"]}</div>')
        fvg_ob_html = rows if rows else '<div class="stat-row"><span class="stat-label">None detected this run</span></div>'
    else:
        fvg_ob_html = '<div class="stat-row"><span class="stat-label">No intraday data / none detected this run</span></div>'

    summary_lines = generate_research_summary(gex, vol, season, cot)
    summary_html = ""
    for line in summary_lines:
        if line.startswith("BOTTOM LINE:"):
            summary_html += f'<div class="summary-line bottom-line">{line}</div>'
        else:
            summary_html += f'<div class="summary-line"><span class="bullet">&#9656;</span>{line}</div>'

    html = HTML_TEMPLATE
    replacements = {
        "__DATE_STR__": datetime.now().strftime("%A, %B %d, %Y"),
        "__TIME_STR__": datetime.now().strftime("%I:%M %p"),
        "__EXPIRY__": gex["expiry"],
        "__DTE__": str(gex["dte"]),
        "__SPOT__": f"{gex['spot']:.2f}",
        "__VIX_NOW__": f"{vol['vix_now']:.2f}",
        "__VIX_REGIME__": vix_regime,
        "__VIX_REGIME_CLASS__": vix_regime.lower(),
        "__HV20__": f"{vol['hv20']:.1f}",
        "__HV60__": f"{vol['hv60']:.1f}",
        "__NET_GEX_M__": f"{gex['net_gex_total']/1e6:.1f}",
        "__GEX_REGIME__": gex_regime,
        "__GEX_REGIME_CLASS__": gex_regime.lower(),
        "__CALL_WALL__": f"${gex['call_wall']:.0f}" if not np.isnan(gex["call_wall"]) else "n/a",
        "__PUT_WALL__": f"${gex['put_wall']:.0f}" if not np.isnan(gex["put_wall"]) else "n/a",
        "__COT_HTML__": cot_html,
        "__ZERO_DTE_HTML__": zdte_html,
        "__CORR_HTML__": corr_html,
        "__ON_HIGH__": on_high,
        "__ON_LOW__": on_low,
        "__PDH__": pdh, "__PDL__": pdl,
        "__WEEKLY_OPEN__": weekly_open, "__MONTHLY_OPEN__": monthly_open,
        "__ATR_VAL__": atr_val, "__ATR_RAW__": atr_raw,
        "__RANGE_LOW__": range_low, "__RANGE_HIGH__": range_high,
        "__ALERT_CLASS__": alert_class, "__ALERT_HTML__": alert_html,
        "__HISTORY_NOTE__": history_note,
        "__FVG_OB_HTML__": fvg_ob_html,
        "__N_YEARS__": str(int(season["n_years"].max())),
        "__SUMMARY_HTML__": summary_html,
        "__GEX_JSON__": gex_json,
        "__SEASON_JSON__": season_json,
        "__PRICE_JSON__": price_json,
        "__HISTORY_JSON__": history_json,
        "__EXPIRY_JSON__": expiry_json,
        "__HURST_HTML__": hurst_html,
        "__CONFLUENCE_LABEL__": confluence["label"],
        "__CONFLUENCE_PARTS__": parts_text,
        "__CONFLUENCE_SCORE__": str(confluence["score"]),
        "__RADAR_JSON__": radar_json,
        "__FORWARD_JSON__": forward_json,
        "__EVENT_HTML__": event_html,
        "__PINE_TEXT__": pine_text,
        "__ADVOCATE_TEXT__": advocate_text,
        "__DIFF_HTML__": diff_html,
        "__ANOMALY_HTML__": anomaly_html,
        "__VELOCITY_HTML__": velocity_html,
        "__LEADLAG_HTML__": leadlag_html,
        "__SURFACE_HTML__": surface_html,
        "__VEX_CHEX_HTML__": vex_chex_html,
        "__VRP_SKEW_HTML__": vrp_skew_html,
        "__FVG_OB_JSON__": fvg_ob_json,
        "__GAMMA_FLIP_JS__": "null" if np.isnan(gex["gamma_flip"]) else f"{gex['gamma_flip']:.2f}",
        "__CALL_WALL_JS__": "null" if np.isnan(gex["call_wall"]) else f"{gex['call_wall']:.2f}",
        "__PUT_WALL_JS__": "null" if np.isnan(gex["put_wall"]) else f"{gex['put_wall']:.2f}",
        "__PDH_JS__": pdh_js, "__PDL_JS__": pdl_js,
        "__RANGE_HIGH_JS__": range_high_js, "__RANGE_LOW_JS__": range_low_js,
    }
    for token, value in replacements.items():
        html = html.replace(token, str(value))
    return html


ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")


def save_archive_copy(html):
    """Saves a dated snapshot alongside latest.html so you can look back at
    exactly what the dashboard said on a specific past morning. Unbounded --
    at ~30-40KB per day this is negligible even after a full year (~10MB)."""
    try:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        archive_path = os.path.join(ARCHIVE_DIR, f"{date_str}.html")
        with open(archive_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        log_error(f"Archive save failed (non-fatal): {e}")


def build_dashboard():
    gex = get_gex_data()
    vol = get_vol_data()
    season = get_seasonality()
    cot = get_cot_data()
    zero_dte = get_0dte_data(gex["spot"])
    overnight = get_overnight_range()
    intraday_hist = get_intraday_data()
    intraday = build_price_series(intraday_hist)
    fvg_ob = detect_fvg_and_ob(intraday_hist)
    corr = get_correlation_data()
    ref_levels = get_reference_levels()
    atr = get_atr_data()
    alerts = generate_alerts(gex, vol, cot)
    expiry_gex = get_multi_expiry_gex(gex["spot"])
    hurst = get_hurst_data()
    forward_proj = get_forward_projection(gex)
    confluence = compute_confluence_score(gex, cot, season)
    radar = compute_radar_data(gex, vol, cot, hurst, season)
    nq_ratio = get_nq_scale_ratio(gex["spot"])
    pine_export = generate_pine_export(gex, ref_levels, atr, nq_ratio, zero_dte, overnight, confluence, hurst, cot, vol)
    event_cal = get_event_calendar_data()

    # Build today's history entry BEFORE logging, so diff/anomaly/velocity can
    # compare against everything logged up to and including today.
    vix_regime = ("Elevated" if vol["vix_now"] > vol["vix_avg5y"] * 1.15
                  else "Low" if vol["vix_now"] < vol["vix_avg5y"] * 0.85 else "Normal")
    gex_regime = "Dampening" if gex["net_gex_total"] > 0 else "Amplifying"
    atm_iv_pct = gex["atm_iv"] * 100 if gex.get("atm_iv") is not None and not np.isnan(gex["atm_iv"]) else None

    history_entry = {
        "net_gex_m": round(gex["net_gex_total"] / 1e6, 2),
        "spot": round(gex["spot"], 2),
        "vix": round(vol["vix_now"], 2),
        "atm_iv_pct": round(atm_iv_pct, 2) if atm_iv_pct is not None else None,
        "cot_pct_oi": round(cot["pct_oi"], 2) if cot else None,
        "hurst_h": hurst["h"] if hurst else None,
        "vix_regime": vix_regime, "gex_regime": gex_regime,
        "hurst_regime": hurst["regime"] if hurst else None,
    }
    history = update_and_load_history(history_entry)

    anomalies = get_anomaly_data(history, history_entry)
    diff_view = get_diff_view(history)
    velocity = get_gamma_velocity(history)
    vol_surface = get_vol_surface_data(gex["spot"])
    lead_lag = get_lead_lag_data()

    html = render_html(gex, vol, season, cot, zero_dte, overnight, intraday, corr,
                        ref_levels, atr, history, alerts, fvg_ob, expiry_gex,
                        hurst, forward_proj, confluence, radar, pine_export, event_cal,
                        anomalies, diff_view, velocity, vol_surface, lead_lag)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    save_archive_copy(html)
    log_error(f"Dashboard build completed -- saved to {OUT_PATH}")


if __name__ == "__main__":
    build_dashboard()