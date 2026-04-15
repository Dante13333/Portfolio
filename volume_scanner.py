"""
volume_scanner.py — HKEX Swing + Volume Hunter
Dynamically discovers liquid HKEX stocks (no fixed list).
Pipeline:
  Step 1: Generate candidate tickers (codes 0001–9999 formatted as XXXX.HK)
          filtered to the top ~300 by market cap using yfinance batch info.
  Step 2: Score each by swing size + choppiness + volume.
  Step 3: Rank and display with deep-dive charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import time
import pytz

def _apply_rangebreaks(fig, df, is_intraday=False, show_week_lines=True):
    """
    Remove closed-market gaps from x-axis and optionally mark week starts.
    is_intraday=True  → hide overnight + weekend gaps (for 1m-60m charts)
    is_intraday=False → hide weekend gaps only (for daily charts)
    """
    breaks = []
    if is_intraday:
        # Hide non-trading hours: 16:00 to 09:30 next day (HKT)
        breaks.append(dict(bounds=[16, 9.5], pattern="hour"))
        # Hide lunch break: 11:30 to 13:00 HKT
        breaks.append(dict(bounds=[11.5, 13], pattern="hour"))
        # Hide weekends
        breaks.append(dict(bounds=["sat", "mon"]))
    else:
        # Daily charts: just hide weekends
        breaks.append(dict(bounds=["sat", "mon"]))

    xaxis_update = dict(rangebreaks=breaks)

    # Week boundary lines (Monday marks)
    if show_week_lines and df is not None and len(df) > 0:
        idx = df.index
        # Find Mondays
        try:
            mondays = [t for t in idx if hasattr(t, "weekday") and t.weekday() == 0]
            for mon in mondays:
                fig.add_vline(
                    x=mon, line_dash="dot",
                    line_color="rgba(100,116,139,0.35)",
                    line_width=1)
        except Exception:
            pass

    fig.update_xaxes(xaxis_update)
    return fig

HK_TZ = pytz.timezone("Asia/Hong_Kong")

# ── DYNAMIC UNIVERSE BUILDER ──────────────────────────────────────────
# HKEX codes run 0001–9999 as 4-digit strings.
# We pre-seed with known liquid bands + let user expand.
# Liquid bands by code range (where most tradeable stocks live):
LIQUID_BANDS = list(range(1,   500))   # Blue chips, banks, SOEs
LIQUID_BANDS += list(range(500, 1200)) # Mid-large caps
LIQUID_BANDS += list(range(1200,2000)) # Consumer, pharma, property
LIQUID_BANDS += list(range(2000,2600)) # H-shares, dual-listed
LIQUID_BANDS += list(range(2600,3000)) # Mixed
LIQUID_BANDS += list(range(6000,6900)) # New economy listings
LIQUID_BANDS += list(range(9600,9999)) # Tech/AI recent IPOs

def code_to_ticker(code: int) -> str:
    return f"{code:04d}.HK"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_cap_batch(codes: list[int], batch_size: int = 100) -> pd.DataFrame:
    """
    Fetch market cap + price for a batch of tickers.
    Returns DataFrame with ticker, name, market_cap, price, volume.
    Only keeps stocks where market_cap > min_cap and price > 0.
    """
    results = []
    tickers = [code_to_ticker(c) for c in codes]
    
    # Process in batches
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                batch, period="2d", interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True
            )
            for t in batch:
                try:
                    if len(batch) == 1:
                        close = data["Close"].dropna()
                        vol   = data["Volume"].dropna()
                    else:
                        if t not in data.columns.get_level_values(0):
                            continue
                        close = data[t]["Close"].dropna()
                        vol   = data[t]["Volume"].dropna()
                    
                    if len(close) == 0 or float(close.iloc[-1]) <= 0:
                        continue
                    price = float(close.iloc[-1])
                    volume= int(vol.iloc[-1]) if len(vol) > 0 else 0
                    # Rough market cap proxy: price × avg_volume × 20
                    # (we don't have shares outstanding without info call)
                    liquidity = price * volume
                    results.append({
                        "ticker":    t,
                        "price":     price,
                        "volume":    volume,
                        "liquidity": liquidity,
                    })
                except Exception:
                    continue
        except Exception:
            continue
    
    return pd.DataFrame(results) if results else pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def scan_ticker_full(ticker: str, lookback: int) -> dict | None:
    """Full scan: fetch 3mo daily + compute all swing/volume metrics."""
    try:
        tk   = yf.Ticker(ticker)
        info = tk.fast_info
        df   = tk.history(period="3mo", interval="1d", auto_adjust=True)

        if df is None or len(df) < lookback + 3:
            return None

        price      = getattr(info, "last_price",      None)
        prev_close = getattr(info, "previous_close",  None)
        day_high   = getattr(info, "day_high",        None)
        day_low    = getattr(info, "day_low",         None)
        mkt_cap    = getattr(info, "market_cap",      None)
        name       = getattr(info, "exchange",        ticker)

        # Try to get a real name
        try:
            long_name = tk.info.get("longName") or tk.info.get("shortName") or ticker
        except Exception:
            long_name = ticker

        if not price or price <= 0:
            return None

        # ── Volume ────────────────────────────────────────────────────
        today_vol  = int(df["Volume"].iloc[-1])
        avg_vol_20 = float(df["Volume"].tail(20).mean())
        vol_ratio  = today_vol / avg_vol_20 if avg_vol_20 > 0 else 0
        turnover   = price * today_vol  # HKD turnover today

        # ── Daily range ───────────────────────────────────────────────
        ranges       = df["High"] - df["Low"]
        avg_range    = float(ranges.tail(lookback).mean())
        max_range    = float(ranges.tail(lookback).max())
        today_range  = (day_high - day_low) if (day_high and day_low) else float(ranges.iloc[-1])
        pct_above_20 = float((ranges.tail(lookback) >= 20).mean() * 100)
        pct_above_30 = float((ranges.tail(lookback) >= 30).mean() * 100)

        # ── Choppiness ────────────────────────────────────────────────
        ci = _choppiness(df, 14)

        # ── Direction changes ─────────────────────────────────────────
        closes  = df["Close"].tail(lookback + 1).values
        dirs    = np.sign(np.diff(closes))
        dir_chg = int(np.sum(dirs[:-1] != dirs[1:]))

        # ── Day change ────────────────────────────────────────────────
        day_pct = (price - prev_close) / prev_close * 100 if prev_close else None

        # ── Meso cycle position ───────────────────────────────────────
        try:
            c_    = df["Close"]
            d_    = c_.diff()
            g_    = d_.clip(lower=0).ewm(com=13, adjust=False).mean()
            l_    = (-d_.clip(upper=0)).ewm(com=13, adjust=False).mean()
            rsi_c = float((100 - 100 / (1 + g_ / l_.replace(0, np.nan))).dropna().iloc[-1])
            mid_c = float(c_.rolling(20).mean().iloc[-1])
            std_c = float(c_.rolling(20).std().iloc[-1])
            bb_c  = float(((c_.iloc[-1] - mid_c + 2*std_c) / (4*std_c + 1e-9)) * 100)
            bb_c  = max(0, min(100, bb_c))
            vr5   = float(df["Volume"].tail(5).mean()) / float(df["Volume"].rolling(20).mean().iloc[-1] + 1e-9)
            vol_w = min(vr5 / 2, 1.0) * 100
            meso_pct = round(max(0, min(100, rsi_c*0.40 + bb_c*0.40 + vol_w*0.20)), 1)
            if meso_pct < 35:   cycle_label = "✅ Early"
            elif meso_pct < 55: cycle_label = "🟢 Mid"
            elif meso_pct < 70: cycle_label = "🟡 Late"
            else:               cycle_label = "🔴 Peak"
        except Exception:
            meso_pct = 50; cycle_label = "—"

        # ── Score ─────────────────────────────────────────────────────
        swing_s  = min(avg_range / 80.0, 1.0) * 100
        chop_s   = max(0, min((ci - 30) / 40 * 100, 100)) if ci else 50
        vol_s    = min(vol_ratio / 3.0, 1.0) * 100
        cons_s   = pct_above_20
        cycle_bonus = 10 if meso_pct < 35 else 5 if meso_pct < 50 else -10 if meso_pct > 70 else 0
        score    = swing_s*0.33 + chop_s*0.24 + vol_s*0.24 + cons_s*0.14 + cycle_bonus

        return {
            "ticker":       ticker,
            "name":         long_name,
            "price":        price,
            "day_pct":      day_pct,
            "today_range":  today_range,
            "avg_range":    round(avg_range, 1),
            "max_range":    round(max_range, 1),
            "pct_above_20": round(pct_above_20, 0),
            "pct_above_30": round(pct_above_30, 0),
            "today_vol":    today_vol,
            "avg_vol_20":   avg_vol_20,
            "vol_ratio":    round(vol_ratio, 2),
            "turnover_hkd": turnover,
            "mkt_cap":      mkt_cap,
            "choppiness":   ci,
            "dir_changes":  dir_chg,
            "meso_pct":     meso_pct,
            "cycle_label":  cycle_label,
            "score":        round(score, 1),
        }
    except Exception as e:
        if "rate" in str(e).lower() or "429" in str(e) or "Too Many" in str(e):
            time.sleep(15)  # back off on rate limit
        return None

def _choppiness(df: pd.DataFrame, period: int = 14) -> float | None:
    if len(df) < period + 2:
        return None
    h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
           for i in range(1, len(df))]
    trs = np.array(trs)
    if len(trs) < period:
        return None
    atr_sum = trs[-period:].sum()
    rng     = h[-period:].max() - l[-period:].min()
    if rng <= 0 or atr_sum <= 0:
        return None
    return round(float(np.clip(
        100 * np.log10(atr_sum / rng) / np.log10(period), 0, 100)), 1)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_daily(ticker, period="3mo"):
    try:
        return yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday(ticker, interval="15m"):
    try:
        df = yf.Ticker(ticker).history(period="5d", interval=interval, auto_adjust=True)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(HK_TZ)
        return df
    except Exception:
        return pd.DataFrame()

# ── COLOUR HELPERS ────────────────────────────────────────────────────
def fc(v):   return "#16a34a" if (v or 0) >= 0 else "#dc2626"
def fhkd(v):
    if not v: return "—"
    return f"HKD {v:,.1f}"
def fvol(v):
    if not v: return "—"
    if v >= 1e9: return f"{v/1e9:.1f}B"
    if v >= 1e6: return f"{v/1e6:.1f}M"
    if v >= 1e3: return f"{v/1e3:.0f}K"
    return str(int(v))
def fcap(v):
    if not v: return "—"
    if v >= 1e12: return f"HKD {v/1e12:.1f}T"
    if v >= 1e9:  return f"HKD {v/1e9:.1f}B"
    return f"HKD {v/1e6:.0f}M"
def score_color(v):
    if v >= 65: return "#16a34a"
    if v >= 40: return "#f59e0b"
    return "#dc2626"
def chop_color(v):
    if v is None: return "#94a3b8"
    if v >= 61.8: return "#16a34a"
    if v >= 50:   return "#f59e0b"
    return "#dc2626"
def range_color(v, thr=20):
    if not v: return "#94a3b8"
    if v >= thr*2: return "#dc2626"
    if v >= thr:   return "#16a34a"
    return "#94a3b8"
def vol_color(v):
    if not v: return "#94a3b8"
    if v >= 2.0: return "#dc2626"
    if v >= 1.3: return "#f59e0b"
    return "#94a3b8"

# ════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ════════════════════════════════════════════════════════════════════
def render():
    now_hk = datetime.now(HK_TZ)

    st.markdown(
        "## 🔍 HKEX Market-Wide Swing + Volume Hunter &nbsp;"
        "<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        "padding:2px 7px;border-radius:5px'>DYNAMIC SCAN</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"HKT {now_hk.strftime('%Y-%m-%d %H:%M')} · "
        f"Scans the full HKEX universe dynamically · "
        f"No fixed stock list · Finds unknown opportunities</span>",
        unsafe_allow_html=True)

    with st.expander("📖 How the scanner works"):
        st.markdown("""
**Two-stage pipeline:**

**Stage 1 — Universe filter** (fast, ~2 min):
Generates HKEX ticker codes across all liquid code ranges, downloads 2 days of price+volume,
keeps only stocks above your minimum turnover (price × volume). This finds all liquid names
across the entire HKEX without a fixed list.

**Stage 2 — Swing scan** (deeper, ~3-5 min for top candidates):
Fetches 3 months of daily data for the filtered candidates, computes:
- Avg daily High−Low range in HKD
- Choppiness Index (>61.8 = oscillating both ways ✅)
- Volume ratio vs 20-day average
- % of days that hit your minimum range threshold

**Score** = 35% swing + 25% choppiness + 25% volume + 15% consistency

**The result:** stocks you've never looked at that match your exact trading style.
        """)

    with st.expander("📖 Metric explanations"):
        st.markdown("""
**Min daily turnover (HKD)** — Price x volume. Filters out illiquid stocks where
you cannot enter and exit cleanly. 5M+ ensures enough liquidity for swing trades.

**Avg daily range (HKD)** — Average High minus Low over the lookback period.
How much swing is available per day. Aim for >20 HKD consistently for your style.

**Vol ratio** — Today's volume vs 20-day average.
>2x = unusually active (something is happening). <0.7x = thin (wide spreads, hard to trade).

**Choppiness Index** — 0-100. Measures oscillation vs trending.
>61.8 = choppy/oscillating (price swings both ways) = good for range trading.
<38.2 = trending one direction = dangerous (no upper swing if downtrending).

**% Days >=20 HKD** — How often this stock actually hits your target range size.
>60% = reliable source of big swings. <30% = rarely gives you what you need.

**Swing score** — Composite: range size (35%) + choppiness (25%) + volume (25%) + consistency (15%).
65+ = strong candidate. Below 40 = avoid.

**Stage 1 (Universe filter)** — Downloads 2 days of data for thousands of HKEX codes,
keeps only those with turnover above your threshold. Fast but shallow.

**Stage 2 (Deep scan)** — Fetches 3 months of daily data for the filtered candidates,
computes all swing metrics. Slower but gives the full picture.
        """)

    st.markdown("---")
    st.markdown("### Step 1 — Set your filters")

    # ── Filter controls ──────────────────────────────────────────────
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)

    min_turnover = r1c1.number_input(
        "Min daily turnover (HKD)", min_value=0.0,
        value=5_000_000.0, step=1_000_000.0,
        format="%.0f", key="sc_turnover",
        help="Price × volume today. Higher = more liquid. 5M+ ensures you can enter/exit.")
    min_price = r1c2.number_input(
        "Min price (HKD)", min_value=0.0, value=5.0, step=1.0,
        key="sc_minprice",
        help="Stocks below this are penny stocks — spreads too wide for swing trading.")
    max_candidates = r1c3.slider(
        "Max candidates to deep-scan", 30, 200, 80, key="sc_maxcand",
        help="After liquidity filter, scan the top N by turnover. More = slower but broader.")
    lookback = r1c4.slider(
        "Lookback days", 5, 20, 10, key="sc_lookback",
        help="How many recent days to measure swing consistency.")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    min_range  = r2c1.number_input(
        "Min avg range (HKD)", min_value=0.0, value=15.0, step=5.0,
        key="sc_minrange",
        help="Minimum average daily High−Low. Set 20–30 for your target stocks.")
    min_chop   = r2c2.slider(
        "Min choppiness", 0, 80, 40, key="sc_minchop",
        help="61.8+ = oscillates both ways. Lower to cast wider net.")
    min_vol_r  = r2c3.number_input(
        "Min volume ratio", min_value=0.0, value=0.5, step=0.1,
        key="sc_minvol",
        help="Today's vol ÷ 20d avg. 1.0 = normal, 2.0 = double.")
    top_n      = r2c4.slider(
        "Show top N results", 5, 50, 20, key="sc_topn")

    r3c1, r3c2 = st.columns(2)
    cycle_filter = r3c1.select_slider(
        "Cycle position filter",
        options=["Any", "Early only (<35%)", "Early + Mid (<55%)", "Exclude peak (>70%)"],
        value="Early + Mid (<55%)",
        key="sc_cycle",
        help="Filter by meso cycle position (daily RSI+BB+volume). Early = best entry zone.")
    sort_col = r3c2.selectbox(
        "Sort by", ["Score", "Cycle Position", "Avg Range (HKD)", "Today Range",
                    "Vol Ratio", "Choppiness", "Turnover (HKD)"],
        key="sc_sort")

    run_btn = st.button(
        "🚀 Run Full Market Scan", key="sc_run",
        help="Stage 1: ~2 min to filter universe. Stage 2: ~3-5 min to deep-scan candidates.")

    if not run_btn and "sc_results" not in st.session_state:
        st.info(
            "👆 Set your filters above and click **Run Full Market Scan**. "
            "The scanner will search the entire HKEX universe (~2,700 stocks) "
            "and surface only the ones matching your criteria.")
        return

    if run_btn:
        # ── Stage 1: Universe filter ─────────────────────────────────
        st.markdown("---")
        st.markdown("### Stage 1 — Filtering HKEX universe by liquidity…")
        prog1 = st.progress(0, text="Generating candidates…")

        # Split into batches for display
        all_codes = LIQUID_BANDS
        batch_size = 200
        liquid = []
        total_batches = len(all_codes) // batch_size + 1

        for batch_i in range(total_batches):
            batch_codes = all_codes[batch_i*batch_size:(batch_i+1)*batch_size]
            if not batch_codes:
                continue
            pct = (batch_i + 1) / total_batches
            prog1.progress(pct,
                text=f"Stage 1: checking codes "
                     f"{batch_codes[0]:04d}–{batch_codes[-1]:04d}… "
                     f"({len(liquid)} liquid so far)")

            tickers = [code_to_ticker(c) for c in batch_codes]
            try:
                raw = yf.download(
                    tickers, period="2d", interval="1d",
                    group_by="ticker", auto_adjust=True,
                    progress=False, threads=False
                )
                for t in tickers:
                    try:
                        if len(tickers) == 1:
                            close = raw["Close"].dropna()
                            vol   = raw["Volume"].dropna()
                        else:
                            if t not in raw.columns.get_level_values(0):
                                continue
                            close = raw[t]["Close"].dropna()
                            vol   = raw[t]["Volume"].dropna()
                        if len(close) == 0: continue
                        price = float(close.iloc[-1])
                        v     = int(vol.iloc[-1]) if len(vol) > 0 else 0
                        if price < min_price: continue
                        turnover = price * v
                        if turnover < min_turnover: continue
                        liquid.append({"ticker": t, "price": price,
                                       "volume": v, "turnover": turnover})
                    except Exception:
                        continue
                time.sleep(1.5)   # respect rate limit between batches
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    st.warning("⚠️ Rate limited — pausing 30s then resuming…")
                    time.sleep(30)
                continue

        prog1.empty()

        if not liquid:
            st.error("No liquid stocks found. Try lowering the min turnover.")
            return

        df_liquid = pd.DataFrame(liquid).sort_values(
            "turnover", ascending=False).head(max_candidates).reset_index(drop=True)

        st.success(
            f"Stage 1 complete — found **{len(df_liquid)} liquid stocks** "
            f"(from {len(all_codes)} candidates, min turnover HKD {min_turnover:,.0f})")

        # ── Stage 2: Deep scan ───────────────────────────────────────
        st.markdown("### Stage 2 — Deep scanning for swing quality…")
        prog2 = st.progress(0, text="Starting deep scan…")

        results = []
        for i, row in df_liquid.iterrows():
            t = row["ticker"]
            prog2.progress(
                (i+1)/len(df_liquid),
                text=f"Deep scanning {t}… ({i+1}/{len(df_liquid)}, "
                     f"{len(results)} passing so far)")
            r = scan_ticker_full(t, lookback)
            if r:
                results.append(r)
            time.sleep(0.5)   # pace requests to avoid rate limit

        prog2.empty()
        st.session_state["sc_results"] = results
        st.session_state["sc_params"]  = {
            "min_range": min_range, "min_chop": min_chop,
            "min_vol_r": min_vol_r, "lookback": lookback,
        }

    # ── Display results ──────────────────────────────────────────────
    results = st.session_state.get("sc_results", [])
    params  = st.session_state.get("sc_params", {})
    min_range = params.get("min_range", min_range)
    min_chop  = params.get("min_chop",  min_chop)
    min_vol_r = params.get("min_vol_r", min_vol_r)

    if not results:
        st.warning("No results yet — run the scan first.")
        return

    df = pd.DataFrame(results)
    df = df[df["avg_range"]  >= min_range]
    df = df[df["choppiness"].fillna(0) >= min_chop]
    df = df[df["vol_ratio"]  >= min_vol_r]

    # Cycle filter
    cycle_filter = st.session_state.get("sc_cycle", "Early + Mid (<55%)")
    if "meso_pct" in df.columns:
        if cycle_filter == "Early only (<35%)":
            df = df[df["meso_pct"] < 35]
        elif cycle_filter == "Early + Mid (<55%)":
            df = df[df["meso_pct"] < 55]
        elif cycle_filter == "Exclude peak (>70%)":
            df = df[df["meso_pct"] < 70]

    sort_map = {
        "Score":           "score",
        "Cycle Position":  "meso_pct",
        "Avg Range (HKD)": "avg_range",
        "Today Range":     "today_range",
        "Vol Ratio":       "vol_ratio",
        "Choppiness":      "choppiness",
        "Turnover (HKD)":  "turnover_hkd",
    }
    _asc = sort_col == "Cycle Position"
    df = df.sort_values(sort_map.get(sort_col, "score"),
                        ascending=_asc).head(top_n).reset_index(drop=True)

    if df.empty:
        st.warning(
            "No stocks passed all filters. "
            "Try: lower min range, lower choppiness, or lower vol ratio.")
        return

    st.markdown("---")

    # ── Summary metrics ───────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Passed filters",  len(df))
    m2.metric("#1 swing stock",  df.iloc[0]["name"][:18],
              delta=f"HKD {df.iloc[0]['avg_range']:.1f} avg range")
    m3.metric("Avg daily range", f"HKD {df['avg_range'].mean():.1f}")
    m4.metric("Avg choppiness",
              f"{df['choppiness'].mean():.1f}" if df['choppiness'].notna().any() else "—")
    m5.metric("Avg vol ratio",   f"{df['vol_ratio'].mean():.1f}×")

    # ── Scatter: Range × Volume, colour = choppiness ─────────────────
    st.markdown("### Volume × Swing Scatter")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Top-right green zone = big swing + high volume + oscillating. "
        "These are your best candidates.</span>",
        unsafe_allow_html=True)

    x_max = float(df["avg_range"].max()) * 1.15
    y_max = float(df["vol_ratio"].max()) * 1.12

    fig = go.Figure()
    fig.add_shape(type="rect", x0=min_range, x1=x_max,
                  y0=max(min_vol_r, 1.0), y1=y_max,
                  fillcolor="rgba(22,163,74,0.07)", line_width=0)
    fig.add_annotation(x=x_max*0.88, y=y_max*0.93,
                       text="✅ IDEAL ZONE", font=dict(size=10, color="#16a34a"),
                       showarrow=False)

    for _, row in df.iterrows():
        ci   = row["choppiness"] or 50
        col  = chop_color(ci)
        sz   = max(10, min(row["score"]/3, 35))
        ideal= row["avg_range"] >= min_range and row["vol_ratio"] >= 1.0 and ci >= 61.8
        fig.add_trace(go.Scatter(
            x=[row["avg_range"]], y=[row["vol_ratio"]],
            mode="markers+text",
            text=[row["name"][:12]],
            textposition="top center",
            textfont=dict(size=9, color="#0f172a" if ideal else "#94a3b8"),
            marker=dict(size=sz, color=col, opacity=0.85,
                        line=dict(color="white", width=2) if ideal else dict(width=0)),
            hovertemplate=(
                f"<b>{row['name']} ({row['ticker']})</b><br>"
                f"Avg range: HKD {row['avg_range']:.1f}<br>"
                f"Vol ratio: {row['vol_ratio']:.1f}×<br>"
                f"Choppiness: {row['choppiness']:.1f}<br>"
                f"Turnover: {fvol(row['turnover_hkd'])}<br>"
                f"Score: {row['score']:.1f}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.add_vline(x=min_range, line_dash="dot", line_color="#f59e0b",
                  line_width=1.5,
                  annotation_text=f"Min {min_range:.0f} HKD",
                  annotation_position="top")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8",
                  line_width=1,
                  annotation_text="Avg volume",
                  annotation_position="right")
    fig.update_layout(
        height=460, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        xaxis=dict(title="Avg daily range (HKD)", gridcolor="#f1f5f9", range=[0,x_max]),
        yaxis=dict(title="Volume ratio (today ÷ 20d avg)", gridcolor="#f1f5f9", range=[0,y_max]),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div style='display:flex;gap:20px;font-size:0.77rem;color:#64748b;margin-bottom:14px'>"
        "<span><span style='color:#16a34a'>●</span> Choppiness ≥61.8 (swinging both ways ✅)</span>"
        "<span><span style='color:#f59e0b'>●</span> Mixed</span>"
        "<span><span style='color:#dc2626'>●</span> Trending ⚠</span>"
        "<span>Size = score</span></div>",
        unsafe_allow_html=True)

    # ── Results table ─────────────────────────────────────────────────
    st.markdown("### Results Table")
    _ar_lbl = "Avg Range\n({:d}d)".format(lookback)
    hdr = st.columns([2.2, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85, 0.8])
    for col_w, lbl in zip(hdr, ["Stock","Price","Day Chg",
                                  _ar_lbl,
                                  "Today\\nRange",
                                  "% Days\\n\u226520 HKD",
                                  "Vol\\nRatio",
                                  "Choppiness",
                                  "Cycle\\nPosition",
                                  "Score"]):
        col_w.markdown(
            f"<span style='font-size:0.7rem;color:#94a3b8;"
            f"font-weight:500;white-space:pre'>{lbl}</span>",
            unsafe_allow_html=True)
    st.markdown("<hr style='margin:4px 0 6px 0;border-color:#e2e8f0'>",
                unsafe_allow_html=True)

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        rc = st.columns([2.2, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85, 0.8])

        rc[0].markdown(
            f"<div style='font-weight:600;font-size:0.86rem'>#{rank} {row['name'][:22]}</div>"
            f"<div style='font-size:0.69rem;color:#64748b'>{row['ticker']}</div>",
            unsafe_allow_html=True)

        rc[1].markdown(
            f"<span style='font-size:0.83rem'>"
            f"{'—' if not row['price'] else 'HKD {:,.1f}'.format(row['price'])}"
            f"</span>", unsafe_allow_html=True)

        dp = row["day_pct"]
        rc[2].markdown(
            f"<span style='color:{fc(dp)};font-size:0.83rem;font-weight:500'>"
            f"{'—' if dp is None else '{:+.2f}%'.format(dp)}</span>",
            unsafe_allow_html=True)

        rc[3].markdown(
            f"<span style='color:{range_color(row['avg_range'],min_range)};"
            f"font-size:0.88rem;font-weight:700'>"
            f"HKD {row['avg_range']:.1f}</span>",
            unsafe_allow_html=True)

        tr = row["today_range"]
        rc[4].markdown(
            f"<span style='color:{range_color(tr,min_range)};font-size:0.85rem'>"
            f"{'—' if not tr else 'HKD {:.1f}'.format(tr)}</span>",
            unsafe_allow_html=True)

        p20c = "#16a34a" if row["pct_above_20"]>=60 else \
               "#f59e0b" if row["pct_above_20"]>=30 else "#94a3b8"
        rc[5].markdown(
            f"<span style='color:{p20c};font-size:0.85rem'>"
            f"{row['pct_above_20']:.0f}%</span>",
            unsafe_allow_html=True)

        rc[6].markdown(
            f"<span style='color:{vol_color(row['vol_ratio'])};"
            f"font-size:0.85rem;font-weight:500'>"
            f"{row['vol_ratio']:.1f}×</span>",
            unsafe_allow_html=True)

        ci = row["choppiness"]
        ci_l = ("✅" if ci and ci>=61.8 else "〰" if ci and ci>=50 else "⚠")
        rc[7].markdown(
            f"<span style='color:{chop_color(ci)};font-size:0.83rem'>"
            f"{'—' if not ci else '{:.0f} {}'.format(ci, ci_l)}</span>",
            unsafe_allow_html=True)

        cyc_p = row.get("meso_pct", 50)
        cyc_l = row.get("cycle_label", "—")
        cyc_c = "#16a34a" if cyc_p < 35 else "#16a34a" if cyc_p < 55 else "#f59e0b" if cyc_p < 70 else "#dc2626"
        rc[8].markdown(
            f"<span style='color:{cyc_c};font-size:0.8rem;font-weight:600'>{cyc_l}</span>"
            f"<br><span style='color:{cyc_c};font-size:0.7rem'>{cyc_p:.0f}%</span>",
            unsafe_allow_html=True)

        rc[9].markdown(
            f"<span style='color:{score_color(row['score'])};"
            f"font-size:1rem;font-weight:800'>{row['score']:.0f}</span>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Deep dive ────────────────────────────────────────────────────
    st.markdown("### 🔍 Deep-Dive Chart")
    dd_opts = {
        f"#{i+1}  {r['name']} ({r['ticker']}) — avg HKD {r['avg_range']:.1f} · score {r['score']:.0f}": r["ticker"]
        for i, (_, r) in enumerate(df.iterrows())
    }
    da1, da2, da3 = st.columns(3)
    sel      = da1.selectbox("Stock to inspect", list(dd_opts.keys()), key="sc_dive")
    d_period = da2.selectbox("History", ["1mo","3mo","6mo"], index=1, key="sc_dperiod")
    d_intra  = da3.selectbox("Intraday interval", ["5m","15m","30m"], index=1, key="sc_dintra")
    dive_sym = dd_opts[sel]
    dive_row = df[df["ticker"]==dive_sym].iloc[0]

    ks = st.columns(5)
    ks[0].metric("Price",         fhkd(dive_row["price"]))
    ks[1].metric("Avg range",     fhkd(dive_row["avg_range"]))
    ks[2].metric("Choppiness",
                 f"{dive_row['choppiness']:.1f}" if dive_row["choppiness"] else "—")
    ks[3].metric("% days ≥20",    f"{dive_row['pct_above_20']:.0f}%")
    ks[4].metric("Mkt cap",       fcap(dive_row["mkt_cap"]))

    with st.spinner("Loading charts…"):
        df_d = fetch_daily(dive_sym, d_period)
        df_i = fetch_intraday(dive_sym, d_intra)

    if df_d is not None and len(df_d) > 5:
        ranges_d = df_d["High"] - df_d["Low"]
        avg_r    = float(ranges_d.mean())
        pct_hit  = float((ranges_d >= min_range).mean() * 100)

        fig_dd = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.50, 0.28, 0.22],
            vertical_spacing=0.03,
            subplot_titles=["Daily candles",
                            f"Daily range (HKD)  avg={avg_r:.1f}",
                            "Direction flips (cumulative)"])

        bc = ["#16a34a" if c>=o else "#dc2626"
              for c,o in zip(df_d["Close"],df_d["Open"])]
        fig_dd.add_trace(go.Candlestick(
            x=df_d.index, open=df_d["Open"], high=df_d["High"],
            low=df_d["Low"], close=df_d["Close"],
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626"), row=1, col=1)

        fig_dd.add_trace(go.Bar(
            x=df_d.index, y=ranges_d,
            marker_color=[range_color(v,min_range) for v in ranges_d],
            opacity=0.85), row=2, col=1)
        fig_dd.add_hline(y=min_range, line_dash="dot",
                          line_color="#f59e0b", line_width=1.5,
                          annotation_text=f"Min {min_range:.0f}",
                          annotation_position="right", row=2, col=1)
        fig_dd.add_hline(y=avg_r, line_dash="dot",
                          line_color="#2563eb", line_width=1,
                          annotation_text=f"Avg {avg_r:.1f}",
                          annotation_position="right", row=2, col=1)

        dirs  = np.sign(df_d["Close"].diff())
        flips = (dirs != dirs.shift()).astype(int).cumsum()
        fig_dd.add_trace(go.Scatter(
            x=df_d.index, y=flips,
            line=dict(color="#8b5cf6", width=1.8),
            fill="tozeroy", fillcolor="rgba(139,92,246,0.08)"), row=3, col=1)

        fig_dd.update_layout(
            height=520, margin=dict(l=0,r=0,t=30,b=0),
            xaxis_rangeslider_visible=False,
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
            yaxis=dict(title="Price",gridcolor="#f1f5f9"),
            yaxis2=dict(title="Range HKD",gridcolor="#f1f5f9"),
            yaxis3=dict(title="Flips",gridcolor="#f1f5f9"),
            xaxis3=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig_dd, use_container_width=True)

        # Range histogram + verdict
        hc1, hc2 = st.columns([2,1])
        with hc1:
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=ranges_d, nbinsx=25,
                marker_color="#2563eb", opacity=0.75))
            fig_h.add_vline(x=min_range, line_dash="dot",
                             line_color="#f59e0b", line_width=2,
                             annotation_text=f"Min {min_range:.0f}",
                             annotation_position="top right")
            fig_h.update_layout(
                height=190, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Daily range (HKD)", gridcolor="#f1f5f9"),
                yaxis=dict(title="Days",gridcolor="#f1f5f9"))
            st.plotly_chart(fig_h, use_container_width=True)
        with hc2:
            vc = "#16a34a" if pct_hit>=60 else "#f59e0b" if pct_hit>=30 else "#dc2626"
            vt = "Consistent ✅" if pct_hit>=60 else "Occasional ⚠️" if pct_hit>=30 else "Rare ❌"
            st.markdown(
                f"<div style='padding:16px;border-radius:10px;"
                f"border:1px solid {vc};background:rgba(0,0,0,0.02);margin-top:8px'>"
                f"<div style='font-size:1.8rem;font-weight:800;color:{vc}'>"
                f"{pct_hit:.0f}%</div>"
                f"<div style='font-size:0.8rem;color:#475569'>"
                f"days hit ≥ HKD {min_range:.0f}</div>"
                f"<div style='font-size:0.85rem;font-weight:600;"
                f"color:{vc};margin-top:6px'>{vt}</div></div>",
                unsafe_allow_html=True)

    if df_i is not None and len(df_i) > 5:
        st.markdown(f"**Intraday ({d_intra}) — last 5 days**")
        vwap = ((df_i["High"]+df_i["Low"]+df_i["Close"])/3*df_i["Volume"]).cumsum() \
               / df_i["Volume"].cumsum()
        bc_i = ["#16a34a" if c>=o else "#dc2626"
                for c,o in zip(df_i["Close"],df_i["Open"])]
        fig_i = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.75,0.25], vertical_spacing=0.03)
        fig_i.add_trace(go.Candlestick(
            x=df_i.index, open=df_i["Open"], high=df_i["High"],
            low=df_i["Low"], close=df_i["Close"],
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626"), row=1, col=1)
        fig_i.add_trace(go.Scatter(
            x=df_i.index, y=vwap,
            line=dict(color="#f59e0b", width=1.5, dash="dash")), row=1, col=1)
        fig_i.add_trace(go.Bar(
            x=df_i.index, y=df_i["Volume"],
            marker_color=bc_i, opacity=0.7), row=2, col=1)
        fig_i.update_layout(
            height=340, margin=dict(l=0,r=0,t=10,b=0),
            xaxis_rangeslider_visible=False,
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
            yaxis=dict(title="Price",gridcolor="#f1f5f9"),
            yaxis2=dict(title="Volume",gridcolor="#f1f5f9"),
            xaxis2=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig_i, use_container_width=True)

    st.markdown(
        "<span style='color:#94a3b8;font-size:0.74rem'>"
        "Results cached in session — click Scan again to refresh. "
        "Data via yfinance · Not financial advice.</span>",
        unsafe_allow_html=True)
