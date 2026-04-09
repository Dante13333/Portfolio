"""
time_analysis.py
Time & Weekday Player Behaviour Analysis
─────────────────────────────────────────
Three danger zones:  09:30–10:00  |  11:30–13:00  |  14:30–16:00
Three day types:     Monday trap  |  Tue/Wed clean  |  Friday squeeze

Answers:
  - What time is it NOW and what does that mean for player behaviour?
  - What day is it and how does that change who is active?
  - Hourly volume profile: where does the real money move?
  - Historical: how has THIS stock behaved at THIS hour across past weeks?
  - Gap fill tracker: do Monday gaps fill by Wednesday?
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import time
from datetime import datetime, timedelta
import pytz

HK_TZ = pytz.timezone("Asia/Hong_Kong")

STOCKS = {
    "0100.HK": "MiniMax Group",
    "2513.HK": "Zhipu / Knowledge Atlas",
}

# ── TIME ZONE DEFINITIONS ─────────────────────────────────────────────
SESSIONS = {
    "open_auction":  (9, 30,  10,  0),
    "inst_flow":     (10,  0, 11, 30),
    "lunch":         (11, 30, 13,  0),
    "afternoon":     (13,  0, 14, 30),
    "close_auction": (14, 30, 16,  0),
}

SESSION_LABELS = {
    "open_auction":  "Opening Auction",
    "inst_flow":     "Institutional Flow",
    "lunch":         "Lunch Lull",
    "afternoon":     "Afternoon Re-open",
    "close_auction": "Closing Auction",
}

SESSION_COLORS = {
    "open_auction":  "#dc2626",
    "inst_flow":     "#16a34a",
    "lunch":         "#f59e0b",
    "afternoon":     "#2563eb",
    "close_auction": "#8b5cf6",
}

# Player behaviour description per session
SESSION_BEHAVIOUR = {
    "open_auction": {
        "danger": "HIGH",
        "who":    "Retail orders from overnight + market makers",
        "game":   "First 30min move is often a TRAP. MM fill overnight retail orders, "
                  "push price to obvious levels to trigger stops, then reverse.",
        "avoid":  "Never chase the opening spike. Wait for the reversal candle.",
        "signal": "If open spike reverses within 15min → fade it hard. "
                  "Volume will be high but price won't hold = trapped retail.",
        "color":  "#dc2626",
    },
    "inst_flow": {
        "danger": "LOW",
        "who":    "Institutional desks, algorithm flow",
        "game":   "Real trend forms here. Institutions execute their morning orders. "
                  "Volume is high AND price moves — this is genuine directional flow.",
        "avoid":  "Don't fade a clean move in this window — it is backed by real money.",
        "signal": "Trade WITH the trend in this window. RSI divergence here is very reliable.",
        "color":  "#16a34a",
    },
    "lunch": {
        "danger": "HIGH",
        "who":    "Thin order book, local retail traders, algos hunting stops",
        "game":   "Low liquidity = easy for small players to push price. "
                  "Moves in this window are frequently reversed in the afternoon. "
                  "Classic stop-hunt territory.",
        "avoid":  "Any breakout in this window is suspect. "
                  "High risk of being whipsawed both ways.",
        "signal": "If a key level breaks at lunch and volume is LOW → it is a fake. "
                  "Wait for 13:00 re-open to confirm.",
        "color":  "#f59e0b",
    },
    "afternoon": {
        "danger": "MEDIUM",
        "who":    "Shanghai-HK connect flow, afternoon institutional rebalancing",
        "game":   "Often reverses the morning direction as mainland money flows in. "
                  "Strong institutional rebalancing at 13:00–13:30.",
        "avoid":  "Be careful holding morning positions into the afternoon — "
                  "the reversal can be sharp.",
        "signal": "If afternoon open gaps against morning trend → morning trend is exhausted. "
                  "Trade the reversal.",
        "color":  "#2563eb",
    },
    "close_auction": {
        "danger": "HIGH",
        "who":    "Fund managers window-dressing, position squaring, stop runs",
        "game":   "Funds push favourite stocks up at close (window dressing). "
                  "Stops get hunted aggressively in last 30min. "
                  "Volume spikes near close are mostly positioning, not real signals.",
        "avoid":  "Never initiate new positions in the last 30min unless you have "
                  "a very clear trapped-player setup.",
        "signal": "A big close-hour volume spike with a long wick = stop hunt. "
                  "Fade next morning open.",
        "color":  "#8b5cf6",
    },
}

WEEKDAY_BEHAVIOUR = {
    0: {  # Monday
        "name": "Monday",
        "type": "TRAP DAY",
        "color": "#dc2626",
        "icon": "🪤",
        "who": "Weekend news traders, gap chasers, early retail",
        "game": "Weekend news creates gaps. Retail chases the gap on open. "
                "Institutions fade the gap — they don't trust weekend news. "
                "Monday gaps fill by Wednesday ~65% of the time.",
        "strategy": "FADE the Monday gap. Wait for the opening fake move to exhaust "
                    "(usually 09:30–10:15), then trade the reversal back toward Friday's close. "
                    "Do NOT hold short positions over the weekend into Monday.",
        "traps": [
            "Gap up open → retail FOMO buys → smart money distributes → fades back",
            "Gap down open → retail panic sells → smart money absorbs → recovers",
            "Stop hunt below Friday's low → then rips higher (classic Monday trap)",
        ],
        "best_window": "10:00–11:30 after the open trap plays out",
    },
    1: {  # Tuesday
        "name": "Tuesday",
        "type": "TREND DAY",
        "color": "#16a34a",
        "icon": "📈",
        "who": "Institutional desks, systematic funds",
        "game": "Cleanest trend day. Institutions have digested Monday's news "
                "and execute their weekly plans. Less manipulation, more genuine flow.",
        "strategy": "Trade WITH momentum. RSI divergences are most reliable today. "
                    "A Tuesday breakout has higher follow-through than any other day.",
        "traps": [
            "Opening move is usually more reliable than Monday",
            "Lunchtime dips are buyable if morning trend was up",
        ],
        "best_window": "10:00–11:30 and 13:00–14:00",
    },
    2: {  # Wednesday
        "name": "Wednesday",
        "type": "TREND DAY",
        "color": "#16a34a",
        "icon": "📊",
        "who": "Institutional desks, macro flow",
        "game": "Mid-week continuation. Strongest volume profile. "
                "Institutional flow cleanest. Trends started Tuesday often extend.",
        "strategy": "Best day for momentum trades. Smart money divergence signals "
                    "most reliable. Monday gaps often fully filled by today.",
        "traps": [
            "Wednesday afternoon can see pre-Thursday profit taking",
            "Watch for reversal near Wednesday close if Tue/Wed trend was strong",
        ],
        "best_window": "09:45–11:30 and 13:00–14:30",
    },
    3: {  # Thursday
        "name": "Thursday",
        "type": "REVERSAL DAY",
        "color": "#f59e0b",
        "icon": "🔄",
        "who": "Short-term funds taking profits, pre-weekend hedgers",
        "game": "Often reverses Tuesday/Wednesday trend as short-term players "
                "take profits before Friday risk. Watch for exhaustion signals.",
        "strategy": "If Tue/Wed trend was strong → Thursday is the profit-taking day. "
                    "Look for RSI divergence and volume fade as signal to exit longs/shorts.",
        "traps": [
            "Continuation of Tue/Wed move often traps latecomers on Thursday",
            "Smart money exits Thursday → retail still buying the trend",
        ],
        "best_window": "Morning: still Tue/Wed trend. Afternoon: watch for reversal",
    },
    4: {  # Friday
        "name": "Friday",
        "type": "SQUEEZE DAY",
        "color": "#8b5cf6",
        "icon": "⚡",
        "who": "Weekend risk-off traders, stop hunters, position squarers",
        "game": "Position squaring before the weekend. Stops get hunted aggressively "
                "in final 2 hours. Moves are exaggerated and often fake. "
                "Do NOT hold losing positions into the weekend.",
        "strategy": "Morning: trade normally. After 14:00: be very cautious. "
                    "Friday close spikes are almost always stop hunts — fade them. "
                    "Cut losing positions before 15:30 — do not hold over weekend.",
        "traps": [
            "Friday afternoon stop hunt above/below week's key levels",
            "Late Friday spike up → retail chases → dumps Monday open",
            "Late Friday dump → retail panics → Monday gap fill",
        ],
        "best_window": "09:45–11:30 only. Avoid afternoon except for exits.",
    },
}

# ── DATA FETCH ────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday(ticker, interval="5m", period="30d"):
    try:
        df = yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(HK_TZ)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily(ticker, period="6mo"):
    try:
        df = yf.Ticker(ticker).history(
            period=period, interval="1d", auto_adjust=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(HK_TZ)
        return df
    except Exception:
        return pd.DataFrame()

# ── ANALYSIS FUNCTIONS ────────────────────────────────────────────────
def get_current_session(now_hk):
    h, m = now_hk.hour, now_hk.minute
    t = h * 60 + m
    for key, (sh, sm, eh, em) in SESSIONS.items():
        if sh * 60 + sm <= t < eh * 60 + em:
            return key
    return None

def build_hourly_profile(df):
    """
    For each hour 9–16, compute:
    - avg volume
    - avg absolute price change %
    - up % vs down %
    - avg candle range as % of price
    """
    if df.empty:
        return pd.DataFrame()
    df2 = df.copy()
    df2["hour"]       = df2.index.hour
    df2["abs_chg_pct"]= (df2["Close"] - df2["Open"]).abs() / df2["Open"] * 100
    df2["range_pct"]  = (df2["High"] - df2["Low"]) / df2["Open"] * 100
    df2["up"]         = (df2["Close"] >= df2["Open"]).astype(int)
    grp = df2[df2["hour"].between(9, 15)].groupby("hour").agg(
        avg_volume   =("Volume",      "mean"),
        avg_move_pct =("abs_chg_pct", "mean"),
        avg_range_pct=("range_pct",   "mean"),
        up_pct       =("up",          "mean"),
        count        =("Volume",      "count"),
    ).reset_index()
    grp["up_pct"] *= 100
    return grp

def build_weekday_profile(df_daily):
    """Per weekday: avg return, avg volume, up%, avg range"""
    if df_daily.empty:
        return pd.DataFrame()
    df2 = df_daily.copy()
    df2["weekday"]    = df2.index.weekday
    df2["day_ret_pct"]= (df2["Close"] - df2["Open"]) / df2["Open"] * 100
    df2["range_pct"]  = (df2["High"]  - df2["Low"])  / df2["Open"] * 100
    df2["up"]         = (df2["Close"] >= df2["Open"]).astype(int)
    grp = df2.groupby("weekday").agg(
        avg_ret   =("day_ret_pct", "mean"),
        avg_range =("range_pct",   "mean"),
        avg_volume=("Volume",      "mean"),
        up_pct    =("up",          "mean"),
        count     =("weekday",     "count"),
    ).reset_index()
    grp["up_pct"] *= 100
    grp["day_name"] = grp["weekday"].map(
        {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri"})
    return grp

def build_session_profile(df):
    """Per session: avg volume, avg move, up%"""
    if df.empty:
        return {}
    df2  = df.copy()
    df2["hour"] = df2.index.hour
    df2["min"]  = df2.index.minute
    df2["t"]    = df2["hour"] * 60 + df2["min"]
    df2["up"]   = (df2["Close"] >= df2["Open"]).astype(int)
    df2["move"] = (df2["Close"] - df2["Open"]).abs() / df2["Open"] * 100

    result = {}
    for key, (sh, sm, eh, em) in SESSIONS.items():
        s_t = sh * 60 + sm
        e_t = eh * 60 + em
        sub = df2[(df2["t"] >= s_t) & (df2["t"] < e_t)]
        if len(sub) == 0:
            continue
        result[key] = {
            "avg_volume": sub["Volume"].mean(),
            "avg_move":   sub["move"].mean(),
            "up_pct":     sub["up"].mean() * 100,
            "n_bars":     len(sub),
        }
    return result

def find_monday_gaps(df_daily):
    """Find all Monday gaps and whether they filled by Wednesday close."""
    if df_daily.empty or len(df_daily) < 5:
        return pd.DataFrame()
    df2 = df_daily.copy()
    df2["weekday"] = df2.index.weekday
    rows = []
    mondays = df2[df2["weekday"] == 0]
    for idx, row in mondays.iterrows():
        # Find prior Friday
        prior = df2[df2.index < idx]
        if prior.empty:
            continue
        fri = prior.iloc[-1]
        gap = row["Open"] - fri["Close"]
        gap_pct = gap / fri["Close"] * 100
        if abs(gap_pct) < 0.3:
            continue  # Not a meaningful gap
        # Check if filled by Wednesday
        after = df2[df2.index > idx].head(3)
        filled = False
        fill_day = None
        if gap > 0:  # gap up — filled when price drops back to fri close
            for fill_idx, fill_row in after.iterrows():
                if fill_row["Low"] <= fri["Close"]:
                    filled = True
                    fill_day = fill_idx.strftime("%a")
                    break
        else:  # gap down — filled when price rises back to fri close
            for fill_idx, fill_row in after.iterrows():
                if fill_row["High"] >= fri["Close"]:
                    filled = True
                    fill_day = fill_idx.strftime("%a")
                    break
        rows.append({
            "monday":   idx.strftime("%Y-%m-%d"),
            "gap_pct":  round(gap_pct, 2),
            "direction":"UP" if gap > 0 else "DOWN",
            "filled":   filled,
            "fill_day": fill_day if filled else "Not filled",
            "fri_close":round(float(fri["Close"]), 2),
            "mon_open": round(float(row["Open"]), 2),
        })
    return pd.DataFrame(rows)

def hourly_heatmap_by_weekday(df):
    """Build hour × weekday matrix of avg return %"""
    if df.empty:
        return pd.DataFrame()
    df2 = df.copy()
    df2["hour"]    = df2.index.hour
    df2["weekday"] = df2.index.weekday
    df2["ret"]     = (df2["Close"] - df2["Open"]) / df2["Open"] * 100
    pivot = df2[df2["hour"].between(9,15)].groupby(
        ["weekday","hour"])["ret"].mean().unstack("hour")
    pivot.index = [["Mon","Tue","Wed","Thu","Fri"][i]
                   for i in pivot.index if i < 5]
    return pivot

# ── RENDER HELPERS ────────────────────────────────────────────────────
def danger_pill(level):
    colors = {"HIGH":"#dc2626","MEDIUM":"#f59e0b","LOW":"#16a34a"}
    c = colors.get(level,"#94a3b8")
    return (f"<span style='background:{c};color:white;font-size:0.68rem;"
            f"padding:2px 8px;border-radius:4px;font-weight:600'>{level} RISK</span>")

def session_card(key, data=None):
    b  = SESSION_BEHAVIOUR[key]
    c  = b["color"]
    sh, sm, eh, em = SESSIONS[key]
    time_str = f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d} HKT"
    vol_str  = ""
    if data and key in data:
        v = data[key]
        vol_str = (f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:4px'>"
                   f"Historical: avg move {v['avg_move']:.2f}% · "
                   f"up {v['up_pct']:.0f}% of bars</div>")
    st.markdown(
        f"<div style='border:1px solid {c};border-left:4px solid {c};"
        f"border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:10px;"
        f"background:rgba(0,0,0,0.02)'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='font-weight:600;color:{c}'>{SESSION_LABELS[key]}</span>"
        f"<span style='font-size:0.78rem;color:#64748b'>{time_str}</span>"
        f"{danger_pill(b['danger'])}</div>"
        f"<div style='font-size:0.8rem;color:#374151;margin-top:6px'>"
        f"<b>Who:</b> {b['who']}</div>"
        f"<div style='font-size:0.8rem;color:#374151;margin-top:3px'>"
        f"<b>Game:</b> {b['game']}</div>"
        f"<div style='font-size:0.8rem;color:#166534;margin-top:3px'>"
        f"<b>Signal:</b> {b['signal']}</div>"
        f"<div style='font-size:0.78rem;color:#991b1b;margin-top:3px'>"
        f"<b>Avoid:</b> {b['avoid']}</div>"
        f"{vol_str}</div>",
        unsafe_allow_html=True)

def weekday_card(wd, profile_row=None):
    b = WEEKDAY_BEHAVIOUR[wd]
    c = b["color"]
    hist = ""
    if profile_row is not None:
        hist = (f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:4px'>"
                f"Historical ({int(profile_row['count'])} sessions): "
                f"avg return {profile_row['avg_ret']:+.2f}% · "
                f"up {profile_row['up_pct']:.0f}% · "
                f"avg range {profile_row['avg_range']:.2f}%</div>")
    traps_html = "".join(
        f"<div style='font-size:0.78rem;color:#991b1b'>⚠ {t}</div>"
        for t in b["traps"])
    st.markdown(
        f"<div style='border:1px solid {c};border-left:4px solid {c};"
        f"border-radius:0 8px 8px 0;padding:14px 16px;margin-bottom:10px;"
        f"background:rgba(0,0,0,0.02)'>"
        f"<div style='display:flex;gap:10px;align-items:center;margin-bottom:8px'>"
        f"<span style='font-size:1.4rem'>{b['icon']}</span>"
        f"<span style='font-weight:700;color:{c};font-size:1rem'>{b['name']}</span>"
        f"<span style='background:{c};color:white;font-size:0.68rem;"
        f"padding:2px 8px;border-radius:4px'>{b['type']}</span></div>"
        f"<div style='font-size:0.8rem;color:#374151'>"
        f"<b>Who is active:</b> {b['who']}</div>"
        f"<div style='font-size:0.8rem;color:#374151;margin-top:3px'>"
        f"<b>Their game:</b> {b['game']}</div>"
        f"<div style='font-size:0.8rem;color:#166534;margin-top:3px'>"
        f"<b>Your strategy:</b> {b['strategy']}</div>"
        f"<div style='font-size:0.8rem;color:#374151;margin-top:3px'>"
        f"<b>Best window:</b> {b['best_window']}</div>"
        f"<div style='margin-top:6px'>{traps_html}</div>"
        f"{hist}</div>",
        unsafe_allow_html=True)

# ── MAIN RENDER ───────────────────────────────────────────────────────
def render():
    now_hk   = datetime.now(HK_TZ)
    weekday  = now_hk.weekday()
    cur_sess = get_current_session(now_hk)
    h_now    = now_hk.hour
    m_now    = now_hk.minute

    st.markdown(
        "## ⏰ Time & Weekday Player Behaviour &nbsp;"
        "<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        "padding:2px 7px;border-radius:5px'>TIMING</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"HKT {now_hk.strftime('%H:%M:%S')} · "
        f"{['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][weekday]}"
        f"</span>",
        unsafe_allow_html=True)
    with st.expander("📖 Metric explanations"):
        st.markdown("""
**Win rate %** — % of bars/days that closed higher than they opened in that time slot.
>55% = bullish bias. <45% = bearish bias. 50% = random (no consistent edge).

**Vol ratio** — Volume in that hour/session vs the overall average.
>1.5x = institutional orders active (real flow). <0.7x = thin market, easy to manipulate.

**Avg range %** — Average High-Low as % of price in that session.
High range + low win rate = TRAP ZONE (big moves, no direction, stop hunts).
High range + high win rate = TREND ZONE (directional institutional flow).

**Choppiness** — How much the session oscillates vs trends in a single direction.
High choppiness = fake moves and reversals. Avoid chasing in choppy sessions.

**Gap fill rate %** — % of Monday gaps that fill (price returns to Friday close) within N days.
>60% fill by Wednesday = your rule-of-thumb is statistically supported for this instrument.

**Weekday win rate** — % of that weekday where the day closed up.
Monday and Friday historically lower (gap traps, position squaring into weekend).
Tuesday-Wednesday typically highest (cleanest institutional flow days).

**Session labels:**
TRAP ZONE = high vol + low directional win rate. Retail chases, institutions fade.
INST FLOW = high vol + directional. Follow the institutional direction.
QUIET = low vol + small range. Skip or wait for a better setup.
        """)

    st.markdown("---")

    # ── CONTROLS ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    ticker   = col1.selectbox("Stock", list(STOCKS.keys()), key="ta_ticker",
                               format_func=lambda x: f"{x} — {STOCKS[x]}")
    interval = col2.selectbox("Intraday bar size", ["1m","5m","15m","30m"], index=1, key="ta_interval")

    if st.button("🔄 Refresh", key="ta_btn_refresh"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Loading time-series data…"):
        df_intra = fetch_intraday(ticker, interval, period="30d")
        df_daily = fetch_daily(ticker, period="6mo")

    # ════════════════════════════════════════════════════════════════
    # SECTION 1 — RIGHT NOW: what session + what day
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 1 · Right Now")

    now_col1, now_col2 = st.columns(2)

    with now_col1:
        st.markdown("**Current session**")
        is_market_open = (weekday < 5 and
                          (h_now > 9 or (h_now == 9 and m_now >= 30)) and
                          h_now < 16)
        if not is_market_open:
            st.markdown(
                "<div style='border:1px solid #94a3b8;border-left:4px solid #94a3b8;"
                "border-radius:0 8px 8px 0;padding:12px 16px;background:rgba(0,0,0,0.02)'>"
                "<div style='font-weight:600;color:#64748b'>Market Closed</div>"
                "<div style='font-size:0.82rem;color:#94a3b8;margin-top:4px'>"
                "HKEX opens Mon–Fri 09:30 HKT. "
                "Use this time to plan tomorrow's trades based on today's close.</div>"
                "</div>",
                unsafe_allow_html=True)
        elif cur_sess:
            b = SESSION_BEHAVIOUR[cur_sess]
            c = b["color"]
            sh, sm, eh, em = SESSIONS[cur_sess]
            mins_elapsed = (h_now*60+m_now) - (sh*60+sm)
            mins_left    = (eh*60+em) - (h_now*60+m_now)
            st.markdown(
                f"<div style='border:2px solid {c};border-radius:8px;"
                f"padding:14px 18px;background:rgba(0,0,0,0.02)'>"
                f"<div style='font-weight:700;color:{c};font-size:1.05rem'>"
                f"🔴 LIVE: {SESSION_LABELS[cur_sess]}</div>"
                f"<div style='font-size:0.78rem;color:#64748b;margin-top:2px'>"
                f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d} HKT · "
                f"{mins_elapsed}min in · {mins_left}min remaining</div>"
                f"<div style='font-size:0.85rem;color:#374151;margin-top:8px'>"
                f"<b>Who:</b> {b['who']}</div>"
                f"<div style='font-size:0.85rem;color:#374151;margin-top:4px'>"
                f"{b['game']}</div>"
                f"<div style='font-size:0.85rem;font-weight:500;color:#166534;"
                f"margin-top:8px;padding:6px 10px;background:rgba(255,255,255,0.7);"
                f"border-radius:6px'>💡 {b['signal']}</div>"
                f"<div style='font-size:0.8rem;color:#991b1b;margin-top:6px'>"
                f"⚠ {b['avoid']}</div>"
                f"</div>",
                unsafe_allow_html=True)

    with now_col2:
        st.markdown("**Today's market personality**")
        if weekday >= 5:
            st.markdown(
                "<div style='border:1px solid #94a3b8;border-left:4px solid #94a3b8;"
                "border-radius:0 8px 8px 0;padding:12px 16px;'>"
                "<div style='font-weight:600;color:#64748b'>Weekend</div>"
                "<div style='font-size:0.82rem;color:#94a3b8;margin-top:4px'>"
                "Plan for Monday gap trap. Read the weekend news but don't react — "
                "your job Monday morning is to watch who is wrong.</div></div>",
                unsafe_allow_html=True)
        else:
            b  = WEEKDAY_BEHAVIOUR[weekday]
            c  = b["color"]
            st.markdown(
                f"<div style='border:2px solid {c};border-radius:8px;"
                f"padding:14px 18px;background:rgba(0,0,0,0.02)'>"
                f"<div style='display:flex;gap:10px;align-items:center'>"
                f"<span style='font-size:1.4rem'>{b['icon']}</span>"
                f"<span style='font-weight:700;color:{c};font-size:1.05rem'>"
                f"It's {b['name']} — {b['type']}</span></div>"
                f"<div style='font-size:0.85rem;color:#374151;margin-top:8px'>"
                f"{b['game']}</div>"
                f"<div style='font-size:0.85rem;font-weight:500;color:#166534;"
                f"margin-top:8px;padding:6px 10px;background:rgba(255,255,255,0.7);"
                f"border-radius:6px'>💡 {b['strategy']}</div>"
                f"<div style='font-size:0.78rem;color:#64748b;margin-top:6px'>"
                f"Best window: {b['best_window']}</div>"
                f"</div>",
                unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 2 — HOURLY VOLUME PROFILE (historical)
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 2 · Hourly Volume & Move Profile")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Where does the real money move? Based on last 30 days of intraday data.</span>",
        unsafe_allow_html=True)

    hourly = build_hourly_profile(df_intra)
    if not hourly.empty:
        tab_vol, tab_move, tab_range = st.tabs(
            ["📊 Volume by hour", "📈 Avg move % by hour", "↕ Avg range % by hour"])

        # Colour bars by session
        def hour_color(h):
            if h < 10:   return SESSION_COLORS["open_auction"]
            if h < 11:   return SESSION_COLORS["inst_flow"]
            if h < 13:   return SESSION_COLORS["lunch"]
            if h < 14:   return SESSION_COLORS["afternoon"]
            return SESSION_COLORS["close_auction"]

        bar_colors = [hour_color(int(h)) for h in hourly["hour"]]
        hour_labels= [f"{int(h):02d}:00" for h in hourly["hour"]]

        with tab_vol:
            fig_v = go.Figure(go.Bar(
                x=hour_labels, y=hourly["avg_volume"],
                marker_color=bar_colors,
                text=[f"{v/1e6:.1f}M" if v>=1e6 else f"{v/1e3:.0f}K"
                      for v in hourly["avg_volume"]],
                textposition="outside"))
            # Session boundary markers (add_vline doesn't work with string x-axis)
            for sess, (sh,sm,eh,em) in SESSIONS.items():
                lbl = f"{sh:02d}:00"
                if lbl in hour_labels:
                    idx = hour_labels.index(lbl)
                    fig_v.add_shape(
                        type="line",
                        x0=idx - 0.5, x1=idx - 0.5,
                        y0=0, y1=1, yref="paper",
                        line=dict(color=SESSION_COLORS[sess],
                                  width=1, dash="dot"))
            fig_v.update_layout(height=300,
                                margin=dict(l=0,r=0,t=20,b=0),
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                showlegend=False,
                                yaxis=dict(title="Avg Volume",
                                           gridcolor="#f1f5f9"))
            st.plotly_chart(fig_v, use_container_width=True)

        with tab_move:
            fig_m = go.Figure(go.Bar(
                x=hour_labels, y=hourly["avg_move_pct"],
                marker_color=bar_colors,
                text=[f"{v:.2f}%" for v in hourly["avg_move_pct"]],
                textposition="outside"))
            fig_m.update_layout(height=300,
                                margin=dict(l=0,r=0,t=20,b=0),
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                yaxis=dict(title="Avg |move| %",
                                           gridcolor="#f1f5f9"))
            st.plotly_chart(fig_m, use_container_width=True)
            st.markdown(
                "<span style='font-size:0.75rem;color:#94a3b8'>"
                "Colour = session: "
                "<span style='color:#dc2626'>■</span> Open "
                "<span style='color:#16a34a'>■</span> Inst "
                "<span style='color:#f59e0b'>■</span> Lunch "
                "<span style='color:#2563eb'>■</span> PM "
                "<span style='color:#8b5cf6'>■</span> Close"
                "</span>",
                unsafe_allow_html=True)

        with tab_range:
            fig_r = go.Figure(go.Bar(
                x=hour_labels, y=hourly["avg_range_pct"],
                marker_color=bar_colors,
                text=[f"{v:.2f}%" for v in hourly["avg_range_pct"]],
                textposition="outside"))
            fig_r.update_layout(height=300,
                                margin=dict(l=0,r=0,t=20,b=0),
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                yaxis=dict(title="Avg High–Low %",
                                           gridcolor="#f1f5f9"))
            st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 3 — HOUR × WEEKDAY HEATMAP
    # "This stock at 09:30 on Monday vs 10:00 on Wednesday"
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 3 · Hour × Weekday Return Heatmap")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Average candle return % at each hour on each weekday. "
        "Red = bearish on average. Green = bullish. "
        "This shows you the recurring player patterns for THIS stock.</span>",
        unsafe_allow_html=True)

    pivot = hourly_heatmap_by_weekday(df_intra)
    if not pivot.empty:
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{int(c):02d}:00" for c in pivot.columns],
            y=list(pivot.index),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values, 2),
            texttemplate="%{text}%",
            textfont=dict(size=10),
            colorbar=dict(title="Avg ret %", thickness=12, len=0.8),
            hoverongaps=False,
        ))
        # Highlight danger zones
        for sess, (sh, sm, eh, em) in SESSIONS.items():
            if SESSION_BEHAVIOUR[sess]["danger"] == "HIGH":
                fig_heat.add_vrect(
                    x0=f"{sh:02d}:00", x1=f"{eh:02d}:00" if eh < 16
                    else "15:00",
                    fillcolor=SESSION_COLORS[sess],
                    opacity=0.07, line_width=0)
        fig_heat.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="white",
            xaxis=dict(title="Hour (HKT)"),
            yaxis=dict(title="Weekday"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown(
            "<span style='font-size:0.75rem;color:#94a3b8'>"
            "Shaded columns = danger zones (open/lunch/close). "
            "Use this to find which hour + day combinations have the most predictable behaviour "
            "for this specific stock.</span>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 4 — WEEKDAY PROFILES
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 4 · Weekday Player Profiles")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Who is trading each day and what game they play.</span>",
        unsafe_allow_html=True)

    wd_profile = build_weekday_profile(df_daily)

    wd_tabs = st.tabs(["🪤 Mon", "📈 Tue", "📊 Wed", "🔄 Thu", "⚡ Fri"])
    for i, tab in enumerate(wd_tabs):
        with tab:
            prof_row = None
            if not wd_profile.empty:
                row = wd_profile[wd_profile["weekday"] == i]
                if not row.empty:
                    prof_row = row.iloc[0]
            weekday_card(i, prof_row)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 5 — WEEKDAY VOLUME + RETURN CHART
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 5 · This Stock by Weekday (historical)")
    if not wd_profile.empty:
        fig_wd = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Avg day return % by weekday",
                            "Avg volume by weekday"])

        wd_colors = [WEEKDAY_BEHAVIOUR[int(w)]["color"]
                     for w in wd_profile["weekday"]
                     if int(w) in WEEKDAY_BEHAVIOUR]

        fig_wd.add_trace(go.Bar(
            x=wd_profile["day_name"],
            y=wd_profile["avg_ret"],
            marker_color=wd_colors,
            text=[f"{v:+.2f}%" for v in wd_profile["avg_ret"]],
            textposition="outside",
            name="Avg return"), row=1, col=1)
        fig_wd.add_hline(y=0, line_color="#e2e8f0",
                          line_width=1, row=1, col=1)

        fig_wd.add_trace(go.Bar(
            x=wd_profile["day_name"],
            y=wd_profile["avg_volume"],
            marker_color=wd_colors,
            text=[f"{v/1e6:.1f}M" if v >= 1e6
                  else f"{v/1e3:.0f}K"
                  for v in wd_profile["avg_volume"]],
            textposition="outside",
            name="Avg volume"), row=1, col=2)

        fig_wd.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            yaxis=dict(gridcolor="#f1f5f9"),
            yaxis2=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig_wd, use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 6 — SESSION REFERENCE CARDS
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 6 · Session Danger Zone Reference")
    sess_profile = build_session_profile(df_intra)
    for key in SESSIONS:
        session_card(key, sess_profile)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 7 — MONDAY GAP TRACKER
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 7 · Monday Gap Fill Tracker")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Tracks every Monday gap for this stock. "
        "Shows fill rate and which day it typically fills. "
        "Use this to calibrate how aggressively to fade Monday gaps.</span>",
        unsafe_allow_html=True)

    gaps_df = find_monday_gaps(df_daily)
    if not gaps_df.empty:
        fill_rate = gaps_df["filled"].mean() * 100
        up_gaps   = gaps_df[gaps_df["direction"] == "UP"]
        dn_gaps   = gaps_df[gaps_df["direction"] == "DOWN"]
        up_fill   = up_gaps["filled"].mean() * 100 if len(up_gaps) else 0
        dn_fill   = dn_gaps["filled"].mean() * 100 if len(dn_gaps) else 0

        mg1, mg2, mg3, mg4 = st.columns(4)
        mg1.metric("Total Monday gaps", len(gaps_df))
        mg2.metric("Fill rate (all)",   f"{fill_rate:.0f}%")
        mg3.metric("Gap-up fill rate",  f"{up_fill:.0f}%",
                   help="% of gap-up Mondays that filled back to Friday close")
        mg4.metric("Gap-down fill rate",f"{dn_fill:.0f}%")

        # Chart gap sizes
        colors_gap = ["#dc2626" if d == "DOWN" else "#16a34a"
                      for d in gaps_df["direction"]]
        fig_gap = go.Figure(go.Bar(
            x=gaps_df["monday"],
            y=gaps_df["gap_pct"],
            marker_color=colors_gap,
            text=[("✓" if f else "✗") for f in gaps_df["filled"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Gap: %{y:.2f}%<br>"
                "<extra></extra>"
            )))
        fig_gap.add_hline(y=0, line_color="#e2e8f0", line_width=1)
        fig_gap.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            yaxis=dict(title="Gap %", gridcolor="#f1f5f9"),
            xaxis=dict(title="Monday date"))
        st.plotly_chart(fig_gap, use_container_width=True)
        st.markdown(
            "<span style='font-size:0.75rem;color:#94a3b8'>"
            "✓ = gap filled by Wednesday  ✗ = not filled  "
            "Green = gap up  Red = gap down</span>",
            unsafe_allow_html=True)

        # Table
        with st.expander("📋 Full gap table"):
            st.dataframe(
                gaps_df.style.applymap(
                    lambda v: "color:#16a34a" if v is True
                              else "color:#dc2626" if v is False else "",
                    subset=["filled"]
                ).format({"gap_pct": "{:+.2f}%"}),
                use_container_width=True,
                hide_index=True)
    else:
        st.info("Not enough daily data to compute Monday gaps. "
                "Try fetching 6mo+ of history.")

    st.markdown(
        "<br><span style='color:#94a3b8;font-size:0.74rem'>"
        "Historical patterns are tendencies, not guarantees. "
        "Data via yfinance. Not financial advice.</span>",
        unsafe_allow_html=True)
