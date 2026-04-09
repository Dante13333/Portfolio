"""
daily_strategy.py — Daily Strategy Briefing
Real-time strategy for every position in portfolio + watchlist.
Runs each morning before market open or during session.

Per position combines:
  1. Technical signals  — RSI, MACD, BB, volume, cycle state
  2. Market context     — gap, weekday, current session
  3. Portfolio view     — allocation vs optimal, cash deployment

Output per position:
  - TODAY action: HOLD / ADD / REDUCE / EXIT / WAIT
  - Best session to act
  - Updated stop & target
  - Confidence level
  - Plain-English reasoning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import time, pytz

from db_manager import (
    get_portfolio_full, get_latest_capital,
    init_portfolio_extended, init_activity_log, log_activity,
    upsert_position_full, get_conn,
)

# ── DAILY FORECAST DB ────────────────────────────────────────────────
def _init_forecast_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_forecast (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT NOT NULL,
            ticker       TEXT NOT NULL,
            name         TEXT,
            action       TEXT,
            est_pnl      REAL,
            est_pnl_pct  REAL,
            est_note     TEXT,
            open_price   REAL,
            close_price  REAL,
            actual_pnl   REAL,
            actual_pct   REAL,
            correct      INTEGER,
            logged_at    TEXT DEFAULT (datetime('now')),
            UNIQUE(date, ticker)
        )
    """)
    conn.commit(); conn.close()

def save_forecast(date, ticker, name, action, est_pnl, est_pnl_pct, est_note, open_price):
    _init_forecast_db()
    conn = get_conn()
    conn.execute("""
        INSERT INTO daily_forecast
        (date,ticker,name,action,est_pnl,est_pnl_pct,est_note,open_price)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(date,ticker) DO UPDATE SET
            action=excluded.action, est_pnl=excluded.est_pnl,
            est_pnl_pct=excluded.est_pnl_pct, est_note=excluded.est_note,
            open_price=excluded.open_price, logged_at=datetime('now')
    """, (date,ticker,name,action,est_pnl,est_pnl_pct,est_note,open_price))
    conn.commit(); conn.close()

def update_actual(date, ticker, close_price, qty, avg_cost):
    _init_forecast_db()
    conn = get_conn()
    row = conn.execute(
        "SELECT est_pnl, open_price, action FROM daily_forecast WHERE date=? AND ticker=?",
        (date, ticker)).fetchone()
    if not row:
        conn.close(); return
    open_p = row[1] or avg_cost
    actual_pnl = (close_price - open_p) * qty if qty > 0 else 0
    actual_pct = (close_price - open_p) / open_p * 100 if open_p > 0 else 0
    est_pnl    = row[0] or 0
    # Correct if actual_pnl and est_pnl have same sign, or action was EXIT and pnl > 0
    correct = 1 if (est_pnl * actual_pnl > 0) or                    ("EXIT" in str(row[2]) and actual_pnl >= 0) else 0
    conn.execute("""
        UPDATE daily_forecast SET close_price=?,actual_pnl=?,actual_pct=?,correct=?
        WHERE date=? AND ticker=?
    """, (close_price, round(actual_pnl,2), round(actual_pct,2), correct, date, ticker))
    conn.commit(); conn.close()

def get_forecast_history(days=30):
    _init_forecast_db()
    conn = get_conn()
    import pandas as pd
    df = pd.read_sql_query("""
        SELECT * FROM daily_forecast
        WHERE date >= date('now',?)
        ORDER BY date DESC, ticker
    """, conn, params=(f'-{days} days',))
    conn.close()
    return df

from portfolio_manager import get_monitor_pos
try:
    from money_flow import get_ticker_flow as _get_flow
except Exception:
    _get_flow = lambda t, p='1mo': {}

HK_TZ = pytz.timezone("Asia/Hong_Kong")

SESSIONS = [
    ("Open",      9,  30, 10,  0),
    ("Morning",  10,   0, 11, 30),
    ("Lunch",    11,  30, 13,  0),
    ("Afternoon",13,   0, 14, 30),
    ("Close",    14,  30, 16,  0),
]

WEEKDAY_BIAS = {
    0: ("Monday",   "gap-trap day — wait for direction after 10:00",   -0.10),
    1: ("Tuesday",  "cleanest trend day — follow institutional flow",   +0.15),
    2: ("Wednesday","continuation day — best for adding to winners",    +0.12),
    3: ("Thursday", "pre-Friday caution — reversals common after 14:00",-0.05),
    4: ("Friday",   "position squaring — stops hunted, reduce risk",    -0.15),
}

# ── VARIANTS ──────────────────────────────────────────────────────────
def _var(ticker):
    v=[ticker]; code=ticker.replace(".HK","")
    if code.isdigit():
        v.append(str(int(code))+".HK")
        v.append(code.zfill(4)+".HK")
    return list(dict.fromkeys(v))

# ── DATA FETCH ────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live(ticker):
    for t in _var(ticker):
        try:
            info = yf.Ticker(t).fast_info
            p    = getattr(info,"last_price",None)
            if p:
                return {
                    "price":     float(p),
                    "prev":      getattr(info,"previous_close",None),
                    "day_high":  getattr(info,"day_high",None),
                    "day_low":   getattr(info,"day_low",None),
                    "open":      getattr(info,"open",None),
                }
        except Exception: pass
    return {}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_daily(ticker, period="3mo"):
    for t in _var(ticker):
        try:
            df = yf.Ticker(t).history(period=period,interval="1d",auto_adjust=True)
            if len(df)>=15: return df
            time.sleep(0.2)
        except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday(ticker):
    for t in _var(ticker):
        try:
            df = yf.Ticker(t).history(period="2d",interval="15m",auto_adjust=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize("UTC")
                df.index = df.index.tz_convert(HK_TZ)
            if len(df)>4: return df
        except Exception: pass
    return pd.DataFrame()

# ── INDICATORS ────────────────────────────────────────────────────────
def _rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0).ewm(com=p-1,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(com=p-1,adjust=False).mean()
    r=100-100/(1+g/l.replace(0,np.nan))
    v=r.dropna()
    return float(v.iloc[-1]) if len(v) else 50

def _macd(s):
    ml=s.ewm(span=12,adjust=False).mean()-s.ewm(span=26,adjust=False).mean()
    sl=ml.ewm(span=9,adjust=False).mean()
    hist=ml-sl
    return float(ml.iloc[-1]),float(sl.iloc[-1]),float(hist.iloc[-1])

def _bb(s,p=20):
    if len(s)<p: return 50
    mid=s.rolling(p).mean(); std=s.rolling(p).std()
    pct=((s-mid+2*std)/(4*std+1e-9)*100).clip(0,100)
    return float(pct.iloc[-1])

def _atr(df,p=14):
    tr=pd.concat([df["High"]-df["Low"],
                  (df["High"]-df["Close"].shift()).abs(),
                  (df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    return float(tr.ewm(com=p-1,adjust=False).mean().iloc[-1])

def _chop(df,p=14):
    if len(df)<p+2: return 50
    tr=pd.concat([df["High"]-df["Low"],
                  (df["High"]-df["Close"].shift()).abs(),
                  (df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    ci=100*np.log10(tr.rolling(p).sum()/(
        df["High"].rolling(p).max()-df["Low"].rolling(p).min()+1e-9))/np.log10(p)
    return float(ci.clip(0,100).iloc[-1])

def _vol_ratio(df):
    avg=df["Volume"].rolling(20).mean()
    return float(df["Volume"].iloc[-1]/avg.iloc[-1]) if float(avg.iloc[-1])>0 else 1

def _trend_slope(s,n=10):
    if len(s)<n: return 0
    x=np.arange(n); y=s.tail(n).values
    slope=float(np.polyfit(x,y,1)[0])
    return slope/float(s.mean())*100

# ── CORE STRATEGY ENGINE ──────────────────────────────────────────────
def analyse_position(ticker, name, avg_cost, qty, target, stop,
                     status, now_hk, capital, total_invested):
    """
    Return full strategy recommendation for one position.
    """
    q   = fetch_live(ticker)
    df  = fetch_daily(ticker)
    dfi = fetch_intraday(ticker)

    price = q.get("price")
    prev  = q.get("prev")
    open_ = q.get("open")
    dh    = q.get("day_high")
    dl    = q.get("day_low")

    if not price:
        return {"ticker":ticker,"name":name,"error":"No live price"}

    # ── Basic position math ───────────────────────────────────────────
    pnl     = (price-avg_cost)*qty if qty>0 else 0
    pnl_pct = (price-avg_cost)/avg_cost*100 if avg_cost>0 else 0
    gap_pct = (open_-prev)/prev*100 if open_ and prev else 0
    day_pct = (price-prev)/prev*100 if prev else 0
    alloc   = (qty*avg_cost)/capital*100 if capital>0 else 0

    # ── Technical signals ─────────────────────────────────────────────
    signals = {}
    tech_score = 0  # positive = bullish, negative = bearish

    if not df.empty and len(df)>=20:
        rsi_v  = _rsi(df["Close"])
        ml,sl,macd_h = _macd(df["Close"])
        bb_v   = _bb(df["Close"])
        chop_v = _chop(df)
        vol_r  = _vol_ratio(df)
        atr_v  = _atr(df)
        slope  = _trend_slope(df["Close"])
        ma20   = float(df["Close"].rolling(20).mean().iloc[-1])
        ma50   = float(df["Close"].rolling(50).mean().iloc[-1]) if len(df)>=50 else None

        signals["rsi"]     = round(rsi_v,1)
        signals["macd_h"]  = round(macd_h,4)
        signals["bb_pct"]  = round(bb_v,1)
        signals["chop"]    = round(chop_v,1)
        signals["vol_r"]   = round(vol_r,2)
        signals["atr"]     = round(atr_v,2)
        signals["slope"]   = round(slope,3)
        signals["ma20"]    = round(ma20,2)
        signals["above_ma20"] = price > ma20
        signals["above_ma50"] = price > ma50 if ma50 else None

        # RSI
        if rsi_v<30:   tech_score+=2;  signals["rsi_signal"]="Oversold — bounce likely"
        elif rsi_v<40: tech_score+=1;  signals["rsi_signal"]="Near oversold"
        elif rsi_v>70: tech_score-=2;  signals["rsi_signal"]="Overbought — reversal risk"
        elif rsi_v>60: tech_score-=1;  signals["rsi_signal"]="Near overbought"
        else:          signals["rsi_signal"]="Neutral"

        # MACD
        if macd_h>0 and macd_h>0.001:
            tech_score+=1; signals["macd_signal"]="Bullish momentum"
        elif macd_h<0 and macd_h<-0.001:
            tech_score-=1; signals["macd_signal"]="Bearish momentum"
        else:
            signals["macd_signal"]="Flat / crossing"

        # BB position
        if bb_v<15:    tech_score+=2;  signals["bb_signal"]="At lower band — oversold"
        elif bb_v<30:  tech_score+=1;  signals["bb_signal"]="Below midline"
        elif bb_v>85:  tech_score-=2;  signals["bb_signal"]="At upper band — overbought"
        elif bb_v>70:  tech_score-=1;  signals["bb_signal"]="Above midline"
        else:          signals["bb_signal"]="Midrange"

        # Volume
        if vol_r>2.0:  signals["vol_signal"]="Volume spike — institutional activity"
        elif vol_r>1.3:signals["vol_signal"]="Above average volume"
        elif vol_r<0.7:signals["vol_signal"]="Low volume — thin market"
        else:          signals["vol_signal"]="Normal volume"

        # Trend
        if slope<-0.3: tech_score-=2; signals["trend_signal"]="Downtrending"
        elif slope<0:  tech_score-=1; signals["trend_signal"]="Mild downtrend"
        elif slope>0.3:tech_score+=1; signals["trend_signal"]="Uptrending"
        else:          signals["trend_signal"]="Flat"

        # Choppiness
        if chop_v>61.8:
            signals["chop_signal"]="Oscillating — good for range trading"
        elif chop_v<38:
            tech_score-=1; signals["chop_signal"]="Trending — not oscillating"
        else:
            signals["chop_signal"]="Mixed"
    else:
        atr_v=price*0.02; chop_v=50; rsi_v=50

    # ── Intraday context ──────────────────────────────────────────────
    if not dfi.empty and len(dfi)>4:
        vwap = ((dfi["High"]+dfi["Low"]+dfi["Close"])/3*dfi["Volume"]).cumsum()\
               /dfi["Volume"].cumsum()
        signals["vwap"]    = round(float(vwap.iloc[-1]),4)
        signals["above_vwap"] = price > float(vwap.iloc[-1])
        if signals["above_vwap"]:
            tech_score+=0.5; signals["vwap_signal"]="Price above VWAP — intraday bullish"
        else:
            tech_score-=0.5; signals["vwap_signal"]="Price below VWAP — intraday bearish"

    # ── Gap analysis ──────────────────────────────────────────────────
    if abs(gap_pct)>1.0:
        signals["gap"]  = round(gap_pct,2)
        wd = now_hk.weekday()
        if gap_pct>0:
            signals["gap_signal"]=(
                "Gap up on Monday — likely trap, fade if no follow-through" if wd==0
                else "Gap up — watch for continuation vs fade")
            tech_score-=0.5  # gaps often fade
        else:
            signals["gap_signal"]=(
                "Gap down on Monday — wait, likely fills by Wednesday" if wd==0
                else "Gap down — watch for recovery vs continuation")
            tech_score+=0.3

    # ── Money flow context ───────────────────────────────────────────
    flow_ctx = {}
    try:
        flow_ctx = _get_flow(ticker, "1mo")
    except Exception:
        flow_ctx = {}

    flow_score   = flow_ctx.get("flow_score", 0)
    flow_signal  = flow_ctx.get("signal", "—")
    flow_driver  = flow_ctx.get("driver", "—")
    flow_sector  = flow_ctx.get("sector", "—")
    flow_drv_note= flow_ctx.get("driver_note", "")

    # Adjust tech_score based on sector money flow
    if flow_score >= 50:
        tech_score += 1.5
        signals["flow_note"] = f"Strong sector inflow ({flow_sector}) — tailwind"
    elif flow_score >= 20:
        tech_score += 0.8
        signals["flow_note"] = f"Sector inflow ({flow_sector}) — mild tailwind"
    elif flow_score <= -50:
        tech_score -= 1.5
        signals["flow_note"] = f"Strong sector outflow ({flow_sector}) — headwind"
    elif flow_score <= -20:
        tech_score -= 0.8
        signals["flow_note"] = f"Sector outflow ({flow_sector}) — mild headwind"
    else:
        signals["flow_note"] = f"Sector neutral ({flow_sector})"

    # Smart money note
    if "Smart" in flow_driver and flow_score > 10:
        signals["flow_note"] += " · Smart money accumulating sector"
        tech_score += 0.5
    elif "Crowd" in flow_driver and flow_score > 20:
        signals["flow_note"] += " · Crowd chasing — reversal risk elevated"
        tech_score -= 0.3

    signals["flow_score"]  = flow_score
    signals["flow_signal"] = flow_signal
    signals["flow_driver"] = flow_driver
    signals["flow_sector"] = flow_sector

    # ── Weekday context ───────────────────────────────────────────────
    wd      = now_hk.weekday()
    wd_name, wd_note, wd_adj = WEEKDAY_BIAS.get(wd, ("?","",0))
    tech_score += wd_adj * 2   # weekday bias influences score

    # ── Current session ───────────────────────────────────────────────
    h,m = now_hk.hour, now_hk.minute
    mins= h*60+m
    curr_session = None
    for sname,sh,sm,eh,em in SESSIONS:
        if sh*60+sm<=mins<eh*60+em:
            curr_session=sname; break

    # ── Action recommendation ─────────────────────────────────────────
    # For active positions (qty > 0)
    # For watchlist (qty == 0) → entry signal

    if qty > 0:
        # Position management
        # Check stop hit
        if stop and price <= stop:
            action="🔴 EXIT — Stop hit"
            action_color="#dc2626"
            confidence="HIGH"
            action_reason=f"Price {price:,.4f} at or below stop {stop:,.4f}. Exit now."
        elif stop and (price-stop)/price < 0.02:
            action="⚠️ REDUCE — Near stop"
            action_color="#f59e0b"
            confidence="HIGH"
            action_reason=f"Price within 2% of stop. Reduce size or tighten stop."
        elif target and price >= target:
            action="🎯 EXIT — Target reached"
            action_color="#16a34a"
            confidence="HIGH"
            action_reason=f"Price {price:,.4f} reached target {target:,.4f}. Take profit."
        elif tech_score <= -3 and chop_v < 45:
            action="🔴 EXIT / REDUCE"
            action_color="#dc2626"
            confidence="HIGH"
            action_reason="Multiple bearish signals + trending down. Not suitable for range trading."
        elif tech_score <= -2:
            action="⚠️ REDUCE"
            action_color="#f59e0b"
            confidence="MEDIUM"
            action_reason="Bearish signals accumulating. Reduce exposure."
        elif tech_score >= 3 and chop_v >= 55:
            action="✅ ADD / HOLD"
            action_color="#16a34a"
            confidence="HIGH"
            action_reason="Strong bullish signals + good choppiness. Consider adding on pullbacks."
        elif tech_score >= 1:
            action="✅ HOLD"
            action_color="#16a34a"
            confidence="MEDIUM"
            action_reason="Mildly bullish. Hold current size."
        elif abs(tech_score) < 1:
            action="⏸ HOLD / WAIT"
            action_color="#64748b"
            confidence="LOW"
            action_reason="Mixed signals. No clear edge today — wait for better setup."
        else:
            action="⚠️ MONITOR"
            action_color="#f59e0b"
            confidence="LOW"
            action_reason="Slight bearish bias. Watch for confirmation before reducing."
    else:
        # Watchlist — entry signal
        if tech_score >= 3 and chop_v >= 55:
            action="🟢 ENTER — Strong signal"
            action_color="#16a34a"
            confidence="HIGH"
            action_reason="Multiple bullish signals + oscillating. Good entry opportunity."
        elif tech_score >= 2:
            action="🟡 WATCH — Near entry"
            action_color="#f59e0b"
            confidence="MEDIUM"
            action_reason="Bullish bias building. Wait for confirmation candle."
        elif tech_score <= -2:
            action="❌ AVOID"
            action_color="#dc2626"
            confidence="HIGH"
            action_reason="Bearish signals. Not the time to enter."
        else:
            action="⏸ WAIT"
            action_color="#94a3b8"
            confidence="LOW"
            action_reason="No clear edge. Stay on watchlist."

    # ── Best session today ────────────────────────────────────────────
    if wd in [0,4]:
        best_session="Morning 10:00–11:30"
        session_note="Avoid Open — gap trap risk on Mon/Fri. Wait for direction to clear."
    elif rsi_v<35 or signals.get("bb_pct",50)<20:
        best_session="Open 09:30–10:00"
        session_note="Oversold — opening bounce possible. Act early."
    elif rsi_v>65 or signals.get("bb_pct",50)>80:
        best_session="Afternoon 13:00–14:30"
        session_note="Overbought — wait for afternoon pullback before adding."
    elif tech_score>=2:
        best_session="Morning 10:00–11:30"
        session_note="Bullish setup — institutional flow session is cleanest."
    else:
        best_session="Afternoon 13:00–14:30"
        session_note="No strong morning signal — afternoon reversal may give better entry."

    # ── Dynamic stop & target suggestion ─────────────────────────────
    atr_stop  = round(price - 1.5*atr_v, 4)
    atr_target= round(price + 2.5*atr_v, 4)

    if stop and abs(stop-atr_stop)/price<0.03:
        stop_note="Current stop aligned with ATR — OK"
        stop_color="#16a34a"
    elif stop and stop < atr_stop:
        stop_note=f"Current stop too tight — consider loosening to {atr_stop:,.4f}"
        stop_color="#f59e0b"
    elif stop and stop > atr_stop:
        stop_note=f"Suggested tighter stop: {atr_stop:,.4f} (1.5× ATR)"
        stop_color="#2563eb"
    else:
        stop_note=f"No stop set — suggest: {atr_stop:,.4f} (1.5× ATR)"
        stop_color="#dc2626"

    if target and price<target:
        tgt_note=f"Target {target:,.4f} — {(target-price)/price*100:+.1f}% away"
        tgt_color="#16a34a"
    else:
        tgt_note=f"Suggested target: {atr_target:,.4f} (2.5× ATR)"
        tgt_color="#2563eb"

    # ── Estimated P&L if recommendation followed ─────────────────────
    # Uses: effective_stop and effective_target (set or ATR-based)
    eff_stop   = stop   if stop   else atr_stop
    eff_target = target if target else atr_target

    if qty > 0:
        # Existing position
        if "EXIT" in action:
            # Exit now at current price
            est_pnl     = round((price - avg_cost) * qty, 2)
            est_pnl_pct = round((price - avg_cost) / avg_cost * 100, 2) if avg_cost else 0
            est_note    = "Exit at current price"
        elif "REDUCE" in action:
            # Reduce to half
            est_pnl     = round((price - avg_cost) * qty * 0.5, 2)
            est_pnl_pct = round((price - avg_cost) / avg_cost * 100, 2) if avg_cost else 0
            est_note    = "Reduce 50% at current price"
        elif "ADD" in action:
            # Add same size again, target = eff_target
            new_avg = (avg_cost * qty + price * qty) / (qty * 2)
            est_pnl     = round((eff_target - new_avg) * qty * 2, 2)
            est_pnl_pct = round((eff_target - new_avg) / new_avg * 100, 2) if new_avg else 0
            est_note    = f"Add same size → target {eff_target:,.4f}"
        else:
            # Hold to target, risking to stop
            est_pnl_win  = round((eff_target - price) * qty, 2)
            est_pnl_lose = round((eff_stop   - price) * qty, 2)
            est_pnl      = est_pnl_win   # optimistic scenario
            est_pnl_pct  = round((eff_target - price) / price * 100, 2) if price else 0
            est_note     = (f"Hold: +{est_pnl_win:,.2f} to target / "
                            f"{est_pnl_lose:,.2f} to stop")
    else:
        # Watchlist — new entry
        if "ENTER" in action or "ADD" in action:
            entry_qty = 100  # placeholder — user sets actual size
            est_pnl     = round((eff_target - price) * entry_qty, 2)
            est_pnl_pct = round((eff_target - price) / price * 100, 2) if price else 0
            est_note    = f"Entry × 100 units → target {eff_target:,.4f}"
        else:
            est_pnl = 0; est_pnl_pct = 0; est_note = "No action"

    # Risk: worst case to stop
    if qty > 0:
        est_risk = round((eff_stop - price) * qty, 2)
    else:
        est_risk = round((eff_stop - price) * 100, 2)

    est_rr = round(abs(est_pnl / est_risk), 2) if est_risk != 0 else 0

    return {
        "ticker":        ticker,
        "name":          name,
        "price":         price,
        "prev":          prev,
        "gap_pct":       round(gap_pct,2),
        "day_pct":       round(day_pct,2),
        "avg_cost":      avg_cost,
        "qty":           qty,
        "pnl":           round(pnl,2),
        "pnl_pct":       round(pnl_pct,2),
        "alloc":         round(alloc,1),
        "action":        action,
        "action_color":  action_color,
        "confidence":    confidence,
        "action_reason": action_reason,
        "best_session":  best_session,
        "session_note":  session_note,
        "signals":       signals,
        "tech_score":    round(tech_score,1),
        "curr_session":  curr_session,
        "wd_name":       wd_name,
        "wd_note":       wd_note,
        "stop":          stop,
        "target":        target,
        "atr_stop":      atr_stop,
        "atr_target":    atr_target,
        "stop_note":     stop_note,
        "stop_color":    stop_color,
        "tgt_note":      tgt_note,
        "tgt_color":     tgt_color,
        "status":        status,
        # Money flow context
        "flow_score":    flow_score,
        "flow_signal":   flow_signal,
        "flow_driver":   flow_driver,
        "flow_sector":   flow_sector,
        # Estimated P&L
        "est_pnl":       est_pnl,
        "est_pnl_pct":   est_pnl_pct,
        "est_risk":      est_risk,
        "est_rr":        est_rr,
        "est_note":      est_note,
    }

# ── RENDER ────────────────────────────────────────────────────────────
def render():
    init_portfolio_extended()
    init_activity_log()

    now_hk  = datetime.now(HK_TZ)
    capital = get_latest_capital()
    wd      = now_hk.weekday()
    wd_name, wd_note, _ = WEEKDAY_BIAS.get(wd,("?","—",0))

    h,m   = now_hk.hour, now_hk.minute
    mins  = h*60+m
    curr_session="Pre-market"
    for sname,sh,sm,eh,em in SESSIONS:
        if sh*60+sm<=mins<eh*60+em:
            curr_session=sname; break
    if mins>=16*60: curr_session="After-hours"

    st.markdown(
        "## 📅 Daily Strategy Briefing &nbsp;"
        "<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        "padding:2px 7px;border-radius:5px'>LIVE</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"{now_hk.strftime('%A %d %b %Y  %H:%M HKT')} · "
        f"Session: **{curr_session}** · "
        f"Today is **{wd_name}** — {wd_note}</span>",
        unsafe_allow_html=True)

    with st.expander("📖 How recommendations are generated"):
        st.markdown("""
Each position is scored by combining three layers:

**1. Technical signals** (main weight)
RSI zone, MACD histogram direction, BB position, volume ratio, trend slope, choppiness, VWAP.
Each signal adds +1 to +2 (bullish) or -1 to -2 (bearish) to a composite score.

**2. Market context**
Gap size and direction, weekday bias (Mon/Fri trap risk vs Tue-Wed clean trend),
current session (Open is trap zone on Mondays, Morning is cleanest for institutional flow).

**3. Position management**
Stop hit → immediate EXIT regardless of other signals.
Near target → scale out. Near stop → reduce size.
Trend trap conditions → EXIT or REDUCE even without stop hit.

**Confidence levels**
HIGH: multiple signals agree, or hard trigger (stop hit, target reached).
MEDIUM: majority signals agree.
LOW: mixed signals — best to wait.

**Stop suggestion**: 1.5× ATR below current price (gives room for normal volatility).
**Target suggestion**: 2.5× ATR above current price (R:R ≈ 1:1.67).
        """)
    st.markdown("---")

    # ── Load portfolio ────────────────────────────────────────────────
    stock_df = get_portfolio_full()
    mon_df   = get_monitor_pos()

    all_pos=[]
    if not stock_df.empty:
        for _,r in stock_df[stock_df["status"].isin(["OPEN","WATCH"])].iterrows():
            all_pos.append({
                "ticker":   r["ticker"],
                "name":     r.get("name",r["ticker"]),
                "avg_cost": float(r.get("avg_cost",0) or 0),
                "qty":      float(r.get("shares",0) or 0),
                "target":   r.get("target_price"),
                "stop":     r.get("stop_price"),
                "status":   r.get("status","OPEN"),
                "src":      "stock",
            })
    if not mon_df.empty:
        for _,r in mon_df[mon_df["status"].isin(["OPEN","WATCH"])].iterrows():
            all_pos.append({
                "ticker":   r["ticker"],
                "name":     r.get("name",r["ticker"]),
                "avg_cost": float(r.get("avg_cost",0) or 0),
                "qty":      float(r.get("quantity",0) or 0),
                "target":   r.get("target"),
                "stop":     r.get("stop"),
                "status":   r.get("status","OPEN"),
                "src":      "monitor",
            })

    # Add study universe instruments not already in portfolio (as opportunities)
    try:
        from portfolio_study import get_study_universe
        study_df = get_study_universe()
        if not study_df.empty:
            existing_tickers = {p["ticker"] for p in all_pos}
            # Top 10 by study_score only — avoid slow fetches for 100+ instruments
            top_study = study_df[~study_df["ticker"].isin(existing_tickers)]
            top_study = top_study.nlargest(10, "study_score")
            for _, sr in top_study.iterrows():
                if sr.get("study_score",0) > 40:
                    all_pos.append({
                        "ticker":   sr["ticker"],
                        "name":     sr.get("name", sr["ticker"]),
                        "avg_cost": 0.0,
                        "qty":      0,           # not held — opportunity
                        "target":   None,
                        "stop":     None,
                        "status":   "WATCH",
                        "src":      "study",
                    })
    except Exception:
        pass

    if not all_pos:
        st.info("No positions in portfolio yet. Add stocks, forex or commodities in 📋 Portfolio.")
        return

    total_invested=sum(p["qty"]*p["avg_cost"] for p in all_pos if p["qty"]>0)

    _init_forecast_db()
    col_refresh,col_note=st.columns([1,3])
    if col_refresh.button("🔄 Refresh all",key="ds_refresh"):
        st.cache_data.clear(); st.rerun()
    col_note.markdown(
        f"<span style='color:#64748b;font-size:0.8rem'>"
        f"{len(all_pos)} positions/watchlist · "
        f"Last refresh: {now_hk.strftime('%H:%M:%S')}</span>",
        unsafe_allow_html=True)

    # ── Analyse all positions ─────────────────────────────────────────
    with st.spinner("Analysing all positions…"):
        results=[]
        for p in all_pos:
            r=analyse_position(
                p["ticker"],p["name"],p["avg_cost"],p["qty"],
                p["target"],p["stop"],p["status"],
                now_hk,capital,total_invested)
            r["src"]=p["src"]
            results.append(r)
            time.sleep(0.15)

    # Auto-save today's forecasts to DB
    today_str = now_hk.strftime("%Y-%m-%d")
    for r in results:
        if r.get("error") or not r.get("price"): continue
        try:
            save_forecast(
                today_str, r["ticker"], r["name"],
                r.get("action","—"),
                r.get("est_pnl",0), r.get("est_pnl_pct",0),
                r.get("est_note",""), r.get("price",0))
        except Exception: pass

    # ── Quick summary strip ───────────────────────────────────────────
    exits   =[r for r in results if "EXIT" in r.get("action","")]
    reduces =[r for r in results if "REDUCE" in r.get("action","")]
    adds    =[r for r in results if "ADD" in r.get("action","") or "ENTER" in r.get("action","")]
    holds   =[r for r in results if "HOLD" in r.get("action","")]
    waits   =[r for r in results if "WAIT" in r.get("action","") or "WATCH" in r.get("action","")]

    s1,s2,s3,s4,s5=st.columns(5)
    for col,lbl,lst,color in [
        (s1,"🔴 Exit/Reduce",   exits+reduces, "#dc2626"),
        (s2,"✅ Add/Enter",     adds,          "#16a34a"),
        (s3,"⏸ Hold",          holds,         "#2563eb"),
        (s4,"⌛ Wait/Watch",   waits,          "#94a3b8"),
        (s5,"⚡ High confidence",
         [r for r in results if r.get("confidence")=="HIGH"],"#f59e0b"),
    ]:
        col.markdown(
            f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
            f"border-radius:10px;padding:10px 12px;text-align:center'>"
            f"<div style='font-size:0.7rem;color:#94a3b8'>{lbl}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>"
            f"{len(lst)}</div></div>",
            unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # ── Portfolio estimated P&L if all recommendations followed ─────────
    valid = [r for r in results if not r.get("error") and r.get("qty",0)>0]
    total_est_pnl  = sum(r.get("est_pnl",0) for r in valid)
    total_est_risk = sum(r.get("est_risk",0) for r in valid)
    total_curr_pnl = sum(r.get("pnl",0) for r in valid)
    total_est_win  = sum(r.get("est_pnl",0) for r in valid if r.get("est_pnl",0)>0)
    total_est_lose = sum(r.get("est_pnl",0) for r in valid if r.get("est_pnl",0)<=0)

    st.markdown("#### 📈 Estimated P&L — if you follow all recommendations")
    ep1,ep2,ep3,ep4,ep5 = st.columns(5)
    for col,lbl,val,color in [
        (ep1,"Current P&L",
         f"{'+'if total_curr_pnl>=0 else ''}{total_curr_pnl:,.0f}",
         "#16a34a" if total_curr_pnl>=0 else "#dc2626"),
        (ep2,"Est. P&L (optimistic)",
         f"{'+'if total_est_pnl>=0 else ''}{total_est_pnl:,.0f}",
         "#16a34a" if total_est_pnl>=0 else "#dc2626"),
        (ep3,"Est. Risk (stop hit)",
         f"{total_est_risk:,.0f}",
         "#dc2626"),
        (ep4,"Portfolio R:R",
         f"1:{abs(total_est_pnl/total_est_risk):.1f}" if total_est_risk!=0 else "—",
         "#16a34a" if total_est_pnl>abs(total_est_risk) else "#f59e0b"),
        (ep5,"Win scenario",
         f"+{total_est_win:,.0f}",
         "#16a34a"),
    ]:
        col.markdown(
            f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
            f"border-radius:10px;padding:10px 14px;text-align:center'>"
            f"<div style='font-size:0.68rem;color:#94a3b8'>{lbl}</div>"
            f"<div style='font-size:1.05rem;font-weight:700;color:{color}'>{val}</div>"
            f"</div>",unsafe_allow_html=True)

    # Per-position est P&L table
    with st.expander("📋 Estimated P&L per position"):
        tbl_ep=[]
        for r in sorted(valid, key=lambda x: -abs(x.get("est_pnl",0))):
            ep=r.get("est_pnl",0); er=r.get("est_risk",0)
            tbl_ep.append({
                "Name":       r["name"],
                "Ticker":     r["ticker"],
                "Action":     r["action"].split()[0] + " " + r["action"].split()[1] if len(r["action"].split())>1 else r["action"],
                "Current P&L":f"{'+'if r['pnl']>=0 else ''}{r['pnl']:,.2f}",
                "Est. P&L":   f"{'+'if ep>=0 else ''}{ep:,.2f}",
                "Est. P&L %": f"{'+'if r.get('est_pnl_pct',0)>=0 else ''}{r.get('est_pnl_pct',0):.2f}%",
                "Est. Risk":  f"{er:,.2f}",
                "R:R":        f"1:{r.get('est_rr',0):.1f}" if r.get('est_rr',0)>0 else "—",
                "Scenario":   r.get("est_note","—"),
            })
        if tbl_ep:
            df_ep=pd.DataFrame(tbl_ep)
            def style_ep(df):
                s=pd.DataFrame("",index=df.index,columns=df.columns)
                for i,row in df.iterrows():
                    for c in ["Current P&L","Est. P&L","Est. P&L %"]:
                        v=str(row.get(c,""))
                        if v.startswith("+"): s.at[i,c]="color:#16a34a;font-weight:600"
                        elif v.startswith("-"): s.at[i,c]="color:#dc2626;font-weight:600"
                return s
            st.dataframe(df_ep.style.apply(style_ep,axis=None),
                         use_container_width=True,hide_index=True)
            st.caption(
                "Est. P&L = projected profit if recommendation followed to target. "
                "Est. Risk = loss if stop is hit. R:R = reward/risk ratio. "
                "Hold/Add scenarios use effective target (your set target or ATR-based). "
                "Exit/Reduce scenarios use current price.")


    # ════════════════════════════════════════════════════════════════
    # CAPITAL ROTATION PLAN
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔄 Daily Capital Rotation Plan")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Since you cycle capital daily — here is today's plan: "
        "what to sell (free capital) → what to buy (deploy capital) → "
        "estimated available capital after rotation.</span>",
        unsafe_allow_html=True)

    # Score every result for rotation priority
    rotation_rows = []
    for r in results:
        if r.get("error") or not r.get("price"): continue
        price   = r["price"]
        qty     = r["qty"]
        avg_c   = r["avg_cost"]
        sig     = r.get("signals", {})
        tech_sc = r.get("tech_score", 0)

        # ── Component 1: Cycle ML score (% through cycle) ────────────
        # Use RSI + BB as proxy for cycle position if no cycle data
        rsi_  = sig.get("rsi", 50)
        bb_   = sig.get("bb_pct", 50)
        # 0 = early cycle (trough), 100 = late cycle (peak)
        cycle_pct = (rsi_/100*50 + bb_/100*50)  # blended 0-100

        # ── Component 2: Technical signal score ──────────────────────
        # tech_score: positive = bullish, negative = bearish
        # Normalise to 0-100 (50 = neutral)
        tech_norm = min(max(tech_sc * 10 + 50, 0), 100)

        # ── Component 3: P&L target/stop proximity ────────────────────
        tgt   = r.get("target")
        stp   = r.get("stop")
        pnl_  = r.get("pnl", 0)

        # Distance to target as % (positive = closer to target)
        if tgt and price:
            tgt_dist_pct = (tgt - price) / price * 100   # positive = not yet reached
            # Near target = time to sell → high rotation sell score
            tgt_score = max(0, 100 - tgt_dist_pct * 5)   # 0% away = 100, 20% away = 0
        else:
            tgt_score = 50

        # Distance to stop (negative = danger zone)
        if stp and price:
            stp_dist_pct = (price - stp) / price * 100   # positive = safe distance
            stp_score = min(stp_dist_pct * 5, 100)        # far from stop = 100 (safe to hold)
        else:
            stp_score = 50

        # ── Combined rotation score ───────────────────────────────────
        # SELL score (0-100): high = should sell today to free capital
        # Weights: cycle position 35% + tech reversal signal 35% + near target 30%
        sell_score = round(
            cycle_pct       * 0.35 +
            (100-tech_norm) * 0.35 +   # bearish tech = sell signal
            tgt_score       * 0.30,
            1)

        # BUY score (0-100): high = should buy today
        # Weights: early cycle 35% + bullish tech 35% + far from stop (safety) 30%
        buy_score = round(
            (100-cycle_pct) * 0.35 +
            tech_norm       * 0.35 +
            stp_score       * 0.30,
            1)

        # Net rotation: positive = BUY bias, negative = SELL bias
        net_score = round(buy_score - sell_score, 1)

        # Freed capital if sold
        freed = price * qty if qty > 0 else 0

        rotation_rows.append({
            "ticker":      r["ticker"],
            "name":        r["name"],
            "price":       price,
            "qty":         qty,
            "avg_cost":    avg_c,
            "pnl":         r.get("pnl", 0),
            "cycle_pct":   round(cycle_pct, 1),
            "tech_norm":   round(tech_norm, 1),
            "tgt_score":   round(tgt_score, 1),
            "sell_score":  sell_score,
            "buy_score":   buy_score,
            "net_score":   net_score,
            "freed":       round(freed, 0),
            "action":      r.get("action", "—"),
            "status":      r.get("status", "OPEN"),
        })

    if not rotation_rows:
        st.info("No positions to rotate.")
    else:
        # Classify
        to_sell  = sorted([r for r in rotation_rows if r["sell_score"]>62 and r["qty"]>0],
                           key=lambda x: -x["sell_score"])
        to_buy   = sorted([r for r in rotation_rows if r["buy_score"]>62],
                           key=lambda x: -x["buy_score"])
        to_hold  = [r for r in rotation_rows
                    if r not in to_sell and r not in to_buy and r["qty"]>0]

        freed_total   = sum(r["freed"] for r in to_sell)
        invested_now  = sum(r["qty"]*r["price"] for r in rotation_rows if r["qty"]>0)
        cash_avail    = max(capital - invested_now, 0)
        deployable    = freed_total + cash_avail

        # ── Summary bar ──────────────────────────────────────────────
        rb1,rb2,rb3,rb4 = st.columns(4)
        for col,lbl,val,color in [
            (rb1,"🔴 Sell today",    str(len(to_sell)),    "#dc2626"),
            (rb2,"🟢 Buy today",     str(len(to_buy)),     "#16a34a"),
            (rb3,"Capital freed",    f"HKD {freed_total:,.0f}", "#f59e0b"),
            (rb4,"Total deployable", f"HKD {deployable:,.0f}","#2563eb"),
        ]:
            col.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                f"border-radius:10px;padding:10px 14px;text-align:center'>"
                f"<div style='font-size:0.7rem;color:#94a3b8'>{lbl}</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:{color}'>{val}</div>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SELL queue ────────────────────────────────────────────────
        if to_sell:
            st.markdown(
                "<div style='font-size:0.85rem;font-weight:600;color:#dc2626;"
                "margin-bottom:6px'>🔴 SELL — Free capital from these positions</div>",
                unsafe_allow_html=True)
            for r in to_sell:
                pnl_c = "#16a34a" if r["pnl"]>=0 else "#dc2626"
                price_s = f"{r['price']:,.2f}" if r["price"]>10 else f"{r['price']:,.4f}"

                # Reason breakdown
                reasons = []
                if r["cycle_pct"] > 65:
                    reasons.append(f"cycle {r['cycle_pct']:.0f}% — near peak")
                if r["tech_norm"] < 40:
                    reasons.append("bearish technical signals")
                if r["tgt_score"] > 70:
                    reasons.append("near target — take profit")
                reason_str = " · ".join(reasons) if reasons else "rotation timing"

                st.markdown(
                    f"<div style='border:1px solid #fca5a5;border-radius:10px;"
                    f"padding:12px 16px;margin-bottom:6px;background:rgba(220,38,38,0.03);"
                    f"display:flex;justify-content:space-between;align-items:center'>"
                    f"<div>"
                    f"<span style='font-weight:600;color:#0f172a'>{r['name']}</span> "
                    f"<span style='color:#94a3b8;font-size:0.78rem'>({r['ticker']})</span><br>"
                    f"<span style='font-size:0.78rem;color:#64748b'>{reason_str}</span>"
                    f"</div>"
                    f"<div style='text-align:right'>"
                    f"<div style='font-size:0.9rem;font-weight:700'>{price_s}</div>"
                    f"<div style='font-size:0.75rem;color:{pnl_c}'>"
                    f"P&L: {'+'if r['pnl']>=0 else ''}{r['pnl']:,.0f}</div>"
                    f"<div style='font-size:0.72rem;color:#dc2626'>"
                    f"Frees: HKD {r['freed']:,.0f}</div>"
                    f"<div style='font-size:0.68rem;color:#94a3b8'>"
                    f"Sell score: {r['sell_score']:.0f}/100</div>"
                    f"</div></div>",
                    unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#64748b;font-size:0.82rem;padding:8px 0'>"
                "🟡 No strong sell signals today — hold current positions.</div>",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── BUY queue ─────────────────────────────────────────────────
        if to_buy:
            st.markdown(
                f"<div style='font-size:0.85rem;font-weight:600;color:#16a34a;"
                f"margin-bottom:4px'>🟢 BUY — Deploy HKD {deployable:,.0f} into these</div>",
                unsafe_allow_html=True)
            # Suggest allocation: weight by buy_score
            total_buy_score = sum(r["buy_score"] for r in to_buy)
            for r in to_buy:
                alloc_pct  = r["buy_score"]/total_buy_score*100 if total_buy_score>0 else 0
                alloc_hkd  = deployable * alloc_pct/100
                price_s = f"{r['price']:,.2f}" if r["price"]>10 else f"{r['price']:,.4f}"
                shares_sug = int(alloc_hkd/r["price"]) if r["price"]>0 else 0

                buy_reasons = []
                if r["cycle_pct"] < 35:
                    buy_reasons.append(f"cycle {r['cycle_pct']:.0f}% — early stage")
                if r["tech_norm"] > 60:
                    buy_reasons.append("bullish technical signals")
                if r["qty"] == 0:
                    buy_reasons.append("new entry — watchlist")
                br_str = " · ".join(buy_reasons) if buy_reasons else "rotation opportunity"

                tag = "🆕 NEW ENTRY" if r["qty"]==0 else "➕ ADD"
                tag_c = "#8b5cf6" if r["qty"]==0 else "#16a34a"

                st.markdown(
                    f"<div style='border:1px solid #86efac;border-radius:10px;"
                    f"padding:12px 16px;margin-bottom:6px;background:rgba(22,163,74,0.03);"
                    f"display:flex;justify-content:space-between;align-items:center'>"
                    f"<div>"
                    f"<span style='font-weight:600;color:#0f172a'>{r['name']}</span> "
                    f"<span style='color:#94a3b8;font-size:0.78rem'>({r['ticker']})</span> "
                    f"<span style='background:{tag_c};color:white;font-size:0.65rem;"
                    f"padding:1px 6px;border-radius:4px'>{tag}</span><br>"
                    f"<span style='font-size:0.78rem;color:#64748b'>{br_str}</span>"
                    f"</div>"
                    f"<div style='text-align:right'>"
                    f"<div style='font-size:0.9rem;font-weight:700'>{price_s}</div>"
                    f"<div style='font-size:0.78rem;color:#16a34a;font-weight:600'>"
                    f"Deploy: HKD {alloc_hkd:,.0f} ({alloc_pct:.0f}%)</div>"
                    f"<div style='font-size:0.72rem;color:#64748b'>"
                    f"~{shares_sug:,} shares · Buy score: {r['buy_score']:.0f}/100</div>"
                    f"</div></div>",
                    unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#64748b;font-size:0.82rem;padding:8px 0'>"
                "🟡 No strong buy signals today — wait for better entry.</div>",
                unsafe_allow_html=True)

        # ── Full rotation table ───────────────────────────────────────
        with st.expander("📋 Full rotation scores — all positions"):
            rot_df = pd.DataFrame([{
                "Name":       r["name"],
                "Ticker":     r["ticker"],
                "Sell score": f"{r['sell_score']:.0f}",
                "Buy score":  f"{r['buy_score']:.0f}",
                "Net":        f"{'BUY' if r['net_score']>10 else 'SELL' if r['net_score']<-10 else 'HOLD'} ({r['net_score']:+.0f})",
                "Cycle %":    f"{r['cycle_pct']:.0f}%",
                "Tech":       f"{r['tech_norm']:.0f}",
                "Tgt prox":   f"{r['tgt_score']:.0f}",
                "Current P&L":f"{'+'if r['pnl']>=0 else ''}{r['pnl']:,.0f}" if r["qty"]>0 else "—",
            } for r in sorted(rotation_rows, key=lambda x:-x["sell_score"])])

            def style_rot(df):
                s=pd.DataFrame("",index=df.index,columns=df.columns)
                for i,row in df.iterrows():
                    net=str(row["Net"])
                    if net.startswith("BUY"):
                        s.at[i,"Net"]="color:#16a34a;font-weight:600"
                    elif net.startswith("SELL"):
                        s.at[i,"Net"]="color:#dc2626;font-weight:600"
                    try:
                        ss=float(str(row["Sell score"]).replace("%","").strip())
                        bs=float(str(row["Buy score"]).replace("%","").strip())
                        if ss>62: s.at[i,"Sell score"]="color:#dc2626;font-weight:700"
                        if bs>62: s.at[i,"Buy score"]="color:#16a34a;font-weight:700"
                    except (ValueError, TypeError):
                        pass
                return s
            st.dataframe(rot_df.style.apply(style_rot,axis=None),
                         use_container_width=True, hide_index=True)

            st.caption(
                "Sell score = cycle position(35%) + bearish tech(35%) + near target(30%). "
                "Buy score = early cycle(35%) + bullish tech(35%) + safe from stop(30%). "
                ">62 = actionable signal.")

    st.markdown("---")

    # Critical alerts first
    if exits:
        for r in exits:
            st.error(
                f"**{r['action']} — {r['name']} ({r['ticker']})** · "
                f"{r['action_reason']} · "
                f"Price: {r['price']:,.4f} · P&L: {r['pnl']:+,.2f}")

    st.markdown("---")

    # ── Per-position cards ────────────────────────────────────────────
    # Sort: exits first, then reduces, then adds, holds, waits
    priority={"EXIT":0,"REDUCE":1,"ADD":2,"ENTER":2,"HOLD":3,"WAIT":4,"WATCH":4,"AVOID":5,"MONITOR":3}
    results.sort(key=lambda x: min(
        [priority.get(w,9) for w in x.get("action","").upper().split()],default=9))

    filter_opt=st.selectbox("Show",
        ["All positions","🔴 Action needed (Exit/Reduce/Add)","✅ Active positions only",
         "👁 Watchlist only","⚡ High confidence only"],
        key="ds_filter")

    def should_show(r):
        s=filter_opt
        if s=="All positions": return True
        if s=="🔴 Action needed (Exit/Reduce/Add)":
            return any(w in r.get("action","") for w in ["EXIT","REDUCE","ADD","ENTER"])
        if s=="✅ Active positions only": return r.get("qty",0)>0
        if s=="👁 Watchlist only": return r.get("qty",0)==0
        if s=="⚡ High confidence only": return r.get("confidence")=="HIGH"
        return True

    for r in results:
        if not should_show(r): continue
        if "error" in r: continue

        ac  = r["action_color"]
        conf_c={"HIGH":"#16a34a","MEDIUM":"#f59e0b","LOW":"#94a3b8"}.get(
               r["confidence"],"#94a3b8")

        price_s = f"{r['price']:,.2f}" if r["price"]>10 else f"{r['price']:,.4f}"
        cost_s  = f"{r['avg_cost']:,.2f}" if r["avg_cost"]>10 else f"{r['avg_cost']:,.4f}"
        pnl_c   = "#16a34a" if r["pnl"]>=0 else "#dc2626"

        with st.expander(
            f"{r['action']}  ·  **{r['name']}** ({r['ticker']})  ·  "
            f"{price_s}  ·  "
            f"Confidence: {r['confidence']}",
            expanded=("EXIT" in r["action"] or "REDUCE" in r["action"])):

            # Top strip
            tc=st.columns(6)
            ep_   = r.get("est_pnl",0)
            er_   = r.get("est_risk",0)
            ep_c  = "#16a34a" if ep_>=0 else "#dc2626"
            for col,lbl,val,color in [
                (tc[0],"Price",    price_s,             "#0f172a"),
                (tc[1],"Current P&L",
                 f"{'+'if r['pnl']>=0 else ''}{r['pnl']:,.2f} ({r['pnl_pct']:+.1f}%)"
                 if r["qty"]>0 else "Watchlist",        pnl_c),
                (tc[2],"Est. P&L",
                 f"{'+'if ep_>=0 else ''}{ep_:,.2f} ({r.get('est_pnl_pct',0):+.1f}%)",
                 ep_c),
                (tc[3],"Est. Risk",
                 f"{er_:,.2f}",
                 "#dc2626" if er_<0 else "#94a3b8"),
                (tc[4],"Day chg",  f"{r['day_pct']:+.2f}%",
                 "#16a34a" if r["day_pct"]>=0 else "#dc2626"),
                (tc[5],"Alloc",    f"{r['alloc']:.1f}%" if r["qty"]>0 else "—","#64748b"),
            ]:
                col.markdown(
                    f"<div style='text-align:center;padding:7px 4px;"
                    f"background:#f8fafc;border-radius:8px'>"
                    f"<div style='font-size:0.65rem;color:#94a3b8'>{lbl}</div>"
                    f"<div style='font-size:0.9rem;font-weight:600;color:{color}'>{val}</div>"
                    f"</div>",unsafe_allow_html=True)

            st.markdown("<br>",unsafe_allow_html=True)

            # Action + session
            ac1_,ac2_=st.columns(2)
            with ac1_:
                st.markdown(
                    f"<div style='border-left:4px solid {ac};padding:10px 14px;"
                    f"background:rgba(0,0,0,0.02);border-radius:0 8px 8px 0'>"
                    f"<div style='font-size:0.72rem;color:#64748b'>TODAY\'S ACTION</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:{ac}'>"
                    f"{r['action']}</div>"
                    f"<div style='font-size:0.8rem;color:#475569;margin-top:4px'>"
                    f"{r['action_reason']}</div>"
                    f"<div style='font-size:0.72rem;color:{conf_c};margin-top:6px'>"
                    f"Confidence: {r['confidence']}</div></div>",
                    unsafe_allow_html=True)
            with ac2_:
                st.markdown(
                    f"<div style='border-left:4px solid #8b5cf6;padding:10px 14px;"
                    f"background:rgba(0,0,0,0.02);border-radius:0 8px 8px 0'>"
                    f"<div style='font-size:0.72rem;color:#64748b'>BEST SESSION</div>"
                    f"<div style='font-size:1rem;font-weight:600;color:#8b5cf6'>"
                    f"{r['best_session']}</div>"
                    f"<div style='font-size:0.8rem;color:#475569;margin-top:4px'>"
                    f"{r['session_note']}</div>"
                    f"<div style='font-size:0.72rem;color:#64748b;margin-top:6px'>"
                    f"Now: {r['curr_session'] or 'Between sessions'}</div></div>",
                    unsafe_allow_html=True)

            st.markdown("<br>",unsafe_allow_html=True)

            # Stop & target
            st1_,st2_=st.columns(2)
            with st1_:
                _sc = r["stop_color"]
                st.markdown(
                    f"<div style='padding:9px 12px;border-radius:8px;"
                    f"border:1px solid {r['stop_color']};background:rgba(0,0,0,0.02)'>"
                    f"<div style='font-size:0.7rem;color:#64748b'>STOP LOSS</div>"
                    f"<div style='font-weight:600;color:{_sc}'>"
                    f"{r['stop']:,.4f} (current)" if r["stop"] else "Not set"
                    f"</div>"
                    f"<div style='font-size:0.75rem;color:{_sc}'>"
                    f"{r['stop_note']}</div>"
                    f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:3px'>"
                    f"ATR-based: {r['atr_stop']:,.4f}</div></div>",
                    unsafe_allow_html=True)
            with st2_:
                _tc = r["tgt_color"]
                st.markdown(
                    f"<div style='padding:9px 12px;border-radius:8px;"
                    f"border:1px solid {r['tgt_color']};background:rgba(0,0,0,0.02)'>"
                    f"<div style='font-size:0.7rem;color:#64748b'>TARGET</div>"
                    f"<div style='font-weight:600;color:{_tc}'>"
                    f"{r['target']:,.4f} (set)" if r["target"] else "Not set"
                    f"</div>"
                    f"<div style='font-size:0.75rem;color:{_tc}'>"
                    f"{r['tgt_note']}</div>"
                    f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:3px'>"
                    f"ATR-based: {r['atr_target']:,.4f}</div></div>",
                    unsafe_allow_html=True)

            # Technical signals detail
            sig=r.get("signals",{})
            if sig:
                with st.expander("📊 Technical signals detail"):
                    sc=st.columns(4)
                    for col,lbl,val,note in [
                        (sc[0],"RSI",        f"{sig.get('rsi',50):.0f}",
                         sig.get("rsi_signal","—")),
                        (sc[1],"MACD hist",  f"{sig.get('macd_h',0):+.4f}",
                         sig.get("macd_signal","—")),
                        (sc[2],"BB position",f"{sig.get('bb_pct',50):.0f}%",
                         sig.get("bb_signal","—")),
                        (sc[3],"Vol ratio",  f"{sig.get('vol_r',1):.2f}×",
                         sig.get("vol_signal","—")),
                    ]:
                        col.markdown(
                            f"<div style='background:#f8fafc;padding:8px;border-radius:6px'>"
                            f"<div style='font-size:0.65rem;color:#94a3b8'>{lbl}</div>"
                            f"<div style='font-weight:600;font-size:0.9rem'>{val}</div>"
                            f"<div style='font-size:0.68rem;color:#64748b'>{note}</div>"
                            f"</div>",unsafe_allow_html=True)
                    sc2=st.columns(4)
                    for col,lbl,val,note in [
                        (sc2[0],"Choppiness",f"{sig.get('chop',50):.0f}",
                         sig.get("chop_signal","—")),
                        (sc2[1],"Trend slope",f"{sig.get('slope',0):+.3f}%/d",
                         sig.get("trend_signal","—")),
                        (sc2[2],"Above MA20",
                         "Yes ✅" if sig.get("above_ma20") else "No ⚠️",""),
                        (sc2[3],"VWAP",
                         "Above ✅" if sig.get("above_vwap") else "Below ⚠️",
                         sig.get("vwap_signal","—")),
                    ]:
                        col.markdown(
                            f"<div style='background:#f8fafc;padding:8px;border-radius:6px'>"
                            f"<div style='font-size:0.65rem;color:#94a3b8'>{lbl}</div>"
                            f"<div style='font-weight:600;font-size:0.9rem'>{val}</div>"
                            f"<div style='font-size:0.68rem;color:#64748b'>{note}</div>"
                            f"</div>",unsafe_allow_html=True)

            # Money flow context strip
            flow_s = r.get("flow_score",0)
            if flow_s != 0 or r.get("flow_sector","—") != "—":
                flow_c="#16a34a" if flow_s>=20 else "#dc2626" if flow_s<=-20 else "#64748b"
                sig_ = r.get("signals",{})
                st.markdown(
                    f"<div style='background:#f0f9ff;border:1px solid #bae6fd;"
                    f"border-radius:8px;padding:9px 14px;margin-bottom:6px'>"
                    f"<span style='font-size:0.8rem;color:#0369a1'>"
                    f"💰 <b>Sector flow:</b> {r.get('flow_sector','—')} · "
                    f"<b style='color:{flow_c}'>{r.get('flow_signal','—')} "
                    f"({flow_s:+d})</b> · "
                    f"{r.get('flow_driver','—')} · "
                    f"{sig_.get('flow_note','')}</span></div>",
                    unsafe_allow_html=True)

            # Quick update stop/target
            with st.expander("✏️ Update stop / target"):
                uf1,uf2=st.columns(2)
                new_stp=uf1.number_input("New stop",value=float(r["stop"] or r["atr_stop"]),
                    step=0.0001,format="%.4f",key=f"ds_stp_{r['ticker']}")
                new_tgt=uf2.number_input("New target",value=float(r["target"] or r["atr_target"]),
                    step=0.0001,format="%.4f",key=f"ds_tgt_{r['ticker']}")
                if st.button("💾 Save",key=f"ds_save_{r['ticker']}"):
                    try:
                        if r["src"]=="stock":
                            port=get_portfolio_full()
                            row_=port[port["ticker"]==r["ticker"]].iloc[0]
                            upsert_position_full(
                                r["ticker"], row_.get("name",r["ticker"]),
                                int(row_.get("shares",0) or 0),
                                float(row_.get("avg_cost",0) or 0),
                                new_tgt, new_stp,
                                row_.get("notes",""),
                                row_.get("status","OPEN"),
                                row_.get("entry_date",""))
                        else:
                            from portfolio_manager import upsert_monitor_pos
                            mon=get_monitor_pos()
                            row_=mon[mon["ticker"]==r["ticker"]].iloc[0]
                            upsert_monitor_pos(
                                r["ticker"],row_.get("name",r["ticker"]),
                                row_.get("asset_type","Forex"),
                                row_.get("unit",""),
                                float(row_.get("quantity",0) or 0),
                                float(row_.get("avg_cost",0) or 0),
                                new_tgt, new_stp,
                                row_.get("notes",""),
                                row_.get("status","OPEN"),
                                row_.get("entry_date",""))
                        log_activity("UPDATE STOP/TARGET",r["ticker"],
                                     f"stop={new_stp:.4f} target={new_tgt:.4f}")
                        st.success("Saved!")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")

    # ════════════════════════════════════════════════════════════════
    # SECTION: TODAY'S FORECAST SUMMARY + RECORD ACTUALS
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📋 Today's Forecast & Record Actuals")

    fc_tab1, fc_tab2 = st.tabs(["📊 Today's Estimates", "✅ Record Actual Results"])

    with fc_tab1:
        st.markdown(
            "<span style='color:#64748b;font-size:0.8rem'>"
            "Estimated P&L for each position if today's recommendations are followed. "
            "Saved automatically each time you open this page.</span>",
            unsafe_allow_html=True)

        today_rows = [r for r in results if not r.get("error") and r.get("price")]
        if today_rows:
            fc_data = []
            total_est = 0
            for r in today_rows:
                ep = r.get("est_pnl",0) or 0
                total_est += ep
                fc_data.append({
                    "Name":       r["name"],
                    "Ticker":     r["ticker"],
                    "Action":     r.get("action","—").split("·")[0].strip(),
                    "Confidence": r.get("confidence","—"),
                    "Est. P&L":   f"{'+'if ep>=0 else ''}{ep:,.2f}",
                    "Est. %":     f"{r.get('est_pnl_pct',0):+.2f}%",
                    "Scenario":   r.get("est_note","—"),
                })

            total_c = "#16a34a" if total_est>=0 else "#dc2626"
            st.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                f"border-radius:10px;padding:12px 16px;margin-bottom:10px'>"
                f"<span style='font-size:0.8rem;color:#64748b'>Total estimated P&L today: </span>"
                f"<span style='font-size:1.1rem;font-weight:700;color:{total_c}'>"
                f"{'+'if total_est>=0 else ''}{total_est:,.2f}</span></div>",
                unsafe_allow_html=True)

            fc_df = pd.DataFrame(fc_data)
            def _sty_fc(df):
                s = pd.DataFrame("", index=df.index, columns=df.columns)
                for i,row in df.iterrows():
                    v = str(row["Est. P&L"])
                    if v.startswith("+"): s.at[i,"Est. P&L"] = "color:#16a34a;font-weight:600"
                    elif v.startswith("-"): s.at[i,"Est. P&L"] = "color:#dc2626;font-weight:600"
                    c = str(row["Confidence"])
                    if c == "HIGH": s.at[i,"Confidence"] = "color:#16a34a;font-weight:600"
                    elif c == "LOW": s.at[i,"Confidence"] = "color:#94a3b8"
                return s
            st.dataframe(fc_df.style.apply(_sty_fc, axis=None),
                         use_container_width=True, hide_index=True)

    with fc_tab2:
        st.markdown(
            "<span style='color:#64748b;font-size:0.8rem'>"
            "After market close, enter actual closing prices to record what really happened. "
            "This builds your accuracy tracking over time.</span>",
            unsafe_allow_html=True)

        today_str2 = now_hk.strftime("%Y-%m-%d")
        _init_forecast_db()
        conn_ = get_conn()
        import pandas as _pd2
        today_fc = _pd2.read_sql_query(
            "SELECT * FROM daily_forecast WHERE date=? ORDER BY ticker",
            conn_, params=(today_str2,))
        conn_.close()

        if today_fc.empty:
            st.info("No forecasts saved yet for today. Come back after the analysis runs.")
        else:
            st.markdown(f"**{len(today_fc)} positions forecast for {today_str2}**")

            # Bulk auto-fetch close prices
            bc1, bc2 = st.columns(2)
            if bc1.button("⚡ Auto-fetch all close prices", key="fc_bulk"):
                fetched = 0
                for _, fc_row in today_fc.iterrows():
                    t_ = fc_row["ticker"]
                    qty_   = next((p["qty"] for p in all_pos if p["ticker"]==t_), 0)
                    avg_c_ = next((p["avg_cost"] for p in all_pos if p["ticker"]==t_), 0)
                    try:
                        q_ = fetch_live(t_)
                        cp_ = q_.get("price") or q_.get("prev")
                        if cp_:
                            update_actual(today_str2, t_, float(cp_), qty_, avg_c_)
                            fetched += 1
                        time.sleep(0.1)
                    except Exception: pass
                st.success(f"✅ Auto-recorded {fetched} positions!"); st.rerun()
            bc2.markdown(
                "<span style='font-size:0.78rem;color:#64748b'>"
                "Or enter each price manually below.</span>",
                unsafe_allow_html=True)
            for _, fc_row in today_fc.iterrows():
                ticker_ = fc_row["ticker"]
                name_   = fc_row.get("name", ticker_)
                ep_     = float(fc_row.get("est_pnl",0) or 0)
                open_p_ = float(fc_row.get("open_price",0) or 0)
                close_p_= fc_row.get("close_price")
                correct_= fc_row.get("correct")

                # Find qty from portfolio
                qty_   = next((p["qty"] for p in all_pos if p["ticker"]==ticker_), 0)
                avg_c_ = next((p["avg_cost"] for p in all_pos if p["ticker"]==ticker_), 0)

                status_icon = ("✅" if correct_==1 else "❌" if correct_==0 else "⏳")
                ep_c   = "#16a34a" if ep_>=0 else "#dc2626"

                with st.expander(
                    f"{status_icon} {name_} ({ticker_}) · "
                    f"Forecast: {'+'if ep_>=0 else ''}{ep_:,.0f} · "
                    f"{'Recorded ✓' if close_p_ else 'Pending'}"):
                    rc1,rc2,rc3 = st.columns(3)
                    rc1.markdown(
                        f"<div style='text-align:center;background:#f8fafc;"
                        f"border-radius:8px;padding:10px'>"
                        f"<div style='font-size:0.68rem;color:#94a3b8'>Forecast P&L</div>"
                        f"<div style='font-size:1rem;font-weight:700;color:{ep_c}'>"
                        f"{'+'if ep_>=0 else ''}{ep_:,.2f}</div>"
                        f"<div style='font-size:0.7rem;color:#64748b'>"
                        f"{fc_row.get('action','—')}</div></div>",
                        unsafe_allow_html=True)

                    if close_p_:
                        ap_ = float(fc_row.get("actual_pnl",0) or 0)
                        ap_c = "#16a34a" if ap_>=0 else "#dc2626"
                        rc2.markdown(
                            f"<div style='text-align:center;background:#f8fafc;"
                            f"border-radius:8px;padding:10px'>"
                            f"<div style='font-size:0.68rem;color:#94a3b8'>Actual P&L</div>"
                            f"<div style='font-size:1rem;font-weight:700;color:{ap_c}'>"
                            f"{'+'if ap_>=0 else ''}{ap_:,.2f}</div>"
                            f"<div style='font-size:0.7rem;color:#64748b'>"
                            f"Close: {float(close_p_):,.4f}</div></div>",
                            unsafe_allow_html=True)
                        diff_ = float(fc_row.get("actual_pnl",0) or 0) - ep_
                        _cc = "#16a34a" if correct_==1 else "#dc2626"
                        _cl = "✅ Correct direction" if correct_==1 else "❌ Wrong direction"
                        rc3.markdown(
                            f"<div style='text-align:center;background:#f8fafc;"
                            f"border-radius:8px;padding:10px'>"
                            f"<div style='font-size:0.68rem;color:#94a3b8'>Diff (actual−est)</div>"
                            f"<div style='font-size:1rem;font-weight:700;"
                            f"color:{'#16a34a' if diff_>=0 else '#dc2626'}'>"
                            f"{'+'if diff_>=0 else ''}{diff_:,.2f}</div>"
                            f"<div style='font-size:0.7rem;color:{_cc}'>"
                            f"{_cl}</div>"
                            f"</div>",
                            unsafe_allow_html=True)

                    # Input close price
                    inp_c1, inp_c2 = st.columns(2)
                    new_close = inp_c1.number_input(
                        "Enter close price",
                        value=float(close_p_) if close_p_ else open_p_ or 0.0,
                        step=0.01, format="%.4f",
                        key=f"fc_close_{ticker_}")
                    if inp_c2.button("💾 Record", key=f"fc_rec_{ticker_}"):
                        update_actual(today_str2, ticker_, new_close, qty_, avg_c_)
                        st.success(f"Recorded {ticker_}!"); st.rerun()

    # ════════════════════════════════════════════════════════════════
    # SECTION: ACCURACY HISTORY
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📈 Forecast Accuracy History")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "How accurate were the daily estimates vs actual results over time.</span>",
        unsafe_allow_html=True)

    hist_days = st.slider("Show last N days", 7, 90, 30, key="fc_hist_days")
    hist_df = get_forecast_history(hist_days)

    if hist_df.empty or "correct" not in hist_df.columns:
        st.info("No historical data yet. Record actual results each day to build your accuracy history.")
    else:
        completed = hist_df.dropna(subset=["actual_pnl"])
        if not completed.empty:
            n_correct = int((completed["correct"]==1).sum())
            n_total   = len(completed)
            accuracy  = n_correct/n_total*100 if n_total>0 else 0
            total_est_h = completed["est_pnl"].sum()
            total_act_h = completed["actual_pnl"].sum()
            avg_err   = (completed["actual_pnl"] - completed["est_pnl"]).mean()

            h1,h2,h3,h4 = st.columns(4)
            for col,lbl,val,color in [
                (h1,"Direction accuracy", f"{accuracy:.0f}%",
                 "#16a34a" if accuracy>=60 else "#f59e0b" if accuracy>=50 else "#dc2626"),
                (h2,"Total est. P&L",     f"{'+'if total_est_h>=0 else ''}{total_est_h:,.0f}",
                 "#16a34a" if total_est_h>=0 else "#dc2626"),
                (h3,"Total actual P&L",   f"{'+'if total_act_h>=0 else ''}{total_act_h:,.0f}",
                 "#16a34a" if total_act_h>=0 else "#dc2626"),
                (h4,"Avg forecast error", f"{'+'if avg_err>=0 else ''}{avg_err:,.0f}",
                 "#94a3b8"),
            ]:
                col.markdown(
                    f"<div style='text-align:center;background:#f8fafc;"
                    f"border-radius:10px;padding:10px 12px;border:1px solid #e2e8f0'>"
                    f"<div style='font-size:0.68rem;color:#94a3b8'>{lbl}</div>"
                    f"<div style='font-size:1rem;font-weight:700;color:{color}'>{val}</div>"
                    f"</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Est vs actual chart
            by_day = completed.groupby("date").agg(
                est=("est_pnl","sum"), actual=("actual_pnl","sum")).reset_index()
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=by_day["date"], y=by_day["est"],
                name="Estimated", marker_color="#94a3b8", opacity=0.7))
            fig_cmp.add_trace(go.Bar(
                x=by_day["date"], y=by_day["actual"],
                name="Actual",
                marker_color=["#16a34a" if v>=0 else "#dc2626" for v in by_day["actual"]],
                opacity=0.85))
            fig_cmp.add_hline(y=0, line_color="#e2e8f0", line_width=1)
            fig_cmp.update_layout(
                height=280, barmode="group",
                margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.08),
                xaxis=dict(gridcolor="#f1f5f9"),
                yaxis=dict(title="P&L", gridcolor="#f1f5f9"))
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Accuracy by ticker
            by_ticker_h = completed.groupby("ticker").agg(
                n=("correct","count"),
                correct_n=("correct","sum"),
                est_sum=("est_pnl","sum"),
                act_sum=("actual_pnl","sum"),
            ).reset_index()
            by_ticker_h["accuracy"] = by_ticker_h["correct_n"]/by_ticker_h["n"]*100
            by_ticker_h = by_ticker_h.sort_values("accuracy", ascending=False)
            ticker_df = pd.DataFrame([{
                "Ticker":    r["ticker"],
                "Trades":    r["n"],
                "Accuracy":  f"{r['accuracy']:.0f}%",
                "Est total": f"{'+'if r['est_sum']>=0 else ''}{r['est_sum']:,.0f}",
                "Actual total":f"{'+'if r['act_sum']>=0 else ''}{r['act_sum']:,.0f}",
                "Diff":      f"{'+'if r['act_sum']-r['est_sum']>=0 else ''}{r['act_sum']-r['est_sum']:,.0f}",
            } for _,r in by_ticker_h.iterrows()])
            st.markdown("**Accuracy by position**")
            def _sty_acc(df):
                s = pd.DataFrame("", index=df.index, columns=df.columns)
                for i,row in df.iterrows():
                    try:
                        acc = float(str(row["Accuracy"]).replace("%",""))
                        if acc >= 65: s.at[i,"Accuracy"] = "color:#16a34a;font-weight:600"
                        elif acc < 50: s.at[i,"Accuracy"] = "color:#dc2626;font-weight:600"
                    except: pass
                    for c in ["Est total","Actual total","Diff"]:
                        v = str(row.get(c,""))
                        if v.startswith("+"): s.at[i,c] = "color:#16a34a"
                        elif v.startswith("-"): s.at[i,c] = "color:#dc2626"
                return s
            st.dataframe(ticker_df.style.apply(_sty_acc, axis=None),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Record actual close prices using the tab above to see accuracy statistics.")

    st.markdown(
        "<span style='color:#94a3b8;font-size:0.74rem'>"
        "Recommendations are algorithmic — always apply your own judgment. "
        "Not financial advice.</span>",
        unsafe_allow_html=True)
