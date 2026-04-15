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
    # Deduct sell-side transaction cost from actual P&L
    if qty > 0:
        actual_pnl -= tx_cost(close_price * qty, is_buy=False)
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

# ── TRANSACTION COSTS ─────────────────────────────────────────────────
# HKEX: 0.028% commission (min HKD 28 per side) + 0.1% stamp duty (buy only)
COMMISSION_RATE = 0.00028   # 0.028%
COMMISSION_MIN  = 28        # HKD 28 minimum per transaction
STAMP_DUTY_RATE = 0.001     # 0.1% on buy side only

def tx_cost(value: float, is_buy: bool = True) -> float:
    """
    Total transaction cost for one side of a trade.
    value = trade_value (price × shares)
    is_buy = True for buy, False for sell
    """
    commission = max(value * COMMISSION_RATE, COMMISSION_MIN)
    stamp      = value * STAMP_DUTY_RATE if is_buy else 0
    return round(commission + stamp, 2)

def round_trip_cost(buy_value: float, sell_value: float) -> float:
    """Total cost for a complete buy + sell cycle."""
    return tx_cost(buy_value, True) + tx_cost(sell_value, False)

def net_pnl(gross_pnl: float, buy_value: float, sell_value: float) -> float:
    """Gross P&L minus round-trip transaction costs."""
    return round(gross_pnl - round_trip_cost(buy_value, sell_value), 2)


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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_weekly(ticker):
    """Fetch weekly OHLCV for macro cycle detection."""
    for t in _var(ticker):
        try:
            df = yf.Ticker(t).history(period="12mo", interval="1wk", auto_adjust=True)
            if len(df) >= 10: return df
        except Exception: pass
    return pd.DataFrame()


def detect_cycle(ticker, df_daily, df_intraday, df_weekly=None):
    """
    Multi-timeframe cycle detection.
    Returns dict with macro/meso/micro cycle positions (0-100%)
    and a combined score with conflict flags.

    0% = cycle trough (buy zone)
    100% = cycle peak (sell zone)
    50% = mid-cycle

    Each timeframe uses three signals combined:
      1. ZigZag-like turning points (ATR-based)
      2. RSI position
      3. Volume weight (surge = new cycle start, dry = exhaustion)
    """
    result = {
        "macro_pct":   50, "macro_label": "—", "macro_signal": "—",
        "meso_pct":    50, "meso_label":  "—", "meso_signal":  "—",
        "micro_pct":   50, "micro_label": "—", "micro_signal": "—",
        "combined":    50, "dominant":    "meso",
        "conflict":    False, "conflict_note": "",
        "action_bias": "NEUTRAL",
    }

    # ── MACRO cycle (weekly, 1-3 months) ─────────────────────────────
    try:
        df_w = df_weekly if (df_weekly is not None and len(df_weekly)>=10) else pd.DataFrame()
        if df_w.empty and len(df_daily)>=60:
            # Resample daily to weekly
            df_w = df_daily.resample("W").agg({
                "Open":"first","High":"max","Low":"min",
                "Close":"last","Volume":"sum"}).dropna()

        if len(df_w) >= 10:
            c_w  = df_w["Close"]
            rsi_w = _rsi(c_w, 10)
            # ATR-based range position on weekly
            atr_w = float((df_w["High"]-df_w["Low"]).ewm(com=9,adjust=False).mean().iloc[-1])
            mid_w = float(c_w.rolling(10).mean().iloc[-1])
            pos_w = (float(c_w.iloc[-1]) - (mid_w - 2*atr_w)) / (4*atr_w + 1e-9) * 100
            pos_w = max(0, min(100, pos_w))
            # Volume: recent 4wk vs 12wk avg
            vol_w = df_w["Volume"]
            vr_w  = float(vol_w.tail(4).mean()) / float(vol_w.mean() + 1e-9)
            # Volume surge near low = accumulation = early cycle
            vol_signal_w = max(0, min(1, (vr_w - 0.8) / 1.2))  # 0=low vol, 1=high vol
            # Combine: RSI(40%) + price position(40%) + volume(20%)
            macro_pct = rsi_w * 0.40 + pos_w * 0.40 + vol_signal_w * 100 * 0.20
            macro_pct = round(max(0, min(100, macro_pct)), 1)
            result["macro_pct"] = macro_pct
            if macro_pct >= 75:    result["macro_label"] = "🔴 Peak"; result["macro_signal"] = "Macro cycle near peak — expect pullback"
            elif macro_pct >= 55:  result["macro_label"] = "🟡 Late"; result["macro_signal"] = "Macro late cycle — reduce new positions"
            elif macro_pct >= 35:  result["macro_label"] = "🟢 Mid";  result["macro_signal"] = "Macro mid-cycle — trend intact"
            else:                  result["macro_label"] = "✅ Early"; result["macro_signal"] = "Macro early cycle — strong buy zone"
    except Exception:
        pass

    # ── MESO cycle (daily, 1-3 weeks) ────────────────────────────────
    try:
        if len(df_daily) >= 20:
            c_d   = df_daily["Close"]
            rsi_d = _rsi(c_d, 14)
            # BB position (0=lower band, 100=upper band)
            bb_d  = _bb(c_d, 20)
            # ATR-based cycle position
            atr_d = _atr(df_daily, 14)
            mid_d = float(c_d.rolling(20).mean().iloc[-1])
            pos_d = (float(c_d.iloc[-1]) - (mid_d - 2*atr_d)) / (4*atr_d + 1e-9) * 100
            pos_d = max(0, min(100, pos_d))
            # Volume: 5d vs 20d avg
            vol_d = df_daily["Volume"]
            vr_d  = float(vol_d.tail(5).mean()) / float(vol_d.rolling(20).mean().iloc[-1] + 1e-9)
            # ZigZag: detect recent trough (last time RSI was <30 or BB<10)
            rsi_series = c_d.rolling(1).apply(lambda x: _rsi(c_d.iloc[:c_d.index.get_loc(x.index[-1])+1],14) if len(c_d)>14 else 50, raw=False)
            # Simplified: days since RSI was last oversold
            try:
                rsi_hist = [float(_rsi(c_d.iloc[max(0,i-14):i+1],14)) for i in range(max(14,len(c_d)-30), len(c_d))]
                days_since_trough = next((len(rsi_hist)-1-i for i,v in enumerate(reversed(rsi_hist)) if v<35), 15)
                trough_recency = max(0, 1 - days_since_trough/15)  # 1=just had trough, 0=long ago
            except:
                trough_recency = 0
            # Combine: RSI(35%) + BB(35%) + vol(15%) + trough recency(15%)
            meso_pct = rsi_d*0.35 + bb_d*0.35 + min(vr_d,2)/2*100*0.15 + (1-trough_recency)*100*0.15
            meso_pct = round(max(0, min(100, meso_pct)), 1)
            result["meso_pct"] = meso_pct
            if meso_pct >= 75:    result["meso_label"] = "🔴 Peak"; result["meso_signal"] = "Meso cycle peak — take profit"
            elif meso_pct >= 55:  result["meso_label"] = "🟡 Late"; result["meso_signal"] = "Meso late — tighten stop"
            elif meso_pct >= 35:  result["meso_label"] = "🟢 Mid";  result["meso_signal"] = "Meso mid-cycle — hold"
            else:                 result["meso_label"] = "✅ Early"; result["meso_signal"] = "Meso early — best entry window"
    except Exception:
        pass

    # ── MICRO cycle (15min, 1-3 days) ────────────────────────────────
    try:
        if not df_intraday.empty and len(df_intraday) >= 10:
            c_i   = df_intraday["Close"]
            rsi_i = _rsi(c_i, min(14, len(c_i)-1))
            # BB position on 15min
            bb_i  = _bb(c_i, min(20, len(c_i)-1))
            # VWAP position
            vwap_i = float(((df_intraday["High"]+df_intraday["Low"]+df_intraday["Close"])/3
                            *df_intraday["Volume"]).cumsum()
                           / df_intraday["Volume"].cumsum().iloc[-1])
            price_now = float(c_i.iloc[-1])
            vwap_pos  = 50 + (price_now - vwap_i) / (price_now * 0.02 + 1e-9) * 25
            vwap_pos  = max(0, min(100, vwap_pos))
            # Volume: last 3 bars vs session avg
            vol_i = df_intraday["Volume"]
            vr_i  = float(vol_i.tail(3).mean()) / float(vol_i.mean() + 1e-9)
            # Combine: RSI(35%) + BB(35%) + VWAP(20%) + volume(10%)
            micro_pct = rsi_i*0.35 + bb_i*0.35 + vwap_pos*0.20 + min(vr_i,2)/2*100*0.10
            micro_pct = round(max(0, min(100, micro_pct)), 1)
            result["micro_pct"] = micro_pct
            if micro_pct >= 75:   result["micro_label"] = "🔴 Peak"; result["micro_signal"] = "Micro overbought — wait for pullback"
            elif micro_pct >= 55: result["micro_label"] = "🟡 Late"; result["micro_signal"] = "Micro late — enter smaller"
            elif micro_pct >= 35: result["micro_label"] = "🟢 Mid";  result["micro_signal"] = "Micro neutral"
            else:                 result["micro_label"] = "✅ Early"; result["micro_signal"] = "Micro oversold — intraday buy signal"
    except Exception:
        pass

    # ── Combined score + conflict detection ───────────────────────────
    mac = result["macro_pct"]
    mes = result["meso_pct"]
    mic = result["micro_pct"]

    # Dominant cycle: meso drives the primary trade, macro gives context,
    # micro gives timing. Weight: macro 20%, meso 50%, micro 30%
    combined = round(mac*0.20 + mes*0.50 + mic*0.30, 1)
    result["combined"] = combined

    # Dominant: whichever cycle is most extreme (furthest from 50)
    dists = {"macro": abs(mac-50), "meso": abs(mes-50), "micro": abs(mic-50)}
    result["dominant"] = max(dists, key=dists.get)

    # Conflict detection
    conflicts = []
    if mac < 35 and mes > 65:
        conflicts.append("Macro early but meso late — meso may be topping within larger uptrend")
    if mac > 65 and mes < 35:
        conflicts.append("Macro late but meso early — meso bounce within larger downtrend, caution")
    if mes < 35 and mic > 65:
        conflicts.append("Meso early (buy zone) but micro overbought — wait for micro to reset before entering")
    if mes > 65 and mic < 35:
        conflicts.append("Meso at peak but micro oversold — possible last dip before peak, risky entry")
    if conflicts:
        result["conflict"] = True
        result["conflict_note"] = " · ".join(conflicts)

    # Action bias
    if combined < 30:      result["action_bias"] = "STRONG BUY"
    elif combined < 40:    result["action_bias"] = "BUY"
    elif combined < 55:    result["action_bias"] = "HOLD"
    elif combined < 65:    result["action_bias"] = "REDUCE"
    elif combined < 75:    result["action_bias"] = "SELL"
    else:                  result["action_bias"] = "STRONG SELL"

    return result

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
    df6 = fetch_daily(ticker, "6mo")   # longer history for macro
    dfi = fetch_intraday(ticker)

    # Multi-timeframe cycle detection
    try:
        df_w = fetch_weekly(ticker)
        cycle = detect_cycle(ticker, df6 if not df6.empty else df, dfi, df_w)
    except Exception:
        cycle = {"macro_pct":50,"meso_pct":50,"micro_pct":50,"combined":50,
                 "macro_label":"—","meso_label":"—","micro_label":"—",
                 "conflict":False,"conflict_note":"","action_bias":"NEUTRAL",
                 "macro_signal":"—","meso_signal":"—","micro_signal":"—",
                 "dominant":"meso"}

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
        # Earning efficiency
        _avg_r_pct = float((df["High"]-df["Low"]).mean()) / float(df["Close"].mean()) * 100
        _ann_vol   = float(df["Close"].pct_change().dropna().std() * (252**0.5) * 100)
        _chop_f    = max((chop_v-38)/(61.8-38), 0)
        _wr        = float((df["Close"].pct_change().dropna()>0).mean()*100)
        earn_eff   = (_avg_r_pct * _chop_f * (_wr/100)) / (_ann_vol/100 + 0.01)
        ma20   = float(df["Close"].rolling(20).mean().iloc[-1])
        ma50   = float(df["Close"].rolling(50).mean().iloc[-1]) if len(df)>=50 else None

        signals["rsi"]     = round(rsi_v,1)
        signals["macd_h"]  = round(macd_h,4)
        signals["bb_pct"]  = round(bb_v,1)
        # BB absolute levels for entry price calculation
        _bb_mid   = float(df["Close"].rolling(20).mean().iloc[-1])
        _bb_std   = float(df["Close"].rolling(20).std().iloc[-1])
        bb_lower  = round(_bb_mid - 2*_bb_std, 4)
        bb_upper  = round(_bb_mid + 2*_bb_std, 4)
        bb_mid    = round(_bb_mid, 4)
        signals["bb_lower"] = bb_lower
        signals["bb_upper"] = bb_upper
        signals["bb_mid"]   = bb_mid
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
        atr_v=price*0.02; chop_v=50; rsi_v=50; slope=0.0; earn_eff=0.0
        bb_lower=price*0.96; bb_upper=price*1.04; bb_mid=price

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

    # ── Target Type 1: Intraday high-volatility target ─────────────────
    # Based on today's expected range from ATR + opening range behaviour
    # Goal: capture 60-70% of daily range within the session
    avg_daily_range = atr_v           # ATR ≈ avg daily range
    today_remaining = avg_daily_range  # conservative: full ATR from here

    # If day high/low already set, use remaining range
    if dh and dl:
        range_used  = dh - dl
        range_left  = max(avg_daily_range - range_used, avg_daily_range * 0.3)
    else:
        range_left  = avg_daily_range

    # Intraday target: price + 65% of expected remaining range
    intraday_target = round(price + range_left * 0.65, 4)
    intraday_stop   = round(price - range_left * 0.40, 4)  # tighter — intraday exit fast
    intraday_rr     = round((intraday_target-price) / max(price-intraday_stop, 0.0001), 2)

    # Intraday P&L estimate (net of sell-side commission — already bought)
    intraday_gross  = (intraday_target - price) * qty if qty > 0 else (intraday_target - price) * 100
    intraday_pnl    = round(intraday_gross - tx_cost(intraday_target * (qty or 100), False), 2)

    # ── Target Type 2: 2-3 day swing target ───────────────────────────
    # Based on short-cycle amplitude: 2-3 days of range
    # Use ATR × 2.0 (2 days of average range) as the swing target
    # Adjusted for choppiness — oscillating stocks have cleaner 2-3d cycles
    chop_mult  = 1.2 if chop_v >= 61.8 else 1.0 if chop_v >= 50 else 0.8
    swing_days = 2.5  # target: capture 2-3 day range
    swing_amp  = atr_v * swing_days * chop_mult

    swing_target = round(price + swing_amp, 4)
    swing_stop   = round(price - atr_v * 1.2, 4)   # wider than intraday
    swing_rr     = round((swing_target-price) / max(price-swing_stop, 0.0001), 2)

    # Swing P&L estimate (round trip costs for new entry scenario)
    swing_qty     = qty if qty > 0 else 100
    swing_gross   = (swing_target - price) * swing_qty
    swing_pnl     = round(swing_gross - tx_cost(swing_target * swing_qty, False), 2)

    # ── Suitability filter ────────────────────────────────────────────
    # Don't recommend targets when the instrument is not suitable for range trading:
    # 1. Downtrend (slope < -0.15 = consistent downward drift)
    # 2. Not oscillating enough (choppiness < 45 = directional / trending)
    # 3. R:R too low to be worth the trade (< 1.0)
    _downtrend   = slope < -0.15
    _not_oscillating = chop_v < 45
    _poor_rr_intraday = intraday_rr < 1.0
    _poor_rr_swing    = swing_rr    < 1.2

    _intraday_suitable = not _downtrend and not _poor_rr_intraday
    _swing_suitable    = not _downtrend and not _not_oscillating and not _poor_rr_swing

    # Build rejection reason strings
    _intraday_reason = None
    _swing_reason    = None
    if _downtrend:
        _intraday_reason = f"Downtrend (slope {slope:+.3f}) — avoid long targets"
        _swing_reason    = f"Downtrend (slope {slope:+.3f}) — not suitable for swing"
    if _not_oscillating and not _swing_reason:
        _swing_reason = f"Choppiness {chop_v:.0f} < 45 — trending, not oscillating enough"
    if _poor_rr_intraday and not _intraday_reason:
        _intraday_reason = f"R:R {intraday_rr:.1f} < 1.0 — range too small vs risk"
    if _poor_rr_swing and not _swing_reason:
        _swing_reason = f"R:R {swing_rr:.1f} < 1.2 — swing amplitude insufficient"

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

    # ── Suggested entry price ─────────────────────────────────────────
    # For ADD / ENTER / WAIT — where is the ideal price to buy?
    # Priority: 1) BB lower band (oversold zone, best for range entry)
    #           2) VWAP (intraday fair value — institutional reference)
    #           3) MA20 (medium-term support)
    #           4) ATR-based pullback from current price
    # Use whichever is closest to current price from below
    _vwap = signals.get("vwap")
    _ma20 = signals.get("ma20", price)

    # Candidate entry levels (must be below or at current price)
    entry_candidates = []
    if bb_lower and bb_lower < price:
        entry_candidates.append(("BB lower band", round(bb_lower, 4)))
    if bb_lower and bb_lower >= price:  # already at lower band — buy now
        entry_candidates.append(("BB lower band (at band)", round(bb_lower, 4)))
    if _vwap and _vwap < price * 1.005:
        entry_candidates.append(("VWAP", round(float(_vwap), 4)))
    if _ma20 and _ma20 < price * 1.01:
        entry_candidates.append(("MA20", round(float(_ma20), 4)))
    # ATR pullback: current price minus 0.5 ATR
    entry_candidates.append(("ATR pullback (−0.5×ATR)", round(price - 0.5*atr_v, 4)))

    # Best entry = closest level to current price from below, but above stop
    valid = [(lbl, lvl) for lbl, lvl in entry_candidates
             if lvl >= eff_stop and lvl <= price * 1.01]
    if valid:
        # Pick the highest (closest to current price) = soonest hit
        best_entry_lbl, best_entry = max(valid, key=lambda x: x[1])
    else:
        best_entry_lbl = "ATR pullback"
        best_entry = round(price - 0.5*atr_v, 4)

    # Entry R:R with this entry price
    entry_rr = round((eff_target - best_entry) / max(best_entry - eff_stop, 0.0001), 2)

    # If already at or below entry → buy now
    if price <= best_entry * 1.005:
        entry_note = f"Buy now — price at {best_entry_lbl}"
    else:
        dist_pct = (price - best_entry) / price * 100
        entry_note = f"Wait for {best_entry_lbl} · {dist_pct:.2f}% below current"

    if qty > 0:
        # Existing position
        if "EXIT" in action:
            # Exit now at current price
            _gross      = (price - avg_cost) * qty
            _buy_val    = avg_cost * qty
            _sell_val   = price * qty
            est_pnl     = net_pnl(_gross, _buy_val, _sell_val)
            est_pnl_pct = round(est_pnl / _buy_val * 100, 2) if _buy_val else 0
            est_note    = f"Exit at current price (after tx costs: HKD {round_trip_cost(_buy_val,_sell_val):,.0f})"
        elif "REDUCE" in action:
            # Reduce to half
            _gross_r    = (price - avg_cost) * qty * 0.5
            _buy_val_r  = avg_cost * qty * 0.5
            _sell_val_r = price * qty * 0.5
            est_pnl     = net_pnl(_gross_r, _buy_val_r, _sell_val_r)
            est_pnl_pct = round(est_pnl / _buy_val_r * 100, 2) if _buy_val_r else 0
            est_note    = f"Reduce 50% (after tx costs: HKD {round_trip_cost(_buy_val_r,_sell_val_r):,.0f})"
        elif "ADD" in action:
            # Add same size again, target = eff_target
            new_avg = (avg_cost * qty + price * qty) / (qty * 2)
            _gross_a    = (eff_target - new_avg) * qty * 2
            _buy_val_a  = price * qty           # new buy
            _sell_val_a = eff_target * qty * 2  # sell all at target
            est_pnl     = net_pnl(_gross_a, _buy_val_a, _sell_val_a)
            est_pnl_pct = round(est_pnl / (new_avg * qty * 2) * 100, 2) if new_avg else 0
            est_note    = f"Add same size → target {eff_target:,.4f} (after tx costs: HKD {round_trip_cost(_buy_val_a,_sell_val_a):,.0f})"
        else:
            # Hold to target, risking to stop
            _buy_val_h   = avg_cost * qty
            _sell_tgt    = eff_target * qty
            _sell_stp    = eff_stop   * qty
            _tc_win      = tx_cost(_sell_tgt, False)      # only sell cost (already bought)
            _tc_lose     = tx_cost(_sell_stp, False)
            est_pnl_win  = round((eff_target - price) * qty - _tc_win,  2)
            est_pnl_lose = round((eff_stop   - price) * qty - _tc_lose, 2)
            est_pnl      = est_pnl_win
            est_pnl_pct  = round(est_pnl / _buy_val_h * 100, 2) if _buy_val_h else 0
            est_note     = (f"Hold: +{est_pnl_win:,.2f} to target / "
                            f"{est_pnl_lose:,.2f} to stop (tx costs incl.)")
    else:
        # Watchlist — new entry
        if "ENTER" in action or "ADD" in action:
            entry_qty   = 100
            _buy_val_e  = price * entry_qty
            _sell_val_e = eff_target * entry_qty
            _gross_e    = (eff_target - price) * entry_qty
            est_pnl     = net_pnl(_gross_e, _buy_val_e, _sell_val_e)
            est_pnl_pct = round(est_pnl / _buy_val_e * 100, 2) if _buy_val_e else 0
            est_note    = f"Entry × 100 → target {eff_target:,.4f} (after tx: HKD {round_trip_cost(_buy_val_e,_sell_val_e):,.0f})"
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
        # Target recommendations
        "intraday_target": intraday_target,
        "intraday_stop":   intraday_stop,
        "intraday_rr":     intraday_rr,
        "intraday_pnl":    intraday_pnl,
        "swing_target":    swing_target,
        "swing_stop":      swing_stop,
        "swing_rr":        swing_rr,
        "swing_pnl":       swing_pnl,
        "atr_v":           round(atr_v,4),
        "range_left":      round(range_left,4),
        "earn_eff":           round(earn_eff, 4),
        "cycle":              cycle,
        "intraday_suitable":  _intraday_suitable,
        "swing_suitable":     _swing_suitable,
        "intraday_reason":    _intraday_reason,
        "swing_reason":       _swing_reason,
        "chop_v":             round(chop_v,1),
        "slope":              round(slope,4),
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
        "best_entry":    best_entry,
        "best_entry_lbl":best_entry_lbl,
        "entry_rr":      entry_rr,
        "entry_note":    entry_note,
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

    # Asset type filter — defined early so all sections can use it
    ds_asset = st.radio(
        "Instrument type",
        ["📈 Stocks", "🌍 Forex & Commodities", "📊 All"],
        horizontal=True, key="ds_asset",
        help="Separate stocks from forex/commodities — different trading hours and liquidity")

    _fmt = lambda v: f"{v:,.4f}" if v and v < 10 else f"{v:,.2f}" if v else "—"
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

    # Cache results in session state so Summary page can show Buy at prices
    st.session_state["_ds_results_cache"] = {
        r["ticker"]: r for r in results if not r.get("error")
    }

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
            be_v   = r.get("best_entry")
            be_lbl = r.get("best_entry_lbl","")
            be_rr  = r.get("entry_rr",0)
            act_w  = r.get("action","")
            _is_buy_act = any(w in act_w for w in ["ADD","ENTER","WAIT","WATCH"])
            _fmt_be = (f"{be_v:,.4f}" if be_v and be_v<10 else f"{be_v:,.2f}" if be_v else "—")
            tbl_ep.append({
                "Name":       r["name"],
                "Ticker":     r["ticker"],
                "Action":     r["action"].split()[0] + " " + r["action"].split()[1] if len(r["action"].split())>1 else r["action"],
                "Buy at":     (_fmt_be if _is_buy_act else "—"),
                "Entry basis":be_lbl if _is_buy_act and be_v else "—",
                "Entry R:R":  f"1:{be_rr:.1f}" if _is_buy_act and be_rr>0 else "—",
                "Current P&L":f"{'+'if r['pnl']>=0 else ''}{r['pnl']:,.2f}",
                "Est. P&L":   f"{'+'if ep>=0 else ''}{ep:,.2f}",
                "R:R":        f"1:{r.get('est_rr',0):.1f}" if r.get('est_rr',0)>0 else "—",
            })
        if tbl_ep:
            df_ep=pd.DataFrame(tbl_ep)
            def style_ep(df):
                s=pd.DataFrame("",index=df.index,columns=df.columns)
                for i,row in df.iterrows():
                    if str(row.get("Buy at","—")) != "—":
                        s.at[i,"Buy at"]="color:#2563eb;font-weight:700;font-size:0.95rem"
                    for c in ["Current P&L","Est. P&L"]:
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
        "estimated available capital after rotation. "
        "Filtered to match the instrument type selected above.</span>",
        unsafe_allow_html=True)

    # Score every result for rotation priority
    rotation_rows = []
    for r in results:
        if r.get("error") or not r.get("price"): continue
        # Apply same asset filter
        _is_stk = r.get("src","stock") in ["stock","study"] or r.get("type","Stock") in ["Stock","Stock (HK)","Stock (US)"]
        if ds_asset == "📈 Stocks" and not _is_stk: continue
        if ds_asset == "🌍 Forex & Commodities" and _is_stk: continue
        price   = r["price"]
        qty     = r["qty"]
        avg_c   = r["avg_cost"]
        sig     = r.get("signals", {})
        tech_sc = r.get("tech_score", 0)

        # ── Component 1: Cycle ML score (% through cycle) ────────────
        # Use RSI + BB as proxy for cycle position if no cycle data
        # Use multi-timeframe cycle if available, else fallback to RSI+BB
        _cyc_r = r.get("cycle", {})
        if _cyc_r and _cyc_r.get("combined",0) != 50:
            cycle_pct = _cyc_r.get("combined", 50)
        else:
            rsi_  = sig.get("rsi", 50)
            bb_   = sig.get("bb_pct", 50)
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

        # ── Earning efficiency boost ──────────────────────────────────
        # High earn_eff = oscillating, good range, good win_rate
        # Scale 0–1: earn_eff 0.5+ = top performer, 0.1 = low quality
        ee_r    = r.get("earn_eff", 0) or 0
        ee_mult = min(ee_r / 0.3, 1.5)  # normalise: 0.3 = baseline, cap at 1.5×

        # ── Combined rotation score ───────────────────────────────────
        sell_score = round(
            (cycle_pct       * 0.35 +
             (100-tech_norm) * 0.35 +
             tgt_score       * 0.30),
            1)

        # Buy score boosted by earn_eff — better oscillators surface higher
        buy_score = round(
            ((100-cycle_pct) * 0.35 +
             tech_norm       * 0.35 +
             stp_score       * 0.30) * ee_mult,
            1)
        # Boost buy_score if multi-cycle says BUY, reduce if SELL
        _bias = r.get("cycle",{}).get("action_bias","NEUTRAL")
        if "STRONG BUY" in _bias:   buy_score = min(buy_score * 1.25, 100)
        elif "BUY" in _bias:        buy_score = min(buy_score * 1.10, 100)
        elif "STRONG SELL" in _bias:buy_score = max(buy_score * 0.60, 0)
        elif "SELL" in _bias:       buy_score = max(buy_score * 0.80, 0)
        buy_score = min(buy_score, 100)

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
            "earn_eff":    round(ee_r, 4),
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
                    f"<div style='font-size:0.82rem;font-weight:700;color:#2563eb'>"
                    f"Buy at: {_fmt(r.get('best_entry'))} "
                    f"<span style='font-size:0.72rem;font-weight:400;color:#64748b'>"
                    f"({r.get('best_entry_lbl','ATR pullback')})</span></div>"
                    f"<div style='font-size:0.72rem;color:#64748b'>"
                    f"~{shares_sug:,} shares · Buy score: {r['buy_score']:.0f}/100 · "
                    f"Earn Eff: {r.get('earn_eff',0):.3f}</div>"
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
                "Earn Eff":   f"{r.get('earn_eff',0):.3f}",
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

    _STOCK_SRC = ["stock"]  # src field set in all_pos

    def should_show(r):
        # Asset type filter
        src_ = r.get("src","stock")
        is_stock = src_ in ["stock","study"] or r.get("type","Stock") in ["Stock","Stock (HK)","Stock (US)"]
        if ds_asset == "📈 Stocks" and not is_stock: return False
        if ds_asset == "🌍 Forex & Commodities" and is_stock: return False

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

            # ── Multi-timeframe cycle display ─────────────────────
            cyc = r.get("cycle", {})
            if cyc:
                mac_p = cyc.get("macro_pct",50); mac_l = cyc.get("macro_label","—")
                mes_p = cyc.get("meso_pct",50);  mes_l = cyc.get("meso_label","—")
                mic_p = cyc.get("micro_pct",50); mic_l = cyc.get("micro_label","—")
                com_p = cyc.get("combined",50);  dom   = cyc.get("dominant","meso")
                bias  = cyc.get("action_bias","NEUTRAL")
                conf  = cyc.get("conflict",False)
                conf_n= cyc.get("conflict_note","")
                bias_c= ("#16a34a" if "BUY" in bias else "#dc2626" if "SELL" in bias else "#f59e0b")

                def _bar(pct, w=80):
                    filled = int(pct/100*w)
                    c = "#16a34a" if pct<35 else "#f59e0b" if pct<65 else "#dc2626"
                    return (f"<span style='display:inline-block;width:{filled}px;height:6px;"
                            f"background:{c};border-radius:3px'></span>"
                            f"<span style='display:inline-block;width:{w-filled}px;height:6px;"
                            f"background:#e2e8f0;border-radius:3px'></span>")

                st.markdown(
                    f"<div style='border:1px solid #e2e8f0;border-radius:9px;"
                    f"padding:10px 14px;background:#f8fafc;margin-bottom:8px'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"margin-bottom:6px'>"
                    f"<span style='font-size:0.72rem;font-weight:600;color:#64748b'>🔄 CYCLE POSITION</span>"
                    f"<span style='font-size:0.8rem;font-weight:700;color:{bias_c}'>{bias} ({com_p:.0f}%)</span>"
                    f"</div>"
                    f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px'>"
                    # Macro
                    f"<div>"
                    f"<div style='font-size:0.65rem;color:#94a3b8'>Macro (1-3mo)</div>"
                    f"<div style='font-size:0.8rem;font-weight:600'>{mac_l} {mac_p:.0f}%</div>"
                    f"{_bar(mac_p)}"
                    f"<div style='font-size:0.65rem;color:#64748b;margin-top:2px'>"
                    f"{cyc.get('macro_signal','')[:35]}</div>"
                    f"</div>"
                    # Meso
                    f"<div style='border-left:2px solid {'#2563eb' if dom=='meso' else '#e2e8f0'};padding-left:8px'>"
                    f"<div style='font-size:0.65rem;color:{'#2563eb' if dom=='meso' else '#94a3b8'}'>Meso (1-3wk) {'★' if dom=='meso' else ''}</div>"
                    f"<div style='font-size:0.8rem;font-weight:600'>{mes_l} {mes_p:.0f}%</div>"
                    f"{_bar(mes_p)}"
                    f"<div style='font-size:0.65rem;color:#64748b;margin-top:2px'>"
                    f"{cyc.get('meso_signal','')[:35]}</div>"
                    f"</div>"
                    # Micro
                    f"<div>"
                    f"<div style='font-size:0.65rem;color:#94a3b8'>Micro (1-3d)</div>"
                    f"<div style='font-size:0.8rem;font-weight:600'>{mic_l} {mic_p:.0f}%</div>"
                    f"{_bar(mic_p)}"
                    f"<div style='font-size:0.65rem;color:#64748b;margin-top:2px'>"
                    f"{cyc.get('micro_signal','')[:35]}</div>"
                    f"</div>"
                    f"</div>"
                    + (f"<div style='margin-top:6px;font-size:0.72rem;color:#f59e0b;"
                       f"border-top:1px solid #fde68a;padding-top:4px'>⚠️ Cycle conflict: {conf_n}</div>"
                       if conf else "")
                    + f"</div>", unsafe_allow_html=True)

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

            # ── Suggested entry price ────────────────────────────
            be   = r.get("best_entry")
            be_l = r.get("best_entry_lbl","—")
            be_n = r.get("entry_note","—")
            be_rr= r.get("entry_rr",0)
            act_ = r.get("action","")
            if be and any(w in act_ for w in ["ADD","ENTER","WAIT","WATCH"]):
                price_s2 = r.get("price",0)
                at_entry = price_s2 <= be * 1.005 if price_s2 and be else False
                be_c     = "#16a34a" if at_entry else "#2563eb"
                be_bg    = "rgba(22,163,74,0.05)" if at_entry else "rgba(37,99,235,0.03)"
                be_border= "#86efac" if at_entry else "#bfdbfe"
                rr_c_be  = "#16a34a" if be_rr>=2 else "#f59e0b" if be_rr>=1 else "#dc2626"
                fmt_be   = f"{be:,.4f}" if be < 10 else f"{be:,.2f}"
                st.markdown(
                    f"<div style='border:1px solid {be_border};border-radius:9px;"
                    f"padding:10px 14px;background:{be_bg};margin-bottom:8px'>"
                    f"<div style='font-size:0.7rem;font-weight:600;color:{be_c}'>"
                    f"{'🟢 BUY NOW' if at_entry else '📍 SUGGESTED ENTRY PRICE'}</div>"
                    f"<div style='display:flex;gap:20px;align-items:center;margin-top:5px'>"
                    f"<div>"
                    f"<span style='font-size:1.0rem;font-weight:700;color:{be_c}'>{fmt_be}</span>"
                    f"<span style='font-size:0.78rem;color:#64748b;margin-left:6px'>"
                    f"({be_l})</span>"
                    f"</div>"
                    f"<div style='font-size:0.78rem'>"
                    f"R:R if filled: <b style='color:{rr_c_be}'>1:{be_rr:.1f}</b>"
                    f"</div></div>"
                    f"<div style='font-size:0.72rem;color:#64748b;margin-top:3px'>"
                    f"{be_n}</div>"
                    f"</div>", unsafe_allow_html=True)

            # ── Target recommendations ───────────────────────────
            itgt = r.get("intraday_target"); istop = r.get("intraday_stop")
            stgt = r.get("swing_target");    sstop = r.get("swing_stop")
            irr  = r.get("intraday_rr",0);   srr   = r.get("swing_rr",0)
            ipnl = r.get("intraday_pnl",0);  spnl  = r.get("swing_pnl",0)
            atr_ = r.get("atr_v",0)

            if itgt and stgt:
                tc1,tc2 = st.columns(2)
                price_s = r.get("price",0)
                _i_ok = r.get("intraday_suitable", True)
                _s_ok = r.get("swing_suitable", True)
                _i_why = r.get("intraday_reason","")
                _s_why = r.get("swing_reason","")
                # Intraday
                irr_c = "#16a34a" if irr>=1.5 else "#f59e0b" if irr>=1.0 else "#dc2626"
                if not _i_ok:
                    tc1.markdown(
                        f"<div style='border:1px solid #e2e8f0;border-radius:9px;"
                        f"padding:10px 14px;background:#f8fafc'>"
                        f"<div style='font-size:0.7rem;font-weight:600;color:#94a3b8'>"
                        f"⚡ INTRADAY TARGET</div>"
                        f"<div style='font-size:0.78rem;color:#dc2626;margin-top:4px'>"
                        f"⛔ Not recommended</div>"
                        f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:2px'>"
                        f"{_i_why}</div></div>", unsafe_allow_html=True)
                else:
                    tc1.markdown(
                    f"<div style='border:1px solid #bfdbfe;border-radius:9px;"
                    f"padding:10px 14px;background:rgba(37,99,235,0.03)'>"
                    f"<div style='font-size:0.7rem;font-weight:600;color:#2563eb'>"
                    f"⚡ INTRADAY TARGET (today)</div>"
                    f"<div style='display:flex;justify-content:space-between;margin-top:4px'>"
                    f"<span style='font-size:0.82rem'>"
                    f"🎯 <b>{_fmt(itgt)}</b> "
                    f"<span style='color:#16a34a'>({(itgt-price_s)/price_s*100:+.2f}%)</span>"
                    f"</span>"
                    f"<span style='font-size:0.82rem'>"
                    f"🛑 <b>{_fmt(istop)}</b> "
                    f"<span style='color:#dc2626'>({(istop-price_s)/price_s*100:+.2f}%)</span>"
                    f"</span></div>"
                    f"<div style='margin-top:4px;font-size:0.78rem'>"
                    f"R:R <b style='color:{irr_c}'>1:{irr:.1f}</b> · "
                    f"Est: <b>{'+'if ipnl>=0 else ''}{ipnl:,.0f}</b> · "
                    f"ATR: {_fmt(atr_)}</div>"
                    f"<div style='font-size:0.68rem;color:#64748b;margin-top:2px'>"
                    f"Based on 65% of remaining daily range · Exit same day</div>"
                    f"</div>", unsafe_allow_html=True)
                # Swing
                srr_c = "#16a34a" if srr>=2.0 else "#f59e0b" if srr>=1.2 else "#dc2626"
                if not _s_ok:
                    tc2.markdown(
                        f"<div style='border:1px solid #e2e8f0;border-radius:9px;"
                        f"padding:10px 14px;background:#f8fafc'>"
                        f"<div style='font-size:0.7rem;font-weight:600;color:#94a3b8'>"
                        f"📅 SWING TARGET (2-3 days)</div>"
                        f"<div style='font-size:0.78rem;color:#dc2626;margin-top:4px'>"
                        f"⛔ Not recommended</div>"
                        f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:2px'>"
                        f"{_s_why}</div></div>", unsafe_allow_html=True)
                else:
                    tc2.markdown(
                    f"<div style='border:1px solid #bbf7d0;border-radius:9px;"
                    f"padding:10px 14px;background:rgba(22,163,74,0.03)'>"
                    f"<div style='font-size:0.7rem;font-weight:600;color:#16a34a'>"
                    f"📅 SWING TARGET (2-3 days)</div>"
                    f"<div style='display:flex;justify-content:space-between;margin-top:4px'>"
                    f"<span style='font-size:0.82rem'>"
                    f"🎯 <b>{_fmt(stgt)}</b> "
                    f"<span style='color:#16a34a'>({(stgt-price_s)/price_s*100:+.2f}%)</span>"
                    f"</span>"
                    f"<span style='font-size:0.82rem'>"
                    f"🛑 <b>{_fmt(sstop)}</b> "
                    f"<span style='color:#dc2626'>({(sstop-price_s)/price_s*100:+.2f}%)</span>"
                    f"</span></div>"
                    f"<div style='margin-top:4px;font-size:0.78rem'>"
                    f"R:R <b style='color:{srr_c}'>1:{srr:.1f}</b> · "
                    f"Est: <b>{'+'if spnl>=0 else ''}{spnl:,.0f}</b> · "
                    f"2.5×ATR×chop</div>"
                    f"<div style='font-size:0.68rem;color:#64748b;margin-top:2px'>"
                    f"Based on 2-3 day cycle amplitude · Exit before full cycle</div>"
                    f"</div>", unsafe_allow_html=True)

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
