"""
decision_page.py
Market Psychology Decision Support — WHO is trapped and HOW to trade against them.

Five detection engines:
  1. Crowd Trap Detector    — retail FOMO/panic exhaustion
  2. Smart Money Tracker    — institution accumulation / distribution
  3. Stop Hunt Detector     — liquidity grabs at obvious levels
  4. Gap Fade Analyser      — failed continuation = trapped longs/shorts
  5. Divergence Scanner     — price vs volume / price vs RSI/MACD
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import time
from datetime import datetime
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

from db_manager import get_latest_capital

HK_TZ = pytz.timezone("Asia/Hong_Kong")

STOCKS = {
    "0100.HK": "MiniMax Group",
    "2513.HK": "Zhipu / Knowledge Atlas",
}

# ─────────────────────────────────────────────────────────────────────
# INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────────────
def safe_last(s, n=1):
    s2 = pd.Series(s).dropna()
    if len(s2) < n:
        return None
    return float(s2.iloc[-n])

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def calc_macd(s, fast=12, slow=26, sig=9):
    ml = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl, ml - sl

def calc_atr(df, p=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(com=p-1, adjust=False).mean()

def calc_bb(s, p=20, n=2):
    mid = s.rolling(p).mean()
    std = s.rolling(p).std()
    return mid - n*std, mid, mid + n*std

def calc_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

def vol_ma(df, p=20):
    return df["Volume"].rolling(p).mean()

# ─────────────────────────────────────────────────────────────────────
# ENGINE 1 — CROWD TRAP DETECTOR
# Retail weak hands chase price, then get trapped at extremes.
# We fade them: buy when they panic-sell, short when they FOMO-buy.
# ─────────────────────────────────────────────────────────────────────
def detect_crowd_trap(df):
    """
    Returns list of (severity, signal_name, explanation, trade_idea)
    """
    signals = []
    if len(df) < 30:
        return signals

    close  = df["Close"]
    volume = df["Volume"]
    rsi    = calc_rsi(close)
    atr    = calc_atr(df)
    vol_avg= vol_ma(df, 20)

    price  = safe_last(close)
    r      = safe_last(rsi)
    r2     = safe_last(rsi, 2)
    r3     = safe_last(rsi, 3)
    v      = safe_last(volume)
    va     = safe_last(vol_avg)
    a      = safe_last(atr)

    if None in [price, r, v, va, a]:
        return signals

    vol_ratio = v / va if va > 0 else 1.0

    # ── FOMO trap: RSI overbought + volume spike + price slowing ──
    price_change_pct = abs(close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100 \
                       if len(close) > 1 else 0
    if r and r >= 72 and vol_ratio >= 1.5 and price_change_pct < 0.5:
        signals.append((
            "HIGH", "🔴 FOMO Trap — Retail buying exhaustion",
            f"RSI={r:.0f} (overbought), volume {vol_ratio:.1f}× avg but price barely moving. "
            f"Retail crowd chasing but smart money not pushing higher — distribution.",
            "SHORT fade: institutions selling into retail demand. "
            f"Entry near current price, stop above recent high +{a*0.5:.1f}"
        ))

    # ── Panic trap: RSI oversold + volume spike + bounce candle ──
    body = close.iloc[-1] - df["Open"].iloc[-1]
    if r and r <= 28 and vol_ratio >= 1.5 and body > 0:
        signals.append((
            "HIGH", "🟢 Panic Trap — Retail selling exhaustion",
            f"RSI={r:.0f} (oversold), volume {vol_ratio:.1f}× avg, last candle closed green. "
            f"Retail panic-selling into smart money absorption — accumulation.",
            "LONG reversal: institutions absorbing retail panic. "
            f"Entry current price, stop below recent low −{a*0.5:.1f}"
        ))

    # ── RSI divergence from crowd expectation ──
    if r and r2 and r3:
        # Crowd expects continuation — RSI says no
        if close.iloc[-1] > close.iloc[-3] and r < r3:  # price up, RSI down
            signals.append((
                "MEDIUM", "🔴 Crowd Continuation Trap — Bearish divergence",
                f"Price made higher high but RSI dropped ({r3:.0f}→{r:.0f}). "
                f"Retail crowd expects more upside but momentum is dying.",
                "SHORT: fade the crowd — momentum already leaving despite price rising."
            ))
        if close.iloc[-1] < close.iloc[-3] and r > r3:  # price down, RSI up
            signals.append((
                "MEDIUM", "🟢 Crowd Breakdown Trap — Bullish divergence",
                f"Price made lower low but RSI rose ({r3:.0f}→{r:.0f}). "
                f"Retail crowd expects more downside but sellers are exhausted.",
                "LONG: fade the crowd — selling pressure weakening despite lower price."
            ))

    # ── Volume climax: huge spike + reversal candle ──
    if vol_ratio >= 3.0:
        candle_range = df["High"].iloc[-1] - df["Low"].iloc[-1]
        if candle_range > 0 and abs(body) < candle_range * 0.3:
            signals.append((
                "HIGH", "⚡ Volume Climax — Indecision at extreme",
                f"Volume {vol_ratio:.1f}× average with a small candle body — "
                f"huge activity but price went nowhere. Classic battle between trapped side and new side.",
                "Wait for next candle direction — whoever wins that candle wins the battle."
            ))

    return signals


# ─────────────────────────────────────────────────────────────────────
# ENGINE 2 — SMART MONEY TRACKER
# Institutions leave footprints: volume without price, quiet accumulation,
# sudden spread widening before a move.
# ─────────────────────────────────────────────────────────────────────
def detect_smart_money(df):
    signals = []
    if len(df) < 30:
        return signals

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    vol_avg= vol_ma(df, 20)

    # ── Accumulation: price flat + volume rising (quiet buying) ──
    last5_price_range = (close.tail(5).max() - close.tail(5).min()) / close.iloc[-1] * 100
    last5_avg_vol     = volume.tail(5).mean()
    prev5_avg_vol     = volume.iloc[-10:-5].mean() if len(volume) >= 10 else None

    if prev5_avg_vol and last5_price_range < 2.0 and last5_avg_vol > prev5_avg_vol * 1.3:
        signals.append((
            "HIGH", "🏦 Smart Money Accumulation",
            f"Price range last 5 bars: {last5_price_range:.1f}% (tight consolidation) "
            f"but volume rose {last5_avg_vol/prev5_avg_vol:.1f}× vs prior 5 bars. "
            f"Institutions quietly loading up — price suppressed while they buy.",
            "LONG: buy the consolidation before the breakout. "
            "Stop below consolidation low. Target = consolidation height × 1.5 above breakout."
        ))

    # ── Distribution: price rising + volume declining ──
    price_trending_up = close.iloc[-1] > close.iloc[-5] if len(close) >= 5 else False
    vol_declining     = volume.tail(5).is_monotonic_decreasing

    if price_trending_up and vol_declining:
        signals.append((
            "MEDIUM", "📤 Smart Money Distribution",
            "Price grinding higher on falling volume — institutions selling into retail strength. "
            "Each push up is weaker. Smart money is offloading to latecomers.",
            "SHORT: fade the grind up. Stop above last high. "
            "Target: last major support level."
        ))

    # ── Absorption: huge volume + small candle body near support ──
    atr_v = safe_last(calc_atr(df))
    v_now = safe_last(volume)
    va_now = safe_last(vol_avg)
    if atr_v and v_now and va_now:
        body_size  = abs(close.iloc[-1] - df["Open"].iloc[-1])
        full_range = high.iloc[-1] - low.iloc[-1]
        vol_ratio  = v_now / va_now if va_now > 0 else 1

        if vol_ratio >= 2.0 and full_range > 0 and body_size < full_range * 0.25:
            signals.append((
                "HIGH", "🧲 Absorption — Smart money stopping the move",
                f"Volume {vol_ratio:.1f}× average, candle range wide "
                f"but body only {body_size/full_range*100:.0f}% of range. "
                "One side tried hard to move price, the other absorbed everything.",
                "Trade OPPOSITE to the candle direction: "
                "if it was a big down candle → LONG (buyers absorbed sellers). "
                "If big up candle → SHORT (sellers absorbed buyers)."
            ))

    return signals


# ─────────────────────────────────────────────────────────────────────
# ENGINE 3 — STOP HUNT DETECTOR
# Market makers push price beyond obvious levels to trigger stops,
# then reverse. The spike IS the signal.
# ─────────────────────────────────────────────────────────────────────
def detect_stop_hunt(df):
    signals = []
    if len(df) < 20:
        return signals

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    atr_v = safe_last(calc_atr(df))

    if not atr_v:
        return signals

    # Find recent swing highs/lows (potential stop clusters)
    lookback = min(20, len(df) - 3)
    recent_high = high.iloc[-lookback:-1].max()
    recent_low  = low.iloc[-lookback:-1].min()

    last_high  = high.iloc[-1]
    last_low   = low.iloc[-1]
    last_close = close.iloc[-1]
    prev_close = close.iloc[-2] if len(close) > 1 else last_close

    # ── Stop hunt high: wick above recent high then closed below it ──
    wick_above = last_high - recent_high
    if wick_above > atr_v * 0.3 and last_close < recent_high:
        wick_pct = wick_above / atr_v
        signals.append((
            "HIGH", "🎯 Stop Hunt — Liquidity grab ABOVE recent high",
            f"Price spiked {wick_above:.2f} ({wick_pct:.1f}× ATR) above recent high "
            f"({recent_high:.2f}) then closed BELOW it at {last_close:.2f}. "
            "Long stops above that level just got triggered — now those longs are trapped short.",
            "SHORT immediately after the reversal candle confirms. "
            f"Stop above the wick high. Target: next support."
        ))

    # ── Stop hunt low: wick below recent low then closed above it ──
    wick_below = recent_low - last_low
    if wick_below > atr_v * 0.3 and last_close > recent_low:
        wick_pct = wick_below / atr_v
        signals.append((
            "HIGH", "🎯 Stop Hunt — Liquidity grab BELOW recent low",
            f"Price spiked {wick_below:.2f} ({wick_pct:.1f}× ATR) below recent low "
            f"({recent_low:.2f}) then closed ABOVE it at {last_close:.2f}. "
            "Short stops below that level just got triggered — shorts are trapped, forced to cover.",
            "LONG immediately after the reversal candle confirms. "
            f"Stop below the wick low. Target: next resistance."
        ))

    # ── False breakout: closed outside key level but came back ──
    if len(close) >= 3:
        two_ago = close.iloc[-3]
        if prev_close > recent_high and last_close < recent_high and two_ago < recent_high:
            signals.append((
                "MEDIUM", "🪤 False Breakout — Trapped longs above resistance",
                f"Closed above resistance ({recent_high:.2f}) last bar but now back below. "
                "Retail breakout buyers are now trapped above, forced to sell.",
                "SHORT: trapped longs will sell to limit losses — accelerates the drop."
            ))
        if prev_close < recent_low and last_close > recent_low and two_ago > recent_low:
            signals.append((
                "MEDIUM", "🪤 False Breakdown — Trapped shorts below support",
                f"Closed below support ({recent_low:.2f}) last bar but now back above. "
                "Retail breakdown sellers trapped below, forced to cover.",
                "LONG: trapped shorts buying to cover — accelerates the recovery."
            ))

    return signals


# ─────────────────────────────────────────────────────────────────────
# ENGINE 4 — GAP FADE ANALYSER
# Gaps open on news/sentiment. Retail chases the gap.
# If it fails to continue → fade it (trapped chasers).
# ─────────────────────────────────────────────────────────────────────
def detect_gap_fade(df_daily):
    signals = []
    if len(df_daily) < 3:
        return signals

    opens  = df_daily["Open"]
    closes = df_daily["Close"]
    highs  = df_daily["High"]
    lows   = df_daily["Low"]
    volume = df_daily["Volume"]
    atr_v  = safe_last(calc_atr(df_daily))

    if not atr_v:
        return signals

    # Today vs yesterday
    gap         = opens.iloc[-1] - closes.iloc[-2]
    gap_pct     = gap / closes.iloc[-2] * 100
    vol_ratio   = volume.iloc[-1] / volume.tail(10).mean() if volume.tail(10).mean() > 0 else 1
    current_price = closes.iloc[-1]

    # ── Gap up fail: gapped up but now trading below open ──
    if gap_pct > 1.5 and current_price < opens.iloc[-1]:
        filled_pct = (opens.iloc[-1] - current_price) / gap * 100 if gap > 0 else 0
        signals.append((
            "HIGH", "🔻 Gap-Up Fade — Retail trapped long above gap",
            f"Gapped up +{gap_pct:.1f}% on open. Price now {filled_pct:.0f}% back into gap. "
            f"News chasers bought the open are now underwater — they will sell to limit losses.",
            f"SHORT: fade the gap. Target = yesterday close ({closes.iloc[-2]:.2f}). "
            "Stop above today's high. This is a high-probability fade when gap>2% and fails in first hour."
        ))

    # ── Gap down fail: gapped down but recovering ──
    elif gap_pct < -1.5 and current_price > opens.iloc[-1]:
        recovery_pct = (current_price - opens.iloc[-1]) / abs(gap) * 100 if gap != 0 else 0
        signals.append((
            "HIGH", "🔺 Gap-Down Fade — Retail trapped short below gap",
            f"Gapped down {gap_pct:.1f}% on open. Price now {recovery_pct:.0f}% recovered. "
            f"Panic sellers who shorted the open are now trapped short.",
            f"LONG: fade the gap. Target = yesterday close ({closes.iloc[-2]:.2f}). "
            "Stop below today's low."
        ))

    # ── Gap up continuation (not a fade — confirm the move) ──
    elif gap_pct > 1.5 and current_price > opens.iloc[-1] * 1.005 and vol_ratio > 1.5:
        signals.append((
            "LOW", "✅ Gap-Up Continuation — Smart money confirming",
            f"Gapped up +{gap_pct:.1f}% AND holding above open on {vol_ratio:.1f}× volume. "
            "This is NOT a fade — institutions participating, gap may extend.",
            "LONG momentum: buy any pullback to the open price. "
            "Stop below today's open. Target = gap size added above yesterday's high."
        ))

    # ── Island reversal: gap up after run, then gap down ──
    if len(df_daily) >= 4:
        prev_gap = opens.iloc[-2] - closes.iloc[-3]
        if prev_gap > atr_v * 0.5 and gap < -atr_v * 0.3:
            signals.append((
                "HIGH", "🏝️ Island Reversal — Everyone trapped on the wrong side",
                "Price gapped up then gapped back down — classic island top. "
                "All buyers from yesterday's high are now trapped with no easy exit.",
                "SHORT aggressively. Target = start of the island gap up."
            ))

    return signals


# ─────────────────────────────────────────────────────────────────────
# ENGINE 5 — DIVERGENCE SCANNER
# Price and momentum/volume disagree → someone is wrong → trade that.
# ─────────────────────────────────────────────────────────────────────
def detect_divergence(df):
    signals = []
    if len(df) < 30:
        return signals

    close  = df["Close"]
    volume = df["Volume"]
    rsi    = calc_rsi(close)
    ml, sl, _ = calc_macd(close)
    obv    = (np.sign(close.diff()) * volume).cumsum()

    # Use last N bars to find divergence
    N = min(15, len(df) - 2)
    p_now  = float(close.iloc[-1]);      p_prev = float(close.iloc[-N])
    r_now  = safe_last(rsi);              r_prev = safe_last(rsi, N)
    m_now  = safe_last(ml);              m_prev = safe_last(ml, N)
    o_now  = float(obv.iloc[-1]);        o_prev = float(obv.iloc[-N])
    v_now  = float(volume.iloc[-1]);     v_avg  = float(volume.tail(20).mean())

    if None in [p_now, r_now, r_prev, m_now, m_prev]:
        return signals

    # ── Bearish RSI divergence: price higher high, RSI lower high ──
    if p_now > p_prev * 1.005 and r_now < r_prev - 5:
        signals.append((
            "HIGH", "📉 Bearish RSI Divergence",
            f"Price: {p_prev:.1f} → {p_now:.1f} (+{(p_now/p_prev-1)*100:.1f}%) "
            f"but RSI: {r_prev:.0f} → {r_now:.0f} (falling). "
            "Buyers pushing price up but with less and less force. "
            "The people holding longs are getting weaker — price will follow RSI down.",
            "SHORT on next bearish candle. Longs are trapped at the top with weakening support."
        ))

    # ── Bullish RSI divergence: price lower low, RSI higher low ──
    elif p_now < p_prev * 0.995 and r_now > r_prev + 5:
        signals.append((
            "HIGH", "📈 Bullish RSI Divergence",
            f"Price: {p_prev:.1f} → {p_now:.1f} ({(p_now/p_prev-1)*100:.1f}%) "
            f"but RSI: {r_prev:.0f} → {r_now:.0f} (rising). "
            "Sellers pushing price lower but running out of energy. "
            "Shorts are trapped and their fuel is nearly gone.",
            "LONG on next bullish candle. Shorts will be forced to cover."
        ))

    # ── MACD divergence ──
    if p_now > p_prev * 1.005 and m_now < m_prev - 0.5:
        signals.append((
            "MEDIUM", "📉 Bearish MACD Divergence",
            f"Price higher but MACD falling ({m_prev:.2f} → {m_now:.2f}). "
            "Trend momentum deteriorating while retail still buying the highs.",
            "SHORT: institutions withdrawing support while retail holds the bag."
        ))
    elif p_now < p_prev * 0.995 and m_now > m_prev + 0.5:
        signals.append((
            "MEDIUM", "📈 Bullish MACD Divergence",
            f"Price lower but MACD rising ({m_prev:.2f} → {m_now:.2f}). "
            "Downward momentum fading while retail still panicking.",
            "LONG: smart money stepping in while retail sells the bottom."
        ))

    # ── OBV divergence (volume doesn't confirm price) ──
    if p_now > p_prev * 1.01 and o_now < o_prev:
        signals.append((
            "MEDIUM", "📦 Bearish OBV Divergence — Distribution",
            "Price rising but On-Balance Volume falling — "
            "more volume is happening on down days than up days. "
            "Smart money distributing into retail buying.",
            "SHORT: the volume tells the real story. Price will follow OBV down."
        ))
    elif p_now < p_prev * 0.99 and o_now > o_prev:
        signals.append((
            "MEDIUM", "📦 Bullish OBV Divergence — Accumulation",
            "Price falling but On-Balance Volume rising — "
            "more volume on up days despite lower price. "
            "Smart money accumulating while retail sells.",
            "LONG: the volume tells the real story. Price will follow OBV up."
        ))

    # ── Volume/price divergence ──
    if v_now > v_avg * 2 and abs(p_now - p_prev) / p_prev < 0.005:
        signals.append((
            "MEDIUM", "🔄 Volume/Price Divergence — Battle at this level",
            f"Volume {v_now/v_avg:.1f}× average but price flat. "
            "Major battle happening here — one side will win, explosive move coming.",
            "Wait for breakout candle then trade in its direction with urgency."
        ))

    return signals


# ─────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_df(ticker, interval="15m", period=None):
    if period is None:
        period = "10d" if interval not in ["1d"] else "6mo"
    safe_period = period if interval != "1m" else ("7d" if period not in ["1d","5d","7d"] else period)
    try:
        df = yf.Ticker(ticker).history(period=safe_period, interval=interval, auto_adjust=True)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            if interval != "1d":
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize("UTC")
                df.index = df.index.tz_convert(HK_TZ)
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────
SEV_COLOR = {"HIGH": "#dc2626", "MEDIUM": "#f59e0b", "LOW": "#16a34a"}
SEV_BG    = {"HIGH": "rgba(220,38,38,0.06)", "MEDIUM": "rgba(245,158,11,0.06)", "LOW": "rgba(22,163,74,0.06)"}

def signal_card(sev, name, explanation, trade_idea):
    c  = SEV_COLOR.get(sev, "#94a3b8")
    bg = SEV_BG.get(sev, "rgba(0,0,0,0.03)")
    st.markdown(
        f"<div style='border:1px solid {c};border-left:4px solid {c};"
        f"border-radius:0 8px 8px 0;background:{bg};"
        f"padding:12px 16px;margin-bottom:10px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>"
        f"<span style='font-weight:600;color:{c};font-size:0.9rem'>{name}</span>"
        f"<span style='font-size:0.68rem;background:{c};color:white;"
        f"padding:2px 7px;border-radius:4px'>{sev}</span></div>"
        f"<div style='font-size:0.81rem;color:#374151;margin-bottom:8px'>{explanation}</div>"
        f"<div style='font-size:0.81rem;font-weight:500;color:#0f172a;"
        f"background:rgba(255,255,255,0.7);padding:6px 10px;border-radius:6px'>"
        f"💡 {trade_idea}</div></div>",
        unsafe_allow_html=True)

def score_bar(score, label, max_signals=4):
    pct   = min(score / max_signals, 1.0)
    color = "#dc2626" if pct >= 0.75 else "#f59e0b" if pct >= 0.4 else "#94a3b8"
    filled = int(pct * 10)
    bar    = "█" * filled + "░" * (10 - filled)
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>"
        f"<span style='font-size:0.78rem;color:#64748b;min-width:160px'>{label}</span>"
        f"<span style='font-family:monospace;color:{color};font-size:0.85rem'>{bar}</span>"
        f"<span style='font-size:0.78rem;color:{color};font-weight:600'>{score} signal{'s' if score!=1 else ''}</span>"
        f"</div>",
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# MAIN CHART
# ─────────────────────────────────────────────────────────────────────
def render_chart(df, df_daily, ticker):
    if df is None or len(df) < 20:
        return

    plot_df = df.tail(100)
    close   = plot_df["Close"]
    volume  = plot_df["Volume"]
    vol_avg = vol_ma(plot_df, 20)
    rsi_s   = calc_rsi(close)
    ml_s, sl_s, hist_s = calc_macd(close)
    vwap_s  = calc_vwap(plot_df)
    obv_s   = (np.sign(close.diff()) * volume).cumsum()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.42, 0.18, 0.20, 0.20],
        vertical_spacing=0.025,
        subplot_titles=["Price + VWAP", "Volume vs Avg", "RSI (divergence)", "OBV (smart money)"]
    )

    # ── Row 1: Candlestick + VWAP ──
    bc = ["#16a34a" if c >= o else "#dc2626"
          for c, o in zip(plot_df["Close"], plot_df["Open"])]
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"],  close=plot_df["Close"],
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=vwap_s,
        line=dict(color="#f59e0b", width=1.8, dash="dash"),
        name="VWAP"), row=1, col=1)

    # Highlight high-volume candles (> 2× avg)
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        va = vol_avg.iloc[i] if i < len(vol_avg) else None
        if va and row["Volume"] > va * 2:
            fig.add_vline(
                x=idx, line_color="rgba(139,92,246,0.25)",
                line_width=8, row=1, col=1)

    # ── Row 2: Volume bars + 20d avg line ──
    fig.add_trace(go.Bar(
        x=plot_df.index, y=volume,
        marker_color=bc, opacity=0.7, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=vol_avg,
        line=dict(color="#8b5cf6", width=1.5, dash="dot"),
        name="Vol 20MA"), row=2, col=1)

    # ── Row 3: RSI ──
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=rsi_s,
        line=dict(color="#f59e0b", width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#16a34a", line_width=1, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(220,38,38,0.05)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(22,163,74,0.05)",  line_width=0, row=3, col=1)

    # ── Row 4: OBV ──
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=obv_s,
        line=dict(color="#2563eb", width=1.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        name="OBV"), row=4, col=1)

    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        yaxis=dict(title="Price", gridcolor="#f1f5f9"),
        yaxis2=dict(title="Volume", gridcolor="#f1f5f9"),
        yaxis3=dict(title="RSI", gridcolor="#f1f5f9"),
        yaxis4=dict(title="OBV", gridcolor="#f1f5f9"),
        xaxis4=dict(gridcolor="#f1f5f9"),
    )
    _dp_is_intra = len(plot_df) > 1 and (plot_df.index[1]-plot_df.index[0]).total_seconds() < 86400
    _apply_rangebreaks(fig, plot_df, is_intraday=_dp_is_intra)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<span style='font-size:0.72rem;color:#94a3b8'>"
        "Purple columns = volume > 2× average (high-activity candles to watch closely)</span>",
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# POSITION SIZING (risk-based)
# ─────────────────────────────────────────────────────────────────────
def render_sizing(price_now, atr_now):
    capital = get_latest_capital()
    st.markdown("### 💰 Position Sizing — based on who is trapped")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Size by how wrong the trapped side is, not by conviction.</span>",
        unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    max_loss_pct = sc1.number_input("Max loss per trade (%)", 0.1, 5.0, 1.5, 0.1, "%.1f", key="dp_001")
    entry_price  = sc2.number_input("Entry price", 0.0,
                                     value=float(round(price_now, 1)) if price_now else 0.0,
                                     step=0.1, format="%.2f", key="dec_001")
    atr_stop     = sc3.number_input("Stop (× ATR)", 0.5, 5.0, 1.0, 0.5, "%.1f",
                                     help="Place stop where trapped players would also be wrong = beyond the fake level", key="dec_002")
    target_rr    = sc4.number_input("Target R:R", 0.5, 10.0, 2.0, 0.5, "%.1f", key="dp_004")

    if atr_now and entry_price > 0:
        direction   = st.session_state.get("direction", "LONG")
        stop_dist   = atr_now * atr_stop
        stop_price  = entry_price - stop_dist if direction == "LONG" else entry_price + stop_dist
        tgt_price   = entry_price + stop_dist * target_rr if direction == "LONG" \
                      else entry_price - stop_dist * target_rr
        max_loss_hkd= capital * max_loss_pct / 100
        shares      = int(max_loss_hkd / stop_dist) if stop_dist > 0 else 0
        pos_val     = shares * entry_price
        max_gain    = max_loss_hkd * target_rr
        partial_tgt = entry_price + stop_dist if direction == "LONG" else entry_price - stop_dist

        rc = st.columns(5)
        for col, label, val, color in [
            (rc[0], "Shares",          f"{shares:,}",                "#0f172a"),
            (rc[1], "Stop loss",       f"HKD {stop_price:,.2f}",     "#dc2626"),
            (rc[2], "Partial (1:1)",   f"HKD {partial_tgt:,.2f}",    "#f59e0b"),
            (rc[3], "Full target",     f"HKD {tgt_price:,.2f}",      "#16a34a"),
            (rc[4], "Max loss / gain", f"−{max_loss_hkd:,.0f} / +{max_gain:,.0f}", "#8b5cf6"),
        ]:
            col.markdown(
                f"<div style='background:#f4f6fb;border-radius:8px;padding:12px 14px;"
                f"border:1px solid #e2e8f0;text-align:center'>"
                f"<div style='font-size:0.7rem;color:#64748b'>{label}</div>"
                f"<div style='font-size:1.05rem;font-weight:700;color:{color}'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────
def render():
    now_hk = datetime.now(HK_TZ)
    st.markdown(
        "## 🎯 Market Psychology — Who Is Trapped? &nbsp;"
        "<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        "padding:2px 7px;border-radius:5px'>SMART MONEY</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"HKT {now_hk.strftime('%H:%M:%S')} · "
        f"Read retail traps · Smart money footprints · Stop hunts · Gap fades · Divergence</span>",
        unsafe_allow_html=True)
    st.markdown("---")

    # ── Controls ─────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    ticker    = ctrl1.selectbox("Stock", list(STOCKS.keys()),
                                 format_func=lambda x: f"{x} — {STOCKS[x]}", key="dp_ticker")
    tf        = ctrl2.selectbox("Intraday timeframe",
                                 ["1m","5m","15m","30m","60m"], index=2, key="dp_tf")
    intra_p   = ctrl3.selectbox("Intraday history",
                                 ["1d","5d","10d","14d","1mo"], index=2,
                                 help="1m capped at 7d",
                                 key="dp_intra_period")
    direction = ctrl4.selectbox("Your intended direction", ["LONG", "SHORT"], key="dp_dir")
    st.session_state["direction"] = direction

    with st.expander("📖 Signal explanations"):
        st.markdown("""
**Crowd Trap** — Detects when the majority of traders are caught on the wrong side.
Gap up + high volume + reversal = retail chased the open, institutions faded them.
Signal: fade the crowd direction (if gap up trap = look short, if gap down trap = look long).

**Smart Money** — Detects institutional accumulation/distribution.
Clean directional candles with steady volume = institutions building a position quietly.
Opposite of retail: they don't chase, they absorb.

**Stop Hunt** — Price briefly spikes through an obvious level (round number, recent high/low)
then immediately reverses. Market makers trigger retail stops to get liquidity, then reverse.
Signal: enter after the reversal back through the level, not before.

**Gap Fade** — Opening gaps that statistically fill.
Large gap + high vol + no follow-through = likely to fill back to previous close.
Best on Tuesday-Wednesday when institutional flow is clearest.

**Divergence** — Price makes a new high but RSI/MACD does not = hidden weakness (look short).
Price makes a new low but indicator does not = hidden strength (look long).
Divergence tells you the move is losing momentum before price confirms it.

**OBV (On-Balance Volume)** — Cumulative volume indicator.
OBV rising + price flat = accumulation (bullish). OBV falling + price flat = distribution (bearish).
Confirms whether volume supports the price direction.

**Position sizing** — Kelly-inspired: risk only what you can afford to lose on one trade.
Default max 1-2% of capital per trade. Stop distance determines your share size.
        """)

    if st.button("🔄 Refresh", key="dp_btn_refresh"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Scanning market participant behaviour…"):
        df_tf    = load_df(ticker, tf)
        df_daily = load_df(ticker, "1d", period="6mo")

    if df_tf is None or len(df_tf) < 20:
        st.warning("Insufficient data — market may be closed or try a longer timeframe.")
        return

    price_now = safe_last(df_tf["Close"])
    atr_now   = safe_last(calc_atr(df_tf))

    # ── Run all 5 engines ────────────────────────────────────────────
    crowd_sigs = detect_crowd_trap(df_tf)
    smart_sigs = detect_smart_money(df_tf)
    stop_sigs  = detect_stop_hunt(df_tf)
    gap_sigs   = detect_gap_fade(df_daily) if len(df_daily) >= 3 else []
    div_sigs   = detect_divergence(df_tf)

    all_sigs   = crowd_sigs + smart_sigs + stop_sigs + gap_sigs + div_sigs
    high_sigs  = [s for s in all_sigs if s[0] == "HIGH"]
    total      = len(all_sigs)

    # ── Trap score summary ───────────────────────────────────────────
    st.markdown("### Signal Overview")
    ov1, ov2 = st.columns([1, 2])

    with ov1:
        overall_color = "#dc2626" if len(high_sigs) >= 2 else \
                        "#f59e0b" if total >= 2 else "#94a3b8"
        st.markdown(
            f"<div style='text-align:center;padding:20px;border-radius:10px;"
            f"border:2px solid {overall_color};background:rgba(0,0,0,0.02)'>"
            f"<div style='font-size:2.5rem;font-weight:800;color:{overall_color}'>{total}</div>"
            f"<div style='font-size:0.8rem;color:#64748b'>active signals</div>"
            f"<div style='font-size:0.75rem;color:{overall_color};margin-top:6px;font-weight:600'>"
            f"{len(high_sigs)} HIGH severity</div>"
            f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:4px'>"
            f"{'⚡ Strong setup — act' if len(high_sigs)>=2 else '👀 Setup forming' if total>=2 else '😴 Wait for setup'}"
            f"</div></div>",
            unsafe_allow_html=True)

    with ov2:
        st.markdown("**Signal breakdown by engine**")
        score_bar(len(crowd_sigs), "👥 Crowd trap detector",  4)
        score_bar(len(smart_sigs), "🏦 Smart money tracker",  3)
        score_bar(len(stop_sigs),  "🎯 Stop hunt detector",   3)
        score_bar(len(gap_sigs),   "📰 Gap fade analyser",    3)
        score_bar(len(div_sigs),   "📊 Divergence scanner",   4)

    # ── Current price strip ──────────────────────────────────────────
    rsi_now  = safe_last(calc_rsi(df_tf["Close"]))
    vwap_now = safe_last(calc_vwap(df_tf))
    v_now    = safe_last(df_tf["Volume"])
    va_now   = safe_last(vol_ma(df_tf, 20))
    vol_r    = v_now / va_now if (v_now and va_now and va_now > 0) else None

    ps = st.columns(5)
    ps[0].metric("Price",      f"HKD {price_now:,.2f}" if price_now else "—")
    ps[1].metric("RSI",        f"{rsi_now:.1f}" if rsi_now else "—",
                 delta="Overbought" if rsi_now and rsi_now>=70
                       else "Oversold" if rsi_now and rsi_now<=30 else "Neutral")
    ps[2].metric("vs VWAP",    f"{price_now-vwap_now:+.2f}" if (price_now and vwap_now) else "—",
                 delta="Above" if (price_now and vwap_now and price_now>vwap_now) else "Below")
    ps[3].metric("Volume ×avg",f"{vol_r:.1f}×" if vol_r else "—",
                 delta="Unusual" if vol_r and vol_r>=2 else "Normal")
    ps[4].metric("ATR",        f"{atr_now:.2f}" if atr_now else "—",
                 help="Expected move per candle")

    st.markdown("---")

    # ── Signal cards ─────────────────────────────────────────────────
    if not all_sigs:
        st.info("No strong participant behaviour signals detected right now. "
                "Good setups require patience — check back when volume picks up.")
    else:
        # Sort: HIGH first
        sorted_sigs = sorted(all_sigs, key=lambda x: {"HIGH":0,"MEDIUM":1,"LOW":2}.get(x[0],3))

        # Filter to direction-relevant signals
        dir_filter = st.checkbox(
            f"Show only {direction}-relevant signals", value=False, key="dp_dirfilter")

        long_keywords  = ["LONG","long","Bullish","bullish","accumul","bounce","recover",
                          "oversold","Oversold","absorption","cover"]
        short_keywords = ["SHORT","short","Bearish","bearish","distribut","fade",
                          "overbought","Overbought","distribution","sell"]

        tabs_sigs = st.tabs([
            f"👥 Crowd ({len(crowd_sigs)})",
            f"🏦 Smart Money ({len(smart_sigs)})",
            f"🎯 Stop Hunt ({len(stop_sigs)})",
            f"📰 Gap Fade ({len(gap_sigs)})",
            f"📊 Divergence ({len(div_sigs)})",
            f"⚡ All HIGH ({len(high_sigs)})",
        ])

        def show_signals(sigs, tab, dir_only=False, kws=None):
            with tab:
                if not sigs:
                    st.markdown(
                        "<span style='color:#94a3b8;font-size:0.85rem'>"
                        "No signals from this engine right now.</span>",
                        unsafe_allow_html=True)
                    return
                for sev, name, expl, idea in sigs:
                    if dir_only and kws:
                        if not any(k in idea for k in kws):
                            continue
                    signal_card(sev, name, expl, idea)

        kws = long_keywords if direction == "LONG" else short_keywords
        show_signals(crowd_sigs, tabs_sigs[0], dir_filter, kws)
        show_signals(smart_sigs, tabs_sigs[1], dir_filter, kws)
        show_signals(stop_sigs,  tabs_sigs[2], dir_filter, kws)
        show_signals(gap_sigs,   tabs_sigs[3], dir_filter, kws)
        show_signals(div_sigs,   tabs_sigs[4], dir_filter, kws)
        show_signals(high_sigs,  tabs_sigs[5])

    st.markdown("---")

    # ── Chart ────────────────────────────────────────────────────────
    st.markdown("### Chart — Price · Volume · RSI · OBV")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "OBV (On-Balance Volume) shows where smart money is really going. "
        "When OBV and price diverge — that is your signal.</span>",
        unsafe_allow_html=True)
    render_chart(df_tf, df_daily, ticker)

    st.markdown("---")

    # ── Position sizing ──────────────────────────────────────────────
    render_sizing(price_now, atr_now)

    st.markdown(
        "<br><span style='color:#94a3b8;font-size:0.74rem'>"
        "All signals are pattern-based detections from price/volume data. "
        "Not financial advice. Data via yfinance.</span>",
        unsafe_allow_html=True)
