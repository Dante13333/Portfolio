"""
analysis_page.py
Short-trade analysis module for HK Portfolio Dashboard.
Sections:
  1. Psychology Indicators  — RSI, MACD, Bollinger Bands (multi-timeframe)
  2. Industry Beta          — beta vs HSI, sector correlation, relative strength
  3. Historical Patterns    — similar candle sequences, open/close patterns
  4. Entry/Exit Signal Zone — live signal scanner + support/resistance
  5. Trade Journal          — log trades, track P&L, emotion tagging
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

from db_manager import (
    init_db, upsert_daily, upsert_intraday,
    get_daily, get_intraday, get_daily_stats,
    save_trade, get_trades, update_trade_exit, delete_trade,
)

HK_TZ   = pytz.timezone("Asia/Hong_Kong")
_FALLBACK_STOCKS = {
    "0100.HK": {"name": "MiniMax",  "color": "#2563eb"},
    "2513.HK": {"name": "Zhipu",    "color": "#16a34a"},
}
PEERS   = ["0700.HK", "9999.HK", "1024.HK", "9888.HK"]
BENCH   = "^HSI"

def _get_all_tickers():
    colors = ["#2563eb","#16a34a","#f59e0b","#8b5cf6","#dc2626",
              "#0891b2","#ec4899","#14b8a6","#f97316","#84cc16"]
    out = {}
    try:
        from db_manager import get_portfolio_full
        port = get_portfolio_full()
        if not port.empty:
            for i,(_,r) in enumerate(port[port["status"].isin(["OPEN","WATCH"])].iterrows()):
                out[r["ticker"]]={"name":r.get("name",r["ticker"]),"color":colors[i%len(colors)]}
    except Exception: pass
    try:
        from portfolio_manager import get_monitor_pos
        mon = get_monitor_pos()
        if not mon.empty:
            for i,(_,r) in enumerate(mon[mon["status"].isin(["OPEN","WATCH"])].iterrows()):
                if r["ticker"] not in out:
                    out[r["ticker"]]={"name":r.get("name",r["ticker"]),"color":colors[(len(out)+i)%len(colors)]}
    except Exception: pass
    return out if out else _FALLBACK_STOCKS

# ── INDICATOR MATH ────────────────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bb(series: pd.Series, period: int = 20, std: float = 2.0):
    mid  = series.rolling(period).mean()
    band = series.rolling(period).std()
    return mid - std * band, mid, mid + std * band

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def calc_vwap(df: pd.DataFrame) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    cv  = (tp * df["Volume"]).cumsum()
    return cv / df["Volume"].cumsum()

def calc_stoch(df: pd.DataFrame, k=14, d=3) -> tuple:
    low_min  = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    k_line   = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    d_line   = k_line.rolling(d).mean()
    return k_line, d_line

def find_support_resistance(df: pd.DataFrame, window: int = 10, n: int = 5):
    highs  = df["High"].rolling(window, center=True).max()
    lows   = df["Low"].rolling(window, center=True).min()
    res_levels = df["High"][df["High"] == highs].dropna().tail(n).values
    sup_levels = df["Low"][df["Low"] == lows].dropna().tail(n).values
    return sorted(set(res_levels.round(1))), sorted(set(sup_levels.round(1)))

def calc_beta(stock_returns: pd.Series, bench_returns: pd.Series) -> float:
    aligned = pd.concat([stock_returns, bench_returns], axis=1).dropna()
    if len(aligned) < 10:
        return None
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return round(cov[0, 1] / cov[1, 1], 3)

def pattern_similarity(ref: np.ndarray, window: np.ndarray) -> float:
    """Cosine similarity between two normalised return sequences."""
    if len(ref) != len(window) or np.std(ref) == 0 or np.std(window) == 0:
        return 0.0
    r = (ref - ref.mean()) / (ref.std() + 1e-9)
    w = (window - window.mean()) / (window.std() + 1e-9)
    return float(np.dot(r, w) / (np.linalg.norm(r) * np.linalg.norm(w) + 1e-9))

def find_similar_patterns(df: pd.DataFrame, lookback: int = 10, top_n: int = 5):
    """Find historical windows most similar to the most recent `lookback` candles."""
    closes  = df["Close"].values
    returns = np.diff(closes) / closes[:-1]
    if len(returns) < lookback + 5:
        return []
    ref     = returns[-lookback:]
    results = []
    for i in range(len(returns) - lookback - 1):
        window = returns[i: i + lookback]
        sim    = pattern_similarity(ref, window)
        date   = df.index[i + lookback]
        fwd_ret= (closes[i + lookback + 1] - closes[i + lookback]) / closes[i + lookback] * 100 \
                 if i + lookback + 1 < len(closes) else None
        results.append({"date": date, "similarity": sim, "fwd_return": fwd_ret, "idx": i})
    return sorted(results, key=lambda x: -x["similarity"])[:top_n]

# ── DATA FETCH ────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def load_ohlcv(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    # 1m bars capped at 7d by Yahoo Finance
    safe = period if interval != "1m" else ("7d" if period not in ["1d","5d","7d"] else period)
    try:
        df = yf.Ticker(ticker).history(period=safe, interval=interval, auto_adjust=True)
        if not df.empty:
            if interval == "1d":
                upsert_daily(ticker, df)
            else:
                upsert_intraday(ticker, df, interval)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def load_multi(tickers: list, period: str = "3mo") -> pd.DataFrame:
    frames = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=True)
            if not df.empty:
                frames[t] = df["Close"]
            time.sleep(0.3)
        except Exception:
            continue
    if frames:
        return pd.DataFrame(frames).dropna()
    return pd.DataFrame()

# ── COLOUR HELPERS ───────────────────────────────────────────────────
def pos_neg(v): return "#16a34a" if (v or 0) >= 0 else "#dc2626"
def rsi_color(v):
    if v >= 70: return "#dc2626"
    if v <= 30: return "#16a34a"
    return "#f59e0b"

# ── CHART HELPERS ────────────────────────────────────────────────────
def indicator_fig(df, title=""):
    """Base OHLCV fig with 4 rows: candle / volume / indicator / indicator."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.15, 0.20, 0.20],
        vertical_spacing=0.025,
        subplot_titles=["", "", "", ""]
    )
    bc = ["#16a34a" if c >= o else "#dc2626"
          for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        name="Price"), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], marker_color=bc, opacity=0.65, name="Vol"), row=2, col=1)
    is_intra = len(df) > 1 and (df.index[1]-df.index[0]).total_seconds() < 86400
    fig.update_layout(
        title=title, height=680,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        yaxis=dict(title="Price", gridcolor="#f1f5f9"),
        yaxis2=dict(title="Vol",  gridcolor="#f1f5f9"),
        yaxis3=dict(gridcolor="#f1f5f9"),
        yaxis4=dict(gridcolor="#f1f5f9"),
    )
    _apply_rangebreaks(fig, df, is_intraday=is_intra)
    return fig

# ═════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION — called from hk_dashboard.py
# ═════════════════════════════════════════════════════════════════════
def render(chart_interval="5m", daily_period="3mo"):
    import strategy_page as _sp
    import cycle_ml as _cm
    import fundamentals as _fund

    st.markdown(
        "## 🔬 Analysis &amp; Strategy",
        unsafe_allow_html=True)
    st.markdown(
        "<span style='color:#64748b;font-size:0.79rem'>"
        "Technical analysis · Market study · ML rules · Cycle detection — all in one place</span>",
        unsafe_allow_html=True)

    master_tabs = st.tabs([
        "📊 Technical Analysis",
        "🧠 Market Study",
        "🔄 Cycle ML",
        "📈 Fundamentals",
    ])

    with master_tabs[0]:
        _render_analysis(chart_interval=chart_interval, daily_period=daily_period)

    with master_tabs[1]:
        _sp.render()

    with master_tabs[2]:
        _cm.render()

    with master_tabs[3]:
        _fund.render_fundamentals_page()


def _render_analysis(chart_interval="5m", daily_period="3mo"):
    """Original analysis page content."""
    now_hk = datetime.now(HK_TZ)
    st.markdown(
        f"## 🔬 Short-Trade Analysis &nbsp;"
        f"<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        f"padding:2px 7px;border-radius:5px'>LIVE</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"HKT {now_hk.strftime('%Y-%m-%d %H:%M:%S')} · "
        f"Industry beta · Psychology indicators · Pattern matching · Signal zones</span>",
        unsafe_allow_html=True)
    with st.expander("📖 Metric explanations"):
        st.markdown("""
**RSI (Relative Strength Index)** — Momentum 0-100.
>70 = overbought (potential reversal down). <30 = oversold (potential bounce). 40-60 = neutral.

**MACD** — Moving average convergence/divergence. Histogram above zero = bullish momentum.
Crossover (line crosses signal) = momentum shift signal.

**Bollinger Bands** — Price channel 2 std devs around 20-day MA.
Price at upper band = extended, watch for reversal. At lower band = oversold, watch for bounce.
BB% position: 100% = at upper band, 0% = at lower band.

**ATR (Average True Range)** — Average daily price range. Use for stop placement:
stop = entry +/- 1.5x ATR gives room for normal volatility without being stopped out by noise.

**VWAP** — Volume Weighted Average Price. Institutional benchmark.
Price above VWAP = bullish intraday bias. Below = bearish. Mean-reversion trades fade moves away from VWAP.

**Beta** — How much this stock moves relative to the HSI index.
Beta 2.0 = moves twice as much as the index. Higher beta = more volatile, bigger swings, more risk.

**Choppiness Index** — 0-100. >61.8 = oscillating (good for range trading). <38.2 = trending one way.

**Support / Resistance** — Price levels where buying/selling has historically clustered.
Strong support = where price bounced multiple times. Strong resistance = where it was rejected.
        """)

    st.markdown("---")

    # ── Controls ─────────────────────────────────────────────────────
    STOCKS = _get_all_tickers()
    cc1, cc2, cc3, cc4 = st.columns(4)
    sym      = cc1.selectbox("Instrument", list(STOCKS.keys()),
                             format_func=lambda x: f"{x} — {STOCKS[x]['name']}", key="ap_sym")
    tf_label = cc2.selectbox("Indicator timeframe",
                             ["5m","15m","30m","60m","1d"], index=2, key="ap_tf")
    intra_period = cc3.selectbox("Intraday history",
                             ["1d","5d","10d","14d","1mo"], index=2,
                             help="1m capped at 7d by Yahoo",
                             key="ap_intra_period")
    period   = cc4.selectbox("Daily history",
                             ["1mo","3mo","6mo","1y"], index=1, key="ap_period")

    if st.button("🔄 Refresh analysis data", key="ap_btn_refresh"):
        st.cache_data.clear()
        st.rerun()

    # Load data
    with st.spinner("Loading data…"):
        _ip    = intra_period if tf_label != "1d" else period
        df_tf  = load_ohlcv(sym, period=_ip, interval=tf_label)
        df_day = load_ohlcv(sym, period=period, interval="1d")
        df_hsi = load_ohlcv(BENCH, period=period, interval="1d")
        peers  = [sym] + PEERS
        df_multi = load_multi(peers, period=period)

    if df_tf.empty or df_day.empty:
        st.warning("Not enough data fetched — try again during market hours or choose a longer window.")
        return

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1 — PSYCHOLOGY INDICATORS
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 1 · Psychology Indicators")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "RSI · MACD · Bollinger Bands · Stochastic · ATR · VWAP</span>",
        unsafe_allow_html=True)

    ind_tab1, ind_tab2, ind_tab3 = st.tabs(["RSI + MACD", "Bollinger Bands + Stoch", "VWAP + ATR"])

    # ── RSI + MACD ──
    with ind_tab1:
        rsi = calc_rsi(df_tf["Close"])
        macd_line, sig_line, hist = calc_macd(df_tf["Close"])
        last_rsi  = rsi.iloc[-1]
        last_macd = macd_line.iloc[-1]
        last_sig  = sig_line.iloc[-1]

        # Signal pills
        p1, p2, p3 = st.columns(3)
        rsi_label = "OVERBOUGHT ▼" if last_rsi >= 70 else ("OVERSOLD ▲" if last_rsi <= 30 else "NEUTRAL")
        macd_label= "BULLISH ▲" if last_macd > last_sig else "BEARISH ▼"
        p1.markdown(
            f"<div style='text-align:center;padding:10px;border-radius:8px;"
            f"background:{rsi_color(last_rsi)}1a;border:1px solid {rsi_color(last_rsi)}'>"
            f"<div style='font-size:0.7rem;color:#64748b'>RSI ({tf_label})</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{rsi_color(last_rsi)}'>{last_rsi:.1f}</div>"
            f"<div style='font-size:0.72rem;color:{rsi_color(last_rsi)}'>{rsi_label}</div></div>",
            unsafe_allow_html=True)
        p2.markdown(
            f"<div style='text-align:center;padding:10px;border-radius:8px;"
            f"background:rgba(37,99,235,0.10);border:1px solid #2563eb'>"
            f"<div style='font-size:0.7rem;color:#64748b'>MACD ({tf_label})</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:#2563eb'>{last_macd:.2f}</div>"
            f"<div style='font-size:0.72rem;color:#2563eb'>{macd_label}</div></div>",
            unsafe_allow_html=True)
        cross = "BULLISH CROSS ▲" if (macd_line.iloc[-1] > sig_line.iloc[-1] and
                                       macd_line.iloc[-2] <= sig_line.iloc[-2]) else \
                "BEARISH CROSS ▼" if (macd_line.iloc[-1] < sig_line.iloc[-1] and
                                       macd_line.iloc[-2] >= sig_line.iloc[-2]) else "No cross"
        cross_col = "#16a34a" if "BULL" in cross else ("#dc2626" if "BEAR" in cross else "#64748b")
        p3.markdown(
            f"<div style='text-align:center;padding:10px;border-radius:8px;"
            f"background:{cross_col}1a;border:1px solid {cross_col}'>"
            f"<div style='font-size:0.7rem;color:#64748b'>MACD Signal</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{cross_col};margin-top:8px'>{cross}</div></div>",
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig = indicator_fig(df_tf)

        # RSI
        fig.add_trace(go.Scatter(x=df_tf.index, y=rsi, name="RSI",
                                  line=dict(color="#f59e0b", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#16a34a", line_width=1, row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(220,38,38,0.06)",  line_width=0, row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(22,163,74,0.06)",  line_width=0, row=3, col=1)

        # MACD
        hist_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in hist]
        fig.add_trace(go.Bar(x=df_tf.index, y=hist, marker_color=hist_colors,
                              name="Histogram", opacity=0.7), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_tf.index, y=macd_line, name="MACD",
                                  line=dict(color="#2563eb", width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_tf.index, y=sig_line, name="Signal",
                                  line=dict(color="#f59e0b", width=1.2, dash="dot")), row=4, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # ── Bollinger Bands + Stochastic ──
    with ind_tab2:
        bb_lo, bb_mid, bb_hi = calc_bb(df_tf["Close"])
        k_stoch, d_stoch     = calc_stoch(df_tf)
        last_k = k_stoch.iloc[-1]
        last_d = d_stoch.iloc[-1]
        bb_pct = (df_tf["Close"].iloc[-1] - bb_lo.iloc[-1]) / \
                 (bb_hi.iloc[-1] - bb_lo.iloc[-1] + 1e-9) * 100

        q1, q2 = st.columns(2)
        bb_label = "Near Upper Band" if bb_pct > 80 else ("Near Lower Band" if bb_pct < 20 else "Mid Band")
        bb_color = "#dc2626" if bb_pct > 80 else ("#16a34a" if bb_pct < 20 else "#f59e0b")
        q1.markdown(
            f"<div style='text-align:center;padding:10px;border-radius:8px;"
            f"background:{bb_color}1a;border:1px solid {bb_color}'>"
            f"<div style='font-size:0.7rem;color:#64748b'>BB %B</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{bb_color}'>{bb_pct:.1f}%</div>"
            f"<div style='font-size:0.72rem;color:{bb_color}'>{bb_label}</div></div>",
            unsafe_allow_html=True)
        stoch_label = "Overbought" if last_k > 80 else ("Oversold" if last_k < 20 else "Neutral")
        stoch_color = "#dc2626" if last_k > 80 else ("#16a34a" if last_k < 20 else "#f59e0b")
        q2.markdown(
            f"<div style='text-align:center;padding:10px;border-radius:8px;"
            f"background:{stoch_color}1a;border:1px solid {stoch_color}'>"
            f"<div style='font-size:0.7rem;color:#64748b'>Stochastic %K</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{stoch_color}'>{last_k:.1f}</div>"
            f"<div style='font-size:0.72rem;color:{stoch_color}'>{stoch_label}</div></div>",
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig2 = indicator_fig(df_tf)
        fig2.add_trace(go.Scatter(x=df_tf.index, y=bb_hi,  name="BB Upper",
                                   line=dict(color="#dc2626", width=1, dash="dot")), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df_tf.index, y=bb_mid, name="BB Mid",
                                   line=dict(color="#94a3b8", width=1)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df_tf.index, y=bb_lo,  name="BB Lower",
                                   line=dict(color="#16a34a", width=1, dash="dot"),
                                   fill="tonexty", fillcolor="rgba(99,155,34,0.05)"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df_tf.index, y=k_stoch, name="%K",
                                   line=dict(color="#2563eb", width=1.5)), row=3, col=1)
        fig2.add_trace(go.Scatter(x=df_tf.index, y=d_stoch, name="%D",
                                   line=dict(color="#f59e0b", width=1.2, dash="dot")), row=3, col=1)
        fig2.add_hline(y=80, line_dash="dot", line_color="#dc2626", line_width=1, row=3, col=1)
        fig2.add_hline(y=20, line_dash="dot", line_color="#16a34a", line_width=1, row=3, col=1)
        # BB width in row 4
        bb_width = (bb_hi - bb_lo) / bb_mid * 100
        fig2.add_trace(go.Scatter(x=df_tf.index, y=bb_width, name="BB Width %",
                                   line=dict(color="#8b5cf6", width=1.5),
                                   fill="tozeroy", fillcolor="rgba(139,92,246,0.08)"), row=4, col=1)
        fig2.update_yaxes(title_text="Stoch", row=3, col=1)
        fig2.update_yaxes(title_text="BB Width %", row=4, col=1)
        st.plotly_chart(fig2, use_container_width=True)

    # ── VWAP + ATR ──
    with ind_tab3:
        atr  = calc_atr(df_tf)
        vwap = calc_vwap(df_tf)
        last_close = df_tf["Close"].iloc[-1]
        last_vwap  = vwap.iloc[-1]
        last_atr   = atr.iloc[-1]
        vwap_pos   = "Above VWAP ▲" if last_close > last_vwap else "Below VWAP ▼"
        vwap_color = "#16a34a" if last_close > last_vwap else "#dc2626"

        a1, a2 = st.columns(2)
        a1.markdown(
            f"<div style='text-align:center;padding:10px;border-radius:8px;"
            f"background:{vwap_color}1a;border:1px solid {vwap_color}'>"
            f"<div style='font-size:0.7rem;color:#64748b'>vs VWAP</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{vwap_color}'>"
            f"{last_close - last_vwap:+.2f}</div>"
            f"<div style='font-size:0.72rem;color:{vwap_color}'>{vwap_pos}</div></div>",
            unsafe_allow_html=True)
        a2.markdown(
            f"<div style='text-align:center;padding:10px;border-radius:8px;"
            f"background:rgba(139,92,246,0.10);border:1px solid #8b5cf6'>"
            f"<div style='font-size:0.7rem;color:#64748b'>ATR ({tf_label})</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:#8b5cf6'>{last_atr:.2f}</div>"
            f"<div style='font-size:0.72rem;color:#8b5cf6'>Volatility range per candle</div></div>",
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig3 = indicator_fig(df_tf)
        fig3.add_trace(go.Scatter(x=df_tf.index, y=vwap, name="VWAP",
                                   line=dict(color="#f59e0b", width=2, dash="dash")), row=1, col=1)
        fig3.add_trace(go.Scatter(x=df_tf.index, y=atr, name="ATR",
                                   line=dict(color="#8b5cf6", width=1.5),
                                   fill="tozeroy", fillcolor="rgba(139,92,246,0.08)"), row=3, col=1)
        # ATR channels
        fig3.add_trace(go.Scatter(x=df_tf.index, y=df_tf["Close"] + atr,
                                   line=dict(color="#dc2626", width=1, dash="dot"),
                                   name="ATR Upper"), row=1, col=1)
        fig3.add_trace(go.Scatter(x=df_tf.index, y=df_tf["Close"] - atr,
                                   line=dict(color="#16a34a", width=1, dash="dot"),
                                   name="ATR Lower"), row=1, col=1)
        atr_pct = (atr / df_tf["Close"] * 100)
        fig3.add_trace(go.Scatter(x=df_tf.index, y=atr_pct, name="ATR %",
                                   line=dict(color="#f59e0b", width=1.5)), row=4, col=1)
        fig3.update_yaxes(title_text="ATR", row=3, col=1)
        fig3.update_yaxes(title_text="ATR %", row=4, col=1)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2 — INDUSTRY BETA & SECTOR
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 2 · Industry Beta & Sector Comparison")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Beta vs HSI · Relative strength · Peer correlation · Rolling beta</span>",
        unsafe_allow_html=True)

    ret_stock = df_day["Close"].pct_change().dropna()
    ret_hsi   = df_hsi["Close"].pct_change().dropna() if not df_hsi.empty else pd.Series()
    beta_val  = calc_beta(ret_stock, ret_hsi) if not ret_hsi.empty else None

    b1, b2, b3, b4 = st.columns(4)

    # ── Beta card ──
    _bv        = beta_val if beta_val is not None else 0.0
    beta_color = "#dc2626" if _bv > 1.5 else "#f59e0b" if _bv > 1 else "#16a34a"
    beta_str   = "{:.2f}".format(_bv) if beta_val is not None else "—"
    beta_label = "High volatility" if _bv > 1.5 else "Moderate" if _bv > 1 else "Low"
    b1.markdown(
        "<div style='text-align:center;padding:10px;border-radius:8px;"
        "background:rgba(0,0,0,0.04);border:1px solid " + beta_color + "'>"
        "<div style='font-size:0.7rem;color:#64748b'>Beta vs HSI</div>"
        "<div style='font-size:1.6rem;font-weight:700;color:" + beta_color + "'>"
        + beta_str +
        "</div><div style='font-size:0.72rem;color:" + beta_color + "'>"
        + beta_label + "</div></div>",
        unsafe_allow_html=True)

    # ── Correlation card ──
    if not ret_hsi.empty:
        _aligned  = pd.concat([ret_stock, ret_hsi], axis=1).dropna()
        _corr_raw = _aligned.corr().iloc[0, 1] if len(_aligned) > 5 else None
        _corr     = float(_corr_raw) if _corr_raw is not None and not np.isnan(float(_corr_raw)) else None
        corr_str  = "{:.2f}".format(_corr) if _corr is not None else "—"
        corr_lbl  = ("Strong" if abs(_corr) > 0.7 else "Moderate" if abs(_corr) > 0.4 else "Weak") if _corr is not None else "—"
        b2.markdown(
            "<div style='text-align:center;padding:10px;border-radius:8px;"
            "background:rgba(37,99,235,0.08);border:1px solid #2563eb'>"
            "<div style='font-size:0.7rem;color:#64748b'>Corr vs HSI</div>"
            "<div style='font-size:1.6rem;font-weight:700;color:#2563eb'>" + corr_str + "</div>"
            "<div style='font-size:0.72rem;color:#2563eb'>" + corr_lbl + "</div></div>",
            unsafe_allow_html=True)

    # ── Relative strength card ──
    _aligned_rs = pd.concat([ret_stock, ret_hsi], axis=1).dropna() if not ret_hsi.empty else pd.DataFrame()
    if len(_aligned_rs) > 2:
        _s_ret = _aligned_rs.iloc[:, 0]
        _h_ret = _aligned_rs.iloc[:, 1]
        stock_cum = float((1 + _s_ret).cumprod().iloc[-1]) - 1
        hsi_cum   = float((1 + _h_ret).cumprod().iloc[-1]) - 1
        _rs       = stock_cum - hsi_cum
        rs_color  = "#16a34a" if _rs > 0 else "#dc2626"
        rs_str    = "{:+.1f}%".format(_rs * 100)
        b3.markdown(
            "<div style='text-align:center;padding:10px;border-radius:8px;"
            "background:rgba(0,0,0,0.04);border:1px solid " + rs_color + "'>"
            "<div style='font-size:0.7rem;color:#64748b'>Rel. Strength vs HSI</div>"
            "<div style='font-size:1.6rem;font-weight:700;color:" + rs_color + "'>" + rs_str + "</div>"
            "<div style='font-size:0.72rem;color:" + rs_color + "'>Over " + str(period) + "</div></div>",
            unsafe_allow_html=True)
    else:
        b3.markdown(
            "<div style='text-align:center;padding:10px;border-radius:8px;"
            "background:rgba(0,0,0,0.04);border:1px solid #94a3b8'>"
            "<div style='font-size:0.7rem;color:#64748b'>Rel. Strength vs HSI</div>"
            "<div style='font-size:1.6rem;font-weight:700;color:#94a3b8'>—</div>"
            "<div style='font-size:0.72rem;color:#94a3b8'>No aligned data</div></div>",
            unsafe_allow_html=True)

    # ── Volatility card ──
    _vol_raw = ret_stock.std() * np.sqrt(252) * 100
    _vol     = float(_vol_raw) if not np.isnan(float(_vol_raw)) else None
    vol_str  = "{:.1f}%".format(_vol) if _vol is not None else "—"
    b4.markdown(
        "<div style='text-align:center;padding:10px;border-radius:8px;"
        "background:rgba(139,92,246,0.08);border:1px solid #8b5cf6'>"
        "<div style='font-size:0.7rem;color:#64748b'>Annualised Vol</div>"
        "<div style='font-size:1.6rem;font-weight:700;color:#8b5cf6'>" + vol_str + "</div>"
        "<div style='font-size:0.72rem;color:#8b5cf6'>Historical</div></div>",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    beta_col, comp_col = st.columns(2)

    # Rolling 20-day beta
    with beta_col:
        st.markdown("**Rolling 20-day Beta vs HSI**")
        if not ret_hsi.empty and len(ret_stock) > 25:
            rolling_betas, dates = [], []
            aligned = pd.concat([ret_stock, ret_hsi], axis=1).dropna()
            aligned.columns = ["stock", "hsi"]
            for i in range(20, len(aligned)):
                window = aligned.iloc[i-20:i]
                cov    = np.cov(window["stock"], window["hsi"])
                b      = cov[0, 1] / (cov[1, 1] + 1e-12)
                rolling_betas.append(b)
                dates.append(aligned.index[i])
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(
                x=dates, y=rolling_betas, name="Beta",
                line=dict(color="#2563eb", width=2),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"))
            fig_beta.add_hline(y=1, line_dash="dot", line_color="#94a3b8", line_width=1)
            fig_beta.add_hline(y=0, line_color="#e2e8f0", line_width=0.5)
            fig_beta.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                                    plot_bgcolor="white", paper_bgcolor="white",
                                    showlegend=False,
                                    yaxis=dict(title="Beta", gridcolor="#f1f5f9"),
                                    xaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_beta, use_container_width=True)
        else:
            st.info("Insufficient data for rolling beta.")

    # Peer comparison — normalised performance
    with comp_col:
        st.markdown("**Peer Relative Performance (normalised to 100)**")
        if not df_multi.empty:
            normed = df_multi / df_multi.iloc[0] * 100
            fig_peer = go.Figure()
            colors_p = ["#2563eb","#16a34a","#f59e0b","#dc2626","#8b5cf6","#0891b2"]
            for i, col in enumerate(normed.columns):
                lw   = 2.5 if col == sym else 1
                dash = "solid" if col == sym else "dot"
                label= STOCKS.get(col, {}).get("name", col)
                fig_peer.add_trace(go.Scatter(
                    x=normed.index, y=normed[col], name=label,
                    line=dict(color=colors_p[i % len(colors_p)], width=lw, dash=dash)))
            fig_peer.add_hline(y=100, line_dash="dot", line_color="#94a3b8", line_width=1)
            fig_peer.update_layout(
                height=260, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(font=dict(size=10), orientation="h",
                            yanchor="bottom", y=1, xanchor="left", x=0),
                yaxis=dict(title="Indexed", gridcolor="#f1f5f9"),
                xaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_peer, use_container_width=True)

    # Correlation heatmap
    if not df_multi.empty and len(df_multi.columns) >= 2:
        st.markdown("**Peer Correlation Matrix**")
        ret_multi = df_multi.pct_change().dropna()
        corr_mat  = ret_multi.corr().round(2)
        labels    = [STOCKS.get(c, {}).get("name", c) for c in corr_mat.columns]
        fig_heat  = go.Figure(go.Heatmap(
            z=corr_mat.values, x=labels, y=labels,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr_mat.values.round(2),
            texttemplate="%{text}", textfont=dict(size=11),
            colorbar=dict(thickness=12, len=0.8)))
        fig_heat.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                                paper_bgcolor="white")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3 — HISTORICAL PATTERN MATCHING
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 3 · Historical Pattern Matching")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Find past periods whose price sequence most resembles today's recent moves. "
        "See what happened next.</span>", unsafe_allow_html=True)

    pm_col1, pm_col2 = st.columns([1, 2])
    with pm_col1:
        lookback_n = st.slider("Pattern lookback (candles)", 5, 30, 10, key="ap_lookback")
        top_n      = st.slider("Top matches to show", 3, 8, 5, key="ap_topn")

    matches = find_similar_patterns(df_day, lookback=lookback_n, top_n=top_n)

    if matches:
        fwd_rets = [m["fwd_return"] for m in matches if m["fwd_return"] is not None]
        avg_fwd  = np.mean(fwd_rets) if fwd_rets else None
        med_fwd  = np.median(fwd_rets) if fwd_rets else None
        pct_up   = sum(1 for r in fwd_rets if r > 0) / len(fwd_rets) * 100 if fwd_rets else None

        pm_col2.markdown(
            f"<div style='display:flex;gap:16px;flex-wrap:wrap;margin-top:8px'>"
            f"<div style='text-align:center;padding:10px 16px;border-radius:8px;"
            f"background:{'#16a34a' if (avg_fwd or 0)>0 else '#dc2626'}1a;"
            f"border:1px solid {'#16a34a' if (avg_fwd or 0)>0 else '#dc2626'}'>"
            f"<div style='font-size:0.7rem;color:#64748b'>Avg next-day return</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{'#16a34a' if (avg_fwd or 0)>0 else '#dc2626'}'>"
            f"{'—' if avg_fwd is None else f'{avg_fwd:+.2f}%'}</div></div>"
            f"<div style='text-align:center;padding:10px 16px;border-radius:8px;"
            f"background:rgba(37,99,235,0.10);border:1px solid #2563eb'>"
            f"<div style='font-size:0.7rem;color:#64748b'>Median next-day return</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:#2563eb'>"
            f"{'—' if med_fwd is None else f'{med_fwd:+.2f}%'}</div></div>"
            f"<div style='text-align:center;padding:10px 16px;border-radius:8px;"
            f"background:rgba(245,158,11,0.10);border:1px solid #f59e0b'>"
            f"<div style='font-size:0.7rem;color:#64748b'>% times up next day</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:#f59e0b'>"
            f"{'—' if pct_up is None else f'{pct_up:.0f}%'}</div></div></div>",
            unsafe_allow_html=True)

        # Plot matches
        fig_pat = go.Figure()
        ref_closes = df_day["Close"].values[-lookback_n:]
        ref_norm   = ref_closes / ref_closes[0] * 100
        fig_pat.add_trace(go.Scatter(
            y=ref_norm, name="Current (last {})".format(lookback_n),
            line=dict(color="#0f172a", width=3)))

        pat_colors = ["#2563eb","#16a34a","#f59e0b","#dc2626","#8b5cf6","#0891b2","#ec4899","#14b8a6"]
        for i, m in enumerate(matches):
            idx   = m["idx"]
            slice_= df_day["Close"].values[idx: idx + lookback_n]
            norm_ = slice_ / slice_[0] * 100
            fwd   = m["fwd_return"]
            lbl   = (f"{str(m['date'])[:10]}  sim={m['similarity']:.2f}  "
                     f"fwd={fwd:+.2f}%" if fwd is not None else
                     f"{str(m['date'])[:10]}  sim={m['similarity']:.2f}")
            fig_pat.add_trace(go.Scatter(
                y=norm_, name=lbl,
                line=dict(color=pat_colors[i % len(pat_colors)], width=1.5, dash="dot"),
                opacity=0.75))

        fig_pat.update_layout(
            height=320, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Indexed (start=100)", gridcolor="#f1f5f9"),
            xaxis=dict(title="Candle offset", gridcolor="#f1f5f9"),
            legend=dict(font=dict(size=10), orientation="v",
                        yanchor="top", y=1, xanchor="left", x=1.01))
        st.plotly_chart(fig_pat, use_container_width=True)

        # Table
        tbl_data = [{
            "Date": str(m["date"])[:10],
            "Similarity": f"{m['similarity']:.3f}",
            "Next-day return": f"{m['fwd_return']:+.2f}%" if m["fwd_return"] is not None else "—",
        } for m in matches]
        st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough historical data for pattern matching — choose a longer history window.")

    # Cross-stock pattern comparison
    st.markdown("**Cross-stock: compare current pattern vs the other stock**")
    other_sym = "2513.HK" if sym == "0100.HK" else "0100.HK"
    df_other  = load_ohlcv(other_sym, period=period, interval="1d")
    if not df_other.empty and len(df_other) >= lookback_n:
        close_ref   = df_day["Close"].values[-lookback_n:]
        close_other = df_other["Close"].values[-lookback_n:]
        sim_cross   = pattern_similarity(close_ref, close_other)
        norm_ref    = close_ref   / close_ref[0]   * 100
        norm_other  = close_other / close_other[0] * 100
        fig_cross   = go.Figure()
        fig_cross.add_trace(go.Scatter(y=norm_ref,   name=sym,
                                        line=dict(color="#2563eb", width=2)))
        fig_cross.add_trace(go.Scatter(y=norm_other, name=other_sym,
                                        line=dict(color="#16a34a", width=2, dash="dot")))
        fig_cross.update_layout(
            height=220, title=f"Pattern similarity: {sim_cross:.3f}",
            margin=dict(l=0,r=0,t=30,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Indexed", gridcolor="#f1f5f9"),
            xaxis=dict(gridcolor="#f1f5f9"),
            legend=dict(font=dict(size=11)))
        st.plotly_chart(fig_cross, use_container_width=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4 — ENTRY/EXIT SIGNAL ZONE
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 4 · Entry / Exit Signal Zone")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Support & resistance · Signal scanner · Risk/reward estimator</span>",
        unsafe_allow_html=True)

    if df_day.empty or len(df_day) < 20:
        st.warning("Not enough daily data for signal analysis.")
        return

    res_levels, sup_levels = find_support_resistance(df_day)
    last_price = float(df_day["Close"].iloc[-1])
    _atr_s     = calc_atr(df_day).dropna()
    last_atr_d = float(_atr_s.iloc[-1]) if len(_atr_s) > 0 else 1.0

    # Signal scanner — guard each indicator against short series
    _rsi_s  = calc_rsi(df_day["Close"]).dropna()
    rsi_d   = float(_rsi_s.iloc[-1])   if len(_rsi_s)  > 0 else 50.0
    ml, sl, hl = calc_macd(df_day["Close"])
    _ml = ml.dropna(); _sl = sl.dropna()
    macd_d  = float(_ml.iloc[-1])      if len(_ml) > 1 else 0.0
    sig_d   = float(_sl.iloc[-1])      if len(_sl) > 1 else 0.0
    macd_p  = float(_ml.iloc[-2])      if len(_ml) > 1 else 0.0
    sig_p   = float(_sl.iloc[-2])      if len(_sl) > 1 else 0.0
    k_d, _  = calc_stoch(df_day)
    _kd     = k_d.dropna()
    stoch_d = float(_kd.iloc[-1])      if len(_kd) > 0 else 50.0
    bb_lo_d, bb_mid_d, bb_hi_d = calc_bb(df_day["Close"])
    _vwap   = calc_vwap(df_day).dropna()
    vwap_d  = float(_vwap.iloc[-1])    if len(_vwap) > 0 else last_price

    signals = []
    if rsi_d >= 70:   signals.append(("RSI Overbought",  "SELL", f"RSI={rsi_d:.1f}"))
    if rsi_d <= 30:   signals.append(("RSI Oversold",    "BUY",  f"RSI={rsi_d:.1f}"))
    if macd_d > sig_d and macd_p <= sig_p:
        signals.append(("MACD Bullish Cross", "BUY", "MACD crossed above signal"))
    if macd_d < sig_d and macd_p >= sig_p:
        signals.append(("MACD Bearish Cross", "SELL", "MACD crossed below signal"))
    _bb_hi_last = float(bb_hi_d.dropna().iloc[-1]) if len(bb_hi_d.dropna())>0 else last_price*999
    if last_price > _bb_hi_last:
        signals.append(("BB Upper Breakout",  "WATCH", f"Price above upper band"))
    _bb_lo_last = float(bb_lo_d.dropna().iloc[-1]) if len(bb_lo_d.dropna())>0 else 0
    if last_price < _bb_lo_last:
        signals.append(("BB Lower Breakdown", "WATCH", f"Price below lower band"))
    if stoch_d > 80:  signals.append(("Stoch Overbought", "SELL", f"%K={stoch_d:.1f}"))
    if stoch_d < 20:  signals.append(("Stoch Oversold",   "BUY",  f"%K={stoch_d:.1f}"))
    if last_price > vwap_d:
        signals.append(("Above VWAP", "BULLISH", f"Price {last_price-vwap_d:+.2f} above VWAP"))
    else:
        signals.append(("Below VWAP", "BEARISH", f"Price {last_price-vwap_d:+.2f} below VWAP"))

    # Signal pills
    if signals:
        cols_sig = st.columns(min(len(signals), 4))
        sig_colors = {"BUY":"#16a34a","SELL":"#dc2626","WATCH":"#f59e0b",
                      "BULLISH":"#16a34a","BEARISH":"#dc2626"}
        for i, (name, stype, detail) in enumerate(signals):
            c = sig_colors.get(stype, "#64748b")
            cols_sig[i % 4].markdown(
                f"<div style='text-align:center;padding:8px;border-radius:8px;"
                f"background:{c}1a;border:1px solid {c};margin-bottom:8px'>"
                f"<div style='font-size:0.68rem;color:#64748b'>{name}</div>"
                f"<div style='font-size:0.9rem;font-weight:700;color:{c}'>{stype}</div>"
                f"<div style='font-size:0.65rem;color:#64748b'>{detail}</div></div>",
                unsafe_allow_html=True)
    else:
        st.info("No strong signals at this time.")

    st.markdown("<br>", unsafe_allow_html=True)

    # S/R + signal chart
    fig_sr = go.Figure()
    plot_df = df_day.tail(60)
    bc_sr = ["#16a34a" if c >= o else "#dc2626"
             for c, o in zip(plot_df["Close"], plot_df["Open"])]
    fig_sr.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"],
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        name="Price"))
    for lv in res_levels:
        fig_sr.add_hline(y=lv, line_dash="dot", line_color="#dc2626", line_width=1,
                          annotation_text=f"R {lv:.0f}", annotation_position="right")
    for lv in sup_levels:
        fig_sr.add_hline(y=lv, line_dash="dot", line_color="#16a34a", line_width=1,
                          annotation_text=f"S {lv:.0f}", annotation_position="right")
    fig_sr.add_trace(go.Scatter(x=plot_df.index, y=bb_hi_d.tail(60),
                                 line=dict(color="#dc2626", width=1, dash="dot"), name="BB Upper"))
    fig_sr.add_trace(go.Scatter(x=plot_df.index, y=bb_lo_d.tail(60),
                                 line=dict(color="#16a34a", width=1, dash="dot"), name="BB Lower"))
    fig_sr.add_trace(go.Scatter(x=plot_df.index, y=calc_vwap(plot_df),
                                 line=dict(color="#f59e0b", width=1.5, dash="dash"), name="VWAP"))
    fig_sr.update_layout(
        height=400, margin=dict(l=0,r=0,t=10,b=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        yaxis=dict(title="Price (HKD)", gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9"))
    _apply_rangebreaks(fig_sr, df_day, is_intraday=False)
    st.plotly_chart(fig_sr, use_container_width=True)

    # Risk / Reward estimator
    st.markdown("**Risk / Reward Estimator**")
    rr1, rr2, rr3, rr4 = st.columns(4)
    entry_est  = rr1.number_input("Entry price", value=float(round(last_price, 1)), step=0.1, format="%.2f", key="ap_001")
    stop_est   = rr2.number_input("Stop loss",   value=float(round(last_price - last_atr_d, 1)), step=0.1, format="%.2f", key="ap_002")
    target_est = rr3.number_input("Target",      value=float(round(last_price + 2 * last_atr_d, 1)), step=0.1, format="%.2f", key="ap_003")
    shares_est = rr4.number_input("Shares",      value=100, step=10, key="ap_004")

    risk_pp   = abs(entry_est - stop_est)
    reward_pp = abs(target_est - entry_est)
    rr_ratio  = reward_pp / risk_pp if risk_pp > 0 else 0
    risk_amt  = risk_pp * shares_est
    reward_amt= reward_pp * shares_est
    rr_color  = "#16a34a" if rr_ratio >= 2 else ("#f59e0b" if rr_ratio >= 1 else "#dc2626")

    re1, re2, re3, re4 = st.columns(4)
    re1.metric("Risk per share",   f"HKD {risk_pp:.2f}")
    re2.metric("Reward per share", f"HKD {reward_pp:.2f}")
    re3.markdown(
        f"<div style='text-align:center;padding:10px;border-radius:8px;"
        f"background:{rr_color}1a;border:1px solid {rr_color}'>"
        f"<div style='font-size:0.7rem;color:#64748b'>R:R ratio</div>"
        f"<div style='font-size:1.5rem;font-weight:700;color:{rr_color}'>1 : {rr_ratio:.1f}</div>"
        f"<div style='font-size:0.7rem;color:{rr_color}'>"
        f"{'Good' if rr_ratio>=2 else 'OK' if rr_ratio>=1 else 'Poor'}</div></div>",
        unsafe_allow_html=True)
    re4.metric("Total risk / reward",
               f"−{risk_amt:,.0f} / +{reward_amt:,.0f} HKD")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5 — TRADE JOURNAL
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 5 · Trade Journal")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Log trades · Track win rate · Emotion tagging · P&L history</span>",
        unsafe_allow_html=True)

    j_tab1, j_tab2 = st.tabs(["📝 Log a Trade", "📊 Trade History & Stats"])

    with j_tab1:
        with st.form("trade_form"):
            fc1, fc2 = st.columns(2)
            j_ticker    = fc1.selectbox("Ticker", list(STOCKS.keys()),
                                         format_func=lambda x: f"{x} — {STOCKS[x]['name']}", key="ana_001")
            j_direction = fc2.selectbox("Direction", ["LONG", "SHORT"], key="ap_jdir")
            fc3, fc4, fc5 = st.columns(3)
            j_entry     = fc3.number_input("Entry price", min_value=0.0, step=0.1, format="%.2f", key="ap_006")
            j_exit      = fc4.number_input("Exit price (leave 0 if still open)",
                                             min_value=0.0, step=0.1, format="%.2f", key="ana_002")
            j_shares    = fc5.number_input("Shares", min_value=1, value=100, step=10, key="ap_jshares")
            fc6, fc7    = st.columns(2)
            j_entry_t   = fc6.text_input("Entry time", value=now_hk.strftime("%Y-%m-%d %H:%M"), key="ana_003")
            j_exit_t    = fc7.text_input("Exit time (blank if open)", value="", key="ana_004")
            j_strategy  = st.text_input("Strategy / setup",
                                         placeholder="e.g. RSI oversold + VWAP bounce", key="ana_005")
            fc8, fc9    = st.columns(2)
            j_emotion   = fc8.selectbox("Emotion at entry",
                                         ["Calm", "FOMO", "Fear", "Greedy",
                                          "Confident", "Uncertain", "Revenge"], key="ana_006")
            j_notes     = fc9.text_area("Notes", height=68,
                                         placeholder="Why this trade? What did you see?", key="ana_007")
            submitted = st.form_submit_button("💾 Save Trade")
            if submitted:
                exit_p = j_exit if j_exit > 0 else None
                exit_t = j_exit_t.strip() if j_exit_t.strip() else None
                save_trade(j_ticker, j_direction, j_entry, j_shares, j_entry_t,
                           exit_p, exit_t, j_strategy, j_notes, j_emotion)
                st.success("Trade saved to database!")
                st.rerun()

    with j_tab2:
        trades = get_trades()
        if trades.empty:
            st.info("No trades logged yet. Use the form above to start tracking.")
        else:
            # Stats
            closed  = trades[trades["exit_price"].notna()]
            wins    = closed[closed["outcome"] == "WIN"]
            losses  = closed[closed["outcome"] == "LOSS"]
            wr      = len(wins) / len(closed) * 100 if len(closed) else 0
            avg_win = wins["pnl"].mean() if len(wins) else 0
            avg_los = losses["pnl"].mean() if len(losses) else 0
            total_pnl_j = closed["pnl"].sum() if len(closed) else 0
            profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) \
                            if len(losses) and losses["pnl"].sum() != 0 else None

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Total trades", len(trades))
            s2.metric("Win rate", f"{wr:.1f}%")
            pnl_d = f"{'+'if total_pnl_j>=0 else ''}HKD {total_pnl_j:,.0f}"
            s3.metric("Total P&L", pnl_d)
            s4.metric("Avg win / loss",
                       f"+{avg_win:,.0f} / {avg_los:,.0f}")
            s5.metric("Profit factor",
                       f"{profit_factor:.2f}" if profit_factor else "—")

            # Cumulative P&L chart
            if len(closed):
                cum_pnl = closed.sort_values("entry_time")["pnl"].cumsum()
                fig_j = go.Figure(go.Scatter(
                    x=list(range(len(cum_pnl))), y=cum_pnl.values,
                    mode="lines+markers",
                    line=dict(color="#16a34a" if total_pnl_j >= 0 else "#dc2626", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(22,163,74,0.08)" if total_pnl_j >= 0 else "rgba(220,38,38,0.08)",
                    marker=dict(size=6)))
                fig_j.add_hline(y=0, line_color="#e2e8f0")
                fig_j.update_layout(
                    height=200, margin=dict(l=0,r=0,t=10,b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False,
                    yaxis=dict(title="Cumulative P&L (HKD)", gridcolor="#f1f5f9"),
                    xaxis=dict(title="Trade #", gridcolor="#f1f5f9"))
                st.plotly_chart(fig_j, use_container_width=True)

            # Emotion analysis
            if "emotion" in trades.columns and trades["emotion"].notna().any():
                st.markdown("**Emotion → P&L breakdown**")
                em_df = trades[trades["pnl"].notna()].groupby("emotion")["pnl"].agg(
                    ["mean","sum","count"]).reset_index()
                em_df.columns = ["Emotion","Avg P&L","Total P&L","# Trades"]
                em_df = em_df.round(2)
                st.dataframe(em_df.style.format({
                    "Avg P&L":"{:+,.2f}","Total P&L":"{:+,.2f}"
                }).applymap(
                    lambda v: "color:#16a34a" if isinstance(v,float) and v>0
                              else "color:#dc2626" if isinstance(v,float) and v<0 else "",
                    subset=["Avg P&L","Total P&L"]),
                    use_container_width=True, hide_index=True)

            # Full trade table
            st.markdown("**All trades**")
            disp = trades[["id","ticker","direction","entry_price","exit_price",
                           "shares","entry_time","exit_time","pnl","pnl_pct",
                           "strategy","emotion","outcome"]].copy()
            disp = disp.sort_values("entry_time", ascending=False).reset_index(drop=True)
            st.dataframe(disp.style.format({
                "entry_price":"{:.2f}","exit_price":"{:.2f}",
                "pnl":"{:+,.2f}","pnl_pct":"{:+.2f}%"
            }, na_rep="—").applymap(
                lambda v: "color:#16a34a" if isinstance(v,float) and v>0
                          else "color:#dc2626" if isinstance(v,float) and v<0 else "",
                subset=["pnl","pnl_pct"]),
                use_container_width=True)

            # Delete trade
            with st.expander("🗑️ Delete a trade"):
                del_id = st.number_input("Trade ID to delete", min_value=1, step=1, key="ap_delid")
                if st.button("Delete", type="secondary", key="ap_btn_delete"):
                    delete_trade(int(del_id))
                    st.success(f"Trade #{del_id} deleted.")
                    st.rerun()

    st.markdown(
        "<br><span style='color:#94a3b8;font-size:0.74rem'>"
        "All signals are informational only · Not financial advice · "
        "Data via yfinance</span>",
        unsafe_allow_html=True)
