"""
dashboard.py — Portfolio Dashboard
Streamlit + yfinance + SQLite
Tracks stocks, forex, commodities in one place.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import time, pytz

from db_manager import (
    init_db, upsert_daily, get_daily, get_daily_stats,
    upsert_intraday, upsert_position, get_portfolio,
    save_capital, get_latest_capital,
    get_portfolio_full, get_raw_sql,
)
import analysis_page
import volume_scanner
import portfolio_manager
import strategy_page
import cycle_ml
import portfolio_study
import daily_strategy
import money_flow
import risk_tools

# ── INIT ──────────────────────────────────────────────────────────────
init_db()
from portfolio_manager import init_monitor_tables as _init_mon
_init_mon()

st.set_page_config(
    page_title="Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container{padding-top:1.2rem;padding-bottom:1rem}
  .mbox{background:#f4f6fb;border-radius:10px;padding:13px 16px;border:1px solid #e2e8f0}
  .mlabel{font-size:0.7rem;color:#64748b;margin-bottom:2px}
  .mval{font-size:1.2rem;font-weight:600;color:#0f172a}
  .msub{font-size:0.73rem;margin-top:2px}
  .pos{color:#16a34a}.neg{color:#dc2626}
  .dbbadge{background:#0f172a;color:#38bdf8;font-size:0.65rem;
           padding:2px 7px;border-radius:5px;display:inline-block}
  [data-testid="stSidebar"]{background:#0f172a}
  [data-testid="stSidebar"] *{color:#cbd5e1 !important}
  [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3{color:#f1f5f9 !important}
  [data-testid="stSidebar"] label{color:#94a3b8 !important;font-size:0.77rem !important}
  [data-testid="stSidebar"] .stButton>button{
    background:#1e40af;color:#fff !important;border:none;
    border-radius:8px;width:100%;margin-top:4px}
  [data-testid="stSidebar"] .stButton>button:hover{background:#2563eb}
</style>""", unsafe_allow_html=True)

HK_TZ = pytz.timezone("Asia/Hong_Kong")

# ── TICKER HELPERS ────────────────────────────────────────────────────
def _variants(ticker):
    v=[ticker]; code=ticker.replace(".HK","")
    if code.isdigit():
        v.append(str(int(code))+".HK"); v.append(code.zfill(4)+".HK")
    return list(dict.fromkeys(v))

# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Portfolio")
    st.markdown("---")
    st.markdown("### ⚙️ Chart Settings")
    chart_iv = st.selectbox("Intraday interval",
                             ["1m","2m","5m","15m","30m","60m"],
                             index=2, key="hd_iv")
    intra_p  = st.selectbox("Intraday history",
                             ["1d","5d","10d","14d","1mo"],
                             index=2, key="hd_ip")
    daily_p  = st.selectbox("Daily history",
                             ["1mo","3mo","6mo","1y"],
                             index=1, key="hd_dp")
    if st.button("🔄 Refresh", key="hd_refresh"):
        st.cache_data.clear(); st.rerun()
    st.markdown("---")
    st.markdown("### 📌 Navigation")
    page = st.radio("Navigate to", [
        "📊  Summary",
        "📅  Daily",
        "💰  Flow",
        "📋  Portfolio",
        "🔬  Analysis",
        "🔍  Scanner",
        "📐  Study",
        "🛡  Risk",
    ], key="nav_page", label_visibility="collapsed")

now_hk = datetime.now(HK_TZ)

# ── DATA FETCHERS ─────────────────────────────────────────────────────
EMPTY_Q = {k:None for k in
           ["price","prev_close","open","day_high","day_low",
            "mkt_cap","52w_high","52w_low"]}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_q(ticker):
    for t in _variants(ticker):
        try:
            info=yf.Ticker(t).fast_info
            q={k:getattr(info,v,None) for k,v in {
                "price":"last_price","prev_close":"previous_close","open":"open",
                "day_high":"day_high","day_low":"day_low",
                "mkt_cap":"market_cap","52w_high":"year_high","52w_low":"year_low"
            }.items()}
            if q.get("price"): return q
        except Exception: pass
    return EMPTY_Q.copy()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_intra(ticker, interval, period="5d"):
    safe=period if interval!="1m" else ("7d" if period not in ["1d","5d","7d"] else period)
    for t in _variants(ticker):
        try:
            df=yf.Ticker(t).history(period=safe,interval=interval,auto_adjust=True)
            if not df.empty:
                df.index=pd.to_datetime(df.index)
                if df.index.tzinfo is None: df.index=df.index.tz_localize("UTC")
                df.index=df.index.tz_convert(HK_TZ)
                upsert_intraday(ticker,df,interval)
                return df
        except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_day(ticker, period):
    for t in _variants(ticker):
        try:
            df=yf.Ticker(t).history(period=period,interval="1d",auto_adjust=True)
            if len(df)>=5:
                upsert_daily(ticker,df); return df
        except Exception: pass
    return pd.DataFrame()

# ── HELPERS ───────────────────────────────────────────────────────────
def pc(v):  return "pos" if (v or 0)>=0 else "neg"
def fc(v):  return "#16a34a" if (v or 0)>=0 else "#dc2626"
def fh(v):  return f"HKD {v:,.2f}" if v is not None else "—"
def fp(v):
    if v is None: return "—"
    return f"{'+'if v>=0 else ''}HKD {v:,.2f}"
def fpct(v):
    if v is None: return "—"
    return f"{'+'if v>=0 else ''}{v:.2f}%"

def mcard(col, label, main, sub="", sc=""):
    sh=f"<div class='msub {sc}'>{sub}</div>" if sub else ""
    col.markdown(f"<div class='mbox'><div class='mlabel'>{label}</div>"
                 f"<div class='mval'>{main}</div>{sh}</div>",
                 unsafe_allow_html=True)

def rangebreaks(is_intra=False):
    rb=[dict(bounds=["sat","mon"])]
    if is_intra:
        rb+=[dict(bounds=[16,9.5],pattern="hour"),
             dict(bounds=[11.5,13],pattern="hour")]
    return rb

def candle_fig(df, height=360, ipo=None, cost=None, is_intra=False):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
                      row_heights=[0.72,0.28],vertical_spacing=0.03)
    bc=["#16a34a" if c>=o else "#dc2626"
        for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Candlestick(
        x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],
        increasing_line_color="#16a34a",decreasing_line_color="#dc2626"),row=1,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],
        marker_color=bc,opacity=0.7),row=2,col=1)
    if ipo:
        fig.add_hline(y=ipo,line_dash="dot",line_color="#f59e0b",line_width=1.5,
                      annotation_text=f"IPO {ipo}",annotation_position="right",row=1,col=1)
    if cost and cost>0:
        fig.add_hline(y=cost,line_dash="dash",line_color="#8b5cf6",line_width=1.5,
                      annotation_text=f"Avg {cost:.2f}",annotation_position="right",row=1,col=1)
    rb=rangebreaks(is_intra)
    fig.update_layout(height=height,margin=dict(l=0,r=0,t=10,b=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white",paper_bgcolor="white",showlegend=False,
        yaxis=dict(gridcolor="#f1f5f9"),yaxis2=dict(gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9",rangebreaks=rb),
        xaxis2=dict(gridcolor="#f1f5f9",rangebreaks=rb))
    return fig

def sparkline(df, color="#2563eb", height=60):
    if df is None or len(df)<2: return None
    fig=go.Figure(go.Scatter(
        x=df.index,y=df["Close"],mode="lines",
        line=dict(color=color,width=1.5),fill="tozeroy",
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))+(0.1,)}"))
    fig.update_layout(height=height,margin=dict(l=0,r=0,t=0,b=0),
        showlegend=False,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),yaxis=dict(visible=False))
    return fig

# ═════════════════════════════════════════════════════════════════════
# SUMMARY PAGE
# ═════════════════════════════════════════════════════════════════════
def render_summary():
    st.markdown("## 📊 Summary")
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"{now_hk.strftime('%Y-%m-%d  %H:%M HKT')} · "
        f"All positions · portfolio study · recommendations</span>",
        unsafe_allow_html=True)
    st.markdown("---")

    # ── Load positions ────────────────────────────────────────────────
    stock_pos = get_portfolio_full()
    stock_pos = stock_pos[stock_pos["status"]=="OPEN"] if not stock_pos.empty else pd.DataFrame()
    from portfolio_manager import get_monitor_pos
    monitor_pos = get_monitor_pos()
    monitor_pos = monitor_pos[monitor_pos["status"]=="OPEN"] if not monitor_pos.empty else pd.DataFrame()
    capital = get_latest_capital()

    all_rows = []
    if not stock_pos.empty:
        with st.spinner("Fetching prices…"):
            for _, r in stock_pos.iterrows():
                q = fetch_q(r["ticker"])
                price=q.get("price"); prev=q.get("prev_close")
                qty=float(r["shares"] or 0); cost_=float(r["avg_cost"] or 0)
                val=qty*price if price and qty>0 else None
                cb=qty*cost_; pnl=val-cb if val is not None else None
                dc=(price-prev) if price and prev else None
                dp=dc/prev*100 if dc and prev else None
                all_rows.append({
                    "type":"Stock","ticker":r["ticker"],"name":r.get("name",r["ticker"]),
                    "price":price,"qty":qty,"cost":cb,"val":val,"pnl":pnl,
                    "day_pnl":qty*dc if dc and qty>0 else None,"day_pct":dp,
                    "target":r.get("target_price"),"stop":r.get("stop_price"),"color":"#2563eb",
                })

    if not monitor_pos.empty:
        with st.spinner("Fetching forex/commodity prices…"):
            for _, r in monitor_pos.iterrows():
                try:
                    info=yf.Ticker(r["ticker"]).fast_info
                    price=getattr(info,"last_price",None); prev=getattr(info,"previous_close",None)
                except Exception: price=prev=None
                qty=float(r["quantity"] or 0); cost_=float(r["avg_cost"] or 0)
                val=qty*price if price and qty>0 else None
                cb=qty*cost_; pnl=val-cb if val is not None else None
                dc=(price-prev) if price and prev else None
                dp=dc/prev*100 if dc and prev else None
                type_=r.get("asset_type","Forex")
                all_rows.append({
                    "type":type_,"ticker":r["ticker"],"name":r.get("name",r["ticker"]),
                    "price":price,"qty":qty,"cost":cb,"val":val,"pnl":pnl,
                    "day_pnl":qty*dc if dc and qty>0 else None,"day_pct":dp,
                    "target":r.get("target"),"stop":r.get("stop"),
                    "color":"#8b5cf6" if type_=="Forex" else "#f59e0b",
                })

    if not all_rows:
        st.info("No open positions yet — use **📋 Portfolio** to add positions.")
        return

    total_cost=sum(r["cost"] for r in all_rows if r["cost"])
    total_val =sum(r["val"]  for r in all_rows if r["val"])
    total_pnl =sum(r["pnl"]  for r in all_rows if r["pnl"] is not None)
    total_dpnl=sum(r["day_pnl"] for r in all_rows if r["day_pnl"] is not None)
    total_pct =total_pnl/total_cost*100 if total_cost>0 else 0
    cash      =max(capital-total_cost,0)

    # ── Top metrics ───────────────────────────────────────────────────
    t1,t2,t3,t4,t5=st.columns(5)
    mcard(t1,"Total capital",  fh(capital))
    mcard(t2,"Invested",       fh(total_cost), f"Cash: {fh(cash)}")
    mcard(t3,"Market value",   fh(total_val))
    mcard(t4,"Total P&L",      fp(total_pnl),  fpct(total_pct), pc(total_pnl))
    mcard(t5,"Today's P&L",   fp(total_dpnl), "", pc(total_dpnl))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── P&L table ─────────────────────────────────────────────────────
    st.markdown("#### Positions")
    pnl_rows=[r for r in all_rows if r["cost"]>0]
    if pnl_rows:
        def fmt_price(p):
            if not p: return "—"
            return f"{p:,.2f}" if p>100 else f"{p:,.4f}"

        tbl_data=[]
        for r in pnl_rows:
            pnl=r["pnl"]; cb=r["cost"]; val=r["val"]
            pnlp=pnl/cb*100 if (pnl is not None and cb>0) else None
            dp=r["day_pct"]
            pos_pct=cb/capital*100 if capital>0 else 0
            tgt=r.get("target"); stp=r.get("stop"); price=r["price"]
            tgt_d=(tgt-price)/price*100 if tgt and price else None
            stp_d=(stp-price)/price*100 if stp and price else None
            tbl_data.append({
                "Type":    r["type"],
                "Name":    r["name"],
                "Ticker":  r["ticker"],
                "Price":   fmt_price(price),
                "Day %":   fpct(dp),
                "P&L":     fp(pnl) if pnl is not None else "—",
                "Return %":fpct(pnlp),
                "Cost":    f"{cb:,.0f}",
                "Alloc %": f"{pos_pct:.1f}%",
                "Target":  f"{tgt:,.4f} ({tgt_d:+.1f}%)" if tgt and tgt_d is not None else "—",
                "Stop":    f"{stp:,.4f} ({stp_d:+.1f}%)" if stp and stp_d is not None else "—",
            })

        df_tbl=pd.DataFrame(tbl_data)

        # colour P&L and Return % columns
        def style_table(df):
            styles=pd.DataFrame("",index=df.index,columns=df.columns)
            for col in ["P&L","Return %","Day %"]:
                if col in df.columns:
                    for i,v in enumerate(df[col]):
                        if isinstance(v,str) and v.startswith("+"):
                            styles.iloc[i][col]="color:#16a34a;font-weight:600"
                        elif isinstance(v,str) and v.startswith("-"):
                            styles.iloc[i][col]="color:#dc2626;font-weight:600"
            return styles

        st.dataframe(
            df_tbl.style.apply(style_table,axis=None),
            use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # PORTFOLIO STUDY
    # ══════════════════════════════════════════════════════════════════
    st.markdown("#### 📐 Portfolio Study — Efficiency & Risk")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "How efficiently is each position using your capital? "
        "Metrics tuned for range-trading: you profit from volatility, "
        "so high volatility is only good if it's captured as return.</span>",
        unsafe_allow_html=True)

    if st.button("🔬 Run portfolio study", key="hd_study"):
        st.session_state["run_study"]=True

    if not st.session_state.get("run_study"):
        st.info("Click **Run portfolio study** to analyse all positions.")
    else:
        study_rows=[]
        with st.spinner("Fetching 3mo history for all positions…"):
            for r in pnl_rows:
                ticker=r["ticker"]
                try:
                    df_h=fetch_day(ticker,"3mo")
                    if df_h is None or len(df_h)<10:
                        study_rows.append(_empty_study(r)); continue
                    rets=df_h["Close"].pct_change().dropna()
                    ranges=df_h["High"]-df_h["Low"]
                    vol_daily=float(rets.std())               # daily volatility
                    ann_vol  =vol_daily*np.sqrt(252)*100      # annualised %
                    avg_range=float(ranges.mean())            # avg daily HKD range
                    avg_range_pct=avg_range/float(df_h["Close"].mean())*100

                    # Max drawdown from entry (or from 3mo high)
                    prices_=df_h["Close"].values
                    peak_=np.maximum.accumulate(prices_)
                    dd_series=(prices_-peak_)/peak_*100
                    max_dd=float(dd_series.min())

                    # Return metrics
                    pnl_=r["pnl"] or 0; cb=r["cost"] or 1
                    ret_pct=pnl_/cb*100

                    # Capital efficiency score:
                    # How much P&L did you extract per HKD of capital per unit of volatility?
                    # = (return% / ann_vol) — higher = better use of risky capital
                    sharpe_like=ret_pct/ann_vol if ann_vol>0 else 0

                    # Range capture ratio:
                    # Your actual return % vs the total range% the stock offered
                    total_range_offered=float(ranges.sum()/df_h["Close"].iloc[0]*100)
                    range_capture=ret_pct/total_range_offered*100 if total_range_offered>0 else 0

                    # Win rate (% of days position gained)
                    win_rate=float((rets>0).mean()*100)

                    # R:R quality
                    tgt=r.get("target"); stp=r.get("stop"); price=r["price"]
                    if tgt and stp and price and price>stp:
                        rr=(tgt-price)/max(price-stp,1e-9)
                    else:
                        rr=None

                    # Capital weight
                    alloc=cb/capital*100 if capital>0 else 0

                    # ── TREND TRAP DETECTION ──────────────────────────
                    # Is this stock trending DOWN (not swinging)?
                    # Uses: slope of EMA, choppiness, consecutive down days,
                    #       price vs 20d/50d MA, lower highs pattern

                    closes_=df_h["Close"]
                    # 1. Linear regression slope (negative = downtrend)
                    x_=np.arange(len(closes_))
                    slope_=float(np.polyfit(x_, closes_.values, 1)[0])
                    slope_pct=slope_/float(closes_.mean())*100  # % per day

                    # 2. Price vs moving averages
                    ma20_=closes_.rolling(20).mean().iloc[-1] if len(closes_)>=20 else closes_.mean()
                    ma50_=closes_.rolling(50).mean().iloc[-1] if len(closes_)>=50 else closes_.mean()
                    price_now_=float(closes_.iloc[-1])
                    below_ma20=price_now_<float(ma20_)
                    below_ma50=price_now_<float(ma50_) if len(closes_)>=50 else None

                    # 3. Consecutive down days (last 5)
                    last5_=closes_.pct_change().tail(5)
                    consec_down=int((last5_<0).sum())

                    # 4. Choppiness — low = trending (bad for you), high = oscillating (good)
                    chop_now=_choppiness(df_h) or 50

                    # 5. Recent range shrinkage — is the swing getting smaller?
                    recent_ranges=ranges.tail(10).mean()
                    older_ranges =ranges.iloc[-30:-10].mean() if len(ranges)>=30 else ranges.mean()
                    range_shrink =(recent_ranges-older_ranges)/older_ranges*100 if older_ranges>0 else 0

                    # 6. Lower highs: count how many of last 5 highs are below previous high
                    highs_=df_h["High"].tail(10).values
                    lower_highs=sum(1 for i in range(1,len(highs_)) if highs_[i]<highs_[i-1])

                    # Trend trap score 0-100 (higher = more danger)
                    trend_score=0
                    if slope_pct < -0.3:   trend_score += 25   # strong down slope
                    elif slope_pct < -0.1: trend_score += 12
                    if below_ma20:         trend_score += 20
                    if below_ma50 is True: trend_score += 15
                    if consec_down >= 4:   trend_score += 20
                    elif consec_down >= 3: trend_score += 10
                    if chop_now < 38:      trend_score += 15   # trending not choppy
                    elif chop_now < 45:    trend_score += 7
                    if range_shrink < -20: trend_score += 5    # swings shrinking
                    if lower_highs >= 7:   trend_score += 10
                    trend_score=min(int(trend_score),100)

                    # Trend label
                    if trend_score>=70:    trend_label="🔴 TREND TRAP"
                    elif trend_score>=45:  trend_label="🟡 WEAK SWING"
                    elif trend_score>=20:  trend_label="🟢 OK SWING"
                    else:                  trend_label="✅ ACTIVE SWING"

                    study_rows.append({
                        "name":       r["name"],
                        "ticker":     ticker,
                        "type":       r["type"],
                        "ret_pct":    round(ret_pct,2),
                        "ann_vol":    round(ann_vol,1),
                        "sharpe":     round(sharpe_like,3),
                        "avg_range":  round(avg_range,1),
                        "avg_range_pct": round(avg_range_pct,2),
                        "max_dd":     round(max_dd,1),
                        "win_rate":   round(win_rate,1),
                        "range_capture": round(range_capture,2),
                        "alloc":      round(alloc,1),
                        "rr":         round(rr,2) if rr else None,
                        "color":      r["color"],
                        "cost":       cb,
                        # Trend trap fields
                        "trend_score":  trend_score,
                        "trend_label":  trend_label,
                        "slope_pct":    round(slope_pct,3),
                        "below_ma20":   below_ma20,
                        "below_ma50":   below_ma50,
                        "consec_down":  consec_down,
                        "chop_now":     round(chop_now,1),
                        "range_shrink": round(range_shrink,1),
                        "lower_highs":  lower_highs,
                    })
                except Exception:
                    study_rows.append(_empty_study(r))

        if not study_rows:
            st.warning("Could not compute study metrics.")
        else:
            # ── Efficiency table ──────────────────────────────────────
            st.markdown("**Capital efficiency metrics**")
            eff_data=[]
            for s in study_rows:
                sharpe_flag = "⚠️" if s["sharpe"]<0 else ("✅" if s["sharpe"]>0.5 else "")
                dd_flag     = "⚠️" if s["max_dd"]<-20 else ""
                def _f(v, fmt, suffix="", fallback="—"):
                    try: return format(v, fmt) + suffix if v is not None else fallback
                    except: return fallback
                eff_data.append({
                    "Name":          s["name"],
                    "Type":          s["type"],
                    "Return %":      _f(s['ret_pct'],  '+.2f', '%'),
                    "Ann. Vol %":    _f(s['ann_vol'],  '.1f',  '%'),
                    "Sharpe-like":   (_f(s['sharpe'], '+.3f') + f" {sharpe_flag}").strip(),
                    "Avg range HKD": _f(s['avg_range'],     '.1f'),
                    "Avg range %":   _f(s['avg_range_pct'], '.2f', '%'),
                    "Max drawdown":  (_f(s['max_dd'], '.1f', '%') + f" {dd_flag}").strip(),
                    "Win rate":      _f(s['win_rate'],      '.0f', '%'),
                    "Range capture": _f(s['range_capture'], '+.2f', '%'),
                    "Alloc %":       _f(s['alloc'],         '.1f', '%'),
                    "R:R":           f"1:{s['rr']:.1f}" if s["rr"] else "—",
                })

            eff_df=pd.DataFrame(eff_data)

            def style_eff(df):
                styles=pd.DataFrame("",index=df.index,columns=df.columns)
                for i,row in df.iterrows():
                    for col in ["Return %","Sharpe-like","Range capture"]:
                        v=str(row[col])
                        if v.startswith("+") or (v[0].isdigit() and "⚠️" not in v):
                            styles.at[i,col]="color:#16a34a;font-weight:600"
                        elif v.startswith("-") or "⚠️" in v:
                            styles.at[i,col]="color:#dc2626;font-weight:600"
                return styles

            st.dataframe(eff_df.style.apply(style_eff,axis=None),
                         use_container_width=True,hide_index=True)

            # ── Key metric explanations ───────────────────────────────
            with st.expander("📖 What do these metrics mean for your trading style?"):
                st.markdown("""
**Sharpe-like** = Return % ÷ Annualised volatility. For a range trader this is the key number —
it measures how much profit you extracted per unit of risk the position carried.
- Positive + high = working well, you're capturing the volatility as profit
- Near zero = you held a volatile position but barely profited — wasted capital
- Negative = losing money on a volatile position — worst outcome for a swing trader

**Range capture %** = Your actual return ÷ total range the stock offered × 100.
If a stock moved 500 HKD total over 3 months and you made 50 HKD, range capture = 10%.
Low range capture means the stock swung a lot but you didn't capture it.

**Avg range %** = Average daily High−Low as % of price — how big the daily swings are.
High range + low return = you're in a volatile stock but not trading it effectively.

**Max drawdown** = Worst drop from a peak. >20% drawdown on a position you meant to swing trade
means you're holding a losing swing trade as if it were an investment.
                """)

            # ── TREND TRAP TABLE ─────────────────────────────────
            st.markdown("---")
            st.markdown("**🚨 Trend Trap Monitor — your biggest risk**")
            st.markdown(
                "<span style='color:#64748b;font-size:0.8rem'>"
                "Detects positions that are trending DOWN instead of swinging. "
                "A trend trap kills your range-trading strategy — "
                "there is no upper swing to capture and you keep losing.</span>",
                unsafe_allow_html=True)

            trap_data=[]
            for s in study_rows:
                if "trend_score" not in s: continue
                sc=s["trend_score"]
                border_c="#dc2626" if sc>=70 else "#f59e0b" if sc>=45 else "#16a34a"
                trap_data.append({
                    "Status":       s.get("trend_label","—"),
                    "Name":         s["name"],
                    "Ticker":       s["ticker"],
                    "Trend score":  f"{sc}/100",
                    "Daily slope":  f"{s['slope_pct']:+.3f}%/day",
                    "Below MA20":   "⚠️ Yes" if s.get("below_ma20") else "✅ No",
                    "Below MA50":   "⚠️ Yes" if s.get("below_ma50") else ("✅ No" if s.get("below_ma50") is False else "N/A"),
                    "Consec down":  f"{s.get('consec_down',0)}/5 days",
                    "Choppiness":   f"{s.get('chop_now',0):.0f} {'✅' if s.get('chop_now',50)>61 else '⚠️' if s.get('chop_now',50)<45 else ''}",
                    "Range shrink": f"{s.get('range_shrink',0):+.0f}%",
                    "Lower highs":  f"{s.get('lower_highs',0)}/9",
                })

            if trap_data:
                trap_df=pd.DataFrame(trap_data)
                def style_trap(df):
                    styles=pd.DataFrame("",index=df.index,columns=df.columns)
                    for i,row in df.iterrows():
                        status=str(row["Status"])
                        if "TREND TRAP" in status:
                            for c in df.columns:
                                styles.at[i,c]="background:rgba(220,38,38,0.06)"
                            styles.at[i,"Status"]="color:#dc2626;font-weight:700"
                        elif "WEAK" in status:
                            styles.at[i,"Status"]="color:#f59e0b;font-weight:600"
                        elif "ACTIVE" in status:
                            styles.at[i,"Status"]="color:#16a34a;font-weight:600"
                        for c in ["Below MA20","Below MA50"]:
                            if "⚠️" in str(row.get(c,"")):
                                styles.at[i,c]="color:#dc2626"
                            elif "✅" in str(row.get(c,"")):
                                styles.at[i,c]="color:#16a34a"
                    return styles

                st.dataframe(trap_df.style.apply(style_trap,axis=None),
                             use_container_width=True,hide_index=True)

                # Danger alerts for TREND TRAP positions
                traps=[s for s in study_rows if s.get("trend_score",0)>=70]
                for s in traps:
                    reasons=[]
                    if s.get("slope_pct",0)<-0.3: reasons.append(f"price falling {abs(s['slope_pct']):.2f}%/day on average")
                    if s.get("below_ma20"): reasons.append("below 20-day MA")
                    if s.get("below_ma50"): reasons.append("below 50-day MA")
                    if s.get("consec_down",0)>=4: reasons.append(f"{s['consec_down']} consecutive down days")
                    if s.get("chop_now",50)<45: reasons.append(f"choppiness {s['chop_now']:.0f} — trending not swinging")
                    if s.get("range_shrink",0)<-20: reasons.append(f"daily swings shrinking {s['range_shrink']:.0f}%")
                    st.markdown(
                        f"<div style='border:2px solid #dc2626;border-radius:10px;"
                        f"padding:14px 18px;margin:8px 0;background:rgba(220,38,38,0.03)'>"
                        f"<div style='font-size:1rem;font-weight:700;color:#dc2626'>"
                        f"🔴 TREND TRAP: {s['name']} ({s['ticker']})</div>"
                        f"<div style='font-size:0.82rem;color:#475569;margin-top:6px'>"
                        f"This position is trending down, not swinging. "
                        f"Your range-trading strategy cannot work here.</div>"
                        f"<div style='font-size:0.8rem;color:#dc2626;margin-top:6px'>"
                        f"Evidence: {' · '.join(reasons)}</div>"
                        f"<div style='font-size:0.8rem;color:#475569;margin-top:8px'>"
                        f"<b>Action:</b> Cut the position or tighten stop immediately. "
                        f"Wait for choppiness > 62 and price reclaiming MA20 before re-entering.</div>"
                        f"</div>",
                        unsafe_allow_html=True)

                weaks=[s for s in study_rows if 45<=s.get("trend_score",0)<70]
                for s in weaks:
                    st.markdown(
                        f"<div style='border:1px solid #f59e0b;border-radius:8px;"
                        f"padding:10px 14px;margin:6px 0;background:rgba(245,158,11,0.03)'>"
                        f"<div style='font-weight:600;color:#f59e0b'>"
                        f"🟡 WEAK SWING: {s['name']} — score {s['trend_score']}/100</div>"
                        f"<div style='font-size:0.78rem;color:#475569;margin-top:4px'>"
                        f"Swing is weakening. Monitor closely. "
                        f"Choppiness: {s.get('chop_now',0):.0f} · "
                        f"Consec down: {s.get('consec_down',0)}/5 · "
                        f"Slope: {s.get('slope_pct',0):+.3f}%/day</div></div>",
                        unsafe_allow_html=True)

            # ── Efficiency scatter ────────────────────────────────────
            st.markdown("**Risk vs Return — top-right = most efficient**")
            fig_eff=go.Figure()
            for s in study_rows:
                if s["ann_vol"] is None: continue
                fig_eff.add_trace(go.Scatter(
                    x=[s["ann_vol"]], y=[s["ret_pct"]],
                    mode="markers+text",
                    text=[s["name"]],
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(
                        size=max(8,min(s["alloc"]*2,40)),
                        color=s["color"], opacity=0.85,
                        line=dict(color="white",width=1.5)),
                    hovertemplate=(
                        f"<b>{s['name']}</b><br>"
                        f"Return: {s['ret_pct']:+.2f}%<br>"
                        f"Volatility: {s['ann_vol']:.1f}%<br>"
                        f"Sharpe-like: {s['sharpe']:+.3f}<br>"
                        f"Alloc: {s['alloc']:.1f}%<extra></extra>"),
                    showlegend=False))
            fig_eff.add_hline(y=0,line_color="#e2e8f0",line_width=1)
            fig_eff.update_layout(
                height=320,margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white",paper_bgcolor="white",
                xaxis=dict(title="Annualised volatility %",gridcolor="#f1f5f9"),
                yaxis=dict(title="Return %",gridcolor="#f1f5f9"))
            st.markdown(
                "<span style='font-size:0.75rem;color:#64748b'>"
                "Circle size = allocation weight · "
                "Top-right = high return for the risk taken (good) · "
                "Bottom-right = high risk, low return (inefficient)</span>",
                unsafe_allow_html=True)
            st.plotly_chart(fig_eff,use_container_width=True)

            # ── Asset class concentration ─────────────────────────────
            st.markdown("---")
            st.markdown("#### 🏗 Portfolio Structure & Recommendations")

            type_alloc={}
            for s in study_rows:
                type_alloc[s["type"]]=type_alloc.get(s["type"],0)+s["alloc"]
            cash_pct=cash/capital*100 if capital>0 else 0
            if cash_pct>0: type_alloc["Cash"]=round(cash_pct,1)

            rc1,rc2=st.columns([1,2])
            with rc1:
                fig_cls=go.Figure(go.Pie(
                    labels=list(type_alloc.keys()),
                    values=list(type_alloc.values()),
                    hole=0.45,textinfo="label+percent",
                    marker=dict(colors=["#2563eb","#8b5cf6","#f59e0b","#94a3b8"])))
                fig_cls.update_layout(height=220,margin=dict(l=0,r=0,t=10,b=0),
                    showlegend=False,paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cls,use_container_width=True)

            with rc2:
                flags=[]

                # Flag: poor Sharpe
                poor=[s for s in study_rows if s["sharpe"]<0 and s["alloc"]>5]
                if poor:
                    names_=", ".join(s["name"] for s in poor)
                    flags.append(("⚠️ Poor risk/reward",
                        f"**{names_}** — negative Sharpe-like score. "
                        "You're holding volatile positions that are losing money. "
                        "Consider cutting or tightening stop loss.",
                        "#dc2626"))

                # Flag: concentration
                dominant=[t for t,a in type_alloc.items() if a>60 and t!="Cash"]
                if dominant:
                    flags.append(("⚠️ Concentration risk",
                        f"**{dominant[0]}** takes {type_alloc[dominant[0]]:.0f}% of capital. "
                        "High concentration in one asset class amplifies drawdowns. "
                        "Consider spreading across uncorrelated assets.",
                        "#f59e0b"))

                # Flag: too much cash
                if cash_pct>50:
                    flags.append(("💤 High cash ratio",
                        f"{cash_pct:.0f}% of capital is idle. "
                        "For a range trader this means missed opportunities. "
                        "Deploy into swing candidates from the Scanner.",
                        "#94a3b8"))

                # Flag: big drawdown
                big_dd=[s for s in study_rows if s["max_dd"]<-25]
                if big_dd:
                    names_=", ".join(s["name"] for s in big_dd)
                    flags.append(("🩸 Large drawdown",
                        f"**{names_}** — max drawdown > 25%. "
                        "A swing trade should not draw down this much. "
                        "Review whether you are sizing correctly.",
                        "#dc2626"))

                # Flag: trend traps (most critical for range trader)
                traps_=[s for s in study_rows if s.get("trend_score",0)>=70 and s["alloc"]>3]
                if traps_:
                    names_=", ".join(s["name"] for s in traps_)
                    flags.append(("🔴 TREND TRAP — critical",
                        f"**{names_}** — trending DOWN, not swinging. "
                        "This is your biggest risk as a range trader. "
                        "A downtrending position has no upper swing to capture — "
                        "cut it or tighten stop before it drains more capital.",
                        "#dc2626"))

                # Flag: high vol low range capture
                bad_cap=[s for s in study_rows if s["avg_range_pct"]>3 and s["range_capture"]<5]
                if bad_cap:
                    names_=", ".join(s["name"] for s in bad_cap)
                    flags.append(("📉 Low range capture",
                        f"**{names_}** — big daily swings but low return capture. "
                        "The stock is moving but you're not trading it. "
                        "Trade it more actively or replace with a less volatile position.",
                        "#f59e0b"))

                if not flags:
                    st.success("✅ Portfolio looks balanced — no major structural issues detected.")
                for icon,msg,col_ in flags:
                    st.markdown(
                        f"<div style='border-left:3px solid {col_};"
                        f"background:rgba(0,0,0,0.02);padding:10px 14px;"
                        f"border-radius:0 8px 8px 0;margin-bottom:8px;"
                        f"font-size:0.82rem'>"
                        f"<b style='color:{col_}'>{icon}</b><br>{msg}</div>",
                        unsafe_allow_html=True)


    # ── Allocation optimizer ──────────────────────────────────────────
    # Runs independently — loads all portfolio items directly, no study required
    render_allocator_standalone(capital)



def _build_alloc_rows(capital):
    """
    Load ALL portfolio items (open + watchlist + can include historical)
    and compute metrics needed for allocation.
    Returns list of dicts compatible with the allocator.
    """
    import time as _time
    # Get sector flow scores once
    try:
        from money_flow import get_flow_snapshot
        flow_snap = get_flow_snapshot("1mo")
    except Exception:
        flow_snap = {}

    from money_flow import get_ticker_sector as _get_sec
    from lot_size import get_lot as _get_lot, round_to_lot as _round_lot
    rows = []
    colors = ["#2563eb","#16a34a","#f59e0b","#8b5cf6","#dc2626",
              "#0891b2","#ec4899","#14b8a6","#f97316","#84cc16"]

    # Load stock portfolio
    try:
        stock_df = get_portfolio_full()
        if not stock_df.empty:
            for i,(_,r) in enumerate(stock_df.iterrows()):
                ticker = r["ticker"]
                qty    = float(r.get("shares",0) or 0)
                ac     = float(r.get("avg_cost",0) or 0)
                status = r.get("status","OPEN")
                cb     = qty*ac
                alloc_pct = cb/capital*100 if capital>0 else 0
                try:
                    df_h = fetch_day(ticker,"6mo")
                    if df_h is not None and len(df_h)>=15:
                        rets = df_h["Close"].pct_change().dropna()
                        vol  = float(rets.std()*np.sqrt(252)*100)
                        ret_ = float((df_h["Close"].iloc[-1]-df_h["Close"].iloc[0])/
                                     df_h["Close"].iloc[0]*100)
                        sharpe = ret_/vol if vol>0 else 0
                        chop   = _choppiness(df_h) or 50
                        ranges = df_h["High"]-df_h["Low"]
                        avg_r  = float(ranges.mean())
                        wr     = float((rets>0).mean()*100)
                        # trend score
                        closes = df_h["Close"]
                        x_     = np.arange(len(closes))
                        slope_ = float(np.polyfit(x_,closes.values,1)[0])/float(closes.mean())*100
                        ma20   = float(closes.rolling(20).mean().iloc[-1])
                        ts     = 0
                        if slope_<-0.3:  ts+=25
                        elif slope_<-0.1:ts+=12
                        if float(closes.iloc[-1])<ma20: ts+=20
                        chop_now = chop
                        if chop_now<38: ts+=15
                        ts = min(ts,100)
                    else:
                        vol=30; ret_=0; sharpe=0; chop=50; avg_r=0; wr=50; ts=0
                    _time.sleep(0.15)
                except Exception:
                    vol=30; ret_=0; sharpe=0; chop=50; avg_r=0; wr=50; ts=0
                sec_name  = _get_sec(ticker) or ""
                flow_sc   = flow_snap.get(sec_name, 0)
                # Rotation score components
                try:
                    if df_h is not None and len(df_h)>=14:
                        closes_ = df_h["Close"]
                        d_=closes_.diff(); g_=d_.clip(lower=0).ewm(com=13,adjust=False).mean()
                        l_=(-d_.clip(upper=0)).ewm(com=13,adjust=False).mean()
                        rsi_now=float((100-100/(1+g_/l_.replace(0,np.nan))).dropna().iloc[-1])
                        if len(closes_)>=20:
                            mid_=closes_.rolling(20).mean(); std_=closes_.rolling(20).std()
                            bb_now=float(((closes_-mid_+2*std_)/(4*std_+1e-9)*100).clip(0,100).iloc[-1])
                        else: bb_now=50
                        price_now_=float(closes_.iloc[-1])
                        tgt_row=r.get("target_price")
                        stp_row=r.get("stop_price")
                        tgt_score_=max(0,100-(tgt_row-price_now_)/price_now_*100*5) if tgt_row and price_now_ else 50
                        stp_score_=min((price_now_-stp_row)/price_now_*100*5,100) if stp_row and price_now_ else 50
                        cycle_pct_=(rsi_now/100*50+bb_now/100*50)
                        tech_norm_=min(max((wr-50)*2+50,0),100)
                        sell_rot=round(cycle_pct_*0.35+(100-tech_norm_)*0.35+tgt_score_*0.30,1)
                        buy_rot =round((100-cycle_pct_)*0.35+tech_norm_*0.35+stp_score_*0.30,1)
                    else: sell_rot=buy_rot=50
                except: sell_rot=buy_rot=50
                rows.append({
                    "ticker":      ticker,
                    "name":        r.get("name",ticker),
                    "type":        "Stock",
                    "status":      status,
                    "qty":         qty,
                    "avg_cost":    ac,
                    "cost":        cb,
                    "alloc":       round(alloc_pct,1),
                    "sharpe":      round(sharpe,3),
                    "chop":        round(chop,1),
                    "chop_now":    round(chop,1),
                    "avg_range":   round(avg_r,1),
                    "win_rate":    round(wr,1),
                    "trend_score": ts,
                    "color":       colors[i%len(colors)],
                    "sector":      sec_name,
                    "flow_score":  flow_sc,
                    "sell_rot":    sell_rot,
                    "buy_rot":     buy_rot,
                    "lot":         _get_lot(ticker, float(df_h["Close"].iloc[-1]) if df_h is not None and len(df_h)>0 else 0),
                })
    except Exception: pass

    # Load monitor positions (forex/commodity)
    try:
        from portfolio_manager import get_monitor_pos
        mon_df = get_monitor_pos()
        if not mon_df.empty:
            offset = len(rows)
            for i,(_,r) in enumerate(mon_df.iterrows()):
                ticker = r["ticker"]
                qty    = float(r.get("quantity",0) or 0)
                ac     = float(r.get("avg_cost",0) or 0)
                status = r.get("status","OPEN")
                cb     = qty*ac
                alloc_pct = cb/capital*100 if capital>0 else 0
                try:
                    df_h = fetch_day(ticker,"6mo")
                    if df_h is not None and len(df_h)>=15:
                        rets = df_h["Close"].pct_change().dropna()
                        vol  = float(rets.std()*np.sqrt(252)*100)
                        ret_ = float((df_h["Close"].iloc[-1]-df_h["Close"].iloc[0])/
                                     df_h["Close"].iloc[0]*100)
                        sharpe = ret_/vol if vol>0 else 0
                        chop   = _choppiness(df_h) or 50
                        ranges = df_h["High"]-df_h["Low"]
                        avg_r  = float(ranges.mean())
                        wr     = float((rets>0).mean()*100)
                        closes = df_h["Close"]
                        x_     = np.arange(len(closes))
                        slope_ = float(np.polyfit(x_,closes.values,1)[0])/float(closes.mean())*100
                        ts     = 0
                        if slope_<-0.3:  ts+=25
                        elif slope_<-0.1:ts+=12
                        if chop<38: ts+=15
                        ts = min(ts,100)
                    else:
                        vol=20; ret_=0; sharpe=0; chop=50; avg_r=0; wr=50; ts=0
                    _time.sleep(0.15)
                except Exception:
                    vol=20; ret_=0; sharpe=0; chop=50; avg_r=0; wr=50; ts=0
                sec_name  = _get_sec(ticker) or ""
                flow_sc   = flow_snap.get(sec_name, 0)
                try:
                    if df_h is not None and len(df_h)>=14:
                        closes_ = df_h["Close"]
                        d_=closes_.diff(); g_=d_.clip(lower=0).ewm(com=13,adjust=False).mean()
                        l_=(-d_.clip(upper=0)).ewm(com=13,adjust=False).mean()
                        rsi_now=float((100-100/(1+g_/l_.replace(0,np.nan))).dropna().iloc[-1])
                        if len(closes_)>=20:
                            mid_=closes_.rolling(20).mean(); std_=closes_.rolling(20).std()
                            bb_now=float(((closes_-mid_+2*std_)/(4*std_+1e-9)*100).clip(0,100).iloc[-1])
                        else: bb_now=50
                        cycle_pct_=(rsi_now/100*50+bb_now/100*50)
                        tech_norm_=min(max((wr-50)*2+50,0),100)
                        sell_rot=round(cycle_pct_*0.35+(100-tech_norm_)*0.35+50*0.30,1)
                        buy_rot =round((100-cycle_pct_)*0.35+tech_norm_*0.35+50*0.30,1)
                    else: sell_rot=buy_rot=50
                except: sell_rot=buy_rot=50
                rows.append({
                    "ticker":      ticker,
                    "name":        r.get("name",ticker),
                    "type":        r.get("asset_type","Forex"),
                    "status":      status,
                    "qty":         qty,
                    "avg_cost":    ac,
                    "cost":        cb,
                    "alloc":       round(alloc_pct,1),
                    "sharpe":      round(sharpe,3),
                    "chop":        round(chop,1),
                    "chop_now":    round(chop,1),
                    "avg_range":   round(avg_r,4),
                    "win_rate":    round(wr,1),
                    "trend_score": ts,
                    "color":       colors[(offset+i)%len(colors)],
                    "sector":      sec_name,
                    "flow_score":  flow_sc,
                    "sell_rot":    sell_rot,
                    "buy_rot":     buy_rot,
                    "lot":         1,  # forex/commodity — no board lot
                })
    except Exception: pass

    return rows


def render_allocator_standalone(capital):
    """Capital allocation optimizer — loads all portfolio items independently."""
    st.markdown("---")
    st.markdown("#### 🎯 Capital Allocation Optimizer")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Loads all positions and watchlist items directly. "
        "Runs Max Sharpe and Kelly methods side-by-side. "
        "Considers open positions, watchlist, and historically held items.</span>",
        unsafe_allow_html=True)

    with st.expander("📖 How the two methods work"):
        st.markdown("""
**Max Sharpe** — weights by return ÷ volatility. More return per unit of risk = more capital.
Zero weight for negative Sharpe (losing on risk-taking).

**Kelly Fraction** — weights by win rate edge: (win% - loss%) ÷ odds.
Instruments where you have a statistical edge get more capital.

**Both methods apply:**
- Choppiness ×1.3 if oscillating (>61.8), ×0.2 if trending (<38)
- Trend trap penalty — score 100 = 10% of base weight only
- Range bonus — bigger daily swing = more opportunity

**Status filter:** Include/exclude open positions, watchlist, or both.
        """)

    # Controls
    fc1,fc2,fc3,fc4 = st.columns(4)
    max_single   = fc1.slider("Max per position %", 10, 60, 35, 5, key="sa_max")
    cash_reserve = fc2.slider("Min cash reserve %",  0, 50, 15, 5, key="sa_cash")
    min_pos      = fc3.slider("Min position size %", 1, 15,  5, 1, key="sa_min")
    include      = fc4.multiselect("Include",
                                    ["OPEN","WATCH"],
                                    default=["OPEN","WATCH"],
                                    key="sa_include")

    if st.button("🔄 Load & Optimise", key="sa_run", type="primary"):
        with st.spinner("Loading portfolio data and computing metrics…"):
            rows = _build_alloc_rows(capital)
        st.session_state["sa_rows"] = rows

    rows = st.session_state.get("sa_rows", [])
    if not rows:
        st.info("Click **Load & Optimise** to run the optimizer on your full portfolio.")
        return

    # Filter by status
    rows = [r for r in rows if r.get("status","OPEN") in include]
    if not rows:
        st.warning("No items match the selected status filter.")
        return

    deployable = capital * (1 - cash_reserve/100)

    sharpe_raw   = _max_sharpe_weights(rows, max_single, min_pos)
    kelly_raw    = _kelly_weights(rows, max_single, min_pos)
    sharpe_final = _apply_constraints(sharpe_raw,  max_single, min_pos)
    kelly_final  = _apply_constraints(kelly_raw,   max_single, min_pos)

    by_ticker = {r["ticker"]: r for r in rows}

    # Summary
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Items analysed",     len(rows))
    m2.metric("Capital to deploy",  f"HKD {deployable:,.0f}")
    m3.metric("Active (Sharpe)",    str(sum(1 for v in sharpe_final.values() if v>0)))
    m4.metric("Active (Kelly)",     str(sum(1 for v in kelly_final.values()  if v>0)))

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("**Allocation comparison table**")

    tbl = []
    for t,s in by_ticker.items():
        sh_p = sharpe_final.get(t,0)
        ke_p = kelly_final.get(t,0)
        cur  = s.get("alloc",0)
        consensus = round((sh_p+ke_p)/2,1) if sh_p>0 and ke_p>0 else round(max(sh_p,ke_p)*0.7,1)
        hkd_c = round(consensus/100*deployable,0)
        diff  = consensus-cur
        status_lbl = s.get("status","OPEN")

        if consensus==0:
            ts=s.get("trend_score",0)
            if ts>=70:   reason="Trend trap"
            elif s.get("sharpe",0)<=0: reason="Negative Sharpe"
            else:        reason="Below min size"
            action="CLOSE/AVOID"; ac_="#dc2626"
        elif diff>10:
            reason=f"Underweight vs optimal"
            action=f"↑ +{diff:.0f}%"; ac_="#16a34a"
        elif diff<-10:
            reason=f"Overweight vs optimal"
            action=f"↓ {diff:.0f}%"; ac_="#f59e0b"
        else:
            reason="Near optimal"
            action="✓ HOLD"; ac_="#64748b"

        fl   = s.get("flow_score", 0)
        fl_s = ("🔥" if fl>=50 else "📈" if fl>=20 else "➡" if fl>=-20
                else "📉" if fl>=-50 else "🧊")
        br   = s.get("buy_rot", 50)
        sr   = s.get("sell_rot", 50)
        rot_label = ("🟢 BUY" if br>=62 else "🔴 SELL" if sr>=62 else "⏸ HOLD")
        tbl.append({
            "Status":     status_lbl,
            "Name":       s["name"],
            "Ticker":     t,
            "Type":       s.get("type","—"),
            "Rotation":   f"{rot_label} (B:{br:.0f}/S:{sr:.0f})",
            "Sector Flow":f"{fl_s} {fl:+d}",
            "Current %":  f"{cur:.1f}%",
            "Sharpe":     f"{sh_p:.1f}%" if sh_p>0 else "—",
            "Kelly":      f"{ke_p:.1f}%" if ke_p>0 else "—",
            "Consensus %":f"{consensus:.1f}%" if consensus>0 else "—",
            "HKD":        f"HKD {hkd_c:,.0f}" if consensus>0 else "—",
            "Action":     action,
            "Reason":     reason,
            "_consensus": consensus,
            "_ac":        ac_,
            "_lot":       s.get("lot", 100),
            "_price":     float(s.get("avg_cost",0) or 0),
        })

    # Add lot info
    for r in tbl:
        lot_ = r.get("_lot",100)
        p_   = r.get("_price",0)
        r["Min 1 lot"] = f"{lot_:,} sh · HKD {lot_*p_:,.0f}" if lot_>1 and p_>0 else "—"
        # Round HKD to lot
        if r["_consensus"]>0 and lot_>1 and p_>0:
            deployable_  = capital*(1-cash_reserve/100)
            alloc_hkd    = r["_consensus"]/100*deployable_
            lots_n       = max(int(alloc_hkd/(lot_*p_)),1) if p_>0 else 1
            actual_shares= lots_n*lot_
            actual_hkd   = actual_shares*p_
            r["HKD"] = f"HKD {actual_hkd:,.0f} ({lots_n} lots × {lot_:,})"
    tbl.sort(key=lambda x: -x["_consensus"])
    df_clean = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in tbl])

    def style_tbl(df):
        s=pd.DataFrame("",index=df.index,columns=df.columns)
        for i,row in df.iterrows():
            raw=tbl[i]
            if raw["_consensus"]==0:
                for c in df.columns: s.at[i,c]="color:#94a3b8"
                s.at[i,"Action"]="color:#dc2626;font-weight:600"
            else:
                s.at[i,"Consensus %"]="font-weight:700;color:#2563eb"
                s.at[i,"HKD"]="font-weight:600"
                act=str(row["Action"])
                if "↑" in act: s.at[i,"Action"]="color:#16a34a;font-weight:600"
                elif "↓" in act: s.at[i,"Action"]="color:#f59e0b;font-weight:600"
                elif "AVOID" in act: s.at[i,"Action"]="color:#dc2626;font-weight:600"
            if row["Status"]=="WATCH":
                s.at[i,"Status"]="color:#8b5cf6;font-style:italic"
            rot=str(row.get("Rotation",""))
            if rot.startswith("🟢"): s.at[i,"Rotation"]="color:#16a34a;font-weight:600"
            elif rot.startswith("🔴"): s.at[i,"Rotation"]="color:#dc2626;font-weight:600"
        return s

    st.dataframe(df_clean.style.apply(style_tbl,axis=None),
                 use_container_width=True, hide_index=True)

    # Pies
    pc1,pc2,pc3=st.columns(3)
    def _pie(weights,title):
        active={t:v for t,v in weights.items() if v>0}
        if not active: return
        lbl=[by_ticker[t]["name"] for t in active]+["Cash"]
        val=list(active.values())+[cash_reserve]
        col=[by_ticker[t].get("color","#94a3b8") for t in active]+["#e2e8f0"]
        fig=go.Figure(go.Pie(labels=lbl,values=val,marker=dict(colors=col),
            hole=0.45,textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>"))
        fig.update_layout(height=240,margin=dict(l=0,r=0,t=24,b=0),
            title=dict(text=title,font=dict(size=11)),
            showlegend=False,paper_bgcolor="rgba(0,0,0,0)")
        return fig
    with pc1:
        f=_pie(sharpe_final,"Max Sharpe")
        if f: st.plotly_chart(f,use_container_width=True)
    with pc2:
        f=_pie(kelly_final,"Kelly")
        if f: st.plotly_chart(f,use_container_width=True)
    with pc3:
        cons_w={r["Ticker"]:r["_consensus"] for r in tbl if r["_consensus"]>0}
        f=_pie(cons_w,"Consensus")
        if f: st.plotly_chart(f,use_container_width=True)

    # Alerts
    avoid=[r for r in tbl if r["_consensus"]==0 and float(str(r["Current %"]).replace("%",""))>0]
    if avoid:
        st.error(f"**Reallocate:** {', '.join(r['Name'] for r in avoid)} — "
                 "both methods recommend 0%. Consider closing or reducing.")
    watch_alloc=[r for r in tbl if r["Status"]=="WATCH" and r["_consensus"]>0]
    if watch_alloc:
        st.info(f"**Watchlist opportunities:** "
                f"{', '.join(r['Name'] for r in watch_alloc[:3])} — "
                f"optimizer suggests allocating capital here if you enter.")

    st.caption("Max Sharpe: return/volatility ratio. Kelly: win-rate edge model. "
               "Both penalise trend traps and low choppiness. Not financial advice.")

def _max_sharpe_weights(study_rows, max_single, min_pos):
    """
    Max Sharpe-like: weight proportional to Sharpe score.
    Negative Sharpe = zero weight.
    Also applies sector money flow multiplier.
    """
    weights = {}
    for s in study_rows:
        sh = max(s.get("sharpe", 0), 0)
        # Money flow multiplier
        flow = s.get("flow_score", 0)
        if flow >= 50:    flow_mult = 1.30
        elif flow >= 20:  flow_mult = 1.15
        elif flow >= -20: flow_mult = 1.00
        elif flow >= -50: flow_mult = 0.75
        else:             flow_mult = 0.50
        # Rotation multiplier: buy_rot>62 = early cycle = boost, sell_rot>62 = peak = reduce
        buy_r  = s.get("buy_rot", 50)
        sell_r = s.get("sell_rot", 50)
        if buy_r >= 70:    rot_mult = 1.20
        elif buy_r >= 62:  rot_mult = 1.10
        elif sell_r >= 70: rot_mult = 0.60
        elif sell_r >= 62: rot_mult = 0.80
        else:              rot_mult = 1.00
        weights[s["ticker"]] = sh * flow_mult * rot_mult
    total = sum(weights.values())
    if total == 0:
        return {s["ticker"]: 0 for s in study_rows}
    return {t: w/total*100 for t,w in weights.items()}


def _kelly_weights(study_rows, max_single, min_pos):
    """
    Kelly fraction per position.
    f = (edge) / (odds) where:
      edge = win_rate - (1-win_rate)  (net edge per trade)
      odds = avg_win / avg_loss proxy = (range × 0.6) / (range × 0.4) = 1.5 (approx)
    Then scale to deployable capital.
    Penalise trend traps and low choppiness.
    """
    weights = {}
    for s in study_rows:
        wr   = s.get("win_rate", 50) / 100  # 0-1
        edge = wr - (1 - wr)                 # positive = edge exists
        if edge <= 0:
            weights[s["ticker"]] = 0
            continue

        # Approximate odds from range (60% of range = avg win, 40% = avg loss)
        avg_r  = s.get("avg_range", 20) or 20
        odds   = 1.5                          # simplified: avg win / avg loss
        kelly  = edge / odds                  # fraction of capital to bet
        kelly  = max(kelly, 0)

        # Choppiness multiplier
        chop = s.get("chop_now", 50) or s.get("chop", 50) or 50
        if chop >= 61.8:   cm = 1.3
        elif chop >= 50:   cm = 1.0
        elif chop >= 38:   cm = 0.6
        else:              cm = 0.2

        # Trend trap penalty
        ts   = s.get("trend_score", 0)
        tm   = max(0.1, (100 - ts) / 100)

        # Range bonus
        rm   = min(1 + avg_r/80, 1.4)

        # Money flow multiplier
        flow = s.get("flow_score", 0)
        if flow >= 50:    fm = 1.30
        elif flow >= 20:  fm = 1.15
        elif flow >= -20: fm = 1.00
        elif flow >= -50: fm = 0.75
        else:             fm = 0.50
        # Rotation multiplier
        buy_r  = s.get("buy_rot", 50)
        sell_r = s.get("sell_rot", 50)
        if buy_r >= 70:    rot_m = 1.20
        elif buy_r >= 62:  rot_m = 1.10
        elif sell_r >= 70: rot_m = 0.60
        elif sell_r >= 62: rot_m = 0.80
        else:              rot_m = 1.00

        weights[s["ticker"]] = kelly * cm * tm * rm * fm * rot_m

    total = sum(weights.values())
    if total == 0:
        return {s["ticker"]: 0 for s in study_rows}
    return {t: w/total*100 for t,w in weights.items()}


def _apply_constraints(weights_pct, max_single, min_pos):
    """Iteratively cap + floor weights, then re-normalise."""
    w = dict(weights_pct)
    for _ in range(30):
        capped = False
        for t in list(w):
            if w[t] > max_single:
                excess = w[t] - max_single
                w[t]   = max_single
                others  = [k for k in w if k != t and w[k] < max_single]
                tot_o   = sum(w[k] for k in others) or 1
                for k in others:
                    w[k] += excess * w[k] / tot_o
                capped = True
        if not capped:
            break
    # Floor
    for t in list(w):
        if w[t] < min_pos:
            w[t] = 0
    # Re-normalise
    total = sum(w.values())
    if total > 0:
        w = {t: round(v/total*100, 1) for t,v in w.items()}
    return w


def render_allocator(study_rows, capital):
    st.markdown("---")
    st.markdown("#### 🎯 Capital Allocation Optimizer")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Two methods side-by-side: Max Sharpe (most efficient) and Kelly (edge-based). "
        "Both penalise trend traps and low choppiness heavily.</span>",
        unsafe_allow_html=True)

    with st.expander("📖 How the two methods work"):
        st.markdown("""
**Max Sharpe-like**
Allocates proportionally to each position's Sharpe-like score (return ÷ volatility).
- The higher your return per unit of risk, the more capital it deserves
- Zero weight for any position with negative Sharpe (losing on risk)
- Then multiplied by choppiness and trend health adjusters
- Conservative: ignores win/loss frequency, focuses purely on efficiency

**Kelly Fraction**
Based on your statistical edge: f = (win_rate - loss_rate) ÷ odds
- Win rate from historical daily data for this instrument
- Odds approximated from average daily range (60% of range = win, 40% = loss)
- Naturally sizes larger on high-frequency winning instruments
- Aggressive: can produce concentrated positions, hence the constraints

**Shared adjustments on both methods:**
- Choppiness ×1.3 if >61.8 (ideal oscillation), ×0.2 if <38 (trending = dangerous)
- Trend score penalty: 0/100 = full weight, 100/100 = 10% weight only
- Range bonus: bigger daily range = more opportunity = more weight (capped ×1.4)

**Constraints you control:**
- Max single position %: caps any one instrument
- Min cash reserve %: always kept as dry powder
- Min position size %: positions below this are rounded to zero
        """)

    ac1, ac2, ac3 = st.columns(3)
    max_single   = ac1.slider("Max single position %", 10, 60, 35, 5, key="alloc_max")
    cash_reserve = ac2.slider("Min cash reserve %",    0,  50, 15, 5, key="alloc_cash")
    min_pos      = ac3.slider("Min position size %",   1,  15,  5, 1, key="alloc_min")

    if not study_rows:
        st.info("Run the portfolio study first to enable the optimizer.")
        return

    deployable = capital * (1 - cash_reserve/100)

    # Compute both methods
    sharpe_raw  = _max_sharpe_weights(study_rows, max_single, min_pos)
    kelly_raw   = _kelly_weights(study_rows, max_single, min_pos)
    sharpe_final= _apply_constraints(sharpe_raw,  max_single, min_pos)
    kelly_final = _apply_constraints(kelly_raw,   max_single, min_pos)

    # Build lookup
    by_ticker = {s["ticker"]: s for s in study_rows}

    # ── Summary metrics ───────────────────────────────────────────────
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Capital to deploy", f"HKD {deployable:,.0f}")
    m2.metric("Cash reserve",      f"HKD {capital*cash_reserve/100:,.0f} ({cash_reserve}%)")
    n_sh = sum(1 for v in sharpe_final.values() if v>0)
    n_ke = sum(1 for v in kelly_final.values()  if v>0)
    m3.metric("Active (Sharpe method)", str(n_sh))
    m4.metric("Active (Kelly method)",  str(n_ke))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison table ──────────────────────────────────────────────
    st.markdown("**Side-by-side allocation comparison**")

    tbl = []
    all_tickers = list(by_ticker.keys())
    for t in all_tickers:
        s    = by_ticker[t]
        sh_p = sharpe_final.get(t, 0)
        ke_p = kelly_final.get(t, 0)
        cur  = s.get("alloc", 0)
        chop = s.get("chop_now", 50) or s.get("chop", 50) or 50
        ts   = s.get("trend_score", 0)

        # Consensus: average of both, only if both >0
        if sh_p > 0 and ke_p > 0:
            consensus = round((sh_p + ke_p) / 2, 1)
        elif sh_p > 0 or ke_p > 0:
            consensus = round(max(sh_p, ke_p) * 0.7, 1)  # discount if only one agrees
        else:
            consensus = 0

        hkd_c = round(consensus / 100 * deployable, 0)

        # Action
        diff  = consensus - cur
        if consensus == 0:
            action = "CLOSE / AVOID"
            act_c  = "#dc2626"
        elif diff > 10:
            action = f"↑ INCREASE +{diff:.0f}%"
            act_c  = "#16a34a"
        elif diff < -10:
            action = f"↓ REDUCE {diff:.0f}%"
            act_c  = "#f59e0b"
        else:
            action = "✓ HOLD"
            act_c  = "#64748b"

        # Reason
        if consensus == 0:
            if ts >= 70:    reason = "Trend trap — downtrending, no swing"
            elif s.get("sharpe",0) <= 0: reason = "Negative Sharpe — losing on risk"
            else:           reason = "Below minimum size threshold"
        elif chop >= 61.8 and ts < 45:
            reason = "Active swing + healthy trend"
        elif chop >= 61.8:
            reason = "Good choppiness, some trend risk"
        elif ts >= 70:
            reason = "Trend trap — weight heavily penalised"
        elif ts >= 45:
            reason = "Weakening swing — reduced weight"
        else:
            reason = "Mixed conditions"

        tbl.append({
            "Name":          s["name"],
            "Ticker":        t,
            "Current %":     f"{cur:.1f}%",
            "Sharpe method": f"{sh_p:.1f}%" if sh_p>0 else "—",
            "Kelly method":  f"{ke_p:.1f}%" if ke_p>0 else "—",
            "Consensus %":   f"{consensus:.1f}%" if consensus>0 else "—",
            "HKD (consensus)":f"HKD {hkd_c:,.0f}" if consensus>0 else "—",
            "Action":        action,
            "Reason":        reason,
            "_act_c":        act_c,
            "_consensus":    consensus,
        })

    # Sort: active first
    tbl.sort(key=lambda x: -x["_consensus"])
    df_tbl_clean = pd.DataFrame([{k:v for k,v in row.items()
                                   if not k.startswith("_")} for row in tbl])

    def style_alloc(df):
        s = pd.DataFrame("", index=df.index, columns=df.columns)
        for i, row in df.iterrows():
            raw = tbl[i]
            if raw["_consensus"] == 0:
                for c in df.columns:
                    s.at[i,c] = "color:#94a3b8"
                s.at[i,"Action"] = "color:#dc2626;font-weight:600"
            else:
                s.at[i,"Consensus %"]    = "font-weight:700;color:#2563eb"
                s.at[i,"HKD (consensus)"]= "font-weight:600"
                act = str(row["Action"])
                if "INCREASE" in act:  s.at[i,"Action"] = "color:#16a34a;font-weight:600"
                elif "REDUCE"  in act: s.at[i,"Action"] = "color:#f59e0b;font-weight:600"
                elif "AVOID"   in act: s.at[i,"Action"] = "color:#dc2626;font-weight:600"
        return s

    st.dataframe(df_tbl_clean.style.apply(style_alloc, axis=None),
                 use_container_width=True, hide_index=True)

    # ── Pie charts side by side ───────────────────────────────────────
    st.markdown("**Allocation visualised**")
    pc1, pc2, pc3 = st.columns(3)

    def make_pie(weights_dict, title, deploy):
        active = {t:v for t,v in weights_dict.items() if v>0}
        if not active: return
        labels = [by_ticker[t]["name"] for t in active] + ["Cash"]
        values = list(active.values()) + [cash_reserve]
        colors = [by_ticker[t].get("color","#94a3b8") for t in active] + ["#e2e8f0"]
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors),
            hole=0.45, textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>"))
        fig.update_layout(height=260, margin=dict(l=0,r=0,t=28,b=0),
            title=dict(text=title, font=dict(size=12)),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    with pc1:
        st.markdown("**Max Sharpe**")
        fig1 = make_pie(sharpe_final, "Max Sharpe", deployable)
        if fig1: st.plotly_chart(fig1, use_container_width=True)

    with pc2:
        st.markdown("**Kelly**")
        fig2 = make_pie(kelly_final, "Kelly", deployable)
        if fig2: st.plotly_chart(fig2, use_container_width=True)

    with pc3:
        st.markdown("**Consensus**")
        cons_w = {row["Ticker"]: row["_consensus"]
                  for row in tbl if row["_consensus"]>0}
        fig3 = make_pie(cons_w, "Consensus", deployable)
        if fig3: st.plotly_chart(fig3, use_container_width=True)

    # ── Flagged alerts ────────────────────────────────────────────────
    avoid = [r for r in tbl if r["_consensus"]==0 and r["Current %"]!="0.0%"]
    if avoid:
        names_ = ", ".join(r["Name"] for r in avoid)
        st.error(
            f"**Exit signal:** {names_} — both methods recommend 0% allocation. "
            "Current capital here is inefficient. "
            "Consider closing or reducing before redeploying into better opportunities.")

    big_diff = [r for r in tbl if abs(r["_consensus"]-float(
        str(r["Current %"]).replace("%","")))>15 and r["_consensus"]>0]
    if big_diff:
        for r in big_diff:
            cur_ = float(str(r["Current %"]).replace("%",""))
            diff_ = r["_consensus"]-cur_
            st.warning(
                f"**{r['Name']}**: current {cur_:.1f}% vs recommended {r['_consensus']:.1f}% "
                f"— gap of {diff_:+.1f}%. {r['Reason']}")

    st.caption(
        "Consensus = average of both methods (discounted if only one agrees). "
        "Max Sharpe: weights by return/risk ratio. "
        "Kelly: weights by win rate edge and range odds. "
        "Not financial advice.")



def _empty_study(r):
    return {"name":r["name"],"ticker":r["ticker"],"type":r["type"],
            "ret_pct":0,"ann_vol":None,"sharpe":0,"avg_range":0,
            "avg_range_pct":0,"max_dd":0,"win_rate":0,"range_capture":0,
            "alloc":r["cost"]/get_latest_capital()*100 if get_latest_capital()>0 else 0,
            "rr":None,"color":r["color"],"cost":r["cost"],
            "trend_score":0,"trend_label":"—","slope_pct":0,
            "below_ma20":None,"below_ma50":None,"consec_down":0,
            "chop_now":50,"range_shrink":0,"lower_highs":0}

def _choppiness(df,p=14):
    if len(df)<p+2: return None
    tr=pd.concat([df["High"]-df["Low"],
                  (df["High"]-df["Close"].shift()).abs(),
                  (df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    ci=100*np.log10(tr.rolling(p).sum()/(
        df["High"].rolling(p).max()-df["Low"].rolling(p).min()+1e-9))/np.log10(p)
    return float(ci.clip(0,100).iloc[-1])


# ═════════════════════════════════════════════════════════════════════
# ROUTING
# ═════════════════════════════════════════════════════════════════════
if page == "📊  Summary":
    render_summary()

elif page == "💰  Flow":
    money_flow.render()

elif page == "📋  Portfolio":
    portfolio_manager.render()

elif page == "🔬  Analysis":
    analysis_page.render(chart_interval=chart_iv, daily_period=daily_p)

elif page == "🛡  Risk":
    risk_tools.render()

elif page == "🔍  Scanner":
    volume_scanner.render()

elif page == "📐  Study":
    portfolio_study.render()

elif page == "📅  Daily":
    daily_strategy.render()

elif page == "🧠  Strategy":
    s1, s2 = st.tabs(["📊 Market Study", "🔄 Cycle ML"])
    with s1:
        strategy_page.render()
    with s2:
        cycle_ml.render()
