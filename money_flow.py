"""
money_flow.py — Global Money Flow Tracker
Tracks where institutional and retail money is concentrated
across sectors, asset classes, and geographies.

Signals:
  1. Volume surge vs 20d avg (unusual activity = money moving in/out)
  2. Price momentum (5d, 20d returns — where is trending up)
  3. RSI zone (overbought = crowded, oversold = unloved)

Combined into:
  - Flow heatmap (sector × signal)
  - Flow score per sector (positive = inflow, negative = outflow)
  - Smart money vs crowd detection
  - Rotation signals (what is being sold to buy what)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import time, pytz
warnings_imported = False
try:
    import warnings; warnings.filterwarnings("ignore")
    warnings_imported = True
except: pass

HK_TZ = pytz.timezone("Asia/Hong_Kong")

# ── UNIVERSE — representative tickers per sector/theme ────────────────
# Each sector uses 3-5 liquid names to compute an aggregate signal
SECTORS = {
    # ══════════════════════════════════════════════════════════════════
    # HKEX — full sector universe (~20 names per sector)
    # More names = more accurate flow reading
    # ══════════════════════════════════════════════════════════════════

    "HK AI/Tech": [
        "0700.HK",  # Tencent
        "9988.HK",  # Alibaba
        "9999.HK",  # NetEase
        "9888.HK",  # Baidu
        "1024.HK",  # Kuaishou
        "0020.HK",  # SenseTime
        "1810.HK",  # Xiaomi
        "0268.HK",  # Kingsoft
        "0763.HK",  # ZTE
        "0992.HK",  # Lenovo
        "2382.HK",  # Sunny Optical
        "1347.HK",  # Hua Hong Semi
        "0100.HK",  # MiniMax
        "2513.HK",  # Zhipu / Knowledge Atlas
        "0285.HK",  # BYD Electronics
        "6606.HK",  # Kanzhun
        "0522.HK",  # ASM Pacific
        "0981.HK",  # SMIC
        "0241.HK",  # Ali Health (digital)
        "0799.HK",  # IGG (mobile games/tech)
        "0700.HK",  # Tencent (also listed again as anchor)
        "2121.HK",  # iQIYI
        "0777.HK",  # NetDragon
        "0858.HK",  # Dongfang Electric (power tech)
    ],

    "HK EV": [
        "9866.HK",  # NIO
        "9868.HK",  # Xpeng
        "2015.HK",  # Li Auto
        "1211.HK",  # BYD
        "0175.HK",  # Geely
        "2238.HK",  # GAC Group
        "3750.HK",  # CATL
        "0285.HK",  # BYD Electronics
        "1519.HK",  # FAW Group
        "0489.HK",  # Dongfeng Motor
        "1776.HK",  # Ganfeng Lithium
        "0136.HK",  # Hengdelai
        "0750.HK",  # Wanguo International (mining/battery)
        "1953.HK",  # Akeso (EV adjacent)
        "6837.HK",  # Haitong Sec (EV sector backer)
        "2727.HK",  # Shanghai Electric
        "0816.HK",  # Metallurgical Corp (supply chain)
    ],

    "HK Biotech": [
        "2269.HK",  # Wuxi Biologics
        "6160.HK",  # BeiGene
        "1093.HK",  # CSPC Pharma
        "1177.HK",  # Sino Biopharm
        "2196.HK",  # Shanghai Pharma
        "0241.HK",  # Ali Health
        "1833.HK",  # Ping An Health
        "0867.HK",  # CSPC
        "6998.HK",  # Hua Medicine
        "2552.HK",  # HUTCHMED
        "1530.HK",  # 3SBio
        "0460.HK",  # Sihuan Pharma
        "6618.HK",  # JD Health
        "1579.HK",  # CMG Bioscience
        "2616.HK",  # Kinetic Biopharma
        "1548.HK",  # Genscript Biotech
        "0853.HK",  # Microport Scientific
        "6919.HK",  # Kineta (biotech)
        "1877.HK",  # Sinopharm
        "0570.HK",  # China Traditional Medicine
    ],

    "HK Gaming": [
        "0700.HK",  # Tencent
        "9999.HK",  # NetEase
        "1054.HK",  # Boyaa Interactive
        "0027.HK",  # Galaxy Entertainment
        "1928.HK",  # Sands China
        "0880.HK",  # SJM Holdings
        "1967.HK",  # MGM China
        "0776.HK",  # iDreamSky
        "0799.HK",  # IGG
        "0777.HK",  # NetDragon
        "2121.HK",  # iQIYI
        "0288.HK",  # WH Group (entertainment adjacent)
    ],

    "HK NewEcon": [
        "9988.HK",  # Alibaba
        "9618.HK",  # JD.com
        "9961.HK",  # Trip.com
        "0780.HK",  # Tongcheng Travel
        "6690.HK",  # Haier Smart
        "9987.HK",  # Yum China
        "6862.HK",  # Haidilao
        "2020.HK",  # Anta Sports
        "9896.HK",  # Miniso
        "0291.HK",  # CR Beer
        "1044.HK",  # Hengan International
        "0220.HK",  # Uni-President
        "0322.HK",  # Tingyi
        "0270.HK",  # Guangdong Investment
        "6110.HK",  # Topsports
        "1368.HK",  # Xtep International
        "0551.HK",  # Yue Yuen Industrial
        "3690.HK",  # Meituan
    ],

    "HK Finance": [
        "0388.HK",  # HKEX
        "2318.HK",  # Ping An Insurance
        "1299.HK",  # AIA Group
        "0005.HK",  # HSBC
        "0939.HK",  # CCB
        "1398.HK",  # ICBC
        "3968.HK",  # China Merchants Bank
        "2388.HK",  # BOC HK
        "0011.HK",  # Hang Seng Bank
        "1336.HK",  # New China Life
        "6886.HK",  # HTSC
        "6958.HK",  # CICC
        "3988.HK",  # Bank of China
        "1988.HK",  # China Minsheng Bank
        "2628.HK",  # China Life
        "0966.HK",  # China Taiping
        "2601.HK",  # CPIC
        "6837.HK",  # Haitong Securities
        "3618.HK",  # Chongqing Rural Bank
    ],

    "HK Resources": [
        "2899.HK",  # Zijin Mining
        "1208.HK",  # MMG
        "0883.HK",  # CNOOC
        "0857.HK",  # PetroChina
        "0386.HK",  # Sinopec
        "1088.HK",  # China Shenhua
        "3993.HK",  # China Moly
        "1816.HK",  # CGN Power
        "0941.HK",  # China Mobile (infrastructure)
        "0762.HK",  # China Unicom
        "0836.HK",  # China Resources Power
        "2688.HK",  # ENN Energy
        "0384.HK",  # China Gas
        "1193.HK",  # China Resources Gas
        "3800.HK",  # GCL Technology (solar)
        "0916.HK",  # China Longyuan Power
        "0579.HK",  # Beijing Energy International
        "0750.HK",  # Wanguo International Mining
    ],

    "HK Property": [
        "0016.HK",  # Sun Hung Kai Prop
        "0001.HK",  # CK Asset
        "0012.HK",  # Henderson Land
        "0688.HK",  # China Overseas Land
        "1109.HK",  # CR Land
        "3383.HK",  # Agile Group
        "2202.HK",  # Vanke
        "0823.HK",  # Link REIT
        "0435.HK",  # Sunlight REIT
        "0659.HK",  # NWS Holdings
        "0101.HK",  # Hang Lung Properties
        "0017.HK",  # New World Development
        "0083.HK",  # Sino Land
        "0002.HK",  # CLP Holdings (utility/property)
    ],

    # ══════════════════════════════════════════════════════════════════
    # GLOBAL TECH — expanded
    # ══════════════════════════════════════════════════════════════════

    "US Mega Tech": [
        "NVDA","AAPL","MSFT","META","GOOGL","AMZN","TSLA","NFLX","ADBE","ORCL",
    ],

    "US Semis": [
        "NVDA","AMD","AVGO","QCOM","MU","AMAT","ASML","INTC","KLAC","LRCX",
        "MCHP","TXN","ON","WOLF","MPWR",
    ],

    "US AI/SaaS": [
        "PLTR","SNOW","CRM","NOW","DDOG","MDB","SMCI","AI","PATH","SOUN",
        "BBAI","IONQ","ARRY","RBRK","APP",
    ],

    "US Biotech": [
        "MRNA","BNTX","CRSP","EDIT","BEAM","RXRX","ILMN","REGN","VRTX",
        "ARKG","NTLA","BLUE","SAGE","KRTX",
    ],

    "Crypto-linked": [
        "MSTR","COIN","MARA","RIOT","HUT","CLSK","BTBT","CIFR",
        "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD",
    ],

    # ══════════════════════════════════════════════════════════════════
    # COMMODITIES
    # ══════════════════════════════════════════════════════════════════
    "Gold/Silver":    ["GC=F","SI=F","PL=F","PA=F","GLD","SLV"],
    "Energy":         ["CL=F","BZ=F","NG=F","RB=F","HO=F","USO","UNG"],
    "Industrial Met": ["HG=F","ALI=F","ZN=F","PB=F"],
    "Agriculture":    ["ZC=F","ZW=F","ZS=F","KC=F","SB=F","CT=F","CC=F"],

    # ══════════════════════════════════════════════════════════════════
    # FOREX / MACRO
    # ══════════════════════════════════════════════════════════════════
    "USD Strength":   ["USDJPY=X","USDCNY=X","USDCNH=X","USDCHF=X","DX-Y.NYB"],
    "Risk Forex":     ["AUDUSD=X","USDMXN=X","USDBRL=X","USDZAR=X","USDKRW=X"],
    "Safe Haven":     ["GC=F","USDJPY=X","EURUSD=X","USDCHF=X"],
}

SECTOR_GROUPS = {
    "🇭🇰 HKEX": [
        "HK AI/Tech","HK EV","HK Biotech","HK Gaming",
        "HK NewEcon","HK Finance","HK Resources","HK Property",
    ],
    "🌐 Global Tech": ["US Mega Tech","US Semis","US AI/SaaS","US Biotech","Crypto-linked"],
    "🥇 Commodities": ["Gold/Silver","Energy","Industrial Met","Agriculture"],
    "💱 Macro/Forex":  ["USD Strength","Risk Forex","Safe Haven"],
}

SECTOR_COLOR = {
    "HK AI/Tech":"#2563eb",    "HK EV":"#0891b2",
    "HK Biotech":"#8b5cf6",    "HK Gaming":"#ec4899",
    "HK NewEcon":"#16a34a",    "HK Finance":"#475569",
    "HK Resources":"#b45309",  "HK Property":"#78716c",
    "US Mega Tech":"#dc2626",  "US Semis":"#ea580c",
    "US AI/SaaS":"#f97316",    "US Biotech":"#9333ea",
    "Crypto-linked":"#f59e0b",
    "Gold/Silver":"#ca8a04",   "Energy":"#78350f",
    "Industrial Met":"#64748b","Agriculture":"#15803d",
    "USD Strength":"#1d4ed8",  "Risk Forex":"#7c3aed",
    "Safe Haven":"#0f766e",
}

# ── DATA ──────────────────────────────────────────────────────────────
def _var(ticker):
    v=[ticker]; code=ticker.replace(".HK","")
    if code.isdigit():
        v.append(str(int(code))+".HK")
        v.append(code.zfill(4)+".HK")
    return list(dict.fromkeys(v))

@st.cache_data(ttl=300, show_spinner=False)
def fetch_sector_data(tickers: tuple, period: str = "3mo") -> dict:
    """Fetch OHLCV for a list of tickers, return {ticker: df}."""
    result = {}
    for t in tickers:
        for tv in _var(t):
            try:
                df = yf.Ticker(tv).history(
                    period=period, interval="1d", auto_adjust=True)
                if len(df) >= 20:
                    df.index = pd.to_datetime(df.index)
                    result[t] = df
                    break
                time.sleep(0.15)
            except Exception: pass
    return result

# ── METRICS ───────────────────────────────────────────────────────────
def sector_metrics(tickers, period="3mo"):
    """
    Aggregate metrics for a sector from its representative tickers.
    Returns dict of flow signals.
    """
    data = fetch_sector_data(tuple(tickers), period)
    if not data:
        return None

    ret_5d_list=[]; ret_20d_list=[]; rsi_list=[]; vol_ratio_list=[]
    trend_list=[]; range_list=[]

    for t, df in data.items():
        if len(df) < 21: continue
        c = df["Close"]; v = df["Volume"]
        # Returns
        r5  = (c.iloc[-1]-c.iloc[-5])/c.iloc[-5]*100  if len(c)>=5  else 0
        r20 = (c.iloc[-1]-c.iloc[-20])/c.iloc[-20]*100 if len(c)>=20 else 0
        # RSI
        d=c.diff(); g=d.clip(lower=0).ewm(com=13,adjust=False).mean()
        lo=(-d.clip(upper=0)).ewm(com=13,adjust=False).mean()
        rsi_v=float((100-100/(1+g/lo.replace(0,np.nan))).dropna().iloc[-1])
        # Volume ratio
        avg_vol = float(v.rolling(20).mean().iloc[-1])
        vol_r   = float(v.iloc[-1])/avg_vol if avg_vol>0 else 1
        # Trend slope
        x=np.arange(20); y=c.tail(20).values
        slope=float(np.polyfit(x,y,1)[0])/float(c.mean())*100
        # Daily range %
        ranges=(df["High"]-df["Low"])/c
        avg_r=float(ranges.tail(20).mean()*100)

        ret_5d_list.append(r5); ret_20d_list.append(r20)
        rsi_list.append(rsi_v); vol_ratio_list.append(vol_r)
        trend_list.append(slope); range_list.append(avg_r)

    if not ret_5d_list:
        return None

    ret5  = np.mean(ret_5d_list)
    ret20 = np.mean(ret_20d_list)
    rsi   = np.mean(rsi_list)
    vol_r = np.mean(vol_ratio_list)
    trend = np.mean(trend_list)
    avg_r = np.mean(range_list)
    n     = len(ret_5d_list)

    # ── FLOW SCORE ────────────────────────────────────────────────────
    # Positive = money flowing IN, Negative = flowing OUT
    # Scale: +100 = maximum inflow signal, -100 = maximum outflow
    score = 0

    # Volume surge component (0-40 pts)
    if vol_r >= 2.0:    vol_s = 40
    elif vol_r >= 1.5:  vol_s = 25
    elif vol_r >= 1.2:  vol_s = 12
    elif vol_r >= 0.8:  vol_s = 0
    elif vol_r >= 0.5:  vol_s = -15
    else:               vol_s = -30
    score += vol_s

    # Momentum component (0-40 pts)
    if ret5 >= 5:      mom_s = 30
    elif ret5 >= 2:    mom_s = 18
    elif ret5 >= 0.5:  mom_s = 8
    elif ret5 >= -0.5: mom_s = 0
    elif ret5 >= -2:   mom_s = -10
    elif ret5 >= -5:   mom_s = -22
    else:              mom_s = -35
    score += mom_s

    # RSI zone component (-20 to +20)
    if rsi >= 75:      rsi_s = -20   # overbought = crowded, reversal risk
    elif rsi >= 65:    rsi_s = -8
    elif rsi >= 55:    rsi_s = 5
    elif rsi >= 45:    rsi_s = 10
    elif rsi >= 35:    rsi_s = 15
    else:              rsi_s = 20    # oversold = unloved, potential entry
    score += rsi_s

    score = int(np.clip(score, -100, 100))

    # ── SMART MONEY vs CROWD ─────────────────────────────────────────
    # Smart money: high vol + SMALL candle bodies (accumulating quietly)
    # Crowd money: high vol + LARGE bodies (chasing the move)
    body_ratios=[]
    for t,df in data.items():
        if len(df)<5: continue
        body=abs(df["Close"]-df["Open"]).tail(5).mean()
        rng=(df["High"]-df["Low"]).tail(5).mean()
        if rng>0: body_ratios.append(body/rng)
    avg_body = np.mean(body_ratios) if body_ratios else 0.5
    # Low body ratio + high vol = smart money (absorbing quietly)
    # High body ratio + high vol = crowd (chasing)
    if vol_r > 1.3 and avg_body < 0.4:
        driver = "🏦 Smart Money"
        driver_color = "#2563eb"
        driver_note  = "High vol + small candles = quiet accumulation"
    elif vol_r > 1.3 and avg_body > 0.65:
        driver = "👥 Crowd"
        driver_color = "#f59e0b"
        driver_note  = "High vol + big candles = retail chasing"
    elif vol_r < 0.7:
        driver = "😴 Quiet"
        driver_color = "#94a3b8"
        driver_note  = "Low volume = no strong conviction"
    else:
        driver = "🔀 Mixed"
        driver_color = "#64748b"
        driver_note  = "Normal activity"

    # ── SIGNAL LABEL ─────────────────────────────────────────────────
    if score >= 50:      signal = "🔥 Strong Inflow"
    elif score >= 20:    signal = "📈 Inflow"
    elif score >= 5:     signal = "↗ Mild Inflow"
    elif score >= -5:    signal = "➡ Neutral"
    elif score >= -20:   signal = "↘ Mild Outflow"
    elif score >= -50:   signal = "📉 Outflow"
    else:                signal = "🧊 Strong Outflow"

    return {
        "ret5":       round(ret5,2),
        "ret20":      round(ret20,2),
        "rsi":        round(rsi,1),
        "vol_ratio":  round(vol_r,2),
        "trend":      round(trend,4),
        "avg_range":  round(avg_r,2),
        "flow_score": score,
        "signal":     signal,
        "driver":     driver,
        "driver_color":driver_color,
        "driver_note": driver_note,
        "n_tickers":  n,
    }

# ── RENDER ────────────────────────────────────────────────────────────

# ── TICKER → SECTOR reverse mapping ─────────────────────────────────
_TICKER_SECTOR = {}
for _sec, _tickers in SECTORS.items():
    for _t in _tickers:
        _TICKER_SECTOR[_t] = _sec

def get_ticker_sector(ticker: str) -> str:
    """Return the sector name for a given ticker, or None."""
    return _TICKER_SECTOR.get(ticker)

@st.cache_data(ttl=300, show_spinner=False)
def get_flow_snapshot(period: str = "1mo") -> dict:
    """
    Return flow scores for ALL sectors — used by other pages.
    Returns {sector_name: flow_score} sorted descending.
    """
    results = {}
    for sec, tickers in SECTORS.items():
        m = sector_metrics(tickers, period)
        if m:
            results[sec] = m["flow_score"]
    return results

def get_ticker_flow(ticker: str, period: str = "1mo") -> dict:
    """
    Return flow context for a specific ticker:
    - its sector name
    - sector flow score
    - sector signal label
    - sector driver (smart money / crowd / quiet)
    Returns dict or empty dict if not found.
    """
    sec = get_ticker_sector(ticker)
    if not sec:
        return {}
    m = sector_metrics(tuple(SECTORS[sec]), period)
    if not m:
        return {}
    return {
        "sector":       sec,
        "flow_score":   m["flow_score"],
        "signal":       m["signal"],
        "driver":       m["driver"],
        "driver_color": m["driver_color"],
        "driver_note":  m["driver_note"],
        "vol_ratio":    m["vol_ratio"],
        "ret5":         m["ret5"],
        "rsi":          m["rsi"],
    }

def render():
    now_hk = datetime.now(HK_TZ)
    st.markdown(
        "## 💰 Money Flow Tracker &nbsp;"
        "<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        "padding:2px 7px;border-radius:5px'>GLOBAL</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"Tracks where institutional & retail capital is concentrated · "
        f"Volume surge + Momentum + RSI combined · "
        f"{now_hk.strftime('%Y-%m-%d %H:%M HKT')}</span>",
        unsafe_allow_html=True)

    with st.expander("📖 How money flow is detected"):
        st.markdown("""
**Three signals combined into a Flow Score (-100 to +100):**

**Volume surge (±40 pts)** — Is this sector unusually active?
Sector volume vs its own 20-day average. >2x = major institutional activity (+40).
<0.5x = nobody cares (-30). Volume is the most reliable signal of where money is moving.

**Price momentum (±35 pts)** — Is money actually making prices move?
5-day return for the sector's representative stocks. Strong positive = capital flowing in.
Strong negative = capital leaving regardless of news.

**RSI zone (±20 pts)** — Is the sector crowded or unloved?
RSI >75 = overbought = too many people already in (-20, reversal risk).
RSI <35 = oversold = unloved = potential entry opportunity (+20).
Note: this is contrarian — a sector with high inflow BUT high RSI = late-cycle crowded trade.

**Smart Money vs Crowd detection:**
High volume + SMALL candle bodies = smart money accumulating quietly (institutions don't chase).
High volume + LARGE candle bodies = retail crowd chasing (momentum play, higher reversal risk).

**Rotation signals:** When one sector shows strong outflow while another shows strong inflow,
that is a rotation — money leaving one theme and entering another.
        """)
    st.markdown("---")

    # Controls
    c1, c2, c3 = st.columns(3)
    groups_sel = c1.multiselect(
        "Sector groups",
        list(SECTOR_GROUPS.keys()),
        default=list(SECTOR_GROUPS.keys()),
        key="mf_groups")
    period = c2.selectbox("History", ["1mo","3mo","6mo"], index=1, key="mf_period")
    if c3.button("🔄 Refresh", key="mf_refresh"):
        st.cache_data.clear(); st.rerun()

    # Build sector list from selected groups
    sectors_to_show = []
    for g in groups_sel:
        sectors_to_show += SECTOR_GROUPS.get(g, [])

    if not sectors_to_show:
        st.info("Select at least one sector group.")
        return

    # ── Scan all sectors ──────────────────────────────────────────────
    prog = st.progress(0, "Scanning sectors…")
    results = {}
    for i, sec in enumerate(sectors_to_show):
        tickers = SECTORS.get(sec, [])
        m = sector_metrics(tickers, period)
        if m:
            results[sec] = m
        prog.progress((i+1)/len(sectors_to_show),
                      text=f"Scanning {sec}…")
    prog.empty()

    if not results:
        st.error("Could not fetch data. Check connection.")
        return

    df_res = pd.DataFrame(results).T
    df_res["sector"] = df_res.index
    df_res = df_res.sort_values("flow_score", ascending=False).reset_index(drop=True)

    # ── TOP BANNER — Biggest inflow / outflow ─────────────────────────
    top_in  = df_res.iloc[0]  if len(df_res)>0 else None
    top_out = df_res.iloc[-1] if len(df_res)>1 else None

    b1, b2 = st.columns(2)
    if top_in is not None:
        b1.markdown(
            f"<div style='border:2px solid #16a34a;border-radius:12px;"
            f"padding:14px 18px;background:rgba(22,163,74,0.04)'>"
            f"<div style='font-size:0.72rem;color:#64748b'>🔥 Strongest inflow</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:#16a34a'>"
            f"{top_in['sector']}</div>"
            f"<div style='font-size:0.82rem;color:#475569;margin-top:4px'>"
            f"Score: {int(top_in['flow_score'])} · "
            f"5d ret: {top_in['ret5']:+.2f}% · "
            f"Vol: {top_in['vol_ratio']:.2f}× · "
            f"RSI: {top_in['rsi']:.0f}</div>"
            f"<div style='font-size:0.78rem;color:#16a34a;margin-top:4px'>"
            f"{top_in['signal']} · {top_in['driver']}</div></div>",
            unsafe_allow_html=True)
    if top_out is not None:
        b2.markdown(
            f"<div style='border:2px solid #dc2626;border-radius:12px;"
            f"padding:14px 18px;background:rgba(220,38,38,0.04)'>"
            f"<div style='font-size:0.72rem;color:#64748b'>🧊 Strongest outflow</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:#dc2626'>"
            f"{top_out['sector']}</div>"
            f"<div style='font-size:0.82rem;color:#475569;margin-top:4px'>"
            f"Score: {int(top_out['flow_score'])} · "
            f"5d ret: {top_out['ret5']:+.2f}% · "
            f"Vol: {top_out['vol_ratio']:.2f}× · "
            f"RSI: {top_out['rsi']:.0f}</div>"
            f"<div style='font-size:0.78rem;color:#dc2626;margin-top:4px'>"
            f"{top_out['signal']} · {top_out['driver']}</div></div>",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Flow score bar chart ──────────────────────────────────────────
    st.markdown("#### Flow Score by sector")
    st.markdown(
        "<span style='color:#64748b;font-size:0.79rem'>"
        ">0 = money flowing in · <0 = money flowing out · "
        "Colour = smart money (blue) vs crowd (yellow) vs quiet (grey)</span>",
        unsafe_allow_html=True)

    bar_colors = [SECTOR_COLOR.get(s, "#94a3b8") for s in df_res["sector"]]
    # Override with driver colour for better signal
    driver_colors = [results[s]["driver_color"] for s in df_res["sector"]]

    fig_bar = go.Figure(go.Bar(
        x=df_res["flow_score"].astype(int),
        y=df_res["sector"],
        orientation="h",
        marker_color=driver_colors,
        opacity=0.85,
        text=[f"{int(s):+d}  {results[sec]['signal']}"
              for s,sec in zip(df_res["flow_score"], df_res["sector"])],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Flow score: %{x}<br>"
            "<extra></extra>"
        )))
    fig_bar.add_vline(x=0, line_color="#e2e8f0", line_width=2)
    fig_bar.update_layout(
        height=max(350, len(df_res)*36),
        margin=dict(l=0,r=120,t=10,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Flow Score", gridcolor="#f1f5f9", range=[-110,140]),
        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Multi-signal heatmap ──────────────────────────────────────────
    st.markdown("#### Signal heatmap — sector × indicator")
    st.markdown(
        "<span style='color:#64748b;font-size:0.79rem'>"
        "Each cell shows the normalised signal strength. "
        "Green = bullish/inflow · Red = bearish/outflow. "
        "Strong inflow = green across all three columns.</span>",
        unsafe_allow_html=True)

    # Normalise each signal -1 to +1 for display
    def norm(s, lo, hi):
        return (s-lo)/(hi-lo)*2-1  # -1 to +1

    hm_sectors = df_res["sector"].tolist()
    hm_vol   = [norm(float(results[s]["vol_ratio"]),   0.3, 2.5) for s in hm_sectors]
    hm_mom   = [norm(float(results[s]["ret5"]),        -8,  8)   for s in hm_sectors]
    hm_rsi   = [norm(100-float(results[s]["rsi"]),     0,  100)  for s in hm_sectors]  # inverted: low RSI = green
    hm_score = [norm(float(results[s]["flow_score"]), -100,100)  for s in hm_sectors]

    z = np.array([hm_vol, hm_mom, hm_rsi, hm_score]).T  # sectors × signals

    vol_txt  = [f"{float(results[s]['vol_ratio']):.2f}×" for s in hm_sectors]
    mom_txt  = [f"{float(results[s]['ret5']):+.1f}%" for s in hm_sectors]
    rsi_txt  = [f"RSI {float(results[s]['rsi']):.0f}" for s in hm_sectors]
    score_txt= [f"{int(float(results[s]['flow_score'])):+d}" for s in hm_sectors]
    text_arr = np.array([vol_txt, mom_txt, rsi_txt, score_txt]).T

    fig_hm = go.Figure(go.Heatmap(
        z=z,
        x=["Volume", "5d Return", "RSI (inv.)", "Flow Score"],
        y=hm_sectors,
        text=text_arr,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=[
            [0.0, "rgb(220,38,38)"],
            [0.35,"rgb(254,202,202)"],
            [0.5, "rgb(243,244,246)"],
            [0.65,"rgb(187,247,208)"],
            [1.0, "rgb(22,163,74)"],
        ],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="Signal", thickness=12,
                      tickvals=[-1,0,1],
                      ticktext=["Outflow","Neutral","Inflow"]),
        hovertemplate=(
            "<b>%{y} — %{x}</b><br>"
            "%{text}<extra></extra>"
        )))
    fig_hm.update_layout(
        height=max(300, len(hm_sectors)*38),
        margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="white",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    # ── Rotation detection ────────────────────────────────────────────
    st.markdown("#### 🔄 Rotation signals")
    st.markdown(
        "<span style='color:#64748b;font-size:0.79rem'>"
        "When one sector has strong inflow AND another has strong outflow "
        "simultaneously, capital is rotating from one theme to another.</span>",
        unsafe_allow_html=True)

    strong_in  = df_res[df_res["flow_score"] >= 30]
    strong_out = df_res[df_res["flow_score"] <= -25]

    if not strong_in.empty and not strong_out.empty:
        for _, r_in in strong_in.iterrows():
            for _, r_out in strong_out.iterrows():
                st.markdown(
                    f"<div style='border:1px solid #e2e8f0;border-radius:8px;"
                    f"padding:10px 14px;margin:4px 0;display:flex;gap:12px;"
                    f"align-items:center'>"
                    f"<span style='color:#dc2626;font-weight:600'>"
                    f"← OUT: {r_out['sector']} ({int(r_out['flow_score']):+d})</span>"
                    f"<span style='color:#94a3b8'>→</span>"
                    f"<span style='color:#16a34a;font-weight:600'>"
                    f"→ IN: {r_in['sector']} ({int(r_in['flow_score']):+d})</span>"
                    f"<span style='font-size:0.75rem;color:#64748b'>"
                    f"Rotation opportunity</span></div>",
                    unsafe_allow_html=True)
    elif strong_in.empty and strong_out.empty:
        st.info("No strong rotation signals right now — flows are balanced.")
    else:
        if not strong_in.empty:
            secs = ", ".join(strong_in["sector"].tolist())
            st.success(f"**Capital flowing into:** {secs} — but no clear outflow sector identified.")
        if not strong_out.empty:
            secs = ", ".join(strong_out["sector"].tolist())
            st.warning(f"**Capital leaving:** {secs} — but destination not yet clear.")

    st.markdown("---")

    # ── Smart money concentration ─────────────────────────────────────
    st.markdown("#### 🏦 Smart Money vs Crowd concentration")

    sm_sectors   = [s for s,m in results.items() if "Smart" in m["driver"] and m["flow_score"]>10]
    crowd_sectors= [s for s,m in results.items() if "Crowd" in m["driver"] and m["flow_score"]>10]
    quiet_sectors= [s for s,m in results.items() if "Quiet" in m["driver"]]

    rc1,rc2,rc3=st.columns(3)
    rc1.markdown(
        f"<div style='background:rgba(37,99,235,0.05);border:1px solid #2563eb;"
        f"border-radius:10px;padding:12px 14px'>"
        f"<div style='font-size:0.75rem;font-weight:600;color:#2563eb'>🏦 Smart Money Active</div>"
        f"<div style='margin-top:6px'>"
        + "".join(f"<div style='font-size:0.8rem;color:#0f172a'>• {s}</div>"
                  for s in sm_sectors)
        + ("<div style='font-size:0.78rem;color:#94a3b8;font-style:italic'>None detected</div>"
           if not sm_sectors else "")
        + "</div></div>", unsafe_allow_html=True)

    rc2.markdown(
        f"<div style='background:rgba(245,158,11,0.05);border:1px solid #f59e0b;"
        f"border-radius:10px;padding:12px 14px'>"
        f"<div style='font-size:0.75rem;font-weight:600;color:#f59e0b'>👥 Crowd Chasing</div>"
        f"<div style='margin-top:6px'>"
        + "".join(f"<div style='font-size:0.8rem;color:#0f172a'>• {s}</div>"
                  for s in crowd_sectors)
        + ("<div style='font-size:0.78rem;color:#94a3b8;font-style:italic'>None detected</div>"
           if not crowd_sectors else "")
        + "</div></div>", unsafe_allow_html=True)

    rc3.markdown(
        f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
        f"border-radius:10px;padding:12px 14px'>"
        f"<div style='font-size:0.75rem;font-weight:600;color:#94a3b8'>😴 Quiet / Unloved</div>"
        f"<div style='margin-top:6px'>"
        + "".join(f"<div style='font-size:0.8rem;color:#64748b'>• {s}</div>"
                  for s in quiet_sectors[:5])
        + ("<div style='font-size:0.78rem;color:#94a3b8;font-style:italic'>None</div>"
           if not quiet_sectors else "")
        + "</div></div>", unsafe_allow_html=True)

    # Contrarian note
    if crowd_sectors:
        st.markdown(
            f"<div style='border-left:3px solid #f59e0b;padding:10px 14px;"
            f"background:rgba(245,158,11,0.04);border-radius:0 8px 8px 0;margin-top:12px;"
            f"font-size:0.82rem;color:#475569'>"
            f"<b style='color:#f59e0b'>⚠️ Contrarian note:</b> "
            f"{', '.join(crowd_sectors)} — retail crowd is chasing these. "
            f"Crowd concentration at highs often precedes reversals. "
            f"If RSI is also overbought, risk of sharp correction is elevated.</div>",
            unsafe_allow_html=True)

    if quiet_sectors:
        st.markdown(
            f"<div style='border-left:3px solid #2563eb;padding:10px 14px;"
            f"background:rgba(37,99,235,0.04);border-radius:0 8px 8px 0;margin-top:8px;"
            f"font-size:0.82rem;color:#475569'>"
            f"<b style='color:#2563eb'>💡 Opportunity note:</b> "
            f"{', '.join(quiet_sectors[:3])} — low volume, unloved. "
            f"These are the sectors smart money quietly enters before the crowd notices.</div>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Full data table ───────────────────────────────────────────────
    with st.expander("📋 Full sector data table"):
        tbl=[]
        for s in df_res["sector"]:
            m=results[s]
            tbl.append({
                "Sector":      s,
                "Flow Score":  f"{int(m['flow_score']):+d}",
                "Signal":      m["signal"],
                "Driver":      m["driver"],
                "5d Return":   f"{m['ret5']:+.2f}%",
                "20d Return":  f"{m['ret20']:+.2f}%",
                "Vol Ratio":   f"{m['vol_ratio']:.2f}×",
                "RSI":         f"{m['rsi']:.0f}",
                "Trend/day":   f"{m['trend']:+.3f}%",
                "Avg Range %": f"{m['avg_range']:.2f}%",
                "# Tickers":   m["n_tickers"],
            })
        df_tbl=pd.DataFrame(tbl)
        def style_tbl(df):
            s=pd.DataFrame("",index=df.index,columns=df.columns)
            for i,row in df.iterrows():
                sc=int(str(row["Flow Score"]))
                if sc>=30:
                    s.at[i,"Flow Score"]="color:#16a34a;font-weight:700"
                    s.at[i,"Signal"]="color:#16a34a"
                elif sc<=-25:
                    s.at[i,"Flow Score"]="color:#dc2626;font-weight:700"
                    s.at[i,"Signal"]="color:#dc2626"
                for c in ["5d Return","20d Return"]:
                    v=str(row[c])
                    if v.startswith("+"): s.at[i,c]="color:#16a34a"
                    elif v.startswith("-"): s.at[i,c]="color:#dc2626"
            return s
        st.dataframe(df_tbl.style.apply(style_tbl,axis=None),
                     use_container_width=True,hide_index=True)

    st.markdown(
        "<span style='color:#94a3b8;font-size:0.74rem'>"
        "Flow score = volume(40%) + momentum(35%) + RSI contrarian(20%). "
        "Representative tickers per sector — not exhaustive. "
        "Data via yfinance · Not financial advice.</span>",
        unsafe_allow_html=True)
