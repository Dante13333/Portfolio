"""
portfolio_study.py — Multi-Asset Portfolio Study
A standalone research tool — study any mix of stocks, forex, commodities.
Not limited to current holdings. Compare efficiency, risk, trend health.

Metrics tuned for range traders:
  - Sharpe-like (return per unit of volatility)
  - Range capture (how much of available swing you'd capture)
  - Trend trap score (is it swinging or just falling?)
  - Max drawdown, win rate, avg daily range
  - Correlation matrix (are you doubling up on the same risk?)
  - Recommendations: what to add/drop/watch
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import time, pytz, warnings
warnings.filterwarnings("ignore")

HK_TZ = pytz.timezone("Asia/Hong_Kong")

# ── PRESET UNIVERSE ───────────────────────────────────────────────────
PRESETS = {
    # ── HKEX — full universe per sector ──────────────────────────────
    "🚀 HKEX — AI / Tech": {
        "0700.HK":"Tencent",       "9988.HK":"Alibaba",
        "9999.HK":"NetEase",       "9888.HK":"Baidu",
        "1024.HK":"Kuaishou",      "0020.HK":"SenseTime",
        "1810.HK":"Xiaomi",        "0992.HK":"Lenovo",
        "0763.HK":"ZTE",           "2382.HK":"Sunny Optical",
        "0268.HK":"Kingsoft",      "1347.HK":"Hua Hong Semi",
        "0100.HK":"MiniMax",       "2513.HK":"Zhipu",
        "0285.HK":"BYD Elec",      "6606.HK":"Kanzhun",
        "0522.HK":"ASM Pacific",   "0981.HK":"SMIC",
        "2121.HK":"iQIYI",         "0777.HK":"NetDragon",
        "1548.HK":"Genscript",     "6699.HK":"Burning Rock",
    },
    "⚡ HKEX — EV / Auto": {
        "9866.HK":"NIO",           "9868.HK":"Xpeng",
        "2015.HK":"Li Auto",       "0175.HK":"Geely",
        "1211.HK":"BYD",           "2238.HK":"GAC Group",
        "0285.HK":"BYD Elec",      "3750.HK":"CATL",
        "0489.HK":"Dongfeng",      "1519.HK":"FAW Group",
        "1776.HK":"Ganfeng Lithium","0136.HK":"Hengdelai",
        "2727.HK":"Shanghai Elec", "0816.HK":"Met Corp",
    },
    "🧬 HKEX — Biotech / Healthcare": {
        "2269.HK":"Wuxi Bio",      "6160.HK":"BeiGene",
        "1093.HK":"CSPC Pharma",   "1177.HK":"Sino Biopharm",
        "0241.HK":"Ali Health",    "0867.HK":"CSPC",
        "2196.HK":"Shanghai Pharma","1833.HK":"PA Health",
        "2552.HK":"HUTCHMED",      "6998.HK":"Hua Medicine",
        "1530.HK":"3SBio",         "0460.HK":"Sihuan Pharma",
        "6618.HK":"JD Health",     "1548.HK":"Genscript",
        "0853.HK":"Microport",     "1877.HK":"Sinopharm",
        "0570.HK":"China Trad Med","2616.HK":"Kinetic Bio",
    },
    "🎮 HKEX — Consumer / Gaming": {
        "0027.HK":"Galaxy Ent",    "1928.HK":"Sands China",
        "0880.HK":"SJM Holdings",  "1967.HK":"MGM China",
        "6862.HK":"Haidilao",      "9618.HK":"JD.com",
        "9961.HK":"Trip.com",      "0291.HK":"CR Beer",
        "2020.HK":"Anta Sports",   "1054.HK":"Boyaa",
        "9896.HK":"Miniso",        "9987.HK":"Yum China",
        "6110.HK":"Topsports",     "1368.HK":"Xtep",
        "0551.HK":"Yue Yuen",      "3690.HK":"Meituan",
    },
    "🌐 HKEX — New Economy / Internet": {
        "0241.HK":"Ali Health",    "1833.HK":"PA Health",
        "9988.HK":"Alibaba",       "1024.HK":"Kuaishou",
        "0780.HK":"Tongcheng",     "6606.HK":"Kanzhun",
        "6690.HK":"Haier Smart",   "0270.HK":"GD Investment",
        "0220.HK":"Uni-President", "0322.HK":"Tingyi",
        "1044.HK":"Hengan Intl",   "2232.HK":"Crystal Int",
    },
    "🏦 HKEX — Finance / Exchange": {
        "0388.HK":"HKEX",          "2318.HK":"Ping An",
        "1299.HK":"AIA",           "0005.HK":"HSBC",
        "0939.HK":"CCB",           "1398.HK":"ICBC",
        "3968.HK":"CMB",           "2388.HK":"BOC HK",
        "0011.HK":"Hang Seng Bank","1336.HK":"New China Life",
        "6886.HK":"HTSC",          "6958.HK":"CICC",
        "3988.HK":"Bank of China", "2628.HK":"China Life",
        "0966.HK":"China Taiping", "2601.HK":"CPIC",
    },
    "⚒ HKEX — Resources / Mining": {
        "2899.HK":"Zijin Mining",  "1208.HK":"MMG",
        "0883.HK":"CNOOC",         "0857.HK":"PetroChina",
        "0386.HK":"Sinopec",       "1088.HK":"China Shenhua",
        "3993.HK":"China Moly",    "1816.HK":"CGN Power",
        "0836.HK":"CR Power",      "2688.HK":"ENN Energy",
        "0384.HK":"China Gas",     "3800.HK":"GCL Solar",
        "0916.HK":"Longyuan Power","0750.HK":"Wanguo Mining",
    },
    "🏠 HKEX — Property / REIT": {
        "0016.HK":"Sun Hung Kai",  "0001.HK":"CK Asset",
        "0012.HK":"Henderson Land","1109.HK":"CR Land",
        "3383.HK":"Agile Group",   "2202.HK":"Vanke",
        "0823.HK":"Link REIT",     "0435.HK":"Sunlight REIT",
        "0101.HK":"Hang Lung Prop","0017.HK":"New World Dev",
        "0083.HK":"Sino Land",     "0659.HK":"NWS Holdings",
    },
    # ── Global High-Beta ─────────────────────────────────────────────
    "🇺🇸 US — Mega Tech": {
        "NVDA":"Nvidia",         "TSLA":"Tesla",
        "META":"Meta",           "MSFT":"Microsoft",
        "AAPL":"Apple",          "AMZN":"Amazon",
        "GOOGL":"Alphabet",
    },
    "🇺🇸 US — High Beta / Momentum": {
        "SMCI":"Super Micro",    "PLTR":"Palantir",
        "MSTR":"MicroStrategy",  "COIN":"Coinbase",
        "RKLB":"Rocket Lab",     "LUNR":"Intuitive Machines",
        "RDDT":"Reddit",         "HOOD":"Robinhood",
    },
    "🇺🇸 US — Semiconductors": {
        "NVDA":"Nvidia",         "AMD":"AMD",
        "AVGO":"Broadcom",       "QCOM":"Qualcomm",
        "MU":"Micron",           "AMAT":"Applied Materials",
        "ASML":"ASML",           "INTC":"Intel",
    },
    "🇺🇸 US — Biotech / Pharma": {
        "MRNA":"Moderna",        "BNTX":"BioNTech",
        "CRSP":"CRISPR",         "EDIT":"Editas",
        "BEAM":"Beam Therapeutics","RXRX":"Recursion",
        "ARKG":"ARK Genomics ETF",
    },
    "🪙 Crypto-linked": {
        "MSTR":"MicroStrategy",  "COIN":"Coinbase",
        "MARA":"Marathon Digital","RIOT":"Riot Platforms",
        "HUT":"Hut 8",           "CLSK":"CleanSpark",
        "BTC-USD":"Bitcoin",     "ETH-USD":"Ethereum",
        "SOL-USD":"Solana",
    },
    # ── Forex ─────────────────────────────────────────────────────────
    "💱 Major Forex": {
        "USDHKD=X":"USD/HKD",   "EURUSD=X":"EUR/USD",
        "GBPUSD=X":"GBP/USD",   "USDJPY=X":"USD/JPY",
        "USDCNY=X":"USD/CNY",   "AUDUSD=X":"AUD/USD",
        "USDCNH=X":"USD/CNH",   "EURJPY=X":"EUR/JPY",
    },
    "💱 EM / High-Vol Forex": {
        "USDKRW=X":"USD/KRW",   "USDINR=X":"USD/INR",
        "USDBRL=X":"USD/BRL",   "USDMXN=X":"USD/MXN",
        "USDZAR=X":"USD/ZAR",   "USDTRY=X":"USD/TRY",
    },
    # ── Commodities ───────────────────────────────────────────────────
    "🥇 Precious Metals": {
        "GC=F":"Gold",           "SI=F":"Silver",
        "PL=F":"Platinum",       "PA=F":"Palladium",
    },
    "🛢 Energy": {
        "CL=F":"WTI Crude",      "BZ=F":"Brent",
        "NG=F":"Nat Gas",        "RB=F":"RBOB Gasoline",
        "HO=F":"Heating Oil",
    },
    "🌽 Agriculture": {
        "ZC=F":"Corn",           "ZW=F":"Wheat",
        "ZS=F":"Soybeans",       "KC=F":"Coffee",
        "SB=F":"Sugar",          "CT=F":"Cotton",
    },
    "🏭 Industrial Metals": {
        "HG=F":"Copper",         "ALI=F":"Aluminium",
        "ZN=F":"Zinc",
    },
    # ── Indices (via ETFs) ────────────────────────────────────────────
    "📊 Global Indices (ETFs)": {
        "^HSI":"Hang Seng Index","^HSCE":"H-Share Index",
        "SPY":"S&P 500 ETF",     "QQQ":"Nasdaq 100 ETF",
        "EEM":"EM ETF",          "FXI":"China Large Cap ETF",
        "KWEB":"China Internet ETF","ARKK":"ARK Innovation ETF",
    },
}

TYPE_MAP = {
    "=X":"Forex", "=F":"Commodity",
    ".HK":"Stock (HK)", "":"Stock (US)",
}

def get_type(ticker):
    if ticker.endswith("=X"): return "Forex"
    if ticker.endswith("=F"): return "Commodity"
    if ticker.endswith(".HK"): return "Stock (HK)"
    return "Stock (US)"

TYPE_COLOR = {
    "Stock (HK)":"#2563eb", "Stock (US)":"#0891b2",
    "Forex":"#8b5cf6",      "Commodity":"#f59e0b",
}

# ── DATA ──────────────────────────────────────────────────────────────
def _variants(ticker):
    v=[ticker]; code=ticker.replace(".HK","")
    if code.isdigit():
        v.append(str(int(code))+".HK"); v.append(code.zfill(4)+".HK")
    return list(dict.fromkeys(v))

@st.cache_data(ttl=300, show_spinner=False)
def load_history(ticker, period="6mo"):
    for t in _variants(ticker):
        try:
            df=yf.Ticker(t).history(period=period,interval="1d",auto_adjust=True)
            if len(df)>=15:
                df.index=pd.to_datetime(df.index)
                return df, t
            time.sleep(0.2)
        except Exception: pass
    return pd.DataFrame(), ticker

@st.cache_data(ttl=300, show_spinner=False)
def load_intraday(ticker, period="30d"):
    for t in _variants(ticker):
        try:
            df=yf.Ticker(t).history(period=period,interval="60m",auto_adjust=True)
            if not df.empty:
                df.index=pd.to_datetime(df.index)
                if df.index.tzinfo is None:
                    df.index=df.index.tz_localize("UTC")
                df.index=df.index.tz_convert(HK_TZ)
                return df
        except Exception: pass
    return pd.DataFrame()

# ── METRICS ───────────────────────────────────────────────────────────
def _chop(df, p=14):
    if len(df)<p+2: return None
    tr=pd.concat([df["High"]-df["Low"],
                  (df["High"]-df["Close"].shift()).abs(),
                  (df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    ci=100*np.log10(tr.rolling(p).sum()/(
        df["High"].rolling(p).max()-df["Low"].rolling(p).min()+1e-9))/np.log10(p)
    return float(ci.clip(0,100).iloc[-1])

def _rsi(s, p=14):
    d=s.diff(); g=d.clip(lower=0).ewm(com=p-1,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(com=p-1,adjust=False).mean()
    r=100-100/(1+g/l.replace(0,np.nan))
    v=r.dropna()
    return float(v.iloc[-1]) if len(v) else 50

def compute_metrics(ticker, df, name):
    if df is None or len(df)<15:
        return None
    closes=df["Close"]; highs=df["High"]; lows=df["Low"]
    rets=closes.pct_change().dropna()
    ranges=highs-lows

    price_now    = float(closes.iloc[-1])
    vol_daily    = float(rets.std())
    ann_vol      = vol_daily*np.sqrt(252)*100
    avg_range    = float(ranges.mean())
    avg_range_pct= avg_range/float(closes.mean())*100
    total_range  = float(ranges.sum())
    win_rate     = float((rets>0).mean()*100)

    # Max drawdown
    peak_=np.maximum.accumulate(closes.values)
    max_dd=float(((closes.values-peak_)/peak_*100).min())

    # Sharpe-like (uses period return, not annualised, for fair comparison)
    period_ret=(closes.iloc[-1]-closes.iloc[0])/closes.iloc[0]*100
    sharpe=period_ret/ann_vol if ann_vol>0 else 0

    # Range capture potential (how much of the total range could be captured
    # if trading every day — theoretical max efficiency)
    range_cap_potential=total_range/closes.iloc[0]*100

    # Choppiness
    chop=_chop(df) or 50

    # RSI
    rsi_now=_rsi(closes)

    # Trend metrics
    x_=np.arange(len(closes))
    slope_=float(np.polyfit(x_,closes.values,1)[0])
    slope_pct=slope_/float(closes.mean())*100

    ma20=float(closes.rolling(20).mean().iloc[-1]) if len(closes)>=20 else float(closes.mean())
    ma50=float(closes.rolling(50).mean().iloc[-1]) if len(closes)>=50 else None

    below_ma20=price_now<ma20
    below_ma50=(price_now<ma50) if ma50 else None

    last5=closes.pct_change().tail(5)
    consec_down=int((last5<0).sum())

    recent_r=ranges.tail(10).mean()
    older_r =ranges.iloc[-30:-10].mean() if len(ranges)>=30 else ranges.mean()
    range_shrink=(recent_r-older_r)/older_r*100 if older_r>0 else 0

    highs_10=highs.tail(10).values
    lower_highs=sum(1 for i in range(1,len(highs_10)) if highs_10[i]<highs_10[i-1])

    # Trend trap score
    ts=0
    if slope_pct<-0.3:   ts+=25
    elif slope_pct<-0.1: ts+=12
    if below_ma20:        ts+=20
    if below_ma50 is True:ts+=15
    if consec_down>=4:    ts+=20
    elif consec_down>=3:  ts+=10
    if chop<38:           ts+=15
    elif chop<45:         ts+=7
    if range_shrink<-20:  ts+=5
    if lower_highs>=7:    ts+=10
    ts=min(ts,100)

    if ts>=70:   trap="🔴 TREND TRAP"
    elif ts>=45: trap="🟡 WEAK SWING"
    elif ts>=20: trap="🟢 OK SWING"
    else:        trap="✅ ACTIVE SWING"

    # Volume activity
    avg_vol=float(df["Volume"].tail(5).mean())
    hist_vol=float(df["Volume"].mean())
    vol_ratio=avg_vol/hist_vol if hist_vol>0 else 1

    return {
        "ticker":          ticker,
        "name":            name,
        "type":            get_type(ticker),
        "price":           price_now,
        "period_ret":      round(period_ret,2),
        "ann_vol":         round(ann_vol,1),
        "sharpe":          round(sharpe,3),
        "avg_range":       round(avg_range,2),
        "avg_range_pct":   round(avg_range_pct,3),
        "range_cap_pot":   round(range_cap_potential,1),
        "max_dd":          round(max_dd,1),
        "win_rate":        round(win_rate,1),
        "chop":            round(chop,1),
        "rsi":             round(rsi_now,1),
        "vol_ratio":       round(vol_ratio,2),
        "trend_score":     ts,
        "trend_label":     trap,
        "slope_pct":       round(slope_pct,4),
        "below_ma20":      below_ma20,
        "below_ma50":      below_ma50,
        "consec_down":     consec_down,
        "range_shrink":    round(range_shrink,1),
        "lower_highs":     lower_highs,
        "df":              df,
    }

def _safe(v, fmt, suffix="", fallback="—"):
    try: return format(float(v), fmt)+suffix if v is not None else fallback
    except: return fallback

# ═════════════════════════════════════════════════════════════════════
# RENDER
# ═════════════════════════════════════════════════════════════════════
def render():
    now_hk=datetime.now(HK_TZ)
    st.markdown(
        "## 📐 Portfolio Study &nbsp;"
        "<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        "padding:2px 7px;border-radius:5px'>RESEARCH TOOL</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"Study any mix of stocks · forex · commodities · "
        f"Not limited to current holdings · "
        f"{now_hk.strftime('%Y-%m-%d %H:%M HKT')}</span>",
        unsafe_allow_html=True)
    st.markdown("---")

    # ── Instrument selection ──────────────────────────────────────────
    st.markdown("### 1 · Select instruments to study")

    # ── Source tabs ───────────────────────────────────────────────────
    selected = {}

    sel_presets=st.multiselect(
        "Add from presets",
        list(PRESETS.keys()),
        default=["🚀 HKEX — AI / Tech","⚡ HKEX — EV / Auto",
                 "💱 Major Forex","🥇 Precious Metals"],
        key="ps_presets")
    custom_raw=st.text_input(
        "Add custom tickers (comma-separated)",
        placeholder="0100.HK, 2513.HK, NVDA, GC=F, BTC-USD",
        key="ps_custom")
    for p_ in sel_presets:
        selected.update(PRESETS[p_])
    if custom_raw:
        for t in custom_raw.split(","):
            t=t.strip().upper()
            if t: selected[t]=t

    # ── Summary + run ─────────────────────────────────────────────────
    st.markdown("---")
    if not selected:
        st.info("Select instruments from the tabs above, then click Run Study.")
        return

    period = st.selectbox("History period", ["3mo","6mo","1y","2y"],
                           index=1, key="ps_period")

    col_run,col_clear=st.columns(2)
    run=col_run.button("🔬 Run Study",key="ps_run",type="primary")
    if col_clear.button("🗑 Clear results",key="ps_clear"):
        for k in ["ps_results","ps_hv_results"]:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

    if run:
        results=[]
        prog=st.progress(0,"Fetching data…")
        items=list(selected.items())
        for i,(ticker,name) in enumerate(items):
            prog.progress((i+1)/len(items),
                          text=f"Loading {name} ({ticker})…")
            df_,_=load_history(ticker,period)
            m=compute_metrics(ticker,df_,name)
            if m: results.append(m)
            time.sleep(0.2)
        prog.empty()
        st.session_state["ps_results"]=results

    results=st.session_state.get("ps_results",[])
    if not results:
        return

    st.markdown("---")
    st.markdown("### 2 · Results")

    # ── Summary table ─────────────────────────────────────────────────
    st.markdown("#### Overview table")
    st.markdown(
        "<span style='color:#64748b;font-size:0.77rem'>"
        "Hover column names for explanations. "
        "Green = good for range trading. Red = risk / avoid. "
        "Click **What do these metrics mean?** below for full glossary.</span>",
        unsafe_allow_html=True)
    tbl=[]
    for m in results:
        tbl.append({
            "Status":        m["trend_label"],
            "Name":          m["name"],
            "Ticker":        m["ticker"],
            "Type":          m["type"],
            "Price":         _safe(m["price"],".4f") if m["price"]<10 else _safe(m["price"],",.2f"),
            "Period ret %":  _safe(m["period_ret"],"+.2f","%"),
            "Ann vol %":     _safe(m["ann_vol"],".1f","%"),
            "Sharpe-like":   _safe(m["sharpe"],"+.3f"),
            "Avg range":     _safe(m["avg_range"],".2f"),
            "Range %":       _safe(m["avg_range_pct"],".3f","%"),
            "Range pot %":   _safe(m["range_cap_pot"],".1f","%"),
            "Max DD %":      _safe(m["max_dd"],".1f","%"),
            "Win rate %":    _safe(m["win_rate"],".0f","%"),
            "Choppiness":    _safe(m["chop"],".0f"),
            "RSI":           _safe(m["rsi"],".0f"),
            "Trend score":   str(m["trend_score"]),
        })

    df_tbl=pd.DataFrame(tbl)

    def style_tbl(df):
        s=pd.DataFrame("",index=df.index,columns=df.columns)
        for i,row in df.iterrows():
            # Colour entire row by trend label
            status=str(row["Status"])
            if "TREND TRAP" in status:
                for c in ["Status","Trend score"]:
                    s.at[i,c]="color:#dc2626;font-weight:700"
            elif "WEAK" in status:
                s.at[i,"Status"]="color:#f59e0b;font-weight:600"
            elif "ACTIVE" in status:
                s.at[i,"Status"]="color:#16a34a;font-weight:600"
            # Colour return / sharpe
            for c in ["Period ret %","Sharpe-like"]:
                v=str(row.get(c,""))
                if v.startswith("+"): s.at[i,c]="color:#16a34a;font-weight:600"
                elif v.startswith("-"): s.at[i,c]="color:#dc2626;font-weight:600"
            # Max DD
            try:
                dd=float(str(row["Max DD %"]).replace("%",""))
                if dd<-25: s.at[i,"Max DD %"]="color:#dc2626;font-weight:600"
            except: pass
            # Choppiness
            try:
                ch=float(str(row["Choppiness"]))
                if ch>61.8: s.at[i,"Choppiness"]="color:#16a34a"
                elif ch<38: s.at[i,"Choppiness"]="color:#dc2626"
            except: pass
        return s

    st.dataframe(df_tbl.style.apply(style_tbl,axis=None),
                 use_container_width=True,hide_index=True)


    with st.expander("📖 What do these metrics mean? (click to expand)"):
        st.markdown("""
**Status / Trend label** — Is this instrument swinging or trending down?
- **ACTIVE SWING**: choppiness high, oscillating — ideal for range trading
- **OK SWING**: slight bias but tradeable both ways
- **WEAK SWING**: swing weakening, watch carefully
- **TREND TRAP**: trending down — no upper swing to capture. Your biggest risk.

**Period ret %** — Total price return over the study period.
For a swing trader this is less important than volatility — but deeply negative return + high
volatility = falling volatile instrument, the worst combination.

**Ann vol %** — Annualised daily volatility (daily std dev x sqrt(252) x 100).
Higher = moves more violently. Good for range trading IF matched with high choppiness.
Bad if the volatility is all one-directional (trending down).

**Sharpe-like** — Period return divided by annualised volatility.
How much did you earn per unit of risk taken?
Positive + high (>0.5): efficient. Near zero: wasted capital. Negative: worst outcome.

**Avg range** — Average daily High minus Low in price units.
The raw swing available to capture per day. Bigger = more opportunity per trade.

**Range %** — Average daily range as % of price.
Normalises so you can compare instruments at different price levels.

**Range pot %** — Total range offered over the period divided by starting price x 100.
Theoretical maximum capturable swing if you traded perfectly every day.
High = lots of opportunity. Compare to your actual return to see how much you captured.

**Max DD %** — Worst peak-to-trough drop during the period.
Above 20%: you held a losing position too long.
A swing trade should be cut at your stop — large drawdown means stops were not respected.

**Win rate %** — % of days price closed higher than it opened.
Above 55% = mild upward bias. Below 45% = mild downward bias.
Win rate alone is not enough — size of wins vs losses matters more.

**Choppiness** — Measures whether price is trending or oscillating (0-100).
Above 61.8: oscillating, good for range trading.
Below 38.2: trending one way, hard to trade both sides.
38-62: mixed.

**RSI** — Momentum indicator (0-100).
Above 70: overbought, reversal risk. Below 30: oversold, bounce possible. 40-60: neutral.

**Trend score** — Composite 0-100 from 6 signals: price slope, position vs MA20/MA50,
consecutive down days, choppiness, range shrinkage, lower highs.
0 = healthy swing. Above 70 = TREND TRAP, avoid.

**Study score** — Ranking score for range-trading suitability.
Range potential (35%) + Choppiness (30%) + Volume (15%) + Trend health (20%).
Trend traps are heavily penalised. Higher = better for your style.
        """)
    st.markdown("---")

    # ── Scatter: Risk vs Swing Opportunity ───────────────────────────
    st.markdown("#### Efficiency map — Swing opportunity vs Risk")
    st.markdown(
        "<span style='color:#64748b;font-size:0.79rem'>"
        "X = annual volatility (higher = riskier) · "
        "Y = range capture potential (higher = more swing to capture) · "
        "Colour = trend health · Size = Sharpe-like score · "
        "**Top-left = best for range trading** (big swings, lower volatility risk)</span>",
        unsafe_allow_html=True)

    fig_sc=go.Figure()
    for m in results:
        if m["ann_vol"] is None: continue
        tc={"🔴 TREND TRAP":"#dc2626","🟡 WEAK SWING":"#f59e0b",
            "🟢 OK SWING":"#16a34a","✅ ACTIVE SWING":"#2563eb"}.get(
            m["trend_label"],TYPE_COLOR.get(m["type"],"#94a3b8"))
        sz=max(8,min(abs(m["sharpe"])*40+8,40))
        fig_sc.add_trace(go.Scatter(
            x=[m["ann_vol"]], y=[m["range_cap_pot"]],
            mode="markers+text",
            text=[m["name"][:10]],textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=sz,color=tc,opacity=0.85,
                        line=dict(color="white",width=1.5)),
            hovertemplate=(
                f"<b>{m['name']} ({m['ticker']})</b><br>"
                f"Ann vol: {m['ann_vol']:.1f}%<br>"
                f"Range potential: {m['range_cap_pot']:.1f}%<br>"
                f"Avg daily range: {m['avg_range']:.2f}<br>"
                f"Sharpe-like: {m['sharpe']:+.3f}<br>"
                f"Choppiness: {m['chop']:.0f}<br>"
                f"Status: {m['trend_label']}<extra></extra>"),
            showlegend=False))
    fig_sc.update_layout(
        height=420,margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="white",paper_bgcolor="white",
        xaxis=dict(title="Annual volatility %",gridcolor="#f1f5f9"),
        yaxis=dict(title="Range capture potential %",gridcolor="#f1f5f9"))
    st.plotly_chart(fig_sc,use_container_width=True)

    # ── Top picks for range trading ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏆 Best for your style — ranked for range trading")
    st.markdown(
        "<span style='color:#64748b;font-size:0.79rem'>"
        "Score = range potential(35%) + choppiness(30%) + vol ratio(15%) + "
        "trend health(20%) — penalises trend traps heavily</span>",
        unsafe_allow_html=True)

    for m in results:
        chop_s  = min(m["chop"]/100,1)*100
        range_s = min(m["range_cap_pot"]/200,1)*100
        vol_s   = min(m["vol_ratio"]/3,1)*100
        trap_pen= max(0,(100-m["trend_score"]))/100
        m["study_score"]=round(
            (range_s*0.35 + chop_s*0.30 + vol_s*0.15)*trap_pen*
            (1 if m["period_ret"]>=0 else 0.7), 1)

    top=sorted(results,key=lambda x:-x["study_score"])

    rank_cols=st.columns(min(len(top),4))
    for i,m in enumerate(top[:4]):
        col=rank_cols[i]
        tc=TYPE_COLOR.get(m["type"],"#94a3b8")
        tl_c={"🔴 TREND TRAP":"#dc2626","🟡 WEAK SWING":"#f59e0b",
               "🟢 OK SWING":"#16a34a","✅ ACTIVE SWING":"#2563eb"}.get(
              m["trend_label"],"#94a3b8")
        chop_c="#16a34a" if m["chop"]>61 else "#f59e0b" if m["chop"]>45 else "#dc2626"
        col.markdown(
            f"<div style='border:2px solid {tc};border-radius:12px;"
            f"padding:14px 16px;text-align:center;margin-bottom:8px'>"
            f"<div style='font-size:0.72rem;color:#94a3b8'>#{i+1} range pick</div>"
            f"<div style='font-weight:700;font-size:1rem;color:#0f172a'>{m['name']}</div>"
            f"<div style='font-size:0.72rem;color:{tc}'>{m['ticker']} · {m['type']}</div>"
            f"<div style='font-size:1.4rem;font-weight:800;color:{tc};margin:6px 0'>"
            f"{m['study_score']:.0f}</div>"
            f"<div style='font-size:0.72rem;color:#64748b'>study score</div>"
            f"<hr style='margin:8px 0;border-color:#f1f5f9'>"
            f"<div style='font-size:0.75rem;display:grid;grid-template-columns:1fr 1fr;gap:4px'>"
            f"<div><span style='color:#94a3b8'>Avg range</span><br>"
            f"<b>{m['avg_range']:.2f}</b></div>"
            f"<div><span style='color:#94a3b8'>Choppiness</span><br>"
            f"<b style='color:{chop_c}'>"
            f"{m['chop']:.0f}</b></div>"
            f"<div><span style='color:#94a3b8'>Ann vol</span><br>"
            f"<b>{m['ann_vol']:.1f}%</b></div>"
            f"<div><span style='color:#94a3b8'>Status</span><br>"
            f"<b style='color:{tl_c};font-size:0.68rem'>{m['trend_label']}</b></div>"
            f"</div></div>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Correlation matrix ────────────────────────────────────────────
    st.markdown("#### 🔗 Correlation matrix — are you doubling up on the same risk?")
    st.markdown(
        "<span style='color:#64748b;font-size:0.79rem'>"
        "High correlation (> 0.7) = positions move together — "
        "holding both gives you less diversification than you think. "
        "Negative correlation = natural hedge.</span>",
        unsafe_allow_html=True)

    # Build returns matrix
    ret_dict={}
    for m in results:
        df_=m.get("df")
        if df_ is not None and len(df_)>10:
            ret_dict[m["name"][:12]]=df_["Close"].pct_change().dropna()

    if len(ret_dict)>=2:
        ret_df=pd.DataFrame(ret_dict).dropna()
        if len(ret_df)>5:
            corr=ret_df.corr()
            labels=corr.columns.tolist()
            z=corr.values

            # Colour: red=high corr (bad), white=zero, green=negative (good)
            fig_corr=go.Figure(go.Heatmap(
                z=z,x=labels,y=labels,
                text=[[f"{v:.2f}" for v in row] for row in z],
                texttemplate="%{text}",textfont=dict(size=9),
                colorscale=[
                    [0.0,"rgb(22,163,74)"],
                    [0.5,"rgb(255,255,255)"],
                    [1.0,"rgb(220,38,38)"],
                ],
                zmid=0,zmin=-1,zmax=1,
                colorbar=dict(title="Corr",thickness=12)))
            fig_corr.update_layout(
                height=max(300,len(labels)*45),
                margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="white",
                xaxis=dict(tickangle=30),
                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_corr,use_container_width=True)

            # Flag high correlations
            high_corr=[]
            for i in range(len(labels)):
                for j in range(i+1,len(labels)):
                    c_=float(corr.iloc[i,j])
                    if abs(c_)>0.75:
                        high_corr.append((labels[i],labels[j],round(c_,2)))
            if high_corr:
                st.markdown("**⚠️ Highly correlated pairs (|corr| > 0.75):**")
                for a,b,c_ in sorted(high_corr,key=lambda x:-abs(x[2])):
                    color="#dc2626" if c_>0 else "#16a34a"
                    st.markdown(
                        f"<span style='color:{color}'>"
                        f"{'⚠️' if c_>0 else '↔️'} **{a}** × **{b}** = {c_:+.2f} — "
                        f"{'holding both gives little extra diversification' if c_>0 else 'natural hedge'}"
                        f"</span>",unsafe_allow_html=True)
    else:
        st.info("Need at least 2 instruments with data for correlation analysis.")

    st.markdown("---")

    # ── Trend trap summary ────────────────────────────────────────────
    st.markdown("#### 🚨 Trend trap summary")
    traps   =[m for m in results if m["trend_score"]>=70]
    weaks   =[m for m in results if 45<=m["trend_score"]<70]
    healthy =[m for m in results if m["trend_score"]<45]

    tc1,tc2,tc3=st.columns(3)
    tc1.metric("🔴 Trend traps", len(traps),
               delta="Avoid — no swing to capture" if traps else "None")
    tc2.metric("🟡 Weak swings", len(weaks),
               delta="Monitor carefully" if weaks else "None")
    tc3.metric("✅ Healthy swings",len(healthy))

    if traps:
        for m in traps:
            reasons=[]
            if m["slope_pct"]<-0.3: reasons.append(f"falling {abs(m['slope_pct']):.3f}%/day")
            if m["below_ma20"]:     reasons.append("below MA20")
            if m["below_ma50"] is True: reasons.append("below MA50")
            if m["consec_down"]>=4: reasons.append(f"{m['consec_down']} consec down days")
            if m["chop"]<45:        reasons.append(f"choppiness {m['chop']:.0f} — trending")
            st.markdown(
                f"<div style='border:2px solid #dc2626;border-radius:10px;"
                f"padding:12px 16px;margin:6px 0;background:rgba(220,38,38,0.03)'>"
                f"<b style='color:#dc2626'>🔴 {m['name']} ({m['ticker']}) "
                f"— score {m['trend_score']}/100</b><br>"
                f"<span style='font-size:0.8rem;color:#475569'>"
                f"{' · '.join(reasons)}</span><br>"
                f"<span style='font-size:0.78rem;color:#dc2626'>"
                f"Not suitable for range trading right now. "
                f"Wait for choppiness > 62 + price above MA20.</span></div>",
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Individual deep dives ─────────────────────────────────────────
    st.markdown("#### 🔍 Individual deep-dive")

    # Sort controls
    sv1, sv2 = st.columns(2)
    sort_by = sv1.selectbox("Sort by", [
        "Study score (best first)",
        "Avg range HKD (highest first)",
        "Choppiness (most oscillating first)",
        "Trend score (healthiest first)",
        "Trend score (most dangerous first)",
        "Ann volatility (highest first)",
        "Period return (best first)",
        "Win rate (highest first)",
        "Max drawdown (worst first)",
        "Name (A-Z)",
    ], key="ps_dive_sort")
    asc = sv2.radio("Order", ["Desc ↓", "Asc ↑"],
                    horizontal=True, key="ps_dive_order") == "Asc ↑"

    sort_map = {
        "Study score (best first)":           ("study_score",    False),
        "Avg range HKD (highest first)":      ("avg_range",      False),
        "Choppiness (most oscillating first)":("chop",           False),
        "Trend score (healthiest first)":     ("trend_score",    True),
        "Trend score (most dangerous first)": ("trend_score",    False),
        "Ann volatility (highest first)":     ("ann_vol",        False),
        "Period return (best first)":         ("period_ret",     False),
        "Win rate (highest first)":           ("win_rate",       False),
        "Max drawdown (worst first)":         ("max_dd",         True),
        "Name (A-Z)":                         ("name",           True),
    }
    sort_key, default_asc = sort_map[sort_by]
    reverse = not (asc if sv2 else default_asc)
    sorted_results = sorted(
        results,
        key=lambda x: (x[sort_key] if x[sort_key] is not None else -9999),
        reverse=reverse)



    for m in sorted_results:
        tl_c={"🔴 TREND TRAP":"#dc2626","🟡 WEAK SWING":"#f59e0b",
               "🟢 OK SWING":"#16a34a","✅ ACTIVE SWING":"#2563eb"}.get(
              m["trend_label"],"#94a3b8")
        with st.expander(
            f"{m['trend_label']}  **{m['name']}** ({m['ticker']}) · "
            f"Avg range {m['avg_range']:.2f} · Chop {m['chop']:.0f} · "
            f"Score {m['study_score']:.0f}",
            expanded=False):

            # Metrics row
            mc=st.columns(7)
            METRIC_TIPS = {
                "Period ret":  "Total return over study period. Negative = lost money.",
                "Ann vol":     "Annualised volatility. Higher = bigger moves but more risk.",
                "Sharpe-like": "Return per unit of risk. >0.5 = efficient. <0 = losing on volatile pos.",
                "Avg range":   "Avg daily High-Low. Bigger = more swing to capture per trade.",
                "Choppiness":  ">61.8 = oscillating (good). <38.2 = trending one way (bad for you).",
                "Max DD":      "Worst peak-to-trough drop. >20% = held losing trade too long.",
                "Win rate":    "% of days price closed up. 50% = neutral.",
            }
            for col,lbl,val,color in [
                (mc[0],"Period ret",   f"{m['period_ret']:+.2f}%",
                 "#16a34a" if m["period_ret"]>=0 else "#dc2626"),
                (mc[1],"Ann vol",      f"{m['ann_vol']:.1f}%","#f59e0b"),
                (mc[2],"Sharpe-like",  f"{m['sharpe']:+.3f}",
                 "#16a34a" if m["sharpe"]>0 else "#dc2626"),
                (mc[3],"Avg range",    f"{m['avg_range']:.2f}","#2563eb"),
                (mc[4],"Choppiness",   f"{m['chop']:.0f}",
                 "#16a34a" if m["chop"]>61 else "#f59e0b" if m["chop"]>45 else "#dc2626"),
                (mc[5],"Max DD",       f"{m['max_dd']:.1f}%",
                 "#dc2626" if m["max_dd"]<-20 else "#94a3b8"),
                (mc[6],"Win rate",     f"{m['win_rate']:.0f}%","#94a3b8"),
            ]:
                tip = METRIC_TIPS.get(lbl,"")
                col.markdown(
                    f"<div style='text-align:center;padding:8px 4px;background:#f8fafc;"
                    f"border-radius:8px;border:1px solid #e2e8f0'>"
                    f"<div style='font-size:0.65rem;color:#94a3b8'>{lbl}</div>"
                    f"<div style='font-size:0.95rem;font-weight:700;color:{color}'>{val}</div>"
                    f"<div style='font-size:0.6rem;color:#94a3b8;margin-top:2px'>{tip}</div>"
                    f"</div>",unsafe_allow_html=True)

            st.markdown("<br>",unsafe_allow_html=True)

            # Chart
            df_=m.get("df")
            if df_ is not None and len(df_)>5:
                bc=["#16a34a" if c>=o else "#dc2626"
                    for c,o in zip(df_["Close"],df_["Open"])]
                fig_d=make_subplots(rows=3,cols=1,shared_xaxes=True,
                                    row_heights=[0.55,0.22,0.23],
                                    vertical_spacing=0.03)
                fig_d.add_trace(go.Candlestick(
                    x=df_.index,open=df_["Open"],high=df_["High"],
                    low=df_["Low"],close=df_["Close"],
                    increasing_line_color="#16a34a",
                    decreasing_line_color="#dc2626"),row=1,col=1)
                # MA lines
                if len(df_)>=20:
                    ma20_=df_["Close"].rolling(20).mean()
                    fig_d.add_trace(go.Scatter(x=df_.index,y=ma20_,
                        line=dict(color="#f59e0b",width=1.5,dash="dot"),
                        name="MA20"),row=1,col=1)
                if len(df_)>=50:
                    ma50_=df_["Close"].rolling(50).mean()
                    fig_d.add_trace(go.Scatter(x=df_.index,y=ma50_,
                        line=dict(color="#dc2626",width=1.5,dash="dot"),
                        name="MA50"),row=1,col=1)
                # Daily range bars
                ranges_=df_["High"]-df_["Low"]
                avg_r_=ranges_.mean()
                fig_d.add_trace(go.Bar(x=df_.index,y=ranges_,
                    marker_color=["#16a34a" if v>=avg_r_ else "#94a3b8" for v in ranges_],
                    opacity=0.8,name="Range"),row=2,col=1)
                fig_d.add_hline(y=float(avg_r_),line_dash="dot",
                                line_color="#2563eb",line_width=1,row=2,col=1)
                # Volume
                fig_d.add_trace(go.Bar(x=df_.index,y=df_["Volume"],
                    marker_color=bc,opacity=0.7,name="Vol"),row=3,col=1)
                rb=[dict(bounds=["sat","mon"])]
                fig_d.update_layout(
                    height=500,margin=dict(l=0,r=0,t=10,b=0),
                    xaxis_rangeslider_visible=False,
                    plot_bgcolor="white",paper_bgcolor="white",showlegend=False,
                    yaxis=dict(gridcolor="#f1f5f9"),
                    yaxis2=dict(title="Range",gridcolor="#f1f5f9"),
                    yaxis3=dict(title="Volume",gridcolor="#f1f5f9"),
                    xaxis3=dict(gridcolor="#f1f5f9",rangebreaks=rb),
                    xaxis2=dict(gridcolor="#f1f5f9",rangebreaks=rb),
                    xaxis=dict(gridcolor="#f1f5f9",rangebreaks=rb))
                st.plotly_chart(fig_d,use_container_width=True)

                # Range distribution
                fig_rh=go.Figure(go.Histogram(
                    x=ranges_,nbinsx=25,marker_color="#2563eb",opacity=0.75))
                fig_rh.add_vline(x=float(avg_r_),line_dash="dot",
                                  line_color="#f59e0b",line_width=2,
                                  annotation_text=f"Avg {avg_r_:.2f}",
                                  annotation_position="top right")
                fig_rh.update_layout(height=180,margin=dict(l=0,r=0,t=10,b=0),
                    plot_bgcolor="white",paper_bgcolor="white",
                    xaxis=dict(title="Daily range",gridcolor="#f1f5f9"),
                    yaxis=dict(title="Days",gridcolor="#f1f5f9"))
                st.plotly_chart(fig_rh,use_container_width=True)

    st.markdown(
        "<span style='color:#94a3b8;font-size:0.74rem'>"
        "Data via yfinance · Past performance does not guarantee future results · "
        "Not financial advice</span>",
        unsafe_allow_html=True)
