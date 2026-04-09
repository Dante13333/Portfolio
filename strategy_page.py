"""
strategy_page.py  —  Market Behaviour Study
Studies price/volume/indicator patterns from historical data.
Covers:
  1. Stock selector   — portfolio stocks + free search
  2. Big Move Setup   — what conditions precede large daily swings
  3. Weekday Study    — which days trend cleanest
  4. Session Study    — intraday hour-by-hour behaviour
  5. Gap Study        — fill rate, speed, size
  6. Confluence Study — when multiple signals align, what happens
  7. ML Strategy      — decision tree rules + pattern matching
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import pytz
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score

from db_manager import get_portfolio_full, get_latest_capital

HK_TZ = pytz.timezone("Asia/Hong_Kong")

DAY_NAMES  = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday"}
DAY_SHORT  = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
DAY_COLOR  = {0:"#dc2626",1:"#16a34a",2:"#16a34a",3:"#f59e0b",4:"#dc2626"}
SESSIONS   = {"Open 09:30-10:00":(9.5,10),"Morning 10:00-11:30":(10,11.5),
              "Afternoon 13:00-14:30":(13,14.5),"Close 14:30-16:00":(14.5,16)}

# ── INDICATORS ────────────────────────────────────────────────────────
def rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0).ewm(com=p-1,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(com=p-1,adjust=False).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def macd(s,fast=12,slow=26,sig=9):
    ml=s.ewm(span=fast,adjust=False).mean()-s.ewm(span=slow,adjust=False).mean()
    sl=ml.ewm(span=sig,adjust=False).mean(); return ml,sl,ml-sl

def bb_pct(s,p=20):
    mid=s.rolling(p).mean(); std=s.rolling(p).std()
    return ((s-mid+2*std)/(4*std+1e-9)*100).clip(0,100)

def choppiness(df,p=14):
    if len(df)<p+2: return pd.Series(50,index=df.index)
    tr=pd.concat([df["High"]-df["Low"],
                  (df["High"]-df["Close"].shift()).abs(),
                  (df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    ci=100*np.log10(tr.rolling(p).sum()/(
        df["High"].rolling(p).max()-df["Low"].rolling(p).min()+1e-9))/np.log10(p)
    return ci.clip(0,100)

def atr(df,p=14):
    tr=pd.concat([df["High"]-df["Low"],
                  (df["High"]-df["Close"].shift()).abs(),
                  (df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(com=p-1,adjust=False).mean()

# ── DATA ──────────────────────────────────────────────────────────────

def _ticker_variants(ticker: str) -> list:
    """Return ticker format variants to try for HKEX (handles leading zeros)."""
    variants = [ticker]
    code = ticker.replace(".HK", "")
    if code.isdigit():
        variants.append(str(int(code)) + ".HK")   # no leading zeros: 0100 -> 100
        variants.append(code.zfill(4) + ".HK")     # 4-digit pad: 100 -> 0100
    return list(dict.fromkeys(variants))             # deduplicate

@st.cache_data(ttl=300, show_spinner=False)
def load_daily(ticker:str, period:str="1y") -> pd.DataFrame:
    for t in _ticker_variants(ticker):
        try:
            df=yf.Ticker(t).history(period=period,interval="1d",auto_adjust=True)
            if len(df)>=5:
                df.index=pd.to_datetime(df.index)
                return df
            time.sleep(0.3)
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def load_intraday(ticker:str, period:str="30d") -> pd.DataFrame:
    for t in _ticker_variants(ticker):
        try:
            df=yf.Ticker(t).history(period=period,interval="60m",auto_adjust=True)
            if df.empty:
                time.sleep(0.3)
                continue
            df.index=pd.to_datetime(df.index)
            if df.index.tzinfo is None: df.index=df.index.tz_localize("UTC")
            df.index=df.index.tz_convert(HK_TZ)
            return df
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_ticker_name(ticker:str) -> str:
    for t in _ticker_variants(ticker):
        try:
            info=yf.Ticker(t).info
            name=info.get("longName") or info.get("shortName")
            if name: return name
            time.sleep(0.3)
        except Exception:
            continue
    return ticker

def build_daily_features(df:pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df)<15: return pd.DataFrame()
    f=pd.DataFrame(index=df.index)
    f["close"]      = df["Close"]
    f["open"]       = df["Open"]
    f["high"]       = df["High"]
    f["low"]        = df["Low"]
    f["volume"]     = df["Volume"]
    f["range_hkd"]  = df["High"]-df["Low"]
    f["body_pct"]   = (df["Close"]-df["Open"])/df["Open"]*100
    f["day_ret"]    = df["Close"].pct_change()*100
    f["gap_pct"]    = (df["Open"]-df["Close"].shift())/df["Close"].shift()*100
    f["upper_wick"] = (df["High"]-df[["Open","Close"]].max(axis=1))/(f["range_hkd"]+1e-9)*100
    f["lower_wick"] = (df[["Open","Close"]].min(axis=1)-df["Low"])/(f["range_hkd"]+1e-9)*100
    avg_vol         = df["Volume"].rolling(20).mean()
    f["vol_ratio"]  = df["Volume"]/avg_vol.replace(0,np.nan)
    f["rsi"]        = rsi(df["Close"])
    ml,sl,hist      = macd(df["Close"])
    f["macd_hist"]  = hist
    f["macd_bull"]  = ((ml>sl)&(ml.shift()<=sl.shift())).astype(int)
    f["macd_bear"]  = ((ml<sl)&(ml.shift()>=sl.shift())).astype(int)
    _bb_p = min(20, max(5, len(df)//5))
    f["bb_pct"]     = bb_pct(df["Close"], _bb_p)
    _chop_p = min(14, max(5, len(df)//5))
    f["chop"]       = choppiness(df, _chop_p)
    f["atr"]        = atr(df)
    f["weekday"]    = df.index.dayofweek
    f["week_num"]   = df.index.isocalendar().week.astype(int)
    # Next-day outcomes
    f["next_range"] = (df["High"]-df["Low"]).shift(-1)
    f["next_ret"]   = df["Close"].pct_change().shift(-1)*100
    f["next_up"]    = (f["next_ret"]>0).astype(int)
    f["next_big"]   = (f["next_range"]>=20).astype(int)
    # Gap fill (did next day reach prev close?)
    prev_close      = df["Close"]
    f["gap_filled"] = np.where(
        f["gap_pct"]>0, (df["Low"].shift(-1)<=prev_close).astype(int),
        np.where(f["gap_pct"]<0, (df["High"].shift(-1)>=prev_close).astype(int), np.nan))
    return f.dropna(subset=["rsi","vol_ratio"])

# ── HELPERS ───────────────────────────────────────────────────────────
def pct_bar(v, color="#2563eb", max_w=100):
    w=min(abs(v),max_w)
    return (f"<div style='background:#f1f5f9;border-radius:4px;height:8px;overflow:hidden'>"
            f"<div style='width:{w:.0f}%;height:100%;background:{color};border-radius:4px'>"
            f"</div></div>")

def stat_card(col, label, value, sub="", color="#0f172a"):
    col.markdown(
        f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
        f"padding:12px 14px;text-align:center'>"
        f"<div style='font-size:0.7rem;color:#64748b'>{label}</div>"
        f"<div style='font-size:1.2rem;font-weight:700;color:{color}'>{value}</div>"
        f"{'<div style=font-size:0.72rem;color:#94a3b8>' + sub + '</div>' if sub else ''}"
        f"</div>", unsafe_allow_html=True)

def color_wr(v):
    if v>=60: return "#16a34a"
    if v<=40: return "#dc2626"
    return "#f59e0b"

def feature_bar_chart(names, values, title="", color="#2563eb"):
    fig=go.Figure(go.Bar(y=names,x=values,orientation="h",
        marker_color=color,opacity=0.8,
        text=[f"{v:.1f}" for v in values],textposition="outside"))
    fig.update_layout(height=max(200,len(names)*32),
        margin=dict(l=0,r=60,t=24,b=0),title=dict(text=title,font=dict(size=12)),
        plot_bgcolor="white",paper_bgcolor="white",
        xaxis=dict(gridcolor="#f1f5f9"),yaxis=dict(autorange="reversed"))
    return fig

# ── ML FEATURE COLS ───────────────────────────────────────────────────
ML_COLS=["body_pct","range_hkd","gap_pct","upper_wick","lower_wick",
         "vol_ratio","rsi","macd_hist","macd_bull","macd_bear",
         "bb_pct","chop","weekday"]
ML_NAMES={"body_pct":"Candle body %","range_hkd":"Day range HKD",
           "gap_pct":"Gap %","upper_wick":"Upper wick %","lower_wick":"Lower wick %",
           "vol_ratio":"Volume vs 20d avg","rsi":"RSI","macd_hist":"MACD histogram",
           "macd_bull":"MACD bullish cross","macd_bear":"MACD bearish cross",
           "bb_pct":"BB position %","chop":"Choppiness","weekday":"Weekday (0=Mon)"}

def train_dt(feat, label, depth=4):
    X=feat[ML_COLS].fillna(0); y=feat[label]
    if len(y.unique())<2 or len(y)<20: return None,None
    clf=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=5,
                                class_weight="balanced",random_state=42)
    clf.fit(X,y)
    try:
        cv=cross_val_score(clf,X,y,cv=min(5,len(y)//10),scoring="accuracy")
        acc=float(cv.mean())
    except: acc=float((clf.predict(X)==y).mean())
    return clf,acc

def predict_now(clf,feat):
    if clf is None or feat.empty: return None,None
    last=feat[ML_COLS].fillna(0).iloc[[-1]]
    return int(clf.predict(last)[0]), clf.predict_proba(last)[0]

def imp_chart(clf,title):
    imp=clf.feature_importances_
    names=[ML_NAMES.get(c,c) for c in ML_COLS]
    order=np.argsort(imp)[::-1][:8]
    return feature_bar_chart([names[i] for i in order],
                              [imp[i]*100 for i in order],title,"#2563eb")

# ── PATTERN MATCHING ──────────────────────────────────────────────────
def find_similar_weeks(feat,df,n=5):
    if len(feat)<15: return []
    keys=["body_pct","gap_pct","vol_ratio","rsi","chop","bb_pct","macd_hist","weekday"]
    data=feat[keys].fillna(0)
    mu=data.mean(); sd=data.std().replace(0,1)
    norm=(data-mu)/sd
    w=5; ref=norm.iloc[-w:].values.flatten()
    results=[]
    for i in range(len(norm)-w*2):
        cand=norm.iloc[i:i+w].values.flatten()
        if len(cand)!=len(ref): continue
        sim=float(np.dot(ref,cand)/(np.linalg.norm(ref)*np.linalg.norm(cand)+1e-9))
        fwd=feat.iloc[i+w:i+w+5]
        if len(fwd)<3: continue
        results.append({
            "sim":round(sim,3),"start":feat.index[i],"end":feat.index[i+w-1],
            "fwd_ret":round(float(fwd["day_ret"].sum()),2),
            "fwd_range":round(float(fwd["range_hkd"].mean()),1),
            "up_days":int((fwd["day_ret"]>0).sum()),
            "idx":i,"fwd_idx":i+w,
        })
    return sorted(results,key=lambda x:-x["sim"])[:n]

# ════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ════════════════════════════════════════════════════════════════════
def render():
    now_hk=datetime.now(HK_TZ)
    st.markdown(
        "## 📊 Market Behaviour Study &nbsp;"
        "<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        "padding:2px 7px;border-radius:5px'>DATA-DRIVEN</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"Study price · volume · indicator patterns from real historical data · "
        f"HKT {now_hk.strftime('%Y-%m-%d %H:%M')}</span>",
        unsafe_allow_html=True)
    with st.expander("📖 Metric and ML explanations"):
        st.markdown("""
**Big Move Setups** — Compares market conditions on days with large ranges vs quiet days.
Bigger difference in a metric = stronger signal. E.g. if vol ratio is 2.1x on big days vs 0.9x
on quiet days, volume is a strong predictor of big moves for this instrument.

**RSI zone** — Which RSI level produces the most big next-day swings?
Oversold (<30) often precedes violent bounces. Overbought (>70) can precede sharp drops.
Both extremes tend to produce bigger ranges than neutral RSI.

**Weekday win rate** — Adjusted for current momentum (RSI, week-to-date return, choppiness).
Monday and Friday adjusted downward when market is overbought (trap risk higher).
Tuesday-Wednesday adjusted upward when oversold (bounce probability higher).

**Decision Tree rules** — The model splits historical data at the single threshold that
best separates UP days from DOWN days at each node. Read top to bottom:
each line is a condition; the leaf (end of branch) shows the predicted outcome and
the percentage of matching historical days that went that direction.

**Pattern matching similarity** — Cosine similarity of the last 5 days' normalised feature
vector vs all historical 5-day windows. 1.0 = identical pattern. >0.8 = very similar.
What happened the following week tells you what this pattern historically leads to.

**Gap fill curve** — How quickly Monday gaps fill on average for this instrument.
If 70% fill by Wednesday, your rule of not chasing Monday opens is well-supported by data.
        """)

    st.markdown("---")

    # ── Stock selector ───────────────────────────────────────────────
    st.markdown("### 🔎 Select Stock to Study")
    port=get_portfolio_full()
    port_tickers=port["ticker"].tolist() if not port.empty else []
    port_names={r["ticker"]:r.get("name",r["ticker"])
                for _,r in port.iterrows()} if not port.empty else {}

    sc1,sc2,sc3=st.columns([2,1,1])
    if port_tickers:
        port_opts=["— Search a new stock —"]+[
            f"{t} — {port_names.get(t,t)}" for t in port_tickers]
        sel_port=sc1.selectbox("Portfolio stock",port_opts,key="strat_portsel")
        if sel_port!="— Search a new stock —":
            ticker_input=sel_port.split(" — ")[0]
        else:
            raw=sc1.text_input("Or type any HKEX ticker",
                               placeholder="e.g. 1810.HK",
                               key="strat_search").strip().upper()
            ticker_input=raw if raw.endswith(".HK") else (raw+".HK" if raw else "")
    else:
        raw=sc1.text_input("HKEX ticker (e.g. 0700.HK)",
                            placeholder="0700.HK",key="strat_search2").strip().upper()
        ticker_input=raw if raw.endswith(".HK") else (raw+".HK" if raw else "")

    period=sc2.selectbox("History to study",
                          ["3mo","6mo","1y","2y"],index=2,key="strat_period")
    tree_depth=sc3.slider("Tree depth",2,6,4,key="strat_depth")

    if st.button("🔄 Refresh data",key="strat_refresh"):
        st.cache_data.clear(); st.rerun()

    if not ticker_input or len(ticker_input)<5:
        st.info("👆 Select a portfolio stock or type a ticker to start studying.")
        return

    # ── Load data ────────────────────────────────────────────────────
    with st.spinner(f"Loading {ticker_input} history ({period})…"):
        df=load_daily(ticker_input,period)
        df_1h=load_intraday(ticker_input,"60d")
        name=get_ticker_name(ticker_input)

    min_rows = 20
    if df.empty:
        st.error(
            f"No data returned for **{ticker_input}**. "
            "Check the ticker format (e.g. `0700.HK`, `9988.HK`). "
            "Very new IPOs (< 1 month) may have limited data.")
        return
    if len(df) < min_rows:
        st.warning(
            f"**{ticker_input}** only has {len(df)} trading days of history "
            f"(minimum {min_rows} needed). "
            "This is a very new listing — some analysis sections will be limited. "
            "Try a shorter period like **3mo** or wait for more data to accumulate.")
        # Don't return — let it continue with reduced analysis

    feat=build_daily_features(df)
    if feat.empty:
        st.warning(
            f"Could not build enough features from {len(df)} days of data. "
            "Some indicators need at least 26 days (MACD) or 20 days (Bollinger Bands). "
            "The stock may be too new — try again in a few weeks.")
        return

    days = len(df)
    if days < 60:
        freshness = "🆕 Very new listing"
        note_color = "#f59e0b"
        note = (f"Only **{days} trading days** available. "
                "Gap & weekday analysis needs 60+ days. "
                "ML pattern matching needs 100+ days. "
                "Big move and confluence studies will work with limited data.")
    elif days < 120:
        freshness = "📅 Short history"
        note_color = "#2563eb"
        note = f"{days} trading days — most analysis works, ML pattern matching improves with more history."
    else:
        freshness = "✅ Good history"
        note_color = "#16a34a"
        note = f"{days} trading days — all analysis sections fully available."

    st.markdown(
        f"### Studying: **{name}** ({ticker_input}) · "
        f"<span style='color:{note_color}'>{freshness} · {days} days</span>",
        unsafe_allow_html=True)
    if days < 120:
        st.info(note)
    st.markdown("---")

    # ── TABS ─────────────────────────────────────────────────────────
    tabs=st.tabs([
        "📈 Big Move Setups",
        "📅 Weekday Patterns",
        "⏱ Session Patterns",
        "↕ Gap Behaviour",
        "🔗 Indicator Confluence",
        "🧠 ML Rules + Pattern Match",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — BIG MOVE SETUPS
    # ════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("#### What conditions precede big daily swings (≥20 HKD)?")
        st.markdown(
            "<span style='color:#64748b;font-size:0.8rem'>"
            "Comparing market state on big-swing days vs quiet days. "
            "Bigger difference = stronger signal.</span>",
            unsafe_allow_html=True)

        thr=st.slider("Big swing threshold (HKD)",5,60,20,key="strat_thr")
        big=feat[feat["range_hkd"]>=thr]
        quiet=feat[feat["range_hkd"]<thr]

        if len(big)<5:
            st.warning(f"Only {len(big)} big-swing days found. Try lowering the threshold.")
        else:
            bc1,bc2,bc3,bc4=st.columns(4)
            stat_card(bc1,"Big swing days",f"{len(big)}",
                      f"{len(big)/len(feat)*100:.0f}% of all days","#dc2626")
            stat_card(bc2,"Avg range on big days",
                      f"HKD {big['range_hkd'].mean():.1f}","","#16a34a")
            stat_card(bc3,"Avg vol ratio on big days",
                      f"{big['vol_ratio'].mean():.1f}×","vs quiet: "
                      f"{quiet['vol_ratio'].mean():.1f}×","#f59e0b")
            stat_card(bc4,"Avg RSI on big days",
                      f"{big['rsi'].mean():.1f}","","#2563eb")

            st.markdown("<br>",unsafe_allow_html=True)

            # Compare key metrics big vs quiet
            metrics=["vol_ratio","rsi","bb_pct","chop","gap_pct","atr","macd_hist"]
            m_names={"vol_ratio":"Volume ratio","rsi":"RSI","bb_pct":"BB position %",
                     "chop":"Choppiness","gap_pct":"Gap %","atr":"ATR","macd_hist":"MACD hist"}
            comp=pd.DataFrame({
                "Metric":   [m_names[m] for m in metrics],
                "Big day":  [round(big[m].mean(),2) for m in metrics],
                "Quiet day":[round(quiet[m].mean(),2) for m in metrics],
            })
            comp["Difference"]=comp["Big day"]-comp["Quiet day"]
            comp["Signal strength"]=comp["Difference"].abs()/comp["Quiet day"].abs().replace(0,1)*100

            # Bar chart comparison
            mc1,mc2=st.columns(2)
            with mc1:
                fig_comp=go.Figure()
                fig_comp.add_trace(go.Bar(name="Big day",x=comp["Metric"],
                    y=comp["Big day"],marker_color="#dc2626",opacity=0.8))
                fig_comp.add_trace(go.Bar(name="Quiet day",x=comp["Metric"],
                    y=comp["Quiet day"],marker_color="#94a3b8",opacity=0.8))
                fig_comp.update_layout(height=300,barmode="group",
                    margin=dict(l=0,r=0,t=24,b=0),
                    title=dict(text="Big-swing vs Quiet day conditions",font=dict(size=12)),
                    plot_bgcolor="white",paper_bgcolor="white",
                    legend=dict(orientation="h",y=1.1),
                    yaxis=dict(gridcolor="#f1f5f9"),
                    xaxis=dict(tickangle=30))
                st.plotly_chart(fig_comp,use_container_width=True)

            with mc2:
                # Range distribution
                fig_dist=go.Figure()
                fig_dist.add_trace(go.Histogram(x=feat["range_hkd"],nbinsx=30,
                    marker_color="#2563eb",opacity=0.7,name="All days"))
                fig_dist.add_vline(x=thr,line_dash="dot",line_color="#dc2626",
                    line_width=2,annotation_text=f"Threshold {thr}",
                    annotation_position="top right")
                fig_dist.add_vline(x=feat["range_hkd"].mean(),line_dash="dot",
                    line_color="#94a3b8",line_width=1,
                    annotation_text=f"Avg {feat['range_hkd'].mean():.0f}",
                    annotation_position="top left")
                fig_dist.update_layout(height=300,
                    margin=dict(l=0,r=0,t=24,b=0),
                    title=dict(text="Daily range distribution (HKD)",font=dict(size=12)),
                    plot_bgcolor="white",paper_bgcolor="white",
                    xaxis=dict(title="Range HKD",gridcolor="#f1f5f9"),
                    yaxis=dict(title="Days",gridcolor="#f1f5f9"))
                st.plotly_chart(fig_dist,use_container_width=True)

            # RSI zone × big move breakdown
            st.markdown("**RSI zone when big moves happen**")
            feat["rsi_zone"]=pd.cut(feat["rsi"],bins=[0,30,45,55,70,100],
                                     labels=["Oversold<30","30-45","45-55","55-70","OB>70"])
            rz=feat.groupby("rsi_zone",observed=True).agg(
                big_pct=("next_big","mean"),n=("next_big","count")).reset_index()
            rz["big_pct"]*=100
            fig_rz=go.Figure(go.Bar(
                x=rz["rsi_zone"].astype(str),y=rz["big_pct"],
                marker_color=["#16a34a","#2563eb","#94a3b8","#f59e0b","#dc2626"],
                text=[f"{v:.0f}%\nn={n}" for v,n in zip(rz["big_pct"],rz["n"])],
                textposition="outside"))
            fig_rz.add_hline(y=float(feat["next_big"].mean()*100),
                              line_dash="dot",line_color="#94a3b8",
                              annotation_text="Overall avg",annotation_position="right")
            fig_rz.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white",paper_bgcolor="white",
                xaxis=dict(title="RSI zone"),
                yaxis=dict(title="% next day ≥20 HKD",gridcolor="#f1f5f9",range=[0,100]))
            st.plotly_chart(fig_rz,use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — WEEKDAY PATTERNS
    # ════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("#### Which days of the week have the cleanest behaviour?")

        wd=feat.groupby("weekday").agg(
            n=("day_ret","count"),
            avg_ret=("day_ret","mean"),
            win_rate=("next_up","mean"),
            avg_range=("range_hkd","mean"),
            big_pct=("next_big","mean"),
            avg_vol=("vol_ratio","mean"),
            avg_chop=("chop","mean"),
            gap_fill=("gap_filled","mean"),
        ).reset_index()
        wd=wd[wd["weekday"]<=4]
        wd["win_rate"]*=100; wd["big_pct"]*=100; wd["gap_fill"]*=100
        wd["day"]=wd["weekday"].map(DAY_SHORT)

        # 4-panel weekday dashboard
        wp1,wp2=st.columns(2)
        with wp1:
            fig_wr=go.Figure(go.Bar(
                x=wd["day"],y=wd["win_rate"],
                marker_color=[color_wr(v) for v in wd["win_rate"]],
                text=[f"{v:.0f}%" for v in wd["win_rate"]],textposition="outside"))
            fig_wr.add_hline(y=50,line_dash="dot",line_color="#94a3b8",line_width=1)
            fig_wr.update_layout(height=240,margin=dict(l=0,r=0,t=24,b=0),
                title=dict(text="Win rate by weekday (next day up %)",font=dict(size=12)),
                plot_bgcolor="white",paper_bgcolor="white",
                yaxis=dict(range=[0,100],gridcolor="#f1f5f9"))
            st.plotly_chart(fig_wr,use_container_width=True)

        with wp2:
            fig_rng=go.Figure(go.Bar(
                x=wd["day"],y=wd["avg_range"],
                marker_color=[DAY_COLOR[d] for d in wd["weekday"]],
                text=[f"HKD {v:.1f}" for v in wd["avg_range"]],textposition="outside"))
            fig_rng.add_hline(y=float(feat["range_hkd"].mean()),
                               line_dash="dot",line_color="#94a3b8",line_width=1,
                               annotation_text="Overall avg")
            fig_rng.update_layout(height=240,margin=dict(l=0,r=0,t=24,b=0),
                title=dict(text="Avg daily range by weekday (HKD)",font=dict(size=12)),
                plot_bgcolor="white",paper_bgcolor="white",
                yaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_rng,use_container_width=True)

        wp3,wp4=st.columns(2)
        with wp3:
            fig_vol=go.Figure(go.Bar(
                x=wd["day"],y=wd["avg_vol"],
                marker_color="#8b5cf6",opacity=0.8,
                text=[f"{v:.2f}×" for v in wd["avg_vol"]],textposition="outside"))
            fig_vol.add_hline(y=1,line_dash="dot",line_color="#94a3b8",line_width=1)
            fig_vol.update_layout(height=240,margin=dict(l=0,r=0,t=24,b=0),
                title=dict(text="Avg volume ratio by weekday",font=dict(size=12)),
                plot_bgcolor="white",paper_bgcolor="white",
                yaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_vol,use_container_width=True)

        with wp4:
            fig_big=go.Figure(go.Bar(
                x=wd["day"],y=wd["big_pct"],
                marker_color=["#dc2626" if v>=50 else "#f59e0b" if v>=30 else "#94a3b8"
                               for v in wd["big_pct"]],
                text=[f"{v:.0f}%" for v in wd["big_pct"]],textposition="outside"))
            fig_big.update_layout(height=240,margin=dict(l=0,r=0,t=24,b=0),
                title=dict(text="% of days with next-day range ≥20 HKD",font=dict(size=12)),
                plot_bgcolor="white",paper_bgcolor="white",
                yaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_big,use_container_width=True)

        # Return distribution by weekday (box plot)
        st.markdown("**Return distribution by weekday**")
        fig_box=go.Figure()
        for wd_i in range(5):
            d=feat[feat["weekday"]==wd_i]["day_ret"].dropna()
            if len(d)<3: continue
            fig_box.add_trace(go.Box(y=d,name=DAY_SHORT[wd_i],
                marker_color=DAY_COLOR[wd_i],boxpoints="outliers"))
        fig_box.add_hline(y=0,line_color="#e2e8f0",line_width=1)
        fig_box.update_layout(height=280,margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white",paper_bgcolor="white",
            yaxis=dict(title="Daily return %",gridcolor="#f1f5f9"))
        st.plotly_chart(fig_box,use_container_width=True)

        # Summary table
        with st.expander("📋 Full weekday stats table"):
            disp=wd[["day","n","avg_ret","win_rate","avg_range","big_pct","avg_vol","avg_chop"]].copy()
            disp.columns=["Day","Days","Avg Ret %","Win Rate %","Avg Range HKD",
                          "% Big Swing","Vol Ratio","Choppiness"]
            st.dataframe(disp.style.format({
                "Avg Ret %":"{:+.2f}","Win Rate %":"{:.0f}%","Avg Range HKD":"{:.1f}",
                "% Big Swing":"{:.0f}%","Vol Ratio":"{:.2f}×","Choppiness":"{:.1f}",
            }),use_container_width=True,hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 3 — SESSION PATTERNS
    # ════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("#### Which intraday session produces the real move?")

        if df_1h.empty:
            st.info("Intraday data unavailable. Try again during or after market hours.")
        else:
            df_1h["hour"]=df_1h.index.hour+(df_1h.index.minute/60)
            df_1h["ret_pct"]=(df_1h["Close"]-df_1h["Open"])/df_1h["Open"]*100
            df_1h["range_pct"]=(df_1h["High"]-df_1h["Low"])/df_1h["Open"]*100
            df_1h["weekday"]=df_1h.index.dayofweek
            df_1h["up"]=(df_1h["Close"]>=df_1h["Open"]).astype(int)
            avg_vol_h=df_1h["Volume"].mean()
            df_1h["vol_r"]=df_1h["Volume"]/avg_vol_h

            # Filter to trading hours
            trading=df_1h[(df_1h["hour"]>=9.5)&(df_1h["hour"]<16)&
                          (df_1h["hour"]<11.5)|
                          (df_1h[(df_1h["hour"]>=9.5)&(df_1h["hour"]<16)]["hour"]>=13)]
            trading=df_1h[((df_1h["hour"]>=9.5)&(df_1h["hour"]<11.5))|
                           ((df_1h["hour"]>=13)&(df_1h["hour"]<16))]

            hour_stats=trading.groupby(trading["hour"].apply(lambda h:int(h))).agg(
                avg_range=("range_pct","mean"),
                win_rate=("up","mean"),
                avg_vol=("vol_r","mean"),
                n=("ret_pct","count"),
            ).reset_index()
            hour_stats["win_rate"]*=100
            hour_labels={9:"09:xx",10:"10:xx",13:"13:xx",14:"14:xx",15:"15:xx"}
            hour_stats["hour_lbl"]=hour_stats["hour"].map(lambda h:f"{h:02d}:xx")

            sp1,sp2=st.columns(2)
            with sp1:
                fig_hr_vol=go.Figure(go.Bar(
                    x=hour_stats["hour_lbl"],y=hour_stats["avg_vol"],
                    marker_color=["#dc2626" if v>=2 else "#f59e0b" if v>=1.3 else "#94a3b8"
                                  for v in hour_stats["avg_vol"]],
                    text=[f"{v:.2f}×" for v in hour_stats["avg_vol"]],
                    textposition="outside"))
                fig_hr_vol.add_hline(y=1,line_dash="dot",line_color="#94a3b8",line_width=1)
                fig_hr_vol.update_layout(height=260,margin=dict(l=0,r=0,t=24,b=0),
                    title=dict(text="Volume intensity by hour (×avg)",font=dict(size=12)),
                    plot_bgcolor="white",paper_bgcolor="white",
                    yaxis=dict(gridcolor="#f1f5f9"))
                st.plotly_chart(fig_hr_vol,use_container_width=True)

            with sp2:
                fig_hr_wr=go.Figure(go.Bar(
                    x=hour_stats["hour_lbl"],y=hour_stats["win_rate"],
                    marker_color=[color_wr(v) for v in hour_stats["win_rate"]],
                    text=[f"{v:.0f}%" for v in hour_stats["win_rate"]],
                    textposition="outside"))
                fig_hr_wr.add_hline(y=50,line_dash="dot",line_color="#94a3b8",line_width=1)
                fig_hr_wr.update_layout(height=260,margin=dict(l=0,r=0,t=24,b=0),
                    title=dict(text="% of hours that closed green",font=dict(size=12)),
                    plot_bgcolor="white",paper_bgcolor="white",
                    yaxis=dict(range=[0,100],gridcolor="#f1f5f9"))
                st.plotly_chart(fig_hr_wr,use_container_width=True)

            # Heatmap: weekday × hour win rate
            st.markdown("**Weekday × Hour win rate heatmap**")
            pivot=trading.groupby([trading.index.dayofweek,
                                   trading["hour"].apply(lambda h:int(h))])["up"].mean()*100
            pivot=pivot.unstack(fill_value=np.nan)
            y_lbl=[DAY_SHORT.get(i,str(i)) for i in pivot.index if i<=4]
            x_lbl=[f"{h:02d}:xx" for h in pivot.columns]
            z=pivot.values[:5] if len(pivot)>=5 else pivot.values

            txt=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in z]
            fig_hm=go.Figure(go.Heatmap(
                z=z,x=x_lbl,y=y_lbl[:len(z)],
                text=txt,texttemplate="%{text}",textfont=dict(size=11),
                colorscale=[[0,"rgb(220,38,38)"],[0.4,"rgb(254,202,202)"],
                            [0.5,"rgb(243,244,246)"],[0.6,"rgb(187,247,208)"],
                            [1,"rgb(22,163,74)"]],
                zmid=50,zmin=20,zmax=80,
                colorbar=dict(title="% up",thickness=12)))
            fig_hm.update_layout(height=220,margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="white",
                xaxis=dict(title="Hour"),
                yaxis=dict(title="Weekday",autorange="reversed"))
            st.plotly_chart(fig_hm,use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4 — GAP BEHAVIOUR
    # ════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("#### Do gaps fill? How fast? How big?")

        gaps=feat[feat["gap_pct"].abs()>0.3].copy()
        gaps_up=gaps[gaps["gap_pct"]>0]
        gaps_dn=gaps[gaps["gap_pct"]<0]

        gb1,gb2,gb3,gb4=st.columns(4)
        stat_card(gb1,"Total gaps analysed",str(len(gaps)),"gap >0.3%")
        fill_rate=float(gaps["gap_filled"].mean()*100) if len(gaps) else 0
        stat_card(gb2,"Overall fill rate",f"{fill_rate:.0f}%","filled same day",
                  "#16a34a" if fill_rate>50 else "#dc2626")
        up_fill=float(gaps_up["gap_filled"].mean()*100) if len(gaps_up) else 0
        dn_fill=float(gaps_dn["gap_filled"].mean()*100) if len(gaps_dn) else 0
        stat_card(gb3,"Gap-up fill rate",f"{up_fill:.0f}%",f"n={len(gaps_up)}","#f59e0b")
        stat_card(gb4,"Gap-down fill rate",f"{dn_fill:.0f}%",f"n={len(gaps_dn)}","#8b5cf6")

        st.markdown("<br>",unsafe_allow_html=True)
        gp1,gp2=st.columns(2)

        with gp1:
            # Gap size distribution
            fig_gd=go.Figure()
            fig_gd.add_trace(go.Histogram(x=gaps_up["gap_pct"],nbinsx=20,
                name="Gap up",marker_color="#16a34a",opacity=0.7))
            fig_gd.add_trace(go.Histogram(x=gaps_dn["gap_pct"],nbinsx=20,
                name="Gap down",marker_color="#dc2626",opacity=0.7))
            fig_gd.update_layout(height=260,barmode="overlay",
                margin=dict(l=0,r=0,t=24,b=0),
                title=dict(text="Gap size distribution (%)",font=dict(size=12)),
                plot_bgcolor="white",paper_bgcolor="white",
                xaxis=dict(title="Gap %",gridcolor="#f1f5f9"),
                yaxis=dict(title="Frequency",gridcolor="#f1f5f9"),
                legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_gd,use_container_width=True)

        with gp2:
            # Fill rate by gap size bucket
            gaps["gap_bucket"]=pd.cut(gaps["gap_pct"].abs(),
                bins=[0,0.5,1,1.5,2,5,100],
                labels=["0-0.5%","0.5-1%","1-1.5%","1.5-2%","2-5%",">5%"])
            gbkt=gaps.groupby("gap_bucket",observed=True).agg(
                fill_rate=("gap_filled","mean"),n=("gap_filled","count")).reset_index()
            gbkt["fill_rate"]*=100
            fig_gf=go.Figure(go.Bar(
                x=gbkt["gap_bucket"].astype(str),y=gbkt["fill_rate"],
                marker_color=[color_wr(v) for v in gbkt["fill_rate"]],
                text=[f"{v:.0f}%\nn={n}" for v,n in zip(gbkt["fill_rate"],gbkt["n"])],
                textposition="outside"))
            fig_gf.add_hline(y=50,line_dash="dot",line_color="#94a3b8",line_width=1)
            fig_gf.update_layout(height=260,
                margin=dict(l=0,r=0,t=24,b=0),
                title=dict(text="Same-day fill rate by gap size",font=dict(size=12)),
                plot_bgcolor="white",paper_bgcolor="white",
                xaxis=dict(title="Gap size"),
                yaxis=dict(title="Fill rate %",gridcolor="#f1f5f9",range=[0,100]))
            st.plotly_chart(fig_gf,use_container_width=True)

        # Gap by weekday
        st.markdown("**Gap fill rate by weekday**")
        gwd=gaps.groupby("weekday").agg(
            fill_rate=("gap_filled","mean"),
            avg_gap=("gap_pct","mean"),
            n=("gap_filled","count")).reset_index()
        gwd=gwd[gwd["weekday"]<=4]
        gwd["fill_rate"]*=100
        fig_gwd=go.Figure(go.Bar(
            x=[DAY_SHORT[d] for d in gwd["weekday"]],y=gwd["fill_rate"],
            marker_color=[DAY_COLOR[d] for d in gwd["weekday"]],
            text=[f"{v:.0f}%\nn={n}" for v,n in zip(gwd["fill_rate"],gwd["n"])],
            textposition="outside"))
        fig_gwd.add_hline(y=50,line_dash="dot",line_color="#94a3b8",line_width=1)
        fig_gwd.update_layout(height=240,margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white",paper_bgcolor="white",
            yaxis=dict(title="Fill rate %",gridcolor="#f1f5f9",range=[0,100]))
        st.plotly_chart(fig_gwd,use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 5 — INDICATOR CONFLUENCE
    # ════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("#### When multiple signals align — what happens?")
        st.markdown(
            "<span style='color:#64748b;font-size:0.8rem'>"
            "Filter by any combination of conditions and see the historical outcome.</span>",
            unsafe_allow_html=True)

        cf1,cf2,cf3=st.columns(3)
        rsi_filter=cf1.selectbox("RSI condition",
            ["Any","Oversold (<30)","Near oversold (30-45)",
             "Neutral (45-55)","Near overbought (55-70)","Overbought (>70)"],
            key="conf_rsi")
        vol_filter=cf2.selectbox("Volume condition",
            ["Any","Spike (>2×)","Above avg (1.3-2×)","Normal (0.7-1.3×)","Low (<0.7×)"],
            key="conf_vol")
        wd_filter=cf3.multiselect("Weekday",
            ["Mon","Tue","Wed","Thu","Fri"],default=[],key="conf_wd")

        cf4,cf5=st.columns(2)
        chop_filter=cf4.selectbox("Choppiness",
            ["Any","Oscillating (>61.8)","Mixed (45-61.8)","Trending (<45)"],
            key="conf_chop")
        gap_filter=cf5.selectbox("Gap",
            ["Any","Gap up (>0.5%)","Gap down (<-0.5%)","Flat (within 0.5%)"],
            key="conf_gap")

        # Apply filters
        mask=pd.Series(True,index=feat.index)
        if rsi_filter!="Any":
            if "Oversold" in rsi_filter:      mask&=feat["rsi"]<30
            elif "Near oversold" in rsi_filter: mask&=(feat["rsi"]>=30)&(feat["rsi"]<45)
            elif "Neutral" in rsi_filter:      mask&=(feat["rsi"]>=45)&(feat["rsi"]<55)
            elif "Near overbought" in rsi_filter:mask&=(feat["rsi"]>=55)&(feat["rsi"]<70)
            elif "Overbought" in rsi_filter:   mask&=feat["rsi"]>=70
        if vol_filter!="Any":
            if "Spike" in vol_filter:          mask&=feat["vol_ratio"]>2
            elif "Above" in vol_filter:        mask&=(feat["vol_ratio"]>1.3)&(feat["vol_ratio"]<=2)
            elif "Normal" in vol_filter:       mask&=(feat["vol_ratio"]>0.7)&(feat["vol_ratio"]<=1.3)
            elif "Low" in vol_filter:          mask&=feat["vol_ratio"]<=0.7
        if wd_filter:
            wd_map={"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4}
            mask&=feat["weekday"].isin([wd_map[d] for d in wd_filter])
        if chop_filter!="Any":
            if "Oscillating" in chop_filter:   mask&=feat["chop"]>61.8
            elif "Mixed" in chop_filter:       mask&=(feat["chop"]>=45)&(feat["chop"]<=61.8)
            elif "Trending" in chop_filter:    mask&=feat["chop"]<45
        if gap_filter!="Any":
            if "up" in gap_filter:             mask&=feat["gap_pct"]>0.5
            elif "down" in gap_filter:         mask&=feat["gap_pct"]<-0.5
            elif "Flat" in gap_filter:         mask&=feat["gap_pct"].abs()<=0.5

        filtered=feat[mask]
        rest=feat[~mask]

        if len(filtered)<5:
            st.warning(f"Only {len(filtered)} days match — too few for analysis. "
                       "Relax some filters.")
        else:
            fn=len(filtered); rn=len(rest)
            fc=st.columns(5)
            stat_card(fc[0],"Matching days",str(fn),
                      f"{fn/len(feat)*100:.0f}% of history")
            wr_f=float(filtered["next_up"].mean()*100)
            wr_r=float(rest["next_up"].mean()*100) if rn>0 else 50
            stat_card(fc[1],"Next-day win rate",f"{wr_f:.0f}%",
                      f"vs {wr_r:.0f}% baseline",color_wr(wr_f))
            rng_f=float(filtered["next_range"].mean())
            rng_r=float(rest["next_range"].mean()) if rn>0 else 0
            stat_card(fc[2],"Avg next-day range",f"HKD {rng_f:.1f}",
                      f"vs HKD {rng_r:.1f} baseline","#8b5cf6")
            big_f=float(filtered["next_big"].mean()*100)
            stat_card(fc[3],"% next day ≥20 HKD",f"{big_f:.0f}%",
                      "big swing prob","#f59e0b")
            avg_ret_f=float(filtered["next_ret"].mean())
            stat_card(fc[4],"Avg next-day return",f"{avg_ret_f:+.2f}%","",
                      "#16a34a" if avg_ret_f>0 else "#dc2626")

            # Return distribution comparison
            st.markdown("<br>",unsafe_allow_html=True)
            fig_conf=go.Figure()
            fig_conf.add_trace(go.Histogram(x=filtered["next_ret"],nbinsx=25,
                name=f"With filters (n={fn})",marker_color="#2563eb",opacity=0.75))
            fig_conf.add_trace(go.Histogram(x=rest["next_ret"],nbinsx=25,
                name=f"Without filters (n={rn})",marker_color="#94a3b8",opacity=0.5))
            fig_conf.add_vline(x=0,line_color="#e2e8f0",line_width=1)
            fig_conf.update_layout(height=280,barmode="overlay",
                margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white",paper_bgcolor="white",
                xaxis=dict(title="Next-day return %",gridcolor="#f1f5f9"),
                yaxis=dict(title="Days",gridcolor="#f1f5f9"),
                legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_conf,use_container_width=True)

            # Show matching days on price chart
            st.markdown("**Matching days highlighted on price chart**")
            plot_df=df.tail(min(len(df),252))
            bc_main=["#16a34a" if c>=o else "#dc2626"
                     for c,o in zip(plot_df["Close"],plot_df["Open"])]
            fig_mark=go.Figure(go.Candlestick(
                x=plot_df.index,open=plot_df["Open"],high=plot_df["High"],
                low=plot_df["Low"],close=plot_df["Close"],
                increasing_line_color="#16a34a",decreasing_line_color="#dc2626"))
            match_dates=[d for d in filtered.index if d in plot_df.index]
            for dt in match_dates:
                fig_mark.add_vline(x=dt,line_color="rgba(37,99,235,0.3)",
                                    line_width=6)
            fig_mark.update_layout(height=300,margin=dict(l=0,r=0,t=10,b=0),
                xaxis_rangeslider_visible=False,
                plot_bgcolor="white",paper_bgcolor="white",showlegend=False,
                xaxis=dict(gridcolor="#f1f5f9",
                           rangebreaks=[dict(bounds=["sat","mon"]),
                                        dict(bounds=[16,9.5],pattern="hour")]),
                yaxis=dict(title="Price HKD",gridcolor="#f1f5f9"))
            st.plotly_chart(fig_mark,use_container_width=True)
            st.caption(f"Blue vertical lines = {len(match_dates)} days matching your filters")

    # ════════════════════════════════════════════════════════════════
    # TAB 6 — ML RULES + PATTERN MATCH
    # ════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("#### Decision Tree Rules + Similar Week Pattern Matching")

        with st.spinner("Training models…"):
            clf_dir,acc_dir   =train_dt(feat,"next_up",tree_depth)
            clf_big,acc_big   =train_dt(feat,"next_big",tree_depth)
            clf_trd,acc_trd   =train_dt(feat,"next_big",tree_depth)

        pred_dir,prob_dir=predict_now(clf_dir,feat)
        pred_big,prob_big=predict_now(clf_big,feat)

        # Tomorrow prediction strip
        st.markdown("##### Tomorrow's prediction based on today's market state")
        tp1,tp2,tp3=st.columns(3)

        if pred_dir is not None:
            dl="#16a34a" if pred_dir==1 else "#dc2626"
            dt="📈 UP" if pred_dir==1 else "📉 DOWN"
            conf=f"{max(prob_dir)*100:.0f}%"
            tp1.markdown(
                f"<div style='border:2px solid {dl};border-radius:10px;"
                f"padding:14px;text-align:center'>"
                f"<div style='font-size:0.72rem;color:#64748b'>Direction</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{dl}'>{dt}</div>"
                f"<div style='font-size:0.75rem;color:{dl}'>{conf} confidence</div>"
                f"<div style='font-size:0.68rem;color:#94a3b8'>accuracy: {acc_dir*100:.0f}%</div>"
                f"</div>",unsafe_allow_html=True)

        if pred_big is not None:
            bl="#16a34a" if pred_big==1 else "#94a3b8"
            bt="⚡ BIG SWING" if pred_big==1 else "😴 QUIET"
            conf2=f"{max(prob_big)*100:.0f}%"
            tp2.markdown(
                f"<div style='border:2px solid {bl};border-radius:10px;"
                f"padding:14px;text-align:center'>"
                f"<div style='font-size:0.72rem;color:#64748b'>Tomorrow's range</div>"
                f"<div style='font-size:1.3rem;font-weight:700;color:{bl}'>{bt}</div>"
                f"<div style='font-size:0.75rem;color:{bl}'>{conf2} · ≥20 HKD</div>"
                f"<div style='font-size:0.68rem;color:#94a3b8'>accuracy: {acc_big*100:.0f}%</div>"
                f"</div>",unsafe_allow_html=True)

        today_row=feat.iloc[-1]
        now_wd=DAY_SHORT.get(int(today_row.get("weekday",0)),"?")
        tp3.markdown(
            f"<div style='border:1px solid #e2e8f0;border-radius:10px;"
            f"padding:14px;text-align:center'>"
            f"<div style='font-size:0.72rem;color:#64748b'>Today</div>"
            f"<div style='font-size:1rem;font-weight:600;color:#0f172a'>{now_wd} · "
            f"RSI {today_row.get('rsi',0):.0f}</div>"
            f"<div style='font-size:0.78rem;color:#475569'>"
            f"Vol {today_row.get('vol_ratio',1):.1f}× · "
            f"Chop {today_row.get('chop',50):.0f} · "
            f"BB {today_row.get('bb_pct',50):.0f}%</div></div>",
            unsafe_allow_html=True)

        st.markdown("---")

        # IF-THEN rules side by side
        ml1,ml2=st.columns(2)
        with ml1:
            st.markdown("**Direction rules (0=DOWN, 1=UP)**")
            if clf_dir:
                st.plotly_chart(imp_chart(clf_dir,"Feature importance — direction"),
                                use_container_width=True)
                st.code(export_text(clf_dir,
                    feature_names=[ML_NAMES.get(c,c) for c in ML_COLS],
                    max_depth=tree_depth),language="text")
        with ml2:
            st.markdown("**Big swing rules (0=quiet, 1=big ≥20 HKD)**")
            if clf_big:
                st.plotly_chart(imp_chart(clf_big,"Feature importance — swing size"),
                                use_container_width=True)
                st.code(export_text(clf_big,
                    feature_names=[ML_NAMES.get(c,c) for c in ML_COLS],
                    max_depth=tree_depth),language="text")

        st.markdown("---")

        # Pattern matching
        st.markdown("##### Similar past weeks — what happened next?")
        n_match=st.slider("Matches to show",3,8,5,key="strat_nmatch")
        with st.spinner("Searching for similar past weeks…"):
            matches=find_similar_weeks(feat,df,n_match)

        if not matches:
            st.info("Need more history for pattern matching. Try 1y or 2y.")
        else:
            fwd_rets=[m["fwd_ret"] for m in matches]
            sm1,sm2,sm3,sm4=st.columns(4)
            sm1.metric("Avg next-week return",f"{np.mean(fwd_rets):+.2f}%")
            sm2.metric("Median return",f"{np.median(fwd_rets):+.2f}%")
            sm3.metric("Avg daily range",f"HKD {np.mean([m['fwd_range'] for m in matches]):.1f}")
            sm4.metric("% times up next week",
                       f"{sum(1 for r in fwd_rets if r>0)/len(fwd_rets)*100:.0f}%")

            # Overlay chart
            fig_pm=go.Figure()
            cur=df["Close"].iloc[-5:]/df["Close"].iloc[-5]*100
            fig_pm.add_trace(go.Scatter(x=list(range(5)),y=cur.values,
                name="This week",mode="lines+markers",
                line=dict(color="#0f172a",width=3),marker=dict(size=7)))
            colors=["#2563eb","#16a34a","#f59e0b","#dc2626","#8b5cf6","#0891b2","#ec4899"]
            for i,m in enumerate(matches):
                try:
                    sl=df["Close"].iloc[m["idx"]:m["idx"]+5]
                    if len(sl)<3: continue
                    norm=sl/sl.iloc[0]*100
                    fig_pm.add_trace(go.Scatter(x=list(range(len(norm))),y=norm.values,
                        name=f"{str(m['start'])[:10]} fwd:{m['fwd_ret']:+.1f}%",
                        mode="lines",line=dict(color=colors[i%len(colors)],
                        width=1.5,dash="dot"),opacity=0.65))
                except: pass
            fig_pm.add_hline(y=100,line_color="#e2e8f0",line_width=1)
            fig_pm.update_layout(height=300,margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white",paper_bgcolor="white",
                xaxis=dict(title="Day in week",tickvals=[0,1,2,3,4],
                           ticktext=["Mon","Tue","Wed","Thu","Fri"],
                           gridcolor="#f1f5f9"),
                yaxis=dict(title="Indexed (100=week start)",gridcolor="#f1f5f9"),
                legend=dict(font=dict(size=9),x=1.01,y=1,xanchor="left"))
            st.plotly_chart(fig_pm,use_container_width=True)

            for i,m in enumerate(matches):
                with st.expander(
                    f"#{i+1} · {str(m['start'])[:10]} · "
                    f"Similarity {m['sim']:.3f} · "
                    f"Next week: {m['fwd_ret']:+.1f}% · {m['up_days']}/5 up",
                    expanded=(i==0)):
                    mc=st.columns(4)
                    mc[0].metric("Similarity",f"{m['sim']:.3f}")
                    mc[1].metric("Next-week return",f"{m['fwd_ret']:+.2f}%")
                    mc[2].metric("Avg daily range",f"HKD {m['fwd_range']:.1f}")
                    mc[3].metric("Up days",f"{m['up_days']} / 5")
                    try:
                        cc1,cc2=st.columns(2)
                        with cc1:
                            s=df.iloc[m["idx"]:m["idx"]+5]
                            if len(s)>0:
                                bc=[  "#16a34a" if c>=o else "#dc2626"
                                      for c,o in zip(s["Close"],s["Open"])]
                                fig_s=go.Figure(go.Candlestick(
                                    x=s.index,open=s["Open"],high=s["High"],
                                    low=s["Low"],close=s["Close"],
                                    increasing_line_color="#16a34a",
                                    decreasing_line_color="#dc2626"))
                                fig_s.update_layout(height=150,
                                    margin=dict(l=0,r=0,t=16,b=0),
                                    title=dict(text="Similar past week",font=dict(size=10)),
                                    xaxis_rangeslider_visible=False,
                                    plot_bgcolor="white",paper_bgcolor="white",
                                    showlegend=False,
                                    xaxis=dict(rangebreaks=[dict(bounds=["sat","mon"])]),
                                    yaxis=dict(gridcolor="#f1f5f9",tickformat=",.0f"))
                                st.plotly_chart(fig_s,use_container_width=True)
                        with cc2:
                            f2=df.iloc[m["fwd_idx"]:m["fwd_idx"]+5]
                            if len(f2)>0:
                                fig_f=go.Figure(go.Candlestick(
                                    x=f2.index,open=f2["Open"],high=f2["High"],
                                    low=f2["Low"],close=f2["Close"],
                                    increasing_line_color="#16a34a",
                                    decreasing_line_color="#dc2626"))
                                fig_f.update_layout(height=150,
                                    margin=dict(l=0,r=0,t=16,b=0),
                                    title=dict(text="What happened next",font=dict(size=10)),
                                    xaxis_rangeslider_visible=False,
                                    plot_bgcolor="white",paper_bgcolor="white",
                                    showlegend=False,
                                    xaxis=dict(rangebreaks=[dict(bounds=["sat","mon"])]),
                                    yaxis=dict(gridcolor="#f1f5f9",tickformat=",.0f"))
                                st.plotly_chart(fig_f,use_container_width=True)
                    except: pass

    st.markdown(
        "<span style='color:#94a3b8;font-size:0.74rem'>"
        "Historical patterns are not guarantees. "
        "Data via yfinance · Not financial advice.</span>",
        unsafe_allow_html=True)
