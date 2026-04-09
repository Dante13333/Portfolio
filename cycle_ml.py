"""
cycle_ml.py — Cycle-Aware ML Strategy Engine

Defines cycles using price structure (ZigZag) + volume confirmation.
ML learns from cycle position/state, NOT calendar time.

Outputs:
  1. Cycle detector    — where am I now (early/mid/late/exhaustion)
  2. Cycle statistics  — typical duration, amplitude, volume profile
  3. Cycle high/low    — expected remaining range from current position
  4. Retail vs inst    — who is driving this move
  5. Peer context      — same-sector + high-beta + pattern-similar stocks
  6. ML rules          — decision tree trained on cycle state features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import time, warnings, pytz
warnings.filterwarnings("ignore")

HK_TZ = pytz.timezone("Asia/Hong_Kong")

# ── PEER UNIVERSE (sector + high-beta HKEX names) ─────────────────────
PEERS = {
    # AI / Tech (high-beta, momentum)
    "0700.HK":("Tencent","AI/Tech"),      "9988.HK":("Alibaba","AI/Tech"),
    "9999.HK":("NetEase","AI/Tech"),       "9888.HK":("Baidu","AI/Tech"),
    "1024.HK":("Kuaishou","AI/Tech"),      "0020.HK":("SenseTime","AI/Tech"),
    "0992.HK":("Lenovo","AI/Tech"),        "2382.HK":("Sunny Optical","AI/Tech"),
    "0763.HK":("ZTE","AI/Tech"),
    # EV / High-beta
    "9866.HK":("NIO","EV"),               "9868.HK":("Xpeng","EV"),
    "2015.HK":("Li Auto","EV"),            "0175.HK":("Geely","EV"),
    "1211.HK":("BYD","EV"),
    # New economy / speculative
    "1810.HK":("Xiaomi","NewEcon"),        "6098.HK":("Country Garden Svc","NewEcon"),
    "9618.HK":("JD.com","NewEcon"),        "9961.HK":("Trip.com","NewEcon"),
    "6862.HK":("Haidilao","NewEcon"),
    # Biotech / speculative healthcare
    "2269.HK":("Wuxi Bio","Biotech"),      "6160.HK":("BeiGene","Biotech"),
    "1093.HK":("CSPC","Biotech"),
    # Finance/exchange (high-vol)
    "0388.HK":("HKEX","Finance"),          "2318.HK":("Ping An","Finance"),
}

# ── DATA FETCH ────────────────────────────────────────────────────────
def _variants(ticker):
    v=[ticker]; code=ticker.replace(".HK","")
    if code.isdigit():
        v.append(str(int(code))+".HK"); v.append(code.zfill(4)+".HK")
    return list(dict.fromkeys(v))

@st.cache_data(ttl=300, show_spinner=False)
def fetch(ticker, period="1y", interval="1d"):
    for t in _variants(ticker):
        try:
            df=yf.Ticker(t).history(period=period,interval=interval,auto_adjust=True)
            if len(df)>=10:
                df.index=pd.to_datetime(df.index)
                if interval!="1d":
                    if df.index.tzinfo is None:
                        df.index=df.index.tz_localize("UTC")
                    df.index=df.index.tz_convert(HK_TZ)
                return df
            time.sleep(0.3)
        except Exception: continue
    return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def get_name(ticker):
    for t in _variants(ticker):
        try:
            i=yf.Ticker(t).info
            n=i.get("longName") or i.get("shortName")
            if n: return n
        except Exception: pass
    return ticker

# ═════════════════════════════════════════════════════════════════════
# CYCLE DETECTION ENGINE
# Price structure (ZigZag) + volume confirmation
# ═════════════════════════════════════════════════════════════════════

def detect_cycles(df: pd.DataFrame,
                  min_pct: float = 5.0,
                  vol_confirm: bool = True) -> pd.DataFrame:
    """
    Detect swing cycles using ZigZag + volume.

    A cycle = trough → peak → trough (upswing) or peak → trough → peak (downswing).
    Volume confirmation: a pivot is 'confirmed' if accompanied by vol > 20d avg.

    Returns DataFrame with columns:
      pivot_type : 'H' or 'L'
      price      : pivot price
      date       : pivot date
      vol_ratio  : volume at pivot vs 20d avg
      confirmed  : whether volume confirmed the pivot
    """
    if len(df) < 20:
        return pd.DataFrame()

    prices  = df["Close"].values
    highs   = df["High"].values
    lows    = df["Low"].values
    vols    = df["Volume"].values
    dates   = df.index
    avg_vol = pd.Series(vols).rolling(20).mean().values

    pivots = []
    direction = 0   # 1=looking for high, -1=looking for low
    extreme_idx = 0
    extreme_val = prices[0]

    for i in range(1, len(prices)):
        if direction == 0:
            # initialise
            if prices[i] > prices[extreme_idx]:
                direction = 1; extreme_idx = i; extreme_val = prices[i]
            elif prices[i] < prices[extreme_idx]:
                direction = -1; extreme_idx = i; extreme_val = prices[i]
        elif direction == 1:
            if highs[i] > extreme_val:
                extreme_idx = i; extreme_val = highs[i]
            elif prices[i] < extreme_val * (1 - min_pct/100):
                # Confirmed high pivot
                vr = vols[extreme_idx] / avg_vol[extreme_idx] if avg_vol[extreme_idx] > 0 else 1
                confirmed = vr >= 1.2 if vol_confirm else True
                pivots.append({"pivot_type":"H","price":extreme_val,
                               "date":dates[extreme_idx],"vol_ratio":round(vr,2),
                               "confirmed":confirmed,"idx":extreme_idx})
                direction = -1; extreme_idx = i; extreme_val = lows[i]
        elif direction == -1:
            if lows[i] < extreme_val:
                extreme_idx = i; extreme_val = lows[i]
            elif prices[i] > extreme_val * (1 + min_pct/100):
                # Confirmed low pivot
                vr = vols[extreme_idx] / avg_vol[extreme_idx] if avg_vol[extreme_idx] > 0 else 1
                confirmed = vr >= 1.2 if vol_confirm else True
                pivots.append({"pivot_type":"L","price":extreme_val,
                               "date":dates[extreme_idx],"vol_ratio":round(vr,2),
                               "confirmed":confirmed,"idx":extreme_idx})
                direction = 1; extreme_idx = i; extreme_val = highs[i]

    return pd.DataFrame(pivots) if pivots else pd.DataFrame()

def build_cycles(pivots: pd.DataFrame, df: pd.DataFrame) -> list:
    """
    From pivot list, build cycle records:
    Each cycle: trough → peak (upswing) with stats.
    Returns list of dicts.
    """
    if pivots.empty or len(pivots) < 3:
        return []
    cycles = []
    for i in range(len(pivots)-2):
        p0, p1, p2 = pivots.iloc[i], pivots.iloc[i+1], pivots.iloc[i+2]
        # Upswing cycle: L → H → L
        if p0["pivot_type"]=="L" and p1["pivot_type"]=="H" and p2["pivot_type"]=="L":
            dur_up   = (p1["date"]-p0["date"]).days
            dur_down = (p2["date"]-p1["date"]).days
            amplitude= (p1["price"]-p0["price"])/p0["price"]*100
            drawdown = (p1["price"]-p2["price"])/p1["price"]*100
            # Volume in each leg
            leg_up   = df[(df.index>=p0["date"])&(df.index<=p1["date"])]
            leg_dn   = df[(df.index>=p1["date"])&(df.index<=p2["date"])]
            avg_all  = df["Volume"].mean()
            vol_up   = leg_up["Volume"].mean()/avg_all if len(leg_up)>0 and avg_all>0 else 1
            vol_dn   = leg_dn["Volume"].mean()/avg_all if len(leg_dn)>0 and avg_all>0 else 1
            # Retail vs institutional proxy:
            # Retail = high vol + big % move + small body candles (noise)
            # Inst   = steady vol + directional bodies
            body_ratio_up = (leg_up["Close"]-leg_up["Open"]).abs().mean() / \
                            (leg_up["High"]-leg_up["Low"]+1e-9).mean() \
                            if len(leg_up)>0 else 0.5
            retail_score = min(vol_up * (1-body_ratio_up) * amplitude/10, 1.0)
            cycles.append({
                "start":       p0["date"],
                "peak":        p1["date"],
                "end":         p2["date"],
                "trough_price":p0["price"],
                "peak_price":  p1["price"],
                "end_price":   p2["price"],
                "amplitude_pct":round(amplitude,1),
                "drawdown_pct": round(drawdown,1),
                "dur_up_days": dur_up,
                "dur_dn_days": dur_down,
                "dur_total":   dur_up+dur_down,
                "vol_up":      round(vol_up,2),
                "vol_dn":      round(vol_dn,2),
                "vol_ratio":   round((p1["vol_ratio"]+p0["vol_ratio"])/2,2),
                "confirmed":   p0["confirmed"] and p1["confirmed"],
                "retail_score":round(retail_score,2),
            })
    return cycles

def current_cycle_state(df: pd.DataFrame, pivots: pd.DataFrame,
                         cycles: list) -> dict:
    """
    Determine where we are in the CURRENT (incomplete) cycle.
    Returns state dict with position, pct_through, expected_high/low etc.
    """
    if df.empty or pivots.empty:
        return {}

    price_now = float(df["Close"].iloc[-1])
    last_pivot = pivots.iloc[-1]
    last_type  = last_pivot["pivot_type"]
    last_price = last_pivot["price"]
    last_date  = last_pivot["date"]
    days_since = (df.index[-1]-last_date).days

    # Cycle statistics from history
    if cycles:
        avg_dur_up  = np.mean([c["dur_up_days"]  for c in cycles])
        avg_dur_dn  = np.mean([c["dur_dn_days"]  for c in cycles])
        avg_amp     = np.mean([c["amplitude_pct"] for c in cycles])
        avg_draw    = np.mean([c["drawdown_pct"]  for c in cycles])
        med_amp     = np.median([c["amplitude_pct"] for c in cycles])
        avg_vol_up  = np.mean([c["vol_up"]  for c in cycles])
        avg_vol_dn  = np.mean([c["vol_dn"]  for c in cycles])
        avg_retail  = np.mean([c["retail_score"] for c in cycles])
    else:
        avg_dur_up=avg_dur_dn=10; avg_amp=avg_draw=15; med_amp=15
        avg_vol_up=avg_vol_dn=1; avg_retail=0.5

    # Current move size
    move_pct = (price_now-last_price)/last_price*100

    if last_type=="L":
        # Currently in upswing
        pct_through = min(days_since/max(avg_dur_up,1)*100, 150)
        move_through= min(move_pct/max(avg_amp,1)*100, 150) if move_pct>0 else 0
        combined    = (pct_through*0.4 + move_through*0.6)

        if combined < 25:   state="🌱 EARLY UPSWING"
        elif combined < 55: state="🔼 MID UPSWING"
        elif combined < 80: state="🔺 LATE UPSWING"
        else:               state="⚡ EXHAUSTION / PEAK ZONE"

        expected_high = last_price * (1 + avg_amp/100)
        expected_low  = expected_high * (1 - avg_draw/100)
        remaining_up  = max(expected_high-price_now, 0)
        leg           = "UP"
    else:
        # Currently in downswing
        move_dn       = abs(move_pct)
        pct_through   = min(days_since/max(avg_dur_dn,1)*100, 150)
        move_through  = min(move_dn/max(avg_draw,1)*100, 150) if move_pct<0 else 0
        combined      = (pct_through*0.4 + move_through*0.6)

        if combined < 25:   state="🔽 EARLY DOWNSWING"
        elif combined < 55: state="↘ MID DOWNSWING"
        elif combined < 80: state="🔻 LATE DOWNSWING"
        else:               state="🌀 EXHAUSTION / TROUGH ZONE"

        expected_low  = last_price * (1 - avg_draw/100)
        expected_high = expected_low * (1 + avg_amp/100)
        remaining_up  = 0
        leg           = "DOWN"

    # Current volume vs historical leg average
    recent_vol  = float(df["Volume"].tail(5).mean())
    hist_avg    = float(df["Volume"].mean())
    vol_ratio   = recent_vol/hist_avg if hist_avg>0 else 1
    leg_avg_vol = avg_vol_up if leg=="UP" else avg_vol_dn
    vol_vs_leg  = vol_ratio/leg_avg_vol if leg_avg_vol>0 else 1

    # Retail vs institutional
    recent_body = (df["Close"]-df["Open"]).abs().tail(5).mean()
    recent_range= (df["High"]-df["Low"]).tail(5).mean()
    body_ratio  = recent_body/(recent_range+1e-9)
    retail_now  = min(vol_ratio*(1-body_ratio)*abs(move_pct)/10, 1.0)
    inst_bias   = body_ratio > 0.5 and vol_vs_leg < 1.3

    return {
        "state":          state,
        "leg":            leg,
        "days_in_leg":    days_since,
        "pct_through":    round(combined, 1),
        "move_pct":       round(move_pct, 2),
        "expected_high":  round(expected_high, 1),
        "expected_low":   round(expected_low, 1),
        "remaining_up":   round(remaining_up, 1),
        "avg_cycle_dur":  round(avg_dur_up+avg_dur_dn, 1),
        "avg_amplitude":  round(avg_amp, 1),
        "avg_drawdown":   round(avg_draw, 1),
        "vol_vs_leg":     round(vol_vs_leg, 2),
        "retail_now":     round(retail_now, 2),
        "inst_bias":      inst_bias,
        "n_cycles":       len(cycles),
    }

# ═════════════════════════════════════════════════════════════════════
# CYCLE-STATE FEATURE ENGINEERING (for ML)
# ═════════════════════════════════════════════════════════════════════

def build_cycle_features(df: pd.DataFrame, pivots: pd.DataFrame,
                          cycles: list) -> pd.DataFrame:
    """
    For each day, compute its cycle-state features.
    ML trains on these, NOT on calendar date.
    """
    if df.empty or pivots.empty or not cycles:
        return pd.DataFrame()

    rows = []
    pivot_dates = pivots["date"].tolist()
    pivot_types = pivots["pivot_type"].tolist()
    pivot_prices= pivots["price"].tolist()

    avg_amp   = np.mean([c["amplitude_pct"] for c in cycles])
    avg_dur   = np.mean([c["dur_total"] for c in cycles])
    avg_draw  = np.mean([c["drawdown_pct"] for c in cycles])
    vol_avg   = df["Volume"].mean()

    for i, (date, row) in enumerate(df.iterrows()):
        if i < 26: continue   # need indicators

        # Find which pivot we're after
        prior_pivots = [(d,t,p) for d,t,p in zip(pivot_dates,pivot_types,pivot_prices)
                        if d <= date]
        if len(prior_pivots) < 2: continue
        last_d, last_t, last_p = prior_pivots[-1]
        prev_d, prev_t, prev_p = prior_pivots[-2]

        days_in_leg  = max((date-last_d).days, 1)
        move_pct     = (row["Close"]-last_p)/last_p*100
        leg          = 1 if last_t=="L" else -1    # 1=upswing, -1=downswing
        pct_of_avg_amp = move_pct/avg_amp*100 if avg_amp>0 else 0
        pct_of_avg_dur = days_in_leg/avg_dur*100 if avg_dur>0 else 0

        # Volume features
        vol_5d  = df["Volume"].iloc[max(0,i-5):i].mean()
        vol_r   = vol_5d/vol_avg if vol_avg>0 else 1

        # Price features
        close_  = row["Close"]
        rng_    = row["High"]-row["Low"]
        body_   = abs(row["Close"]-row["Open"])
        body_r  = body_/(rng_+1e-9)
        gap_    = (row["Open"]-df["Close"].iloc[i-1])/df["Close"].iloc[i-1]*100 if i>0 else 0

        # Indicators
        closes_ = df["Close"].iloc[:i+1]
        rsi_v   = _rsi(closes_)
        bb_v    = _bb_pct(closes_)
        atr_v   = _atr(df.iloc[:i+1])

        # Retail vs inst proxy
        retail  = min(vol_r*(1-body_r)*abs(move_pct)/10, 1.0)

        # Label: next-day direction + big move
        if i+1 < len(df):
            next_ret   = (df["Close"].iloc[i+1]-close_)/close_*100
            next_range = df["High"].iloc[i+1]-df["Low"].iloc[i+1]
        else:
            next_ret = next_range = np.nan

        rows.append({
            # Cycle state (key features — what ML learns on)
            "leg":              leg,
            "days_in_leg":      days_in_leg,
            "move_pct":         round(move_pct,2),
            "pct_of_avg_amp":   round(pct_of_avg_amp,1),
            "pct_of_avg_dur":   round(pct_of_avg_dur,1),
            # Volume
            "vol_ratio":        round(vol_r,2),
            "retail_score":     round(retail,2),
            "vol_x_move":       round(vol_r*abs(move_pct),2),
            # Price structure
            "body_ratio":       round(body_r,2),
            "gap_pct":          round(gap_,2),
            "rng_hkd":          round(rng_,1),
            # Indicators
            "rsi":              round(rsi_v,1),
            "bb_pct":           round(bb_v,1),
            "atr":              round(atr_v,2),
            # Weekday (secondary, not primary)
            "weekday":          date.dayofweek,
            # Labels
            "next_up":          int(next_ret>0) if not np.isnan(next_ret) else np.nan,
            "next_big":         int(next_range>=20) if not np.isnan(next_range) else np.nan,
            "date":             date,
        })

    feat = pd.DataFrame(rows).set_index("date")
    return feat.dropna(subset=["next_up","next_big"])

def _rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0).ewm(com=p-1,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(com=p-1,adjust=False).mean()
    r=100-100/(1+g/l.replace(0,np.nan))
    return float(r.iloc[-1]) if len(r)>0 and not np.isnan(r.iloc[-1]) else 50

def _bb_pct(s,p=20):
    if len(s)<p: return 50
    mid=s.rolling(p).mean(); std=s.rolling(p).std()
    lo=mid-2*std; hi=mid+2*std
    v=((s-lo)/(hi-lo+1e-9)*100).clip(0,100)
    return float(v.iloc[-1])

def _atr(df,p=14):
    if len(df)<p+1: return float((df["High"]-df["Low"]).mean())
    tr=pd.concat([df["High"]-df["Low"],
                  (df["High"]-df["Close"].shift()).abs(),
                  (df["Low"]-df["Close"].shift()).abs()],axis=1).max(axis=1)
    return float(tr.ewm(com=p-1,adjust=False).mean().iloc[-1])

# ═════════════════════════════════════════════════════════════════════
# PEER ANALYSIS
# ═════════════════════════════════════════════════════════════════════

CYCLE_FEAT_COLS = ["leg","days_in_leg","move_pct","pct_of_avg_amp",
                   "pct_of_avg_dur","vol_ratio","retail_score","body_ratio",
                   "gap_pct","rsi","bb_pct","atr","weekday"]

@st.cache_data(ttl=600, show_spinner=False)
def fetch_peer_snapshot(ticker, period="6mo", min_pct=5.0):
    """Fetch + detect cycles for one peer. Returns summary dict."""
    try:
        df=fetch(ticker,period)
        if len(df)<30: return None
        pivots=detect_cycles(df,min_pct)
        cycles=build_cycles(pivots,df) if not pivots.empty else []
        state=current_cycle_state(df,pivots,cycles) if cycles else {}
        beta=_calc_beta(df)
        return {
            "ticker":   ticker,
            "name":     PEERS.get(ticker,("?","?"))[0],
            "sector":   PEERS.get(ticker,("?","?"))[1],
            "price":    float(df["Close"].iloc[-1]),
            "cycle_state": state.get("state","—"),
            "leg":         state.get("leg","—"),
            "pct_through": state.get("pct_through",0),
            "retail_now":  state.get("retail_now",0.5),
            "avg_amp":     state.get("avg_amplitude",0),
            "n_cycles":    state.get("n_cycles",0),
            "beta":        beta,
            "vol_ratio":   float(df["Volume"].tail(5).mean()/df["Volume"].mean()) if len(df)>5 else 1,
        }
    except Exception: return None

def _calc_beta(df, bench="^HSI"):
    try:
        hsi=yf.Ticker(bench).history(period="6mo",interval="1d",auto_adjust=True)
        if len(hsi)<20 or len(df)<20: return None
        r1=df["Close"].pct_change().dropna()
        r2=hsi["Close"].pct_change().dropna()
        al=pd.concat([r1,r2],axis=1).dropna()
        if len(al)<10: return None
        cov=np.cov(al.iloc[:,0],al.iloc[:,1])
        return round(cov[0,1]/cov[1,1],2)
    except Exception: return None

def find_pattern_peers(target_feat, all_peers_feat, n=5):
    """Cosine similarity on cycle-state features."""
    if target_feat is None or all_peers_feat is None: return []
    results=[]
    ref=np.array(target_feat)
    for sym,vec in all_peers_feat.items():
        if vec is None: continue
        v=np.array(vec)
        sim=float(np.dot(ref,v)/(np.linalg.norm(ref)*np.linalg.norm(v)+1e-9))
        results.append((sym,sim))
    return sorted(results,key=lambda x:-x[1])[:n]

# ═════════════════════════════════════════════════════════════════════
# ML ENGINE
# ═════════════════════════════════════════════════════════════════════

def train_cycle_model(feat, label, depth=4):
    X=feat[CYCLE_FEAT_COLS].fillna(0); y=feat[label]
    if len(y.unique())<2 or len(y)<15: return None,None
    clf=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=4,
                                class_weight="balanced",random_state=42)
    clf.fit(X,y)
    try:
        cv=cross_val_score(clf,X,y,cv=min(5,len(y)//8),scoring="accuracy")
        acc=float(cv.mean())
    except: acc=float((clf.predict(X)==y).mean())
    return clf,acc

def predict_cycle(clf,feat):
    if clf is None or feat.empty: return None,None
    last=feat[CYCLE_FEAT_COLS].fillna(0).iloc[[-1]]
    return int(clf.predict(last)[0]),clf.predict_proba(last)[0]

# ═════════════════════════════════════════════════════════════════════
# CHARTS
# ═════════════════════════════════════════════════════════════════════

def cycle_chart(df,pivots,cycles,state,ticker_name):
    plot_df=df.tail(min(len(df),252))
    bc=["#16a34a" if c>=o else "#dc2626"
        for c,o in zip(plot_df["Close"],plot_df["Open"])]

    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
                      row_heights=[0.55,0.25,0.20],vertical_spacing=0.03,
                      subplot_titles=["Price + Cycle Pivots","Volume","Retail Score"])

    fig.add_trace(go.Candlestick(
        x=plot_df.index,open=plot_df["Open"],high=plot_df["High"],
        low=plot_df["Low"],close=plot_df["Close"],
        increasing_line_color="#16a34a",decreasing_line_color="#dc2626",
        name="Price"),row=1,col=1)

    # Draw cycle zones (shaded)
    for c in cycles:
        if c["start"]<plot_df.index[0]: continue
        fig.add_vrect(x0=c["start"],x1=c["peak"],
                      fillcolor="rgba(22,163,74,0.06)",line_width=0,row=1,col=1)
        fig.add_vrect(x0=c["peak"],x1=c["end"],
                      fillcolor="rgba(220,38,38,0.06)",line_width=0,row=1,col=1)

    # Draw pivots
    if not pivots.empty:
        for _,p in pivots.iterrows():
            if p["date"]<plot_df.index[0]: continue
            col="#16a34a" if p["pivot_type"]=="L" else "#dc2626"
            sym="triangle-up" if p["pivot_type"]=="L" else "triangle-down"
            sz=14 if p["confirmed"] else 8
            fig.add_trace(go.Scatter(
                x=[p["date"]],y=[p["price"]],mode="markers",
                marker=dict(color=col,size=sz,symbol=sym,
                            line=dict(color="white",width=1.5)),
                showlegend=False),row=1,col=1)

    # Expected high/low lines
    if state:
        price_now=float(df["Close"].iloc[-1])
        fig.add_hline(y=state.get("expected_high",price_now),
                      line_dash="dot",line_color="#16a34a",line_width=1.5,
                      annotation_text=f"Cycle high est. {state.get('expected_high',0):.0f}",
                      annotation_position="right",row=1,col=1)
        fig.add_hline(y=state.get("expected_low",price_now),
                      line_dash="dot",line_color="#dc2626",line_width=1.5,
                      annotation_text=f"Cycle low est. {state.get('expected_low',0):.0f}",
                      annotation_position="right",row=1,col=1)

    # Volume bars
    vol_avg=plot_df["Volume"].mean()
    vol_colors=["#dc2626" if v>vol_avg*1.5 else "#2563eb" if v>vol_avg else "#94a3b8"
                for v in plot_df["Volume"]]
    fig.add_trace(go.Bar(x=plot_df.index,y=plot_df["Volume"],
                          marker_color=vol_colors,opacity=0.75,name="Vol"),row=2,col=1)
    fig.add_hline(y=vol_avg,line_dash="dot",line_color="#94a3b8",line_width=1,row=2,col=1)

    # Retail score (simple proxy: vol spike × body_inverse)
    retail_proxy=plot_df["Volume"]/vol_avg*(
        1-(plot_df["Close"]-plot_df["Open"]).abs()/(plot_df["High"]-plot_df["Low"]+1e-9))
    retail_proxy=retail_proxy.clip(0,3)
    fig.add_trace(go.Scatter(x=plot_df.index,y=retail_proxy,
                              line=dict(color="#f59e0b",width=1.5),
                              fill="tozeroy",fillcolor="rgba(245,158,11,0.1)",
                              name="Retail proxy"),row=3,col=1)
    fig.add_hline(y=1,line_dash="dot",line_color="#94a3b8",line_width=1,row=3,col=1)

    fig.update_layout(
        height=620,margin=dict(l=0,r=0,t=24,b=0),
        title=dict(text=f"{ticker_name} — Cycle Detection",font=dict(size=13)),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white",paper_bgcolor="white",showlegend=False,
        xaxis=dict(rangebreaks=[dict(bounds=["sat","mon"])],gridcolor="#f1f5f9"),
        xaxis2=dict(rangebreaks=[dict(bounds=["sat","mon"])],gridcolor="#f1f5f9"),
        xaxis3=dict(rangebreaks=[dict(bounds=["sat","mon"])],gridcolor="#f1f5f9"),
        yaxis=dict(title="Price HKD",gridcolor="#f1f5f9"),
        yaxis2=dict(title="Volume",gridcolor="#f1f5f9"),
        yaxis3=dict(title="Retail proxy",gridcolor="#f1f5f9"))
    return fig

def cycle_duration_hist(cycles):
    durs=[c["dur_total"] for c in cycles]
    amps=[c["amplitude_pct"] for c in cycles]
    fig=make_subplots(rows=1,cols=2,
                      subplot_titles=["Cycle duration (days)","Cycle amplitude (%)"])
    fig.add_trace(go.Histogram(x=durs,nbinsx=15,marker_color="#2563eb",opacity=0.75,
                                name="Duration"),row=1,col=1)
    fig.add_vline(x=np.mean(durs),line_dash="dot",line_color="#0f172a",
                  annotation_text=f"Avg {np.mean(durs):.0f}d",row=1,col=1)
    fig.add_trace(go.Histogram(x=amps,nbinsx=15,marker_color="#16a34a",opacity=0.75,
                                name="Amplitude"),row=1,col=2)
    fig.add_vline(x=np.mean(amps),line_dash="dot",line_color="#0f172a",
                  annotation_text=f"Avg {np.mean(amps):.0f}%",row=1,col=2)
    fig.update_layout(height=240,margin=dict(l=0,r=0,t=30,b=0),
                       plot_bgcolor="white",paper_bgcolor="white",showlegend=False,
                       xaxis=dict(title="Days",gridcolor="#f1f5f9"),
                       xaxis2=dict(title="%",gridcolor="#f1f5f9"),
                       yaxis=dict(gridcolor="#f1f5f9"),yaxis2=dict(gridcolor="#f1f5f9"))
    return fig

# ═════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ═════════════════════════════════════════════════════════════════════

def render():
    now_hk=datetime.now(HK_TZ)
    st.markdown(
        "## 🔄 Cycle ML Engine &nbsp;"
        "<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        "padding:2px 7px;border-radius:5px'>CYCLE-STATE AWARE</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"ZigZag + volume cycle detection · ML trained on cycle position not calendar · "
        f"Retail vs institutional · Peer comparison · "
        f"HKT {now_hk.strftime('%H:%M')}</span>",
        unsafe_allow_html=True)

    with st.expander("📖 How cycles are defined"):
        st.markdown("""
**Cycle = trough → peak → trough** detected by:
1. **ZigZag** — price must move ≥ your threshold % before a new pivot is confirmed
2. **Volume confirmation** — pivot is "confirmed" only if volume at that bar > 1.2× 20d avg
   (unconfirmed pivots shown as smaller markers)

**Cycle state** is computed from:
- % of typical amplitude already moved (how far into the swing)
- % of typical duration elapsed (how long into the swing)
- Volume vs historical leg average (accelerating or decelerating?)
- Retail score (high vol + choppy bodies = retail driven; steady vol + clean bodies = institutional)

**ML features** are entirely cycle-state based — the model never sees "it's Tuesday"
as a primary feature. It sees "I'm 60% through a typical upswing with declining volume."
        """)

    with st.expander("📖 Cycle metric explanations"):
        st.markdown("""
**ZigZag threshold %** — Minimum % move required before a new pivot is confirmed.
Lower (2-5%) = finds more cycles including short swings.
Higher (8-15%) = only major cycle turning points. Set to match your trading timeframe.

**Cycle amplitude %** — Average % gain from trough to peak in historical cycles.
If avg amplitude is 40%, a typical upswing moves 40% from its low.
Current move vs average amplitude tells you how far you are through the typical cycle.

**Cycle duration (days)** — Average number of days from trough-to-peak and peak-to-trough.
Combined with % of amplitude moved, gives your cycle position estimate.

**Volume confirmation** — A pivot is "confirmed" (large marker) only if volume at that
bar was >1.2x the 20-day average. Unconfirmed pivots (small markers) = price structure
turned but volume did not support it — weaker signal.

**Retail score** — High volume + choppy candle bodies (small close vs open relative to range)
= retail crowd behaviour (noise, likely to reverse). Steady volume + clean directional candles
= institutional (trend more reliable). Score 0-1: >0.6 = retail driven.

**Cycle progress %** — Weighted blend: (% of avg duration elapsed x 0.4) + (% of avg amplitude
moved x 0.6). Price progress weighted more because cycles vary more in size than in time.
>80% = exhaustion zone, expect reversal.

**Expected cycle high/low** — Estimated from historical average amplitude and drawdown.
These are probability centres, not guarantees. One standard deviation above/below the mean.

**Peer phase alignment** — If 5+ sector peers are all in the same late-upswing phase,
that is a sector-wide signal stronger than any single stock showing it alone.
        """)

    st.markdown("---")

    # ── Controls ─────────────────────────────────────────────────────
    from db_manager import get_portfolio_full
    port=get_portfolio_full()
    port_tickers=port["ticker"].tolist() if not port.empty else []

    c1,c2,c3,c4=st.columns(4)
    if port_tickers:
        opts=["— type a ticker —"]+port_tickers
        sel=c1.selectbox("Portfolio stock",opts,key="cml_port")
        if sel!="— type a ticker —":
            ticker=sel
        else:
            raw=c1.text_input("Ticker","0700.HK",key="cml_raw").strip().upper()
            ticker=raw if raw.endswith(".HK") else raw+".HK"
    else:
        raw=c1.text_input("Ticker","0700.HK",key="cml_raw2").strip().upper()
        ticker=raw if raw.endswith(".HK") else raw+".HK"

    period  =c2.selectbox("History",["6mo","1y","2y"],index=1,key="cml_period")
    min_pct =c3.slider("ZigZag threshold %",2,20,5,key="cml_zigzag",
                        help="Min % move required to confirm a new pivot. Lower = more cycles detected.")
    depth   =c4.slider("Tree depth",2,6,4,key="cml_depth")

    if st.button("🔄 Refresh",key="cml_refresh"):
        st.cache_data.clear(); st.rerun()

    # ── Load + detect ─────────────────────────────────────────────────
    with st.spinner(f"Loading {ticker} and detecting cycles…"):
        df=fetch(ticker,period)
        name=get_name(ticker)

    if len(df)<30:
        st.error(f"Not enough data for {ticker}. Try a longer period or check the ticker.")
        return

    with st.spinner("Detecting cycle pivots…"):
        pivots=detect_cycles(df,min_pct,vol_confirm=True)
        cycles=build_cycles(pivots,df) if not pivots.empty else []
        state =current_cycle_state(df,pivots,cycles)

    if not cycles:
        st.warning(
            f"No complete cycles detected with {min_pct}% ZigZag threshold. "
            "Try lowering the threshold or using a longer period.")

    # ════════════════════════════════════════════════════════════════
    # SECTION 1 — CURRENT CYCLE STATE
    # ════════════════════════════════════════════════════════════════
    st.markdown(f"### 🔄 Current Cycle State — {name}")

    if state:
        cs_col = "#16a34a" if "UP" in state.get("leg","") else "#dc2626"
        exh_col= "#dc2626" if "EXHAUST" in state.get("state","") or \
                              "PEAK" in state.get("state","") or \
                              "TROUGH" in state.get("state","") else cs_col

        # Big state card
        st.markdown(
            f"<div style='border:2px solid {exh_col};border-radius:12px;"
            f"padding:18px 22px;background:rgba(0,0,0,0.02);margin-bottom:14px'>"
            f"<div style='font-size:1.5rem;font-weight:800;color:{exh_col}'>"
            f"{state.get('state','—')}</div>"
            f"<div style='display:flex;gap:24px;flex-wrap:wrap;margin-top:10px;"
            f"font-size:0.82rem;color:#475569'>"
            f"<span><b>Days in leg:</b> {state.get('days_in_leg',0)}</span>"
            f"<span><b>Move so far:</b> {state.get('move_pct',0):+.1f}%</span>"
            f"<span><b>% of avg cycle:</b> {state.get('pct_through',0):.0f}%</span>"
            f"<span><b>Avg cycle dur:</b> {state.get('avg_cycle_dur',0):.0f}d</span>"
            f"<span><b>Cycles detected:</b> {state.get('n_cycles',0)}</span>"
            f"</div></div>",
            unsafe_allow_html=True)

        # Progress bar through cycle
        pct=min(state.get("pct_through",0),100)
        bar_c="#16a34a" if pct<60 else "#f59e0b" if pct<85 else "#dc2626"
        st.markdown(
            f"<div style='font-size:0.75rem;color:#64748b;margin-bottom:4px'>"
            f"Cycle progress: {pct:.0f}% through typical {state.get('leg','?')} leg</div>"
            f"<div style='background:#f1f5f9;border-radius:8px;height:12px;overflow:hidden;margin-bottom:14px'>"
            f"<div style='width:{pct:.0f}%;height:100%;background:{bar_c};"
            f"border-radius:8px'></div></div>",
            unsafe_allow_html=True)

        # Key metrics row
        m1,m2,m3,m4,m5,m6=st.columns(6)
        def mc(col,lbl,val,color="#0f172a",sub=""):
            col.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
                f"padding:10px 12px;text-align:center'>"
                f"<div style='font-size:0.68rem;color:#94a3b8'>{lbl}</div>"
                f"<div style='font-size:1.05rem;font-weight:700;color:{color}'>{val}</div>"
                f"{'<div style=font-size:0.68rem;color:#94a3b8>'+sub+'</div>' if sub else ''}"
                f"</div>",unsafe_allow_html=True)

        mc(m1,"Expected cycle high",f"HKD {state.get('expected_high',0):.0f}","#16a34a",
           f"avg amp {state.get('avg_amplitude',0):.0f}%")
        mc(m2,"Expected cycle low", f"HKD {state.get('expected_low',0):.0f}","#dc2626",
           f"avg draw {state.get('avg_drawdown',0):.0f}%")
        mc(m3,"Remaining upside",
           f"HKD {state.get('remaining_up',0):.0f}" if state.get("leg")=="UP" else "N/A",
           "#16a34a" if state.get("remaining_up",0)>0 else "#94a3b8")
        vol_vl=state.get("vol_vs_leg",1)
        mc(m4,"Vol vs leg avg",f"{vol_vl:.1f}×",
           "#dc2626" if vol_vl>1.5 else "#16a34a" if vol_vl<0.8 else "#f59e0b",
           "accelerating" if vol_vl>1.2 else "decelerating" if vol_vl<0.8 else "normal")
        retail=state.get("retail_now",0.5)
        mc(m5,"Retail drive score",f"{retail:.2f}",
           "#dc2626" if retail>0.6 else "#16a34a",
           "retail-driven" if retail>0.6 else "institutional")
        mc(m6,"Driver","🏦 Institutional" if state.get("inst_bias") else "👥 Retail",
           "#2563eb" if state.get("inst_bias") else "#f59e0b")

    # ── Cycle chart ───────────────────────────────────────────────────
    st.markdown("<br>",unsafe_allow_html=True)
    fig_c=cycle_chart(df,pivots,cycles,state,name)
    st.plotly_chart(fig_c,use_container_width=True)
    st.markdown(
        "<span style='font-size:0.72rem;color:#94a3b8'>"
        "🔺 Red triangle = cycle peak · 🔻 Green triangle = cycle trough · "
        "Filled = volume-confirmed · Hollow = unconfirmed · "
        "Green shading = upswing · Red shading = downswing · "
        "Orange line = retail proxy (high + choppy = retail)</span>",
        unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 2 — CYCLE STATISTICS
    # ════════════════════════════════════════════════════════════════
    if cycles:
        st.markdown("### 📊 Cycle Statistics")

        cs1,cs2=st.columns(2)
        with cs1:
            st.plotly_chart(cycle_duration_hist(cycles),use_container_width=True)

        with cs2:
            st.markdown("**Cycle table (most recent first)**")
            cyc_df=pd.DataFrame(cycles).sort_values("start",ascending=False)
            disp=cyc_df[["start","peak","end","amplitude_pct","drawdown_pct",
                          "dur_up_days","dur_dn_days","vol_up","vol_dn",
                          "retail_score","confirmed"]].head(10).copy()
            disp["start"]=disp["start"].dt.strftime("%Y-%m-%d")
            disp["peak"] =disp["peak"].dt.strftime("%Y-%m-%d")
            disp["end"]  =disp["end"].dt.strftime("%Y-%m-%d")
            disp.columns=["Start","Peak","End","Amp %","Draw %",
                           "Up days","Dn days","Vol↑","Vol↓","Retail","Confirmed"]
            st.dataframe(disp.style.format({
                "Amp %":"{:.1f}","Draw %":"{:.1f}",
                "Vol↑":"{:.2f}","Vol↓":"{:.2f}","Retail":"{:.2f}"}),
                use_container_width=True,hide_index=True)

        # Retail vs Institutional across cycles
        st.markdown("**Retail vs Institutional drive per cycle**")
        cyc_df2=pd.DataFrame(cycles)
        fig_ri=go.Figure()
        fig_ri.add_trace(go.Bar(
            x=[str(c["start"])[:10] for c in cycles],
            y=[c["retail_score"] for c in cycles],
            name="Retail score",
            marker_color=["#dc2626" if c["retail_score"]>0.5 else "#16a34a"
                          for c in cycles],opacity=0.8))
        fig_ri.add_hline(y=0.5,line_dash="dot",line_color="#94a3b8",line_width=1,
                          annotation_text="Retail/Inst boundary",
                          annotation_position="right")
        fig_ri.update_layout(height=200,margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white",paper_bgcolor="white",
            xaxis=dict(tickangle=30,gridcolor="#f1f5f9"),
            yaxis=dict(title="Score (>0.5=retail)",gridcolor="#f1f5f9",range=[0,1.2]))
        st.plotly_chart(fig_ri,use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 3 — ML ON CYCLE STATE
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 🧠 ML Rules (Cycle-State Features)")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "Decision tree trained on WHERE YOU ARE in a cycle, not what day of the week it is. "
        "Features: % through cycle amplitude, % through cycle duration, volume vs leg avg, "
        "retail score, RSI, BB position.</span>",
        unsafe_allow_html=True)

    with st.spinner("Building cycle features and training…"):
        feat=build_cycle_features(df,pivots,cycles)

    if feat.empty or len(feat)<15:
        st.info("Not enough cycle data for ML yet — need more complete cycles. "
                "Try a longer period or lower the ZigZag threshold.")
    else:
        clf_dir,acc_dir=train_cycle_model(feat,"next_up",depth)
        clf_big,acc_big=train_cycle_model(feat,"next_big",depth)

        pred_dir,prob_dir=predict_cycle(clf_dir,feat)
        pred_big,prob_big=predict_cycle(clf_big,feat)

        # Prediction cards
        p1,p2,p3=st.columns(3)
        if pred_dir is not None:
            dl="#16a34a" if pred_dir==1 else "#dc2626"
            dt="📈 UP" if pred_dir==1 else "📉 DOWN"
            p1.markdown(
                f"<div style='border:2px solid {dl};border-radius:10px;"
                f"padding:14px;text-align:center'>"
                f"<div style='font-size:0.72rem;color:#64748b'>Next-day direction</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{dl}'>{dt}</div>"
                f"<div style='font-size:0.75rem;color:{dl}'>"
                f"{max(prob_dir)*100:.0f}% confidence</div>"
                f"<div style='font-size:0.68rem;color:#94a3b8'>acc: {acc_dir*100:.0f}%</div>"
                f"</div>",unsafe_allow_html=True)
        if pred_big is not None:
            bl="#dc2626" if pred_big==1 else "#94a3b8"
            bt="⚡ BIG SWING ≥20" if pred_big==1 else "😴 QUIET DAY"
            p2.markdown(
                f"<div style='border:2px solid {bl};border-radius:10px;"
                f"padding:14px;text-align:center'>"
                f"<div style='font-size:0.72rem;color:#64748b'>Tomorrow's range</div>"
                f"<div style='font-size:1.2rem;font-weight:700;color:{bl}'>{bt}</div>"
                f"<div style='font-size:0.75rem;color:{bl}'>"
                f"{max(prob_big)*100:.0f}%</div>"
                f"<div style='font-size:0.68rem;color:#94a3b8'>acc: {acc_big*100:.0f}%</div>"
                f"</div>",unsafe_allow_html=True)
        if state:
            retail=state.get("retail_now",0.5)
            rc="#dc2626" if retail>0.6 else "#2563eb"
            rl="👥 RETAIL" if retail>0.6 else "🏦 INSTITUTIONAL"
            p3.markdown(
                f"<div style='border:2px solid {rc};border-radius:10px;"
                f"padding:14px;text-align:center'>"
                f"<div style='font-size:0.72rem;color:#64748b'>Current driver</div>"
                f"<div style='font-size:1.2rem;font-weight:700;color:{rc}'>{rl}</div>"
                f"<div style='font-size:0.78rem;color:#475569;margin-top:4px'>"
                f"Score: {retail:.2f} · "
                f"{'High vol + choppy = retail crowd' if retail>0.6 else 'Clean directional = institutional'}"
                f"</div></div>",unsafe_allow_html=True)

        # Rules
        st.markdown("<br>",unsafe_allow_html=True)
        r1,r2=st.columns(2)
        fn=[{"leg":"Cycle leg (1=up,-1=dn)","days_in_leg":"Days in current leg",
              "move_pct":"Move % from last pivot","pct_of_avg_amp":"% of avg amplitude moved",
              "pct_of_avg_dur":"% of avg duration elapsed","vol_ratio":"Vol vs 20d avg",
              "retail_score":"Retail drive score","body_ratio":"Candle body ratio",
              "gap_pct":"Gap %","rsi":"RSI","bb_pct":"BB position %",
              "atr":"ATR","weekday":"Weekday"}.get(c,c) for c in CYCLE_FEAT_COLS]
        with r1:
            st.markdown("**Direction rules (0=down, 1=up)**")
            if clf_dir:
                imp=clf_dir.feature_importances_
                order=np.argsort(imp)[::-1][:8]
                fig_i=go.Figure(go.Bar(y=[fn[i] for i in order],
                    x=[imp[i]*100 for i in order],orientation="h",
                    marker_color="#2563eb",opacity=0.8,
                    text=[f"{imp[i]*100:.1f}%" for i in order],textposition="outside"))
                fig_i.update_layout(height=260,margin=dict(l=0,r=60,t=10,b=0),
                    plot_bgcolor="white",paper_bgcolor="white",
                    xaxis=dict(gridcolor="#f1f5f9"),yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_i,use_container_width=True)
                st.code(export_text(clf_dir,feature_names=fn,max_depth=depth),
                        language="text")
        with r2:
            st.markdown("**Big swing rules (0=quiet, 1=big)**")
            if clf_big:
                imp2=clf_big.feature_importances_
                order2=np.argsort(imp2)[::-1][:8]
                fig_i2=go.Figure(go.Bar(y=[fn[i] for i in order2],
                    x=[imp2[i]*100 for i in order2],orientation="h",
                    marker_color="#16a34a",opacity=0.8,
                    text=[f"{imp2[i]*100:.1f}%" for i in order2],textposition="outside"))
                fig_i2.update_layout(height=260,margin=dict(l=0,r=60,t=10,b=0),
                    plot_bgcolor="white",paper_bgcolor="white",
                    xaxis=dict(gridcolor="#f1f5f9"),yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_i2,use_container_width=True)
                st.code(export_text(clf_big,feature_names=fn,max_depth=depth),
                        language="text")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 4 — PEER CONTEXT
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 👥 Peer Context — Same Cycle Phase?")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>"
        "High-beta + sector peers + pattern-similar stocks. "
        "When multiple related stocks are at the same cycle phase → stronger signal.</span>",
        unsafe_allow_html=True)

    peer_opts=st.multiselect(
        "Include peer groups",
        ["AI/Tech","EV","NewEcon","Biotech","Finance"],
        default=["AI/Tech","EV"],key="cml_peers")

    if st.button("🔍 Scan peers",key="cml_scan_peers"):
        selected_peers=[t for t,(n,s) in PEERS.items() if s in peer_opts]
        if ticker in selected_peers: selected_peers.remove(ticker)
        prog=st.progress(0,"Scanning peers…")
        peer_results=[]
        for i,pt in enumerate(selected_peers[:20]):
            r=fetch_peer_snapshot(pt,period,min_pct)
            if r: peer_results.append(r)
            prog.progress((i+1)/min(len(selected_peers),20),
                          text=f"Scanning {pt}…")
        prog.empty()
        st.session_state["cml_peers_data"]=peer_results

    peer_results=st.session_state.get("cml_peers_data",[])
    if peer_results:
        pdf=pd.DataFrame(peer_results).sort_values("pct_through",ascending=False)

        # Phase alignment chart
        st.markdown("**Cycle phase alignment across peers**")
        phase_colors={"UP":"#16a34a","DOWN":"#dc2626","—":"#94a3b8"}
        fig_phase=go.Figure(go.Bar(
            x=pdf["name"],y=pdf["pct_through"],
            marker_color=[phase_colors.get(r,"#94a3b8") for r in pdf["leg"]],
            text=[f"{r['cycle_state'][:12]}\n{r['pct_through']:.0f}%"
                  for _,r in pdf.iterrows()],
            textposition="outside"))
        fig_phase.add_hline(y=100,line_dash="dot",line_color="#94a3b8",line_width=1,
                             annotation_text="100% = full cycle",annotation_position="right")
        fig_phase.update_layout(height=300,margin=dict(l=0,r=0,t=10,b=30),
            plot_bgcolor="white",paper_bgcolor="white",
            xaxis=dict(tickangle=30,gridcolor="#f1f5f9"),
            yaxis=dict(title="% through cycle",gridcolor="#f1f5f9"))
        st.plotly_chart(fig_phase,use_container_width=True)

        # Peer table
        disp_p=pdf[["name","sector","price","cycle_state","pct_through",
                     "retail_now","avg_amp","beta","vol_ratio"]].copy()
        disp_p.columns=["Name","Sector","Price","Cycle state","% through",
                         "Retail","Avg amp %","Beta","Vol ×"]
        st.dataframe(disp_p.style.format({
            "Price":"{:.1f}","% through":"{:.0f}%","Retail":"{:.2f}",
            "Avg amp %":"{:.1f}%","Beta":"{:.2f}","Vol ×":"{:.2f}×"
        }),use_container_width=True,hide_index=True)

        # Convergence alert
        same_phase=pdf[pdf["leg"]==state.get("leg","—")] if state else pdf
        if len(same_phase)>=3:
            st.success(
                f"**{len(same_phase)} peers are in the same {state.get('leg','?')} leg** — "
                f"sector-wide move in progress, not just this stock.")
        elif len(same_phase)>=1:
            st.info(f"{len(same_phase)} peer(s) in same phase — partial alignment.")
        else:
            st.warning("No peers in same cycle phase — this stock is moving alone (idiosyncratic).")

    st.markdown(
        "<span style='color:#94a3b8;font-size:0.74rem'>"
        "Cycle detection is probabilistic. Adjust ZigZag threshold to find "
        "cycles that match your trading timeframe. Not financial advice.</span>",
        unsafe_allow_html=True)
