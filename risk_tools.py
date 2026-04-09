"""
risk_tools.py — Risk Management Toolkit
Four tools in one page:
  1. Pre-market checklist  — daily briefing before 9:30
  2. Position sizing calc  — exact shares given capital + risk %
  3. Trade log            — log trades, track P&L, win rate
  4. Weekly review        — end-of-week performance summary
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time, pytz

from db_manager import get_conn, get_portfolio_full, get_latest_capital, init_db
from portfolio_manager import get_monitor_pos

HK_TZ = pytz.timezone("Asia/Hong_Kong")

# ── DB SETUP ──────────────────────────────────────────────────────────
def init_risk_tables():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trade_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            name        TEXT,
            direction   TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price  REAL,
            shares      REAL NOT NULL,
            stop_price  REAL,
            target_price REAL,
            pnl         REAL,
            pnl_pct     REAL,
            outcome     TEXT,
            setup       TEXT,
            notes       TEXT,
            status      TEXT DEFAULT 'OPEN',
            logged_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS signal_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            signal_type TEXT,
            signal_val  TEXT,
            recommended TEXT,
            actual_next_day_pct REAL,
            was_correct INTEGER,
            logged_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS weekly_review (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start  TEXT NOT NULL,
            week_end    TEXT NOT NULL,
            total_pnl   REAL,
            trades_won  INTEGER,
            trades_lost INTEGER,
            best_trade  TEXT,
            worst_trade TEXT,
            notes       TEXT,
            lessons     TEXT,
            logged_at   TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit(); conn.close()

def get_trades(status=None, limit=200):
    conn = get_conn()
    q = "SELECT * FROM trade_log"
    if status: q += f" WHERE status='{status}'"
    q += " ORDER BY logged_at DESC LIMIT ?"
    df = pd.read_sql_query(q, conn, params=(limit,))
    conn.close()
    return df

def log_trade(date, ticker, name, direction, entry, exit_p, shares,
              stop, target, pnl, pnl_pct, outcome, setup, notes, status):
    conn = get_conn()
    conn.execute("""
        INSERT INTO trade_log
        (date,ticker,name,direction,entry_price,exit_price,shares,
         stop_price,target_price,pnl,pnl_pct,outcome,setup,notes,status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (date,ticker,name,direction,entry,exit_p,shares,
          stop,target,pnl,pnl_pct,outcome,setup,notes,status))
    conn.commit(); conn.close()

def close_trade(trade_id, exit_price, notes=""):
    conn = get_conn()
    row = pd.read_sql_query(
        "SELECT * FROM trade_log WHERE id=?", conn, params=(trade_id,)).iloc[0]
    entry = float(row["entry_price"]); shares = float(row["shares"])
    direction = row["direction"]
    pnl = (exit_price-entry)*shares if direction=="LONG" else (entry-exit_price)*shares
    pnl_pct = (exit_price-entry)/entry*100 if direction=="LONG" else (entry-exit_price)/entry*100
    outcome = "WIN" if pnl>0 else "LOSS" if pnl<0 else "BREAK EVEN"
    conn.execute("""
        UPDATE trade_log SET exit_price=?,pnl=?,pnl_pct=?,outcome=?,status='CLOSED',
        notes=COALESCE(notes,'')||?,logged_at=datetime('now') WHERE id=?
    """, (exit_price, round(pnl,2), round(pnl_pct,2), outcome,
          f" | Exit: {notes}" if notes else "", trade_id))
    conn.commit(); conn.close()

def save_weekly_review(ws, we, pnl, won, lost, best, worst, notes, lessons):
    conn = get_conn()
    conn.execute("""
        INSERT INTO weekly_review
        (week_start,week_end,total_pnl,trades_won,trades_lost,
         best_trade,worst_trade,notes,lessons)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (ws,we,pnl,won,lost,best,worst,notes,lessons))
    conn.commit(); conn.close()

# ── DATA HELPERS ──────────────────────────────────────────────────────
def _var(t):
    v=[t]; code=t.replace(".HK","")
    if code.isdigit():
        v.append(str(int(code))+".HK"); v.append(code.zfill(4)+".HK")
    return list(dict.fromkeys(v))

@st.cache_data(ttl=60, show_spinner=False)
def fetch_q(ticker):
    for t in _var(ticker):
        try:
            info=yf.Ticker(t).fast_info
            p=getattr(info,"last_price",None)
            if p: return {
                "price":float(p),
                "prev":getattr(info,"previous_close",None),
                "open":getattr(info,"open",None),
            }
        except Exception: pass
    return {}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_d(ticker, period="5d"):
    for t in _var(ticker):
        try:
            df=yf.Ticker(t).history(period=period,interval="1d",auto_adjust=True)
            if len(df)>=3: return df
        except Exception: pass
    return pd.DataFrame()

# ── MAIN RENDER ───────────────────────────────────────────────────────
def render():
    init_risk_tables()
    now_hk = datetime.now(HK_TZ)
    capital = get_latest_capital()

    st.markdown(
        "## 🛡 Risk Management &nbsp;"
        "<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        "padding:2px 7px;border-radius:5px'>TOOLKIT</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"Pre-market checklist · Position sizing · Trade log · Weekly review · "
        f"{now_hk.strftime('%A %d %b %Y  %H:%M HKT')}</span>",
        unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "☀️ Pre-Market",
        "📐 Position Sizing",
        "📓 Trade Log",
        "📅 Weekly Review",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — PRE-MARKET CHECKLIST
    # ════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### ☀️ Pre-Market Checklist")
        st.markdown(
            "<span style='color:#64748b;font-size:0.8rem'>"
            "Run this before 09:30. Sets the context for the day.</span>",
            unsafe_allow_html=True)

        if st.button("🔄 Refresh checklist", key="rt_premarket"):
            st.cache_data.clear(); st.rerun()

        # ── Market context ────────────────────────────────────────
        st.markdown("#### 🌍 Market Context")
        with st.spinner("Fetching market data…"):
            benchmarks = {
                "^HSI":   "Hang Seng Index",
                "^HSCE":  "H-Share Index",
                "SPY":    "S&P 500 ETF",
                "QQQ":    "Nasdaq 100 ETF",
                "GC=F":   "Gold",
                "DX-Y.NYB":"USD Index",
                "USDHKD=X":"USD/HKD",
            }
            bm_rows = []
            for sym, name in benchmarks.items():
                q = fetch_q(sym)
                p = q.get("price"); prev = q.get("prev")
                if p and prev:
                    chg = (p-prev)/prev*100
                    bm_rows.append({"name":name,"price":f"{p:,.2f}",
                                    "chg":chg,"sym":sym})
                time.sleep(0.1)

        # Show as cards
        bc = st.columns(4)
        for i,r in enumerate(bm_rows[:8]):
            c = "#16a34a" if r["chg"]>=0 else "#dc2626"
            bc[i%4].markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                f"border-radius:8px;padding:9px 12px;margin-bottom:6px;"
                f"border-left:3px solid {c}'>"
                f"<div style='font-size:0.68rem;color:#94a3b8'>{r['name']}</div>"
                f"<div style='font-size:0.88rem;font-weight:600'>{r['price']}</div>"
                f"<div style='font-size:0.75rem;color:{c}'>{r['chg']:+.2f}%</div>"
                f"</div>", unsafe_allow_html=True)

        # ── Position gaps ─────────────────────────────────────────
        st.markdown("#### 📊 Your Positions — Overnight Gap")
        stock_df = get_portfolio_full()
        mon_df   = get_monitor_pos()
        all_pos  = []
        if not stock_df.empty:
            for _,r in stock_df[stock_df["status"]=="OPEN"].iterrows():
                all_pos.append({"ticker":r["ticker"],"name":r.get("name",r["ticker"]),
                                 "avg_cost":float(r.get("avg_cost",0) or 0),
                                 "qty":float(r.get("shares",0) or 0),
                                 "stop":r.get("stop_price")})
        if not mon_df.empty:
            for _,r in mon_df[mon_df["status"]=="OPEN"].iterrows():
                all_pos.append({"ticker":r["ticker"],"name":r.get("name",r["ticker"]),
                                 "avg_cost":float(r.get("avg_cost",0) or 0),
                                 "qty":float(r.get("quantity",0) or 0),
                                 "stop":r.get("stop")})

        if all_pos:
            gap_rows = []
            with st.spinner("Checking gaps…"):
                for p in all_pos:
                    q = fetch_q(p["ticker"])
                    price=q.get("price"); prev=q.get("prev"); op=q.get("open")
                    gap = (op-prev)/prev*100 if op and prev else None
                    day_chg = (price-prev)/prev*100 if price and prev else None
                    pnl = (price-p["avg_cost"])*p["qty"] if price and p["qty"]>0 else None
                    stop = p.get("stop")
                    near_stop = (price-stop)/price*100 < 3 if (price and stop) else False
                    gap_rows.append({
                        "Name":     p["name"],
                        "Ticker":   p["ticker"],
                        "Price":    f"{price:,.2f}" if price else "—",
                        "Gap":      f"{gap:+.2f}%" if gap else "—",
                        "Day %":    f"{day_chg:+.2f}%" if day_chg else "—",
                        "P&L":      f"{'+'if (pnl or 0)>=0 else ''}{pnl:,.0f}" if pnl else "—",
                        "Stop":     f"{stop:,.2f}" if stop else "—",
                        "⚠️":      "🔴 NEAR STOP" if near_stop else "",
                        "_gap":     gap or 0,
                        "_near":    near_stop,
                    })
                    time.sleep(0.1)

            df_gap = pd.DataFrame(gap_rows)
            def style_gap(df):
                s=pd.DataFrame("",index=df.index,columns=df.columns)
                for i,row in df.iterrows():
                    g=float(str(row["Gap"]).replace("%","").replace("+","").replace("—","0") or 0)
                    if g>=2:    s.at[i,"Gap"]="color:#16a34a;font-weight:700"
                    elif g<=-2: s.at[i,"Gap"]="color:#dc2626;font-weight:700"
                    if row["⚠️"]: 
                        for c in df.columns: s.at[i,c]="background:rgba(220,38,38,0.07)"
                    v=str(row["P&L"])
                    if v.startswith("+"): s.at[i,"P&L"]="color:#16a34a;font-weight:600"
                    elif v.startswith("-"): s.at[i,"P&L"]="color:#dc2626;font-weight:600"
                return s
            disp = df_gap[[c for c in df_gap.columns if not c.startswith("_")]]
            st.dataframe(disp.style.apply(style_gap,axis=None),
                         use_container_width=True, hide_index=True)

        # ── Day type assessment ───────────────────────────────────
        st.markdown("#### 🎯 Today's Setup Assessment")
        wd = now_hk.weekday()
        wd_names = ["Monday","Tuesday","Wednesday","Thursday","Friday"]

        # HSI overnight
        hsi_chg = next((r["chg"] for r in bm_rows if "Hang Seng" in r["name"]),0)
        spx_chg = next((r["chg"] for r in bm_rows if "S&P" in r["name"]),0)
        usd_chg = next((r["chg"] for r in bm_rows if "USD Index" in r["name"]),0)

        flags = []
        score = 50  # neutral

        if wd==0:
            flags.append(("⚠️ Monday","Gap-trap day. Wait for direction after 10:00. Reduce position size.", "#f59e0b"))
            score -= 15
        elif wd==4:
            flags.append(("⚠️ Friday","Position squaring day. Stops get hunted before close. Take profits early.", "#f59e0b"))
            score -= 10
        elif wd in [1,2]:
            flags.append(("✅ Tue/Wed","Cleanest institutional flow days. Best for following trends.", "#16a34a"))
            score += 10

        if abs(hsi_chg) > 1.5:
            if hsi_chg > 0:
                flags.append(("📈 HSI gap up",f"Hang Seng +{hsi_chg:.1f}%. Watch for fade if retail chasing.", "#16a34a"))
            else:
                flags.append(("📉 HSI gap down",f"Hang Seng {hsi_chg:.1f}%. Wait for support before entering.", "#dc2626"))
            score -= 10  # gaps = uncertainty

        if spx_chg < -1:
            flags.append(("🇺🇸 US weak overnight",f"S&P {spx_chg:.1f}%. Expect HK tech to face selling pressure.", "#dc2626"))
            score -= 10
        elif spx_chg > 1:
            flags.append(("🇺🇸 US strong overnight",f"S&P +{spx_chg:.1f}%. Positive for HK tech sentiment.", "#16a34a"))
            score += 8

        if usd_chg > 0.3:
            flags.append(("💵 USD strengthening","Strong USD = headwind for HK growth stocks.", "#f59e0b"))
            score -= 5

        score = max(0,min(100,score))
        sc_c = "#16a34a" if score>=60 else "#f59e0b" if score>=40 else "#dc2626"
        sc_l = "🟢 Good trading day" if score>=60 else "🟡 Caution — mixed signals" if score>=40 else "🔴 High risk — reduce exposure"

        st.markdown(
            f"<div style='border:2px solid {sc_c};border-radius:12px;padding:14px 18px;"
            f"background:rgba(0,0,0,0.02);margin-bottom:12px'>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{sc_c}'>{sc_l}</div>"
            f"<div style='font-size:0.8rem;color:#64748b;margin-top:4px'>"
            f"Day score: {score}/100 · {wd_names[wd] if wd<5 else 'Weekend'}</div>"
            f"</div>", unsafe_allow_html=True)

        for icon, msg, col in flags:
            st.markdown(
                f"<div style='border-left:3px solid {col};padding:8px 12px;"
                f"background:rgba(0,0,0,0.02);border-radius:0 6px 6px 0;margin-bottom:6px;"
                f"font-size:0.82rem'><b style='color:{col}'>{icon}</b> — {msg}</div>",
                unsafe_allow_html=True)

        # ── Risk summary ──────────────────────────────────────────
        st.markdown("#### 🛡 Portfolio Risk Right Now")
        if all_pos:
            total_risk = 0
            risk_rows = []
            for p in all_pos:
                q = fetch_q(p["ticker"])
                price = q.get("price") or p["avg_cost"]
                stop  = p.get("stop")
                qty   = p["qty"]
                if stop and price and qty>0:
                    risk_hkd = (price-stop)*qty
                    risk_pct = risk_hkd/capital*100 if capital>0 else 0
                    total_risk += risk_hkd
                    risk_rows.append({
                        "Position":   p["name"],
                        "Risk HKD":   f"{risk_hkd:,.0f}",
                        "Risk % cap": f"{risk_pct:.2f}%",
                        "Status":     "✅ OK" if risk_pct<=2 else "⚠️ HIGH" if risk_pct<=4 else "🔴 OVER",
                    })

            if risk_rows:
                total_pct = total_risk/capital*100 if capital>0 else 0
                heat_c = "#16a34a" if total_pct<=5 else "#f59e0b" if total_pct<=10 else "#dc2626"
                st.markdown(
                    f"<div style='display:flex;gap:20px;padding:10px 14px;"
                    f"background:#f8fafc;border-radius:8px;margin-bottom:8px;"
                    f"font-size:0.85rem;border:1px solid #e2e8f0'>"
                    f"<span>Portfolio heat: <b style='color:{heat_c}'>"
                    f"HKD {total_risk:,.0f} ({total_pct:.1f}%)</b></span>"
                    f"<span style='color:#64748b'>Target: ≤5% · Max: 10%</span>"
                    f"{'<span style=color:#dc2626;font-weight:600>⚠️ REDUCE RISK</span>' if total_pct>8 else ''}"
                    f"</div>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(risk_rows),
                             use_container_width=True, hide_index=True)
            else:
                st.info("Set stop prices in Portfolio to see risk metrics.")

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — POSITION SIZING
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📐 Position Sizing Calculator")
        st.markdown(
            "<span style='color:#64748b;font-size:0.8rem'>"
            "Calculates exact shares so your dollar risk stays constant "
            "regardless of which stock you trade.</span>",
            unsafe_allow_html=True)

        with st.expander("📖 How position sizing works"):
            st.markdown("""
**Core rule:** Never risk more than 1-2% of total capital on one trade.

**Formula:** `Shares = Max risk HKD ÷ (Entry price − Stop price)`

**Example:** Capital HKD 500,000 · Risk 1% = HKD 5,000 max loss.
Entry 200 HKD · Stop 185 HKD → distance = 15 HKD.
Shares = 5,000 ÷ 15 = **333 shares**.

**Portfolio heat:** Sum of all open position risks. Keep below 5-6% total.
If 4 positions all at 1.5% = 6% heat. If all stops hit → you lose 6%.

**Cycle adjustment:**
- Early cycle (score <25%): full size
- Mid cycle (25-55%): normal size  
- Late cycle (55-80%): 75% size
- Exhaustion (>80%): 50% size or skip
            """)

        pc1, pc2 = st.columns(2)
        cap_   = pc1.number_input("Total capital (HKD)", value=float(capital),
                                    min_value=1000.0, step=10000.0, format="%.0f",
                                    key="ps_cap")
        risk_pct_ = pc2.slider("Max risk per trade %", 0.5, 3.0, 1.0, 0.1,
                                key="ps_risk")

        st.markdown("---")
        sc1,sc2,sc3 = st.columns(3)
        entry_ = sc1.number_input("Entry price", min_value=0.01,
                                    value=100.0, step=0.1, format="%.2f",
                                    key="ps_entry")
        stop_  = sc2.number_input("Stop price", min_value=0.01,
                                    value=90.0, step=0.1, format="%.2f",
                                    key="ps_stop")
        target_= sc3.number_input("Target price", min_value=0.01,
                                    value=120.0, step=0.1, format="%.2f",
                                    key="ps_target")

        cycle_adj = st.select_slider(
            "Cycle position (adjust size)",
            options=["Early (<25%)", "Mid (25-55%)", "Late (55-80%)", "Exhaustion (>80%)"],
            value="Mid (25-55%)", key="ps_cycle")
        cycle_mult = {"Early (<25%)":1.0,"Mid (25-55%)":1.0,
                      "Late (55-80%)":0.75,"Exhaustion (>80%)":0.5}[cycle_adj]

        if entry_ > stop_:
            max_risk_hkd = cap_ * risk_pct_/100
            dist = entry_ - stop_
            shares_raw = max_risk_hkd / dist * cycle_mult
            shares = max(int(shares_raw), 1)

            cost      = shares * entry_
            act_risk  = shares * dist
            act_risk_pct = act_risk/cap_*100
            reward    = shares * (target_ - entry_)
            rr        = reward/act_risk if act_risk>0 else 0
            rr_c      = "#16a34a" if rr>=2 else "#f59e0b" if rr>=1 else "#dc2626"

            # Result cards
            rc1,rc2,rc3,rc4,rc5 = st.columns(5)
            for col,lbl,val,color in [
                (rc1,"Shares to buy",    f"{shares:,}",              "#0f172a"),
                (rc2,"Capital required", f"HKD {cost:,.0f}",         "#0f172a"),
                (rc3,"Max loss (stop)",  f"HKD {act_risk:,.0f}",     "#dc2626"),
                (rc4,"Risk % of capital",f"{act_risk_pct:.2f}%",
                 "#16a34a" if act_risk_pct<=1.5 else "#f59e0b" if act_risk_pct<=2.5 else "#dc2626"),
                (rc5,"R:R ratio",        f"1:{rr:.1f}",              rr_c),
            ]:
                col.markdown(
                    f"<div style='text-align:center;background:#f8fafc;border:1px solid #e2e8f0;"
                    f"border-radius:10px;padding:12px 8px'>"
                    f"<div style='font-size:0.68rem;color:#94a3b8'>{lbl}</div>"
                    f"<div style='font-size:1.05rem;font-weight:700;color:{color}'>{val}</div>"
                    f"</div>", unsafe_allow_html=True)

            st.markdown("<br>",unsafe_allow_html=True)

            # Scenario table
            st.markdown("**Size scenarios**")
            scen_df = pd.DataFrame([{
                "Risk %":  f"{rp:.1f}%",
                "Max loss":f"HKD {cap_*rp/100:,.0f}",
                "Shares":  f"{max(int(cap_*rp/100/dist*cycle_mult),1):,}",
                "Cost":    f"HKD {max(int(cap_*rp/100/dist*cycle_mult),1)*entry_:,.0f}",
                "R:R":     f"1:{reward/max(max(int(cap_*rp/100/dist*cycle_mult),1)*dist,1):.1f}",
            } for rp in [0.5,1.0,1.5,2.0,2.5,3.0]])
            st.dataframe(scen_df, use_container_width=True, hide_index=True)

            # Portfolio heat check
            st.markdown("**Portfolio heat check**")
            open_trades = get_trades(status="OPEN")
            existing_risk = 0
            if not open_trades.empty:
                for _,t in open_trades.iterrows():
                    ep=float(t.get("entry_price",0) or 0)
                    sp=float(t.get("stop_price",0) or 0)
                    sh=float(t.get("shares",0) or 0)
                    if ep>sp>0 and sh>0:
                        existing_risk += (ep-sp)*sh

            new_heat = (existing_risk+act_risk)/cap_*100
            heat_c   = "#16a34a" if new_heat<=5 else "#f59e0b" if new_heat<=8 else "#dc2626"
            st.markdown(
                f"<div style='padding:10px 14px;border-radius:8px;background:#f8fafc;"
                f"border:1px solid #e2e8f0;font-size:0.82rem'>"
                f"Existing risk: HKD {existing_risk:,.0f} · "
                f"New position adds: HKD {act_risk:,.0f} · "
                f"New total heat: <b style='color:{heat_c}'>{new_heat:.1f}%</b>"
                f"{'  ⚠️ Over 5% — consider smaller size' if new_heat>5 else '  ✅ Within limits'}"
                f"</div>", unsafe_allow_html=True)
        else:
            st.warning("Stop price must be below entry price.")

    # ════════════════════════════════════════════════════════════════
    # TAB 3 — TRADE LOG
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 📓 Trade Log")

        log_tab1, log_tab2 = st.tabs(["📝 Log a Trade", "📊 Performance Stats"])

        with log_tab1:
            st.markdown("**Log a new trade**")
            lc1,lc2,lc3 = st.columns(3)
            l_date   = lc1.date_input("Date", value=datetime.now(HK_TZ).date(),
                                        key="tl_date")
            l_ticker = lc2.text_input("Ticker", placeholder="0100.HK",
                                       key="tl_ticker").strip().upper()
            l_name   = lc3.text_input("Name", placeholder="MiniMax",
                                       key="tl_name").strip()
            lc4,lc5,lc6 = st.columns(3)
            l_dir    = lc4.selectbox("Direction", ["LONG","SHORT"], key="tl_dir")
            l_entry  = lc5.number_input("Entry price", min_value=0.01,
                                         value=100.0, step=0.01, format="%.4f",
                                         key="tl_entry")
            l_shares = lc6.number_input("Shares/Qty", min_value=0.01,
                                          value=100.0, step=1.0, format="%.0f",
                                          key="tl_shares")
            lc7,lc8 = st.columns(2)
            l_stop   = lc7.number_input("Stop price", min_value=0.0,
                                          value=0.0, step=0.01, format="%.4f",
                                          key="tl_stop")
            l_target = lc8.number_input("Target price", min_value=0.0,
                                          value=0.0, step=0.01, format="%.4f",
                                          key="tl_target")
            l_setup  = st.selectbox("Setup type",
                ["Breakout","Pullback","Gap fade","Reversal","Range trade",
                 "Trend follow","Oversold bounce","Other"],
                key="tl_setup")
            l_notes  = st.text_area("Notes (why this trade?)", height=68,
                                     key="tl_notes",
                                     placeholder="Signal: RSI oversold + choppiness 65 + sector inflow…")

            if st.button("➕ Log trade", key="tl_log"):
                if l_ticker:
                    log_trade(str(l_date), l_ticker, l_name or l_ticker,
                              l_dir, l_entry, None, l_shares,
                              l_stop or None, l_target or None,
                              None, None, None, l_setup, l_notes, "OPEN")
                    st.success(f"✅ {l_ticker} trade logged!"); st.rerun()

            # Close open trades
            open_t = get_trades(status="OPEN")
            if not open_t.empty:
                st.markdown("---")
                st.markdown("**Close an open trade**")
                trade_opts = [f"#{r['id']} {r['ticker']} {r['direction']} @ {r['entry_price']}"
                              for _,r in open_t.iterrows()]
                sel_trade = st.selectbox("Select trade to close",
                                          trade_opts, key="tl_sel_close")
                if sel_trade:
                    trade_id = int(sel_trade.split("#")[1].split(" ")[0])
                    row_ = open_t[open_t["id"]==trade_id].iloc[0]
                    close_c1, close_c2 = st.columns(2)
                    exit_p = close_c1.number_input(
                        "Exit price",
                        value=float(fetch_q(row_["ticker"]).get("price") or row_["entry_price"]),
                        step=0.01, format="%.4f", key="tl_exit_p")
                    exit_n = close_c2.text_input("Close note", key="tl_exit_n")
                    if st.button("✅ Close trade", key="tl_close_btn"):
                        close_trade(trade_id, exit_p, exit_n)
                        st.success("Trade closed!"); st.rerun()

        with log_tab2:
            all_trades = get_trades()
            if all_trades.empty:
                st.info("No trades logged yet.")
            else:
                closed = all_trades[all_trades["status"]=="CLOSED"]
                open_t = all_trades[all_trades["status"]=="OPEN"]

                # Stats
                if not closed.empty:
                    total_pnl  = closed["pnl"].sum()
                    wins       = closed[closed["pnl"]>0]
                    losses     = closed[closed["pnl"]<=0]
                    win_rate   = len(wins)/len(closed)*100
                    avg_win    = wins["pnl"].mean() if len(wins)>0 else 0
                    avg_loss   = abs(losses["pnl"].mean()) if len(losses)>0 else 0
                    expectancy = (win_rate/100*avg_win) - ((1-win_rate/100)*avg_loss)
                    profit_f   = avg_win/avg_loss if avg_loss>0 else 0

                    s1,s2,s3,s4,s5 = st.columns(5)
                    for col,lbl,val,color in [
                        (s1,"Total P&L",   f"{'+'if total_pnl>=0 else ''}{total_pnl:,.0f}",
                         "#16a34a" if total_pnl>=0 else "#dc2626"),
                        (s2,"Win rate",    f"{win_rate:.0f}%",
                         "#16a34a" if win_rate>=55 else "#f59e0b" if win_rate>=45 else "#dc2626"),
                        (s3,"Avg win",     f"+{avg_win:,.0f}","#16a34a"),
                        (s4,"Avg loss",    f"-{avg_loss:,.0f}","#dc2626"),
                        (s5,"Expectancy",  f"{expectancy:+.0f}/trade",
                         "#16a34a" if expectancy>0 else "#dc2626"),
                    ]:
                        col.markdown(
                            f"<div style='text-align:center;background:#f8fafc;"
                            f"border:1px solid #e2e8f0;border-radius:10px;padding:10px'>"
                            f"<div style='font-size:0.68rem;color:#94a3b8'>{lbl}</div>"
                            f"<div style='font-size:1rem;font-weight:700;color:{color}'>{val}</div>"
                            f"</div>", unsafe_allow_html=True)

                    # Cumulative P&L chart
                    st.markdown("<br>",unsafe_allow_html=True)
                    closed_s = closed.sort_values("logged_at")
                    cumulative = closed_s["pnl"].cumsum()
                    fig_pnl = go.Figure(go.Scatter(
                        x=list(range(len(cumulative))),
                        y=cumulative.values,
                        mode="lines+markers",
                        line=dict(color="#2563eb",width=2),
                        fill="tozeroy",
                        fillcolor="rgba(37,99,235,0.07)"))
                    fig_pnl.add_hline(y=0,line_color="#e2e8f0",line_width=1)
                    fig_pnl.update_layout(height=200,margin=dict(l=0,r=0,t=10,b=0),
                        plot_bgcolor="white",paper_bgcolor="white",
                        xaxis=dict(title="Trade #",gridcolor="#f1f5f9"),
                        yaxis=dict(title="Cumulative P&L",gridcolor="#f1f5f9"))
                    st.plotly_chart(fig_pnl,use_container_width=True)

                    # Setup breakdown
                    if "setup" in closed.columns:
                        setup_stats = closed.groupby("setup").agg(
                            trades=("pnl","count"),
                            total_pnl=("pnl","sum"),
                            win_rate=("outcome",lambda x:(x=="WIN").mean()*100)
                        ).reset_index()
                        st.markdown("**P&L by setup type**")
                        st.dataframe(setup_stats.style.format({
                            "total_pnl":"{:+,.0f}","win_rate":"{:.0f}%"}),
                            use_container_width=True, hide_index=True)

                # Full table
                st.markdown("**All trades**")
                disp = all_trades[["date","ticker","direction","entry_price",
                                   "exit_price","shares","pnl","pnl_pct",
                                   "outcome","setup","status"]].copy()
                disp["pnl"] = disp["pnl"].apply(
                    lambda x: f"{x:+,.0f}" if pd.notna(x) else "—")
                disp["pnl_pct"] = disp["pnl_pct"].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
                def style_trades(df):
                    s=pd.DataFrame("",index=df.index,columns=df.columns)
                    for i,row in df.iterrows():
                        oc=str(row.get("outcome",""))
                        if oc=="WIN":    s.at[i,"outcome"]="color:#16a34a;font-weight:600"
                        elif oc=="LOSS": s.at[i,"outcome"]="color:#dc2626;font-weight:600"
                        pnl=str(row.get("pnl",""))
                        if pnl.startswith("+"): s.at[i,"pnl"]="color:#16a34a"
                        elif pnl.startswith("-"): s.at[i,"pnl"]="color:#dc2626"
                    return s
                st.dataframe(disp.style.apply(style_trades,axis=None),
                             use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4 — WEEKLY REVIEW
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📅 Weekly Review")
        st.markdown(
            "<span style='color:#64748b;font-size:0.8rem'>"
            "Run every Friday after close. Reflect on what worked, "
            "what didn't, and what to do differently next week.</span>",
            unsafe_allow_html=True)

        # Auto-fill from trade log
        today = datetime.now(HK_TZ).date()
        week_start = today - timedelta(days=today.weekday())
        week_end   = week_start + timedelta(days=4)

        all_trades = get_trades()
        week_trades = pd.DataFrame()
        if not all_trades.empty:
            all_trades["date"] = pd.to_datetime(all_trades["date"])
            week_trades = all_trades[
                (all_trades["date"].dt.date>=week_start) &
                (all_trades["date"].dt.date<=week_end) &
                (all_trades["status"]=="CLOSED")]

        # Week summary auto-computed
        if not week_trades.empty:
            w_pnl  = week_trades["pnl"].sum()
            w_won  = int((week_trades["pnl"]>0).sum())
            w_lost = int((week_trades["pnl"]<=0).sum())
            best_t = week_trades.loc[week_trades["pnl"].idxmax()]
            worst_t= week_trades.loc[week_trades["pnl"].idxmin()]
            best_s = f"{best_t['ticker']} +{best_t['pnl']:,.0f}"
            worst_s= f"{worst_t['ticker']} {worst_t['pnl']:,.0f}"
        else:
            w_pnl=0; w_won=0; w_lost=0
            best_s=worst_s=""

        wc1,wc2,wc3 = st.columns(3)
        wc1.metric("Week P&L",    f"HKD {w_pnl:+,.0f}")
        wc2.metric("Wins / Losses",f"{w_won} / {w_lost}")
        wc3.metric("Win rate",    f"{w_won/(w_won+w_lost)*100:.0f}%" if w_won+w_lost>0 else "—")


        # ── STRATEGY EVALUATION ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Strategy Evaluation — Was My Strategy Right?")
        st.markdown(
            "<span style='color:#64748b;font-size:0.8rem'>"
            "Automated scoring of last week's recommendations vs what actually happened.</span>",
            unsafe_allow_html=True)

        if st.button("🔬 Run strategy evaluation", key="wr_eval"):
            st.session_state["run_eval"] = True

        if st.session_state.get("run_eval"):
            import yfinance as yf

            # ── 1. Trade signal accuracy ─────────────────────────────
            st.markdown("#### 1 · Trade Signal Accuracy")
            st.markdown(
                "<span style='color:#64748b;font-size:0.79rem'>"
                "Did the EXIT/REDUCE signals come before further losses? "
                "Did the ADD/ENTER signals come before gains?</span>",
                unsafe_allow_html=True)

            if not week_trades.empty:
                sig_rows = []
                for _, tr in week_trades.iterrows():
                    ticker_ = tr["ticker"]
                    entry_d = pd.to_datetime(tr["date"])
                    exit_d  = pd.to_datetime(tr.get("logged_at", tr["date"]))
                    pnl_    = float(tr.get("pnl", 0) or 0)
                    setup_  = tr.get("setup", "—")
                    outcome_= tr.get("outcome", "—")

                    # Fetch 5 days after exit to see what happened next
                    try:
                        df_after = fetch_d(ticker_, "10d")
                        if not df_after.empty and len(df_after) >= 2:
                            next_ret = float((df_after["Close"].iloc[-1] -
                                              df_after["Close"].iloc[-2]) /
                                             df_after["Close"].iloc[-2] * 100)
                        else:
                            next_ret = None
                    except:
                        next_ret = None

                    direction_ = tr.get("direction", "LONG")
                    # Was the exit timing good?
                    if outcome_ == "WIN":
                        timing = "✅ Good exit" if (next_ret is not None and (
                            (direction_=="LONG" and next_ret < 0) or
                            (direction_=="SHORT" and next_ret > 0))) else "⚠️ Left money"
                    elif outcome_ == "LOSS":
                        timing = "✅ Cut loss" if pnl_ > -500 else "⚠️ Late cut"
                    else:
                        timing = "—"

                    sig_rows.append({
                        "Date":      str(tr["date"])[:10],
                        "Ticker":    ticker_,
                        "Setup":     setup_,
                        "Direction": direction_,
                        "P&L":       f"{'+'if pnl_>=0 else ''}{pnl_:,.0f}",
                        "Outcome":   outcome_,
                        "Post-exit": f"{next_ret:+.2f}%" if next_ret is not None else "—",
                        "Timing":    timing,
                    })

                if sig_rows:
                    sig_df = pd.DataFrame(sig_rows)
                    def _sty_sig(df):
                        s = pd.DataFrame("", index=df.index, columns=df.columns)
                        for i, row in df.iterrows():
                            if str(row["Outcome"]) == "WIN":
                                s.at[i,"Outcome"] = "color:#16a34a;font-weight:600"
                            elif str(row["Outcome"]) == "LOSS":
                                s.at[i,"Outcome"] = "color:#dc2626;font-weight:600"
                            if "✅" in str(row["Timing"]):
                                s.at[i,"Timing"] = "color:#16a34a"
                            elif "⚠️" in str(row["Timing"]):
                                s.at[i,"Timing"] = "color:#f59e0b"
                            p = str(row["P&L"])
                            if p.startswith("+"): s.at[i,"P&L"] = "color:#16a34a"
                            elif p.startswith("-"): s.at[i,"P&L"] = "color:#dc2626"
                        return s
                    st.dataframe(sig_df.style.apply(_sty_sig, axis=None),
                                 use_container_width=True, hide_index=True)
            else:
                st.info("No closed trades this week to evaluate.")

            # ── 2. Cycle timing accuracy ──────────────────────────────
            st.markdown("#### 2 · Cycle Timing — Did You Enter Early or Late?")
            st.markdown(
                "<span style='color:#64748b;font-size:0.79rem'>"
                "For range trading, entering at <35% cycle progress = early = good. "
                ">65% = late = bad timing.</span>",
                unsafe_allow_html=True)

            capital_ = get_latest_capital()
            all_pos_ = []
            try:
                stock_df_ = get_portfolio_full()
                if not stock_df_.empty:
                    for _,r_ in stock_df_.iterrows():
                        all_pos_.append({
                            "ticker": r_["ticker"],
                            "name":   r_.get("name", r_["ticker"]),
                            "avg_cost": float(r_.get("avg_cost", 0) or 0),
                        })
            except: pass

            if all_pos_:
                cycle_rows = []
                for pos_ in all_pos_:
                    try:
                        df_c = fetch_d(pos_["ticker"], "3mo")
                        if df_c is None or len(df_c) < 20:
                            continue
                        closes_ = df_c["Close"]
                        d_=closes_.diff(); g_=d_.clip(lower=0).ewm(com=13,adjust=False).mean()
                        l_=(-d_.clip(upper=0)).ewm(com=13,adjust=False).mean()
                        rsi_=float((100-100/(1+g_/l_.replace(0,np.nan))).dropna().iloc[-1])
                        if len(closes_)>=20:
                            mid_=closes_.rolling(20).mean(); std_=closes_.rolling(20).std()
                            bb_=float(((closes_-mid_+2*std_)/(4*std_+1e-9)*100).clip(0,100).iloc[-1])
                        else: bb_=50
                        cycle_pct = rsi_/100*50 + bb_/100*50
                        price_now_ = float(closes_.iloc[-1])
                        avg_c_ = pos_["avg_cost"]
                        ret_ = (price_now_-avg_c_)/avg_c_*100 if avg_c_>0 else 0

                        if cycle_pct < 35:
                            timing_q = "✅ Early (good)"
                            t_c = "#16a34a"
                        elif cycle_pct < 55:
                            timing_q = "✅ Mid (OK)"
                            t_c = "#16a34a"
                        elif cycle_pct < 70:
                            timing_q = "⚠️ Late"
                            t_c = "#f59e0b"
                        else:
                            timing_q = "🔴 Peak zone"
                            t_c = "#dc2626"

                        cycle_rows.append({
                            "Name":       pos_["name"],
                            "Ticker":     pos_["ticker"],
                            "Avg entry":  f"{avg_c_:,.2f}",
                            "Now":        f"{price_now_:,.2f}",
                            "Return %":   f"{ret_:+.2f}%",
                            "RSI":        f"{rsi_:.0f}",
                            "BB%":        f"{bb_:.0f}",
                            "Cycle %":    f"{cycle_pct:.0f}%",
                            "Timing":     timing_q,
                            "_c":         t_c,
                        })
                    except: pass

                if cycle_rows:
                    cyc_df = pd.DataFrame(cycle_rows)
                    def _sty_cyc(df):
                        s = pd.DataFrame("", index=df.index, columns=df.columns)
                        for i,row in df.iterrows():
                            c_ = cycle_rows[i]["_c"]
                            s.at[i,"Timing"] = f"color:{c_};font-weight:600"
                            s.at[i,"Cycle %"] = f"color:{c_}"
                            ret = str(row["Return %"])
                            if ret.startswith("+"): s.at[i,"Return %"] = "color:#16a34a"
                            elif ret.startswith("-"): s.at[i,"Return %"] = "color:#dc2626"
                        return s
                    disp_cyc = cyc_df[[c for c in cyc_df.columns if not c.startswith("_")]]
                    st.dataframe(disp_cyc.style.apply(_sty_cyc, axis=None),
                                 use_container_width=True, hide_index=True)

                    early = sum(1 for r in cycle_rows if r["Cycle %"] < "50")
                    late  = sum(1 for r in cycle_rows if r["Cycle %"] >= "65%")
                    st.markdown(
                        f"<span style='font-size:0.82rem;color:#64748b'>"
                        f"Early entries: {early} · Late entries: {late} · "
                        f"{'✅ Good timing overall' if early > late else '⚠️ Entering too late — wait for earlier cycle'}"
                        f"</span>", unsafe_allow_html=True)

            # ── 3. Sector flow alignment ──────────────────────────────
            st.markdown("#### 3 · Sector Flow Alignment")
            st.markdown(
                "<span style='color:#64748b;font-size:0.79rem'>"
                "Were you positioned in sectors with strong inflow? "
                "Trading against the flow costs you.</span>",
                unsafe_allow_html=True)

            try:
                from money_flow import get_flow_snapshot, get_ticker_sector
                flow_snap_ = get_flow_snapshot("1mo")
                if all_pos_ and flow_snap_:
                    flow_rows = []
                    for pos_ in all_pos_:
                        sec_ = get_ticker_sector(pos_["ticker"]) or "—"
                        fl_  = flow_snap_.get(sec_, 0)
                        aligned = fl_ >= 20
                        flow_rows.append({
                            "Name":     pos_["name"],
                            "Ticker":   pos_["ticker"],
                            "Sector":   sec_,
                            "Flow score": f"{fl_:+d}",
                            "Aligned":  "✅ With flow" if aligned else
                                        ("⚠️ Neutral" if fl_ > -20 else "🔴 Against flow"),
                        })
                    fl_df = pd.DataFrame(flow_rows)
                    def _sty_fl(df):
                        s = pd.DataFrame("", index=df.index, columns=df.columns)
                        for i,row in df.iterrows():
                            a = str(row["Aligned"])
                            if "✅" in a: s.at[i,"Aligned"] = "color:#16a34a;font-weight:600"
                            elif "🔴" in a: s.at[i,"Aligned"] = "color:#dc2626;font-weight:600"
                            else: s.at[i,"Aligned"] = "color:#f59e0b"
                            fl = int(str(row["Flow score"]).replace("+",""))
                            if fl >= 20: s.at[i,"Flow score"] = "color:#16a34a;font-weight:600"
                            elif fl <= -20: s.at[i,"Flow score"] = "color:#dc2626;font-weight:600"
                        return s
                    st.dataframe(fl_df.style.apply(_sty_fl, axis=None),
                                 use_container_width=True, hide_index=True)
                    aligned_n = sum(1 for r in flow_rows if "✅" in r["Aligned"])
                    against_n = sum(1 for r in flow_rows if "🔴" in r["Aligned"])
                    if against_n > 0:
                        st.warning(
                            f"⚠️ {against_n} position(s) are in sectors with outflow. "
                            "This is a headwind — consider rotating to sectors with better flow.")
            except Exception as e:
                st.info(f"Flow data unavailable: {e}")

            # ── 4. Strategy Health Score ──────────────────────────────
            st.markdown("#### 4 · Strategy Health Score")

            scores = {}
            score_notes = []

            # Signal accuracy
            if not week_trades.empty and w_won + w_lost > 0:
                wr_ = w_won / (w_won + w_lost) * 100
                sig_score = min(wr_ * 1.5, 100)
                scores["Signal accuracy"] = round(sig_score, 0)
                if wr_ >= 55:
                    score_notes.append(f"✅ Win rate {wr_:.0f}% — signals working")
                elif wr_ >= 45:
                    score_notes.append(f"⚠️ Win rate {wr_:.0f}% — marginal edge")
                else:
                    score_notes.append(f"🔴 Win rate {wr_:.0f}% — signals not working")
            else:
                scores["Signal accuracy"] = 50
                score_notes.append("— No trades to evaluate")

            # Cycle timing
            if cycle_rows:
                good_timing = sum(1 for r in cycle_rows
                                  if float(r["Cycle %"].replace("%","")) < 55)
                timing_score = good_timing / len(cycle_rows) * 100
                scores["Cycle timing"] = round(timing_score, 0)
                if timing_score >= 70:
                    score_notes.append(f"✅ Cycle timing {timing_score:.0f}% — entering early")
                elif timing_score >= 50:
                    score_notes.append(f"⚠️ Cycle timing {timing_score:.0f}% — mixed")
                else:
                    score_notes.append(f"🔴 Cycle timing {timing_score:.0f}% — entering too late")
            else:
                scores["Cycle timing"] = 50

            # Flow alignment
            if "flow_rows" in dir() and flow_rows:
                aligned_pct = aligned_n / len(flow_rows) * 100 if flow_rows else 50
                scores["Flow alignment"] = round(aligned_pct, 0)
                if aligned_pct >= 70:
                    score_notes.append(f"✅ Flow alignment {aligned_pct:.0f}% — trading with money")
                else:
                    score_notes.append(f"⚠️ Flow alignment {aligned_pct:.0f}% — fighting some flows")
            else:
                scores["Flow alignment"] = 50

            # P&L sign
            if w_pnl > 0:
                scores["P&L"] = 80
                score_notes.append(f"✅ Profitable week +{w_pnl:,.0f}")
            elif w_pnl == 0:
                scores["P&L"] = 50
            else:
                scores["P&L"] = 20
                score_notes.append(f"🔴 Losing week {w_pnl:,.0f}")

            # Overall
            overall = round(sum(scores.values()) / len(scores), 0)
            overall_c = "#16a34a" if overall >= 65 else "#f59e0b" if overall >= 45 else "#dc2626"
            overall_l = ("✅ Strategy working well" if overall >= 65
                         else "⚠️ Needs adjustment" if overall >= 45
                         else "🔴 Review strategy — not working")

            st.markdown(
                f"<div style='border:2px solid {overall_c};border-radius:12px;"
                f"padding:16px 20px;background:rgba(0,0,0,0.02);margin-bottom:12px'>"
                f"<div style='font-size:1.3rem;font-weight:800;color:{overall_c}'>"
                f"Strategy Health: {overall:.0f}/100 — {overall_l}</div>"
                f"</div>", unsafe_allow_html=True)

            # Component scores radar-style bar
            fig_health = go.Figure()
            cats = list(scores.keys())
            vals = list(scores.values())
            colors_ = ["#16a34a" if v>=65 else "#f59e0b" if v>=45 else "#dc2626" for v in vals]
            fig_health.add_trace(go.Bar(
                x=cats, y=vals,
                marker_color=colors_, opacity=0.85,
                text=[f"{v:.0f}" for v in vals],
                textposition="outside"))
            fig_health.add_hline(y=65, line_dash="dot", line_color="#16a34a",
                                  line_width=1, annotation_text="Good (65)",
                                  annotation_position="right")
            fig_health.add_hline(y=45, line_dash="dot", line_color="#f59e0b",
                                  line_width=1, annotation_text="Marginal (45)",
                                  annotation_position="right")
            fig_health.update_layout(height=220, margin=dict(l=0,r=80,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(range=[0,110], gridcolor="#f1f5f9"),
                xaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_health, use_container_width=True)

            for note in score_notes:
                c_ = "#16a34a" if "✅" in note else "#f59e0b" if "⚠️" in note else "#dc2626"
                st.markdown(
                    f"<div style='font-size:0.82rem;color:{c_};"
                    f"padding:4px 0'>{note}</div>",
                    unsafe_allow_html=True)

            # ── Next week priorities ──────────────────────────────────
            st.markdown("#### 5 · Recommended Focus for Next Week")
            priorities = []
            if scores.get("Signal accuracy", 50) < 45:
                priorities.append("📌 Review signal settings — win rate too low. "
                    "Check if RSI thresholds match this stock's behaviour in Strategy page.")
            if scores.get("Cycle timing", 50) < 50:
                priorities.append("📌 Wait for earlier cycle entry — you are buying too late. "
                    "Only enter when Cycle ML shows <35% progress.")
            if scores.get("Flow alignment", 50) < 60:
                priorities.append("📌 Check Money Flow page before trading — "
                    "align positions with sectors showing inflow (score >20).")
            if w_pnl < 0 and w_won + w_lost >= 3:
                priorities.append("📌 Reduce position sizes this week — "
                    "preserve capital until win rate recovers.")
            if not priorities:
                priorities.append("✅ Strategy looks healthy — maintain discipline "
                    "and keep position sizes consistent.")
            for p_ in priorities:
                st.markdown(
                    f"<div style='border-left:3px solid #2563eb;padding:8px 12px;"
                    f"background:rgba(37,99,235,0.04);border-radius:0 6px 6px 0;"
                    f"font-size:0.82rem;margin-bottom:6px'>{p_}</div>",
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Write your weekly review**")


        wrc1,wrc2 = st.columns(2)
        w_notes   = wrc1.text_area("What worked this week?", height=100,
                                    key="wr_notes",
                                    placeholder="Which signals were reliable? Which setups paid off?")
        w_lessons = wrc2.text_area("What to improve?", height=100,
                                    key="wr_lessons",
                                    placeholder="Where did I break my rules? What cost me money?")

        # Questions to guide reflection
        with st.expander("📝 Weekly reflection prompts"):
            st.markdown("""
1. Did I follow my position sizing rules on every trade?
2. Did I hold any position past its stop (and why)?
3. Were my entries based on signals or emotion?
4. Did I overtrade on any day?
5. Which sector had the best flow this week — was I positioned correctly?
6. How was my portfolio heat? Did I ever exceed 6%?
7. Did my cycle ML readings match what actually happened?
8. What is my #1 rule to focus on next week?
            """)

        if st.button("💾 Save weekly review", key="wr_save"):
            save_weekly_review(str(week_start), str(week_end),
                               w_pnl, w_won, w_lost,
                               best_s, worst_s, w_notes, w_lessons)
            st.success("Weekly review saved!"); st.rerun()

        # Past reviews
        conn = get_conn()
        past = pd.read_sql_query(
            "SELECT * FROM weekly_review ORDER BY logged_at DESC LIMIT 10", conn)
        conn.close()
        if not past.empty:
            st.markdown("---")
            st.markdown("**Past reviews**")
            for _,r in past.iterrows():
                pnl_ = float(r.get("total_pnl",0) or 0)
                c_ = "#16a34a" if pnl_>=0 else "#dc2626"
                with st.expander(
                    f"Week of {r['week_start']} — P&L: {'+'if pnl_>=0 else ''}{pnl_:,.0f} "
                    f"({r['trades_won']}W/{r['trades_lost']}L)"):
                    if r.get("notes"):
                        st.markdown(f"**What worked:** {r['notes']}")
                    if r.get("lessons"):
                        st.markdown(f"**Improvements:** {r['lessons']}")

    st.markdown(
        "<span style='color:#94a3b8;font-size:0.74rem'>"
        "Risk management is probabilistic — no system eliminates losses. "
        "The goal is to stay in the game long enough for your edge to compound.</span>",
        unsafe_allow_html=True)
