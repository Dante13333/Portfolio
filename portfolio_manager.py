"""
portfolio_manager.py  —  Unified Portfolio Manager
All products in one place: Stocks · Forex · Commodities
Tabs: Positions | Add/Edit | Capital | Activity Log
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
import time, pytz

from db_manager import (
    init_db, get_latest_capital, save_capital, get_capital_history,
    get_portfolio_full, upsert_position_full, delete_position, close_position,
    init_portfolio_extended, init_activity_log, log_activity, get_activity_log,
)

# ── MONITOR DB (formerly market_monitor.py) ───────────────────────────
from db_manager import get_conn

def init_monitor_tables():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS monitor_positions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            name        TEXT NOT NULL,
            asset_type  TEXT NOT NULL,
            unit        TEXT,
            quantity    REAL NOT NULL DEFAULT 0,
            avg_cost    REAL NOT NULL DEFAULT 0,
            target      REAL,
            stop        REAL,
            notes       TEXT,
            status      TEXT DEFAULT 'OPEN',
            entry_date  TEXT,
            updated_at  TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker)
        );
    """)
    conn.commit(); conn.close()

def get_monitor_pos() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM monitor_positions ORDER BY updated_at DESC", conn)
    conn.close()
    return df

def upsert_monitor_pos(ticker, name, asset_type, unit, quantity,
                        avg_cost, target=None, stop=None,
                        notes="", status="OPEN", entry_date=""):
    conn = get_conn()
    conn.execute("""
        INSERT INTO monitor_positions
          (ticker,name,asset_type,unit,quantity,avg_cost,target,stop,notes,status,entry_date)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(ticker) DO UPDATE SET
            name=excluded.name, quantity=excluded.quantity,
            avg_cost=excluded.avg_cost, target=excluded.target,
            stop=excluded.stop, notes=excluded.notes,
            status=excluded.status, entry_date=excluded.entry_date,
            updated_at=datetime('now')
    """, (ticker, name, asset_type, unit, quantity, avg_cost,
          target, stop, notes, status, entry_date))
    conn.commit(); conn.close()

def delete_monitor_pos(ticker):
    conn = get_conn()
    conn.execute("DELETE FROM monitor_positions WHERE ticker=?", (ticker,))
    conn.commit(); conn.close()

def close_monitor_pos(ticker, exit_price):
    conn = get_conn()
    conn.execute("""
        UPDATE monitor_positions SET status='CLOSED', updated_at=datetime('now')
        WHERE ticker=?
    """, (ticker,))
    conn.commit(); conn.close()


HK_TZ = pytz.timezone("Asia/Hong_Kong")

# ── CATALOGUE for quick-add ───────────────────────────────────────────
FOREX_CAT = {
    "USDHKD=X":"USD/HKD", "EURUSD=X":"EUR/USD", "USDCNY=X":"USD/CNY",
    "GBPUSD=X":"GBP/USD", "USDJPY=X":"USD/JPY", "AUDUSD=X":"AUD/USD",
    "USDCNH=X":"USD/CNH", "EURJPY=X":"EUR/JPY", "EURGBP=X":"EUR/GBP",
}
# Grouped for display in selectbox
COMMODITY_CAT = {
    # Precious Metals
    "GC=F":"Gold",          "SI=F":"Silver",
    "PL=F":"Platinum",      "PA=F":"Palladium",
    # Energy
    "CL=F":"WTI Crude Oil", "BZ=F":"Brent Crude",
    "NG=F":"Natural Gas",   "RB=F":"RBOB Gasoline",
    "HO=F":"Heating Oil",
    # Industrial Metals
    "HG=F":"Copper",        "ALI=F":"Aluminium",
    "ZN=F":"Zinc",
    # Agriculture
    "ZC=F":"Corn",          "ZW=F":"Wheat",
    "ZS=F":"Soybeans",      "KC=F":"Coffee",
    "SB=F":"Sugar",         "CT=F":"Cotton",
    "OJ=F":"Orange Juice",  "LBS=F":"Lumber",
}

COMMODITY_GROUPS = {
    "Precious Metals": ["GC=F","SI=F","PL=F","PA=F"],
    "Energy":          ["CL=F","BZ=F","NG=F","RB=F","HO=F"],
    "Industrial Metals":["HG=F","ALI=F","ZN=F"],
    "Agriculture":     ["ZC=F","ZW=F","ZS=F","KC=F","SB=F","CT=F","OJ=F","LBS=F"],
}
ASSET_TYPES = ["Stock (HKEX)", "Forex", "Commodity"]

# ── PRICE FETCH ───────────────────────────────────────────────────────
def _variants(ticker):
    v=[ticker]; code=ticker.replace(".HK","")
    if code.isdigit():
        v.append(str(int(code))+".HK"); v.append(code.zfill(4)+".HK")
    return list(dict.fromkeys(v))

@st.cache_data(ttl=120, show_spinner=False)
def fetch_prices(tickers: tuple) -> dict:
    out = {}
    for t in tickers:
        for tv in _variants(t):
            try:
                info = yf.Ticker(tv).fast_info
                p = getattr(info,"last_price",None)
                if p:
                    out[t] = float(p); break
            except Exception: pass
        if t not in out: out[t] = None
        time.sleep(0.2)
    return out

@st.cache_data(ttl=600, show_spinner=False)
def lookup_name(ticker:str) -> str:
    for t in _variants(ticker):
        try:
            i=yf.Ticker(t).info
            n=i.get("longName") or i.get("shortName")
            if n: return n
        except Exception: pass
    return ticker

# ── FORMATTERS ────────────────────────────────────────────────────────
def fc(v):   return "#16a34a" if (v or 0)>=0 else "#dc2626"
def fv(v, d=2):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
    return f"{v:,.{d}f}"
def fpnl(v):
    if v is None: return "—"
    return f"{'+'if v>=0 else ''}{v:,.2f}"
def fpct(v):
    if v is None: return "—"
    return f"{'+'if v>=0 else ''}{v:.2f}%"

def pos_card(col, r, price, capital):
    """Render a compact position expander card."""
    is_stock = r.get("_src") == "stock"
    qty   = float(r.get("shares",0) or r.get("quantity",0) or 0)
    cost_ = float(r.get("avg_cost",0) or 0)
    tgt   = r.get("target_price") or r.get("target")
    stp   = r.get("stop_price")   or r.get("stop")
    notes = r.get("notes","") or ""

    val  = qty*price if price and qty>0 else None
    cb   = qty*cost_
    pnl  = val-cb if val is not None else None
    pnlp = pnl/cb*100 if (pnl is not None and cb>0) else None
    posp = cb/capital*100 if capital>0 else 0

    name_   = r.get("name",r["ticker"])
    ticker_ = r["ticker"]
    ps      = f"{price:,.4f}" if price else "—"
    if price and price>100: ps = f"{price:,.2f}"

    # Expander label
    pnl_str = (f"  ·  {'+'if (pnl or 0)>=0 else ''}{pnl:,.0f}" if pnl is not None else "")
    label   = f"**{name_}** ({ticker_})  ·  {ps}{pnl_str}"

    with col.expander(label, expanded=False):
        mc = st.columns(5)
        for c_,l_,v_ in zip(mc,
            ["Price","Qty / Shares","Cost basis","P&L","% of capital"],
            [ps, fv(qty,2), fv(cb,2), fpnl(pnl), f"{posp:.1f}%"]):
            _vc = fc(pnl) if "P&L" in l_ else "#0f172a"
            c_.markdown(
                f"<div style='text-align:center;padding:6px 4px;background:#f8fafc;"
                f"border-radius:8px'><div style='font-size:0.65rem;color:#94a3b8'>{l_}</div>"
                f"<div style='font-size:0.88rem;font-weight:600;color:{_vc}'>{v_}</div></div>",
                unsafe_allow_html=True)

        if tgt or stp:
            tc = st.columns(2)
            if stp:
                d=(cost_-stp)/cost_*100 if cost_ else 0
                tc[0].markdown(
                    f"<div style='padding:6px 10px;border-radius:6px;"
                    f"border:1px solid #dc2626;background:rgba(220,38,38,0.04);"
                    f"font-size:0.78rem'><b style='color:#dc2626'>Stop:</b> "
                    f"{stp:,.4f} (−{d:.2f}%)</div>",unsafe_allow_html=True)
            if tgt:
                d=(tgt-cost_)/cost_*100 if cost_ else 0
                tc[1].markdown(
                    f"<div style='padding:6px 10px;border-radius:6px;"
                    f"border:1px solid #16a34a;background:rgba(22,163,74,0.04);"
                    f"font-size:0.78rem'><b style='color:#16a34a'>Target:</b> "
                    f"{tgt:,.4f} (+{d:.2f}%)</div>",unsafe_allow_html=True)

        if notes:
            st.markdown(
                f"<div style='font-size:0.78rem;color:#475569;background:#f8fafc;"
                f"padding:7px 10px;border-radius:6px;border-left:3px solid #cbd5e1'>"
                f"📝 {notes}</div>",unsafe_allow_html=True)

        # Inline edit
        ec = st.columns(5)
        key_pf = f"pm_{ticker_}"
        new_qty = ec[0].number_input("Qty/Shares",   value=qty,   step=1.0 if is_stock else 0.01, format="%.4f", key=f"{key_pf}_qty")
        new_ac  = ec[1].number_input("Avg cost",     value=cost_, step=0.01, format="%.4f",       key=f"{key_pf}_ac")
        new_tg  = ec[2].number_input("Target",       value=float(tgt) if tgt else 0.0, step=0.0001, format="%.4f", key=f"{key_pf}_tg")
        new_st  = ec[3].number_input("Stop",         value=float(stp) if stp else 0.0, step=0.0001, format="%.4f", key=f"{key_pf}_st")
        new_nt  = ec[4].text_input("Notes",          value=notes, key=f"{key_pf}_nt")

        b1,b2,b3 = st.columns(3)
        if b1.button("💾 Save", key=f"{key_pf}_save"):
            if is_stock:
                upsert_position_full(ticker_, r.get("name",ticker_),
                    int(new_qty), new_ac,
                    new_tg if new_tg>0 else None,
                    new_st if new_st>0 else None,
                    new_nt, "OPEN", r.get("entry_date",""))
            else:
                upsert_monitor_pos(ticker_, r.get("name",ticker_),
                    r.get("asset_type","Forex"), r.get("unit",""),
                    new_qty, new_ac,
                    new_tg if new_tg>0 else None,
                    new_st if new_st>0 else None,
                    new_nt, "OPEN", r.get("entry_date",""))
            log_activity("EDIT", ticker_, f"qty={new_qty:.4f} avg={new_ac:.4f}")
            st.success(f"✅ Saved {name_}"); st.rerun()

        with b2.form(key=f"{key_pf}_close_form"):
            ep = st.number_input("Exit price",
                                  value=float(price) if price else cost_,
                                  step=0.0001, format="%.4f",
                                  key=f"{key_pf}_ep")
            if st.form_submit_button("✅ Close"):
                if is_stock:
                    close_position(ticker_, ep, datetime.now(HK_TZ).strftime("%Y-%m-%d"))
                else:
                    close_monitor_pos(ticker_, ep)
                log_activity("CLOSE", ticker_,
                             f"exit={ep:.4f} pnl={(ep-cost_)*qty:,.2f}")
                st.success(f"{name_} closed!"); st.rerun()

        if b3.button("🗑️ Delete", key=f"{key_pf}_del", type="secondary"):
            if is_stock: delete_position(ticker_)
            else:        delete_monitor_pos(ticker_)
            log_activity("DELETE", ticker_)
            st.rerun()


# ═════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ═════════════════════════════════════════════════════════════════════
def render():
    init_portfolio_extended()
    init_activity_log()
    init_monitor_tables()

    now_hk  = datetime.now(HK_TZ)
    capital = get_latest_capital()

    st.markdown(
        "## 📋 Portfolio &nbsp;"
        "<span style='background:#0f172a;color:#38bdf8;font-size:0.68rem;"
        "padding:2px 7px;border-radius:5px'>ALL PRODUCTS</span>",
        unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"Stocks · Forex · Commodities · "
        f"{now_hk.strftime('%Y-%m-%d %H:%M HKT')}</span>",
        unsafe_allow_html=True)
    st.markdown("---")

    tab_pos, tab_add, tab_cap, tab_log = st.tabs([
        "📊 Positions",
        "➕ Add / Edit",
        "💰 Capital",
        "📜 Log",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — POSITIONS
    # ════════════════════════════════════════════════════════════════
    with tab_pos:
        # Load all open positions
        stock_df   = get_portfolio_full()
        monitor_df = get_monitor_pos()

        open_s = stock_df[stock_df["status"]=="OPEN"] if not stock_df.empty else pd.DataFrame()
        watch_s= stock_df[stock_df["status"]=="WATCH"] if not stock_df.empty else pd.DataFrame()
        open_m = monitor_df[monitor_df["status"]=="OPEN"] if not monitor_df.empty else pd.DataFrame()
        watch_m= monitor_df[monitor_df["status"]=="WATCH"] if not monitor_df.empty else pd.DataFrame()

        # Mark source
        for df_, src in [(open_s,"stock"),(watch_s,"stock"),
                         (open_m,"monitor"),(watch_m,"monitor")]:
            if not df_.empty: df_["_src"] = src

        # Collect all tickers and fetch prices
        all_tickers = []
        for df_ in [open_s, watch_s, open_m, watch_m]:
            if not df_.empty:
                all_tickers += df_["ticker"].tolist()
        all_tickers = list(dict.fromkeys(all_tickers))

        prices = {}
        if all_tickers:
            with st.spinner("Fetching live prices…"):
                prices = fetch_prices(tuple(all_tickers))

        # Summary metrics
        def calc_totals(df_, is_stock=True):
            if df_.empty: return 0,0,0
            cost=val=pnl=0
            for _,r in df_.iterrows():
                qty  = float(r.get("shares",0) if is_stock else r.get("quantity",0) or 0)
                ac   = float(r.get("avg_cost",0) or 0)
                p    = prices.get(r["ticker"])
                cb   = qty*ac
                v    = qty*p if p and qty>0 else cb
                cost+=cb; val+=v; pnl+=v-cb
            return cost,val,pnl

        sc,sv,sp = calc_totals(open_s, True)
        mc_,mv,mp= calc_totals(open_m, False)
        total_cost= sc+mc_; total_val=sv+mv; total_pnl=sp+mp
        total_pct = total_pnl/total_cost*100 if total_cost>0 else 0
        cash      = max(capital-total_cost,0)

        m1,m2,m3,m4 = st.columns(4)
        for col,lbl,val,sub,color in [
            (m1,"Invested",    f"{total_cost:,.0f}",   f"Cash: {cash:,.0f}",  "#0f172a"),
            (m2,"Market value",f"{total_val:,.0f}",    "",                    "#0f172a"),
            (m3,"Total P&L",   fpnl(total_pnl),        fpct(total_pct),       fc(total_pnl)),
            (m4,"Positions",   str(len(open_s)+len(open_m)), f"{len(watch_s)+len(watch_m)} watching", "#64748b"),
        ]:
            col.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                f"border-radius:10px;padding:12px 14px;text-align:center'>"
                f"<div style='font-size:0.7rem;color:#94a3b8'>{lbl}</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:{color}'>{val}</div>"
                f"<div style='font-size:0.7rem;color:#94a3b8'>{sub}</div></div>",
                unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)

        # ── Stocks ───────────────────────────────────────────────────
        if not open_s.empty:
            st.markdown(
                "<div style='font-size:0.8rem;font-weight:600;color:#2563eb;"
                "text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px'>"
                "📈 Stocks</div>", unsafe_allow_html=True)
            cols_s = st.columns(min(len(open_s),2))
            for i,(_, r) in enumerate(open_s.iterrows()):
                pos_card(cols_s[i%2], r.to_dict()|{"_src":"stock"},
                         prices.get(r["ticker"]), capital)

        # ── Forex ────────────────────────────────────────────────────
        forex_rows = open_m[open_m["asset_type"]=="Forex"] if not open_m.empty else pd.DataFrame()
        if not forex_rows.empty:
            st.markdown(
                "<div style='font-size:0.8rem;font-weight:600;color:#8b5cf6;"
                "text-transform:uppercase;letter-spacing:0.05em;margin:12px 0 6px 0'>"
                "💱 Forex</div>", unsafe_allow_html=True)
            cols_f = st.columns(min(len(forex_rows),2))
            for i,(_, r) in enumerate(forex_rows.iterrows()):
                pos_card(cols_f[i%2], r.to_dict()|{"_src":"monitor"},
                         prices.get(r["ticker"]), capital)

        # ── Commodities ───────────────────────────────────────────────
        comm_rows = open_m[open_m["asset_type"]=="Commodity"] if not open_m.empty else pd.DataFrame()
        if not comm_rows.empty:
            st.markdown(
                "<div style='font-size:0.8rem;font-weight:600;color:#f59e0b;"
                "text-transform:uppercase;letter-spacing:0.05em;margin:12px 0 6px 0'>"
                "🥇 Commodities</div>", unsafe_allow_html=True)
            cols_c = st.columns(min(len(comm_rows),2))
            for i,(_, r) in enumerate(comm_rows.iterrows()):
                pos_card(cols_c[i%2], r.to_dict()|{"_src":"monitor"},
                         prices.get(r["ticker"]), capital)

        # ── Watchlist ─────────────────────────────────────────────────
        all_watch = pd.concat(
            [df_ for df_ in [watch_s, watch_m] if not df_.empty]
        ) if any(not df_.empty for df_ in [watch_s,watch_m]) else pd.DataFrame()

        if not all_watch.empty:
            with st.expander(f"👁 Watchlist ({len(all_watch)} instruments)"):
                for _, r in all_watch.iterrows():
                    p = prices.get(r["ticker"])
                    ps = f"{p:,.4f}" if p else "—"
                    src= "stock" if "shares" in r.index else "monitor"
                    st.markdown(
                        f"**{r.get('name',r['ticker'])}** ({r['ticker']}) · {ps} · "
                        + (r.get("notes","") or ""))
                    wb1,wb2 = st.columns(2)
                    if wb1.button("Convert to position", key=f"pm_watch_conv_{r['ticker']}"):
                        st.session_state[f"conv_{r['ticker']}"] = True
                    if st.session_state.get(f"conv_{r['ticker']}"):
                        wq = st.number_input("Qty/Shares", min_value=0.01, value=1.0,
                                              key=f"pm_wq_{r['ticker']}")
                        wa = st.number_input("Avg cost", min_value=0.0,
                                              value=float(p) if p else 0.0, step=0.0001,
                                              format="%.4f", key=f"pm_wa_{r['ticker']}")
                        if st.button("✅ Confirm", key=f"pm_wconf_{r['ticker']}"):
                            if src=="stock":
                                upsert_position_full(r["ticker"], r.get("name",r["ticker"]),
                                    int(wq), wa, None, None, "", "OPEN", "")
                            else:
                                upsert_monitor_pos(r["ticker"], r.get("name",r["ticker"]),
                                    r.get("asset_type","Forex"), r.get("unit",""),
                                    wq, wa, None, None, "", "OPEN", "")
                            log_activity("ADD", r["ticker"], "converted from watchlist")
                            st.rerun()
                    if wb2.button("🗑️ Remove", key=f"pm_wdel_{r['ticker']}",
                                   type="secondary"):
                        if src=="stock": delete_position(r["ticker"])
                        else:            delete_monitor_pos(r["ticker"])
                        st.rerun()

        # ── Closed ────────────────────────────────────────────────────
        closed_s = stock_df[stock_df["status"]=="CLOSED"] if not stock_df.empty else pd.DataFrame()
        closed_m = monitor_df[monitor_df["status"]=="CLOSED"] if not monitor_df.empty else pd.DataFrame()
        total_closed = len(closed_s)+len(closed_m)
        if total_closed:
            with st.expander(f"📁 Closed positions ({total_closed})"):
                all_cl = pd.concat([df_ for df_ in [closed_s,closed_m] if not df_.empty])
                st.dataframe(all_cl[["ticker","name","updated_at"]].rename(
                    columns={"ticker":"Ticker","name":"Name","updated_at":"Closed at"}),
                    use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — ADD / EDIT
    # ════════════════════════════════════════════════════════════════
    with tab_add:
        st.markdown("### Add or Edit Position")
        st.markdown(
            "<span style='color:#64748b;font-size:0.8rem'>"
            "One form for all product types. Set quantity to 0 to add as watchlist.</span>",
            unsafe_allow_html=True)

        # Asset type selector
        asset_type = st.radio("Asset type", ASSET_TYPES,
                               horizontal=True, key="pm_add_type")

        col1, col2 = st.columns(2)

        # ── Ticker / instrument input ─────────────────────────────────
        if asset_type == "Stock (HKEX)":
            existing = get_portfolio_full()
            ex_t     = existing["ticker"].tolist() if not existing.empty else []
            mode     = col1.radio("", ["Add new","Edit existing"],
                                   horizontal=True, key="pm_stock_mode")
            if mode=="Edit existing" and ex_t:
                ticker  = col1.selectbox("Select", ex_t, key="pm_sel_stock")
                row_    = existing[existing["ticker"]==ticker].iloc[0]
                def_name= row_.get("name",ticker)
                def_qty = int(row_.get("shares",0) or 0)
                def_ac  = float(row_.get("avg_cost",0) or 0)
                def_tg  = row_.get("target_price")
                def_st  = row_.get("stop_price")
                def_nt  = row_.get("notes","") or ""
                def_dt  = row_.get("entry_date","") or ""
            else:
                raw = col1.text_input("HKEX ticker", placeholder="0700.HK",
                                       key="pm_new_stock").strip().upper()
                ticker  = raw if raw.endswith(".HK") else (raw+".HK" if raw else "")
                def_name= def_nt = def_dt = ""
                def_qty = 0; def_ac = 0.0; def_tg = def_st = None
                if ticker and len(ticker)>=5:
                    with st.spinner("Looking up…"):
                        def_name = lookup_name(ticker)
                    col2.info(f"Found: **{def_name}**")
            unit_disp = "HKD"

        elif asset_type == "Forex":
            source = col1.radio("", ["Catalogue","Custom"], horizontal=True, key="pm_fx_src")
            if source=="Catalogue":
                ticker  = col1.selectbox("Pair", list(FOREX_CAT.keys()),
                                          format_func=lambda k:f"{FOREX_CAT[k]} ({k})",
                                          key="pm_fx_sel")
                def_name= FOREX_CAT[ticker]
            else:
                ticker  = col1.text_input("Ticker", placeholder="EURUSD=X",
                                           key="pm_fx_raw").strip().upper()
                def_name= col1.text_input("Name", placeholder="EUR/USD",
                                           key="pm_fx_name").strip()
            unit_disp = col2.text_input("Unit", value="USD", key="pm_fx_unit")
            def_qty=0; def_ac=0.0; def_tg=def_st=None; def_nt=def_dt=""
            existing_m = get_monitor_pos()
            if not existing_m.empty and ticker in existing_m["ticker"].values:
                row_ = existing_m[existing_m["ticker"]==ticker].iloc[0]
                def_qty=float(row_.get("quantity",0) or 0)
                def_ac=float(row_.get("avg_cost",0) or 0)
                def_tg=row_.get("target"); def_st=row_.get("stop")
                def_nt=row_.get("notes","") or ""; def_dt=row_.get("entry_date","") or ""

        else:  # Commodity
            source2 = col1.radio("", ["Catalogue","Custom"], horizontal=True, key="pm_cm_src")
            if source2=="Catalogue":
                grp2 = col1.selectbox("Category",
                                       list(COMMODITY_GROUPS.keys()),
                                       key="pm_cm_grp")
                grp_tickers2 = COMMODITY_GROUPS[grp2]
                ticker  = col1.selectbox("Commodity",
                                          grp_tickers2,
                                          format_func=lambda k: f"{COMMODITY_CAT[k]} ({k})",
                                          key="pm_cm_sel")
                def_name= COMMODITY_CAT[ticker]
            else:
                ticker  = col1.text_input("Ticker", placeholder="GC=F",
                                           key="pm_cm_raw").strip().upper()
                def_name= col1.text_input("Name", placeholder="Gold",
                                           key="pm_cm_name").strip()
            unit_disp = col2.text_input("Unit", value="USD/oz", key="pm_cm_unit")
            def_qty=0; def_ac=0.0; def_tg=def_st=None; def_nt=def_dt=""
            existing_m2 = get_monitor_pos()
            if not existing_m2.empty and ticker and ticker in existing_m2["ticker"].values:
                row_ = existing_m2[existing_m2["ticker"]==ticker].iloc[0]
                def_qty=float(row_.get("quantity",0) or 0)
                def_ac=float(row_.get("avg_cost",0) or 0)
                def_tg=row_.get("target"); def_st=row_.get("stop")
                def_nt=row_.get("notes","") or ""; def_dt=row_.get("entry_date","") or ""

        if not ticker:
            st.info("👆 Enter a ticker to continue.")
        else:
            # Live price
            with st.spinner(f"Fetching {ticker}…"):
                lp_dict = fetch_prices((ticker,))
                live_p  = lp_dict.get(ticker)
            if live_p:
                st.info(f"**Live price:** {live_p:,.4f}  {unit_disp}")

            # Form fields
            f1,f2,f3 = st.columns(3)
            name_  = f1.text_input("Display name", value=def_name or ticker, key="pm_f_name")
            qty_   = f2.number_input("Quantity / Shares", value=float(def_qty),
                                      min_value=0.0, step=1.0 if asset_type=="Stock (HKEX)" else 0.01,
                                      format="%.4f", key="pm_f_qty")
            ac_    = f3.number_input("Avg buy price", value=float(def_ac) if def_ac else (live_p or 0.0),
                                      min_value=0.0, step=0.0001, format="%.4f", key="pm_f_ac")

            f4,f5,f6 = st.columns(3)
            tg_    = f4.number_input("Target", value=float(def_tg) if def_tg else 0.0,
                                      min_value=0.0, step=0.0001, format="%.4f", key="pm_f_tg")
            st_    = f5.number_input("Stop loss", value=float(def_st) if def_st else 0.0,
                                      min_value=0.0, step=0.0001, format="%.4f", key="pm_f_st")
            dt_    = f6.text_input("Entry date",
                                    value=def_dt or now_hk.strftime("%Y-%m-%d"),
                                    key="pm_f_dt")
            nt_    = st.text_area("Notes", value=def_nt, height=72, key="pm_f_nt",
                                   placeholder="Why this trade?")

            # R:R preview
            if ac_>0 and tg_>0 and st_>0:
                risk=abs(ac_-st_); reward=abs(tg_-ac_)
                rr=reward/risk if risk>0 else 0
                rrc="#16a34a" if rr>=2 else "#f59e0b" if rr>=1 else "#dc2626"
                st.markdown(
                    f"<div style='display:flex;gap:20px;padding:9px 14px;"
                    f"background:#f8fafc;border-radius:8px;font-size:0.81rem;margin-bottom:6px'>"
                    f"<span>Risk: <b style='color:#dc2626'>{risk:.4f}</b></span>"
                    f"<span>Reward: <b style='color:#16a34a'>{reward:.4f}</b></span>"
                    f"<span>R:R <b style='color:{rrc}'>1:{rr:.1f}</b> "
                    f"{'✅' if rr>=2 else '⚠️'}</span></div>",
                    unsafe_allow_html=True)

            status = "WATCH" if qty_==0 else "OPEN"
            lbl    = "➕ Add to Watchlist" if status=="WATCH" else "➕ Add / Update"
            if st.button(lbl, key="pm_submit"):
                if asset_type=="Stock (HKEX)":
                    upsert_position_full(ticker, name_, int(qty_), ac_,
                                          tg_ or None, st_ or None,
                                          nt_, status, dt_)
                else:
                    atype = "Forex" if asset_type=="Forex" else "Commodity"
                    upsert_monitor_pos(ticker, name_, atype, unit_disp,
                                        qty_, ac_, tg_ or None, st_ or None,
                                        nt_, status, dt_)
                log_activity("ADD" if status=="OPEN" else "WATCHLIST",
                              ticker, f"qty={qty_:.4f} avg={ac_:.4f}")
                msg = f"✅ {'Watching' if status=='WATCH' else 'Saved'} {name_}"
                if qty_>0: msg += f" · {qty_:.4f} @ {ac_:.4f}"
                st.success(msg); st.rerun()

    # ════════════════════════════════════════════════════════════════
    # TAB 3 — CAPITAL
    # ════════════════════════════════════════════════════════════════
    with tab_cap:
        st.markdown("### Capital Tracker")
        current = get_latest_capital()
        cc1,cc2 = st.columns(2)
        new_cap  = cc1.number_input("Total capital", value=float(current),
                                     min_value=0.0, step=1000.0, format="%.2f",
                                     key="pm_cap")
        cap_note = cc2.text_input("Note", placeholder="Top up / withdrawal",
                                   key="pm_capnote")
        if st.button("💾 Save capital", key="pm_savecap"):
            save_capital(new_cap, cap_note)
            log_activity("CAPITAL", "", f"{new_cap:,.2f} — {cap_note}")
            st.success(f"Saved {new_cap:,.2f}"); st.rerun()

        hist = get_capital_history()
        if hist is not None and not hist.empty:
            fig_c = go.Figure(go.Scatter(
                x=hist.sort_values("recorded_at")["recorded_at"],
                y=hist.sort_values("recorded_at")["amount"],
                mode="lines+markers",
                line=dict(color="#2563eb",width=2),
                fill="tozeroy",fillcolor="rgba(37,99,235,0.07)"))
            fig_c.update_layout(height=220,margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white",paper_bgcolor="white",
                yaxis=dict(gridcolor="#f1f5f9"),xaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_c,use_container_width=True)
            st.dataframe(hist[["amount","note","recorded_at"]].rename(
                columns={"amount":"Capital","note":"Note","recorded_at":"Time"}),
                use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4 — LOG
    # ════════════════════════════════════════════════════════════════
    with tab_log:
        st.markdown("### Activity Log")
        log = get_activity_log(200)
        if log is None or log.empty:
            st.info("No activity yet.")
        else:
            a1,a2,a3 = st.columns(3)
            a1.metric("Total",   len(log))
            a2.metric("Adds",    int((log["action"].str.contains("ADD")).sum()))
            a3.metric("Closes",  int((log["action"].str.contains("CLOSE")).sum()))
            disp = log[["action","ticker","detail","logged_at"]].copy()
            disp.columns = ["Action","Ticker","Detail","Time"]
            def row_col(r):
                if "ADD" in str(r["Action"]):    return ["background:rgba(22,163,74,0.07)"]*4
                if "CLOSE" in str(r["Action"]):  return ["background:rgba(220,38,38,0.07)"]*4
                if "DELETE" in str(r["Action"]): return ["background:rgba(220,38,38,0.04)"]*4
                if "EDIT" in str(r["Action"]):   return ["background:rgba(37,99,235,0.04)"]*4
                return [""]*4
            st.dataframe(disp.style.apply(row_col,axis=1),
                         use_container_width=True, hide_index=True)
