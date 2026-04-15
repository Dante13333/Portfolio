"""
fundamentals.py — Fundamental Study
Fetches key fundamentals for stocks in portfolio + watchlist.
Only works for stocks (HKEX + US) — forex/commodities have no fundamentals.

Data via yfinance .info dict.
Caches aggressively (fundamentals change slowly).

Summary triggers:
  1. Earnings within 7 days
  2. Analyst target >20% above current price
  3. Institutional ownership increased (>50% and rising)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import time, pytz

HK_TZ = pytz.timezone("Asia/Hong_Kong")

def _f(val, fmt=".1f", fallback="—"):
    """Safely format a numeric value that may be a string or None."""
    try:
        return format(float(val), fmt) if val is not None else fallback
    except (TypeError, ValueError):
        return fallback


# ── DATA ──────────────────────────────────────────────────────────────
def _var(ticker):
    v=[ticker]; code=ticker.replace(".HK","")
    if code.isdigit():
        v.append(str(int(code))+".HK"); v.append(code.zfill(4)+".HK")
    return list(dict.fromkeys(v))

@st.cache_data(ttl=3600, show_spinner=False)   # 1 hour cache — fundamentals are slow
def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental data for one stock ticker."""
    for t in _var(ticker):
        try:
            info = yf.Ticker(t).info
            if not info or info.get("quoteType") not in ["EQUITY","ETF",None]:
                if info.get("quoteType") == "CURRENCY": return {}
            # Extract what we need
            price  = info.get("currentPrice") or info.get("regularMarketPrice") or 0

            # Earnings
            earn_date = info.get("earningsTimestamp") or info.get("earningsDate")
            if earn_date:
                try:
                    earn_dt = datetime.fromtimestamp(earn_date, tz=HK_TZ) if isinstance(earn_date,(int,float)) else None
                except: earn_dt = None
            else:
                earn_dt = None

            # Revenue trend (use trailing + forward estimates)
            rev_ttm   = info.get("totalRevenue")
            rev_growth= info.get("revenueGrowth")       # YoY %
            earn_growth=info.get("earningsGrowth")
            profit_margin=info.get("profitMargins")

            # Analyst
            n_analysts  = info.get("numberOfAnalystOpinions") or 0
            target_mean = info.get("targetMeanPrice")
            target_high = info.get("targetHighPrice")
            target_low  = info.get("targetLowPrice")
            recommend   = info.get("recommendationKey","—")   # strong_buy/buy/hold/sell

            # Valuation
            pe   = info.get("trailingPE")
            fpe  = info.get("forwardPE")
            pb   = info.get("priceToBook")
            ps   = info.get("priceToSalesTrailing12Months")
            ev_ebitda = info.get("enterpriseToEbitda")

            # Institutional
            inst_pct  = info.get("heldPercentInstitutions")    # 0-1
            insider_pct=info.get("heldPercentInsiders")
            short_pct  =info.get("shortPercentOfFloat")

            # Sector/industry
            sector   = info.get("sector","—")
            industry = info.get("industry","—")
            mkt_cap  = info.get("marketCap")
            beta_v   = info.get("beta")

            return {
                "ticker":        ticker,
                "name":          info.get("longName") or info.get("shortName") or ticker,
                "price":         float(price) if price else None,
                "sector":        sector,
                "industry":      industry,
                "mkt_cap":       mkt_cap,
                "beta":          beta_v,
                # Earnings
                "earn_date":     earn_dt,
                "earn_days":     (earn_dt - datetime.now(HK_TZ)).days if earn_dt else None,
                # Growth
                "rev_ttm":       rev_ttm,
                "rev_growth":    rev_growth,
                "earn_growth":   earn_growth,
                "profit_margin": profit_margin,
                # Analyst
                "n_analysts":    n_analysts,
                "target_mean":   float(target_mean) if target_mean else None,
                "target_high":   float(target_high) if target_high else None,
                "target_low":    float(target_low)  if target_low  else None,
                "recommend":     recommend,
                "upside_mean":   ((float(target_mean)-float(price))/float(price)*100
                                  if target_mean and price else None),
                # Valuation
                "pe":            pe,
                "fpe":           fpe,
                "pb":            pb,
                "ps":            ps,
                "ev_ebitda":     ev_ebitda,
                # Institutional
                "inst_pct":      inst_pct,
                "insider_pct":   insider_pct,
                "short_pct":     short_pct,
            }
        except Exception:
            pass
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_fundamentals(tickers: tuple) -> dict:
    """Fetch fundamentals for multiple tickers. Returns {ticker: dict}."""
    result = {}
    for t in tickers:
        f = fetch_fundamentals(t)
        if f: result[t] = f
        time.sleep(0.3)
    return result


def get_triggers(f: dict) -> list:
    """
    Return list of trigger dicts for summary page.
    Each: {type, label, color, detail}
    """
    triggers = []
    if not f: return triggers
    price = f.get("price") or 0

    # Trigger 1: Earnings within 7 days
    days = f.get("earn_days")
    if days is not None and 0 <= days <= 7:
        triggers.append({
            "type":   "earnings",
            "label":  f"⚡ Earnings in {days}d",
            "color":  "#dc2626",
            "detail": f"Results due {f['earn_date'].strftime('%b %d') if f.get('earn_date') else '—'} — expect high volatility",
        })
    elif days is not None and 7 < days <= 14:
        triggers.append({
            "type":   "earnings",
            "label":  f"📅 Earnings in {days}d",
            "color":  "#f59e0b",
            "detail": f"Results due {f['earn_date'].strftime('%b %d') if f.get('earn_date') else '—'}",
        })

    # Trigger 2: Analyst target >20% upside
    upside = f.get("upside_mean")
    if upside is not None and float(upside) >= 20:
        rec = f.get("recommend","—").replace("_"," ").title()
        triggers.append({
            "type":   "analyst",
            "label":  f"🎯 Analyst +{upside:.0f}% upside",
            "color":  "#16a34a",
            "detail": f"Consensus target {f['target_mean']:,.2f} · {f['n_analysts']} analysts · {rec}",
        })
    elif upside is not None and float(upside) <= -10:
        triggers.append({
            "type":   "analyst",
            "label":  f"⚠️ Analyst {upside:.0f}% downside",
            "color":  "#dc2626",
            "detail": f"Target {f['target_mean']:,.2f} below current price · {f['n_analysts']} analysts",
        })

    # Trigger 3: Institutional ownership rising (>50% and high)
    inst = f.get("inst_pct")
    if inst is not None and float(inst) > 0.55:
        triggers.append({
            "type":   "institutional",
            "label":  f"🏦 Inst. owned {inst*100:.0f}%",
            "color":  "#2563eb",
            "detail": "High institutional ownership — smart money conviction",
        })

    # Bonus: Short squeeze risk
    short = f.get("short_pct")
    if short is not None and float(short) > 0.15:
        triggers.append({
            "type":   "short",
            "label":  f"🔥 Short {short*100:.0f}% of float",
            "color":  "#f59e0b",
            "detail": "High short interest — squeeze potential if price moves up",
        })

    return triggers


# ── RENDER ────────────────────────────────────────────────────────────
def render_fundamentals_page():
    """Full fundamentals study page (called from Analysis page)."""
    now_hk = datetime.now(HK_TZ)
    st.markdown("### 📊 Fundamental Study")
    st.markdown(
        "<span style='color:#64748b;font-size:0.79rem'>"
        "Earnings · Revenue growth · Analyst targets · Valuation · Institutional ownership · "
        f"{now_hk.strftime('%Y-%m-%d %H:%M HKT')}</span>",
        unsafe_allow_html=True)

    with st.expander("📖 What these metrics mean"):
        st.markdown("""
**Earnings date** — When the company reports results. Earnings = binary event:
beat estimates → stock gaps up. Miss → gaps down. As a range trader, high volatility
around earnings is an opportunity BUT also a risk — spreads widen, gaps are huge.
Rule: don't hold through earnings unless sized very small.

**Revenue growth %** — Year-over-year revenue change. Growing revenue = business expanding.
Shrinking revenue = warning sign even if stock looks cheap.

**Profit margin** — Net profit as % of revenue. High margin = pricing power.
For HK tech stocks, a margin turning positive is often a catalyst for re-rating.

**Analyst consensus** — Average of all analyst price targets. Buy/Hold/Sell rating.
>20% upside = analysts see significant value. Note: analysts are often late and wrong,
but consensus upgrades/downgrades can move the stock short term.

**P/E ratio** — Price / Earnings per share. Expensive vs history and peers = less upside.
N/A = not yet profitable (common for HK growth stocks). Forward P/E uses estimates.

**P/B ratio** — Price / Book value. <1 = trading below asset value (cheap).
>5 = priced for growth expectations.

**Institutional ownership %** — What % of shares big funds hold.
>50% = strong institutional conviction. Rising = accumulation. Falling = distribution.

**Short % of float** — What % of tradeable shares are sold short.
>15% = crowded short = squeeze risk if price moves up (shorts must cover = buy pressure).
        """)

    # Load portfolio tickers
    try:
        from db_manager import get_portfolio_full
        from portfolio_manager import get_monitor_pos
        port = get_portfolio_full()
        mon  = get_monitor_pos()
    except Exception as e:
        st.error(f"Could not load portfolio: {e}"); return

    # Only stocks — filter out forex/commodities
    stock_tickers = {}
    if not port.empty:
        for _,r in port.iterrows():
            t = r["ticker"]
            if t.endswith("=X") or t.endswith("=F"): continue
            stock_tickers[t] = r.get("name", t)

    if not stock_tickers:
        st.info("No stock positions in portfolio. Fundamentals only apply to stocks.")
        return

    col_btn, col_note = st.columns(2)
    if col_btn.button("🔄 Refresh fundamentals", key="fund_refresh"):
        st.cache_data.clear(); st.rerun()
    col_note.markdown(
        f"<span style='color:#64748b;font-size:0.79rem'>"
        f"{len(stock_tickers)} stocks · Data cached 1 hour</span>",
        unsafe_allow_html=True)

    with st.spinner("Fetching fundamentals…"):
        fund_data = fetch_all_fundamentals(tuple(stock_tickers.keys()))

    if not fund_data:
        st.warning("Could not fetch fundamentals. Check connection.")
        return

    # ── Summary trigger cards ─────────────────────────────────────────
    all_triggers = []
    for t, f in fund_data.items():
        for trig in get_triggers(f):
            trig["ticker"] = t
            trig["name"]   = f.get("name", t)
            all_triggers.append(trig)

    if all_triggers:
        st.markdown("#### ⚡ Active Triggers")
        for trig in sorted(all_triggers, key=lambda x: {"earnings":0,"analyst":1,"institutional":2,"short":3}.get(x["type"],4)):
            st.markdown(
                f"<div style='border-left:4px solid {trig['color']};"
                f"padding:9px 14px;background:rgba(0,0,0,0.02);"
                f"border-radius:0 8px 8px 0;margin-bottom:6px'>"
                f"<b style='color:{trig['color']}'>{trig['label']}</b> · "
                f"<b>{trig['name']}</b> ({trig['ticker']}) · "
                f"<span style='color:#475569;font-size:0.82rem'>{trig['detail']}</span>"
                f"</div>", unsafe_allow_html=True)
        st.markdown("---")

    # ── Main fundamentals table ───────────────────────────────────────
    st.markdown("#### 📋 Fundamentals Overview")
    rows = []
    for t, f in fund_data.items():
        upside = f.get("upside_mean")
        rec    = (f.get("recommend","—") or "—").replace("_"," ").title()
        rows.append({
            "Name":       f.get("name",t),
            "Ticker":     t,
            "Sector":     f.get("sector","—"),
            "Mkt Cap":    (f"HKD {float(f['mkt_cap'])/1e9:.1f}B" if f.get("mkt_cap") else "—"),
            "P/E":        _f(f.get("pe")),
            "Fwd P/E":    _f(f.get("fpe")),
            "P/B":        _f(f.get("pb")),
            "Rev growth": f"{float(f['rev_growth'])*100:+.1f}%" if f.get("rev_growth") not in (None,"—") else "—",
            "Net margin": f"{float(f['profit_margin'])*100:.1f}%" if f.get("profit_margin") not in (None,"—") else "—",
            "Analysts":   str(int(f["n_analysts"])) if f.get("n_analysts") else "—",
            "Target":     _f(f.get("target_mean"), ",.2f"),
            "Upside %":   f"{float(upside):+.1f}%" if upside is not None else "—",
            "Rating":     rec,
            "Inst %":     f"{float(f['inst_pct'])*100:.0f}%" if f.get("inst_pct") else "—",
            "Short %":    f"{float(f['short_pct'])*100:.1f}%" if f.get("short_pct") else "—",
            "Earnings":   (f.get("earn_date").strftime("%b %d") if f.get("earn_date") else "—"),
        })

    df_fund = pd.DataFrame(rows)

    def style_fund(df):
        s = pd.DataFrame("", index=df.index, columns=df.columns)
        for i, row in df.iterrows():
            # Upside
            try:
                up = float(str(row["Upside %"]).replace("%","").replace("+",""))
                if up >= 20:   s.at[i,"Upside %"] = "color:#16a34a;font-weight:700"
                elif up >= 10: s.at[i,"Upside %"] = "color:#16a34a"
                elif up <= -10:s.at[i,"Upside %"] = "color:#dc2626;font-weight:700"
                elif up < 0:   s.at[i,"Upside %"] = "color:#dc2626"
            except: pass
            # Rev growth
            try:
                rg = float(str(row["Rev growth"]).replace("%","").replace("+",""))
                if rg > 20:  s.at[i,"Rev growth"] = "color:#16a34a;font-weight:600"
                elif rg > 0: s.at[i,"Rev growth"] = "color:#16a34a"
                elif rg < 0: s.at[i,"Rev growth"] = "color:#dc2626"
            except: pass
            # Rating
            r_ = str(row["Rating"]).lower()
            if "strong buy" in r_ or r_=="buy":  s.at[i,"Rating"] = "color:#16a34a;font-weight:600"
            elif "sell" in r_:                   s.at[i,"Rating"] = "color:#dc2626;font-weight:600"
            # Earnings soon
            try:
                ed = str(row["Earnings"])
                if ed != "—":
                    earn_f = fund_data.get(row["Ticker"],{})
                    if earn_f.get("earn_days") is not None and earn_f["earn_days"] <= 7:
                        s.at[i,"Earnings"] = "color:#dc2626;font-weight:700"
                    elif earn_f.get("earn_days") is not None and earn_f["earn_days"] <= 14:
                        s.at[i,"Earnings"] = "color:#f59e0b;font-weight:600"
            except: pass
        return s

    st.dataframe(df_fund.style.apply(style_fund, axis=None),
                 use_container_width=True, hide_index=True)

    # ── Individual deep-dives ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Individual company deep-dive")

    for t, f in fund_data.items():
        if not f: continue
        triggers = get_triggers(f)
        trig_str = " · ".join(trig["label"] for trig in triggers) if triggers else ""
        with st.expander(
            f"**{f.get('name',t)}** ({t})"
            + (f"  ·  {trig_str}" if trig_str else ""),
            expanded=bool(triggers)):

            # Row 1: price + valuation
            c1,c2,c3,c4,c5 = st.columns(5)
            for col,lbl,val,color in [
                (c1,"Price",      _f(f.get("price"), ",.2f"),"#0f172a"),
                (c2,"P/E",        f"{f['pe']:.1f}" if f.get("pe") else "N/A",
                 "#16a34a" if (float(f["pe"]) if f.get("pe") else 99)<15 else "#f59e0b" if (float(f["pe"]) if f.get("pe") else 99)<30 else "#dc2626"),
                (c3,"Fwd P/E",    f"{f['fpe']:.1f}" if f.get("fpe") else "N/A","#64748b"),
                (c4,"P/B",        _f(f.get("pb")),"#64748b"),
                (c5,"Mkt Cap",    f"HKD {float(f['mkt_cap'])/1e9:.1f}B" if f.get("mkt_cap") else "—","#64748b"),
            ]:
                col.markdown(
                    f"<div style='text-align:center;background:#f8fafc;"
                    f"border-radius:8px;padding:8px;border:1px solid #e2e8f0'>"
                    f"<div style='font-size:0.65rem;color:#94a3b8'>{lbl}</div>"
                    f"<div style='font-size:0.9rem;font-weight:700;color:{color}'>{val}</div>"
                    f"</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Row 2: growth + analyst
            c6,c7,c8,c9,c10 = st.columns(5)
            upside = f.get("upside_mean")
            rec    = (f.get("recommend","—") or "—").replace("_"," ").title()
            for col,lbl,val,color in [
                (c6,"Rev growth",   f"{float(f['rev_growth'])*100:+.1f}%" if f.get("rev_growth") not in (None,"—") else "—",
                 "#16a34a" if (float(f["rev_growth"]) if f.get("rev_growth") else -1)>0 else "#dc2626"),
                (c7,"Earn growth",  f"{float(f['earn_growth'])*100:+.1f}%" if f.get("earn_growth") not in (None,"—") else "—",
                 "#16a34a" if (float(f["earn_growth"]) if f.get("earn_growth") else -1)>0 else "#dc2626"),
                (c8,"Net margin",   f"{float(f['profit_margin'])*100:.1f}%" if f.get("profit_margin") not in (None,"—") else "—",
                 "#16a34a" if (float(f["profit_margin"]) if f.get("profit_margin") else -1)>0 else "#dc2626"),
                (c9,"Analyst target",_f(f.get("target_mean"), ",.2f"),
                 "#16a34a" if (float(upside) if upside is not None else 0)>10 else "#dc2626" if (float(upside) if upside is not None else 0)<-5 else "#64748b"),
                (c10,"Upside",      f"{float(upside):+.1f}%" if upside is not None else "—",
                 "#16a34a" if (float(upside) if upside is not None else 0)>=20 else "#f59e0b" if (float(upside) if upside is not None else 0)>=5 else "#dc2626"),
            ]:
                col.markdown(
                    f"<div style='text-align:center;background:#f8fafc;"
                    f"border-radius:8px;padding:8px;border:1px solid #e2e8f0'>"
                    f"<div style='font-size:0.65rem;color:#94a3b8'>{lbl}</div>"
                    f"<div style='font-size:0.9rem;font-weight:700;color:{color}'>{val}</div>"
                    f"</div>", unsafe_allow_html=True)

            # Row 3: institutional + upcoming events
            st.markdown("<br>", unsafe_allow_html=True)
            inst = f.get("inst_pct"); short = f.get("short_pct")
            earn_days = f.get("earn_days")
            info_parts = []
            if inst:    info_parts.append(f"🏦 Institutional: {inst*100:.0f}%")
            if short:   info_parts.append(f"🩳 Short float: {short*100:.1f}%")
            if f.get("n_analysts"): info_parts.append(f"📊 {f['n_analysts']} analysts · {rec}")
            if earn_days is not None:
                earn_label = f"⚡ Earnings in {earn_days}d" if earn_days<=7 else f"📅 Earnings in {earn_days}d"
                c_earn = "#dc2626" if earn_days<=7 else "#f59e0b"
                info_parts.append(f"<span style='color:{c_earn};font-weight:600'>{earn_label}</span>")
            if info_parts:
                st.markdown(
                    "<div style='background:#f8fafc;border-radius:8px;padding:9px 14px;"
                    "font-size:0.82rem;border:1px solid #e2e8f0'>"
                    + " &nbsp;·&nbsp; ".join(info_parts)
                    + "</div>", unsafe_allow_html=True)

    st.markdown(
        "<span style='color:#94a3b8;font-size:0.74rem'>"
        "Data via yfinance · Updated hourly · "
        "Fundamentals change slowly — check weekly not daily · Not financial advice</span>",
        unsafe_allow_html=True)


def render_summary_triggers(stock_tickers: dict) -> None:
    """
    Compact trigger display for the Summary page.
    Only shows the 3 key triggers: earnings <7d, analyst upside >20%, inst >55%.
    """
    if not stock_tickers:
        return

    with st.spinner("Checking fundamental triggers…"):
        fund_data = fetch_all_fundamentals(tuple(stock_tickers.keys()))

    all_triggers = []
    for t, f in fund_data.items():
        for trig in get_triggers(f):
            trig["ticker"] = t
            trig["name"]   = f.get("name", t)
            all_triggers.append(trig)

    if not all_triggers:
        return

    # Show compact trigger strip
    st.markdown("#### ⚡ Fundamental Triggers")
    cols = st.columns(min(len(all_triggers), 3))
    for i, trig in enumerate(all_triggers[:6]):
        col = cols[i % 3]
        col.markdown(
            f"<div style='border:1px solid {trig['color']};border-radius:8px;"
            f"padding:9px 12px;margin-bottom:6px;"
            f"background:rgba(0,0,0,0.02)'>"
            f"<div style='font-size:0.75rem;font-weight:600;color:{trig['color']}'>"
            f"{trig['label']}</div>"
            f"<div style='font-size:0.78rem;font-weight:600;color:#0f172a'>"
            f"{trig['name']} ({trig['ticker']})</div>"
            f"<div style='font-size:0.7rem;color:#64748b'>{trig['detail']}</div>"
            f"</div>", unsafe_allow_html=True)
