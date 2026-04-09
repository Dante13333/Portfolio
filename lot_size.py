"""
lot_size.py — HKEX Board Lot Size Lookup
HKEX stocks have standard board lots that vary by price tier.
Also supports manual override per ticker stored in SQLite.
"""

import yfinance as yf
import streamlit as st
from db_manager import get_conn

# ── HKEX board lot by price tier (approximate standard) ──────────────
# Source: HKEX rules — lot size generally decreases as share price rises
PRICE_LOT_TABLE = [
    (0.25,   20000),   # < 0.25 → 20,000 shares
    (0.50,   10000),   # 0.25 – 0.50 → 10,000
    (1.00,    5000),   # 0.50 – 1.00 → 5,000
    (5.00,    2000),   # 1.00 – 5.00 → 2,000
    (10.00,   1000),   # 5.00 – 10.00 → 1,000
    (20.00,    500),   # 10 – 20 → 500
    (50.00,    200),   # 20 – 50 → 200
    (100.00,   100),   # 50 – 100 → 100
    (200.00,    50),   # 100 – 200 → 50
    (500.00,    20),   # 200 – 500 → 20 (rare)
    (float("inf"), 10),# >500 → 10 (very high priced stocks)
]

# Known overrides for common stocks
KNOWN_LOTS = {
    "0700.HK": 100,   "9988.HK": 50,    "9999.HK": 200,
    "1024.HK": 100,   "0020.HK": 200,   "1810.HK": 500,
    "9866.HK": 50,    "9868.HK": 100,   "2015.HK": 100,
    "1211.HK": 500,   "0175.HK": 2000,  "2238.HK": 500,
    "2269.HK": 500,   "6160.HK": 100,   "9618.HK": 100,
    "0388.HK": 100,   "2318.HK": 500,   "1299.HK": 500,
    "0005.HK": 400,   "0939.HK": 1000,  "1398.HK": 1000,
    "2899.HK": 500,   "0883.HK": 1000,  "0857.HK": 2000,
    "0100.HK": 100,   "2513.HK": 100,   "3750.HK": 100,
    "0700.HK": 100,   "0268.HK": 500,   "0763.HK": 1000,
}

def init_lot_table():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lot_sizes (
            ticker  TEXT PRIMARY KEY,
            lot     INTEGER NOT NULL,
            source  TEXT DEFAULT 'manual',
            updated TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit(); conn.close()

def get_saved_lot(ticker: str):
    try:
        conn = get_conn()
        r = conn.execute(
            "SELECT lot FROM lot_sizes WHERE ticker=?", (ticker,)).fetchone()
        conn.close()
        return int(r[0]) if r else None
    except: return None

def save_lot(ticker: str, lot: int, source: str = "manual"):
    try:
        conn = get_conn()
        conn.execute("""
            INSERT INTO lot_sizes (ticker,lot,source) VALUES (?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET lot=excluded.lot,
            source=excluded.source, updated=datetime('now')
        """, (ticker, lot, source))
        conn.commit(); conn.close()
    except: pass

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_lot_yf(ticker: str):
    """Try yfinance lotSize field."""
    try:
        info = yf.Ticker(ticker).info
        lot = info.get("lotSize") or info.get("lot_size")
        if lot and int(lot) > 0:
            return int(lot)
    except: pass
    return None

def price_to_lot(price: float) -> int:
    """Estimate lot size from price tier."""
    for threshold, lot in PRICE_LOT_TABLE:
        if price < threshold:
            return lot
    return 10

def get_lot(ticker: str, price: float = None) -> int:
    """
    Get board lot for a ticker. Priority:
    1. User-saved manual override (DB)
    2. Known lots dict
    3. yfinance lotSize field
    4. Price-tier estimate
    """
    # 1. Manual override
    saved = get_saved_lot(ticker)
    if saved: return saved

    # 2. Known lots
    if ticker in KNOWN_LOTS:
        return KNOWN_LOTS[ticker]

    # 3. yfinance
    yf_lot = fetch_lot_yf(ticker)
    if yf_lot:
        save_lot(ticker, yf_lot, "yfinance")
        return yf_lot

    # 4. Price tier
    if price and price > 0:
        return price_to_lot(price)

    return 100  # safe default

def round_to_lot(shares: float, lot: int) -> int:
    """Round shares DOWN to nearest valid lot."""
    if lot <= 0: return int(shares)
    lots = int(shares // lot)
    return max(lots * lot, lot)  # at least 1 lot

def min_cost(ticker: str, price: float) -> float:
    """Minimum cost to buy one board lot."""
    lot = get_lot(ticker, price)
    return lot * price if price > 0 else 0
