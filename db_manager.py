"""
db_manager.py
Handles all SQLite database operations for HK stock price data.
Tables:
  - stocks            : stock metadata
  - daily_prices      : OHLCV per day
  - intraday_prices   : OHLCV per minute/interval
  - portfolio         : user positions
  - capital           : total capital snapshots
  - trade_journal     : manual trade log (NEW)
  - analysis_signals  : saved indicator signals (NEW)
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

DB_PATH = Path("hk_stocks.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_conn()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS stocks (
            ticker      TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            short_name  TEXT,
            ipo_price   REAL,
            currency    TEXT DEFAULT 'HKD',
            exchange    TEXT DEFAULT 'HKEX',
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS daily_prices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            date        TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            pct_change  REAL,
            fetched_at  TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker, date),
            FOREIGN KEY(ticker) REFERENCES stocks(ticker)
        );

        CREATE TABLE IF NOT EXISTS intraday_prices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            ts          TEXT NOT NULL,
            interval    TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            fetched_at  TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker, ts, interval),
            FOREIGN KEY(ticker) REFERENCES stocks(ticker)
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            shares      INTEGER NOT NULL DEFAULT 0,
            avg_cost    REAL NOT NULL DEFAULT 0,
            updated_at  TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker),
            FOREIGN KEY(ticker) REFERENCES stocks(ticker)
        );

        CREATE TABLE IF NOT EXISTS capital (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            amount      REAL NOT NULL,
            note        TEXT,
            recorded_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS trade_journal (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT NOT NULL,
            direction    TEXT NOT NULL CHECK(direction IN ('LONG','SHORT')),
            entry_price  REAL NOT NULL,
            exit_price   REAL,
            shares       INTEGER NOT NULL,
            entry_time   TEXT NOT NULL,
            exit_time    TEXT,
            pnl          REAL,
            pnl_pct      REAL,
            strategy     TEXT,
            notes        TEXT,
            emotion      TEXT,
            outcome      TEXT CHECK(outcome IN ('WIN','LOSS','BREAKEVEN',NULL)),
            created_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS analysis_signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            interval    TEXT NOT NULL,
            ts          TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            value       REAL,
            note        TEXT,
            saved_at    TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_daily_ticker_date
            ON daily_prices(ticker, date DESC);
        CREATE INDEX IF NOT EXISTS idx_intraday_ticker_ts
            ON intraday_prices(ticker, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_journal_ticker
            ON trade_journal(ticker, entry_time DESC);
    """)

    cur.executemany("""
        INSERT OR IGNORE INTO stocks (ticker, name, short_name, ipo_price)
        VALUES (?, ?, ?, ?)
    """, [
        ("0100.HK", "MiniMax Group Inc.",                 "MiniMax", 165.0),
        ("2513.HK", "Knowledge Atlas Technology JSC Ltd", "Zhipu",   116.2),
        ("^HSI",    "Hang Seng Index",                    "HSI",     None),
    ])

    conn.commit()
    conn.close()


# ── DAILY PRICES ──────────────────────────────────────────────────────
def upsert_daily(ticker: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    conn = get_conn()
    rows = []
    closes = df["Close"].tolist()
    for i, (idx, row) in enumerate(df.iterrows()):
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        pct = None
        if i > 0 and closes[i - 1]:
            pct = round((closes[i] - closes[i - 1]) / closes[i - 1] * 100, 4)
        rows.append((
            ticker, date_str,
            round(float(row["Open"]),  4) if pd.notna(row.get("Open"))   else None,
            round(float(row["High"]),  4) if pd.notna(row.get("High"))   else None,
            round(float(row["Low"]),   4) if pd.notna(row.get("Low"))    else None,
            round(float(row["Close"]), 4) if pd.notna(row.get("Close"))  else None,
            int(row["Volume"])             if pd.notna(row.get("Volume")) else None,
            pct,
        ))
    conn.executemany("""
        INSERT INTO daily_prices (ticker, date, open, high, low, close, volume, pct_change)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume,
            pct_change=excluded.pct_change, fetched_at=datetime('now')
    """, rows)
    conn.commit()
    conn.close()


def get_daily(ticker: str, limit: int = 365) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM daily_prices WHERE ticker=? ORDER BY date DESC LIMIT ?",
        conn, params=(ticker, limit)
    )
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


# ── INTRADAY PRICES ───────────────────────────────────────────────────
def upsert_intraday(ticker: str, df: pd.DataFrame, interval: str):
    if df is None or df.empty:
        return
    conn = get_conn()
    rows = []
    for idx, row in df.iterrows():
        ts_str = idx.strftime("%Y-%m-%d %H:%M:%S") if hasattr(idx, "strftime") else str(idx)
        rows.append((
            ticker, ts_str, interval,
            round(float(row["Open"]),  4) if pd.notna(row.get("Open"))   else None,
            round(float(row["High"]),  4) if pd.notna(row.get("High"))   else None,
            round(float(row["Low"]),   4) if pd.notna(row.get("Low"))    else None,
            round(float(row["Close"]), 4) if pd.notna(row.get("Close"))  else None,
            int(row["Volume"])             if pd.notna(row.get("Volume")) else None,
        ))
    conn.executemany("""
        INSERT INTO intraday_prices (ticker, ts, interval, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, ts, interval) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume, fetched_at=datetime('now')
    """, rows)
    conn.commit()
    conn.close()


def get_intraday(ticker: str, interval: str = "5m", limit: int = 500) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM intraday_prices WHERE ticker=? AND interval=? ORDER BY ts DESC LIMIT ?",
        conn, params=(ticker, interval, limit)
    )
    conn.close()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")
    return df


# ── PORTFOLIO ─────────────────────────────────────────────────────────
def upsert_position(ticker: str, shares: int, avg_cost: float):
    conn = get_conn()
    conn.execute("""
        INSERT INTO portfolio (ticker, shares, avg_cost)
        VALUES (?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            shares=excluded.shares, avg_cost=excluded.avg_cost, updated_at=datetime('now')
    """, (ticker, shares, avg_cost))
    conn.commit()
    conn.close()


def get_portfolio() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT p.*, s.name, s.ipo_price FROM portfolio p JOIN stocks s USING(ticker)", conn
    )
    conn.close()
    return df


# ── CAPITAL ───────────────────────────────────────────────────────────
def save_capital(amount: float, note: str = ""):
    conn = get_conn()
    conn.execute("INSERT INTO capital (amount, note) VALUES (?, ?)", (amount, note))
    conn.commit()
    conn.close()


def get_latest_capital() -> float:
    conn = get_conn()
    row = conn.execute(
        "SELECT amount FROM capital ORDER BY recorded_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0] if row else 100000.0


def get_capital_history() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM capital ORDER BY recorded_at DESC LIMIT 50", conn)
    conn.close()
    return df


# ── TRADE JOURNAL ─────────────────────────────────────────────────────
def save_trade(ticker, direction, entry_price, shares, entry_time,
               exit_price=None, exit_time=None, strategy="", notes="", emotion=""):
    pnl = pnl_pct = outcome = None
    if exit_price:
        mult = 1 if direction == "LONG" else -1
        pnl = mult * (exit_price - entry_price) * shares
        pnl_pct = mult * (exit_price - entry_price) / entry_price * 100
        outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
    conn = get_conn()
    conn.execute("""
        INSERT INTO trade_journal
          (ticker, direction, entry_price, exit_price, shares,
           entry_time, exit_time, pnl, pnl_pct, strategy, notes, emotion, outcome)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (ticker, direction, entry_price, exit_price, shares,
          entry_time, exit_time, pnl, pnl_pct, strategy, notes, emotion, outcome))
    conn.commit()
    conn.close()


def get_trades(ticker: str = None, limit: int = 200) -> pd.DataFrame:
    conn = get_conn()
    q = "SELECT * FROM trade_journal"
    params = []
    if ticker:
        q += " WHERE ticker=?"
        params.append(ticker)
    q += " ORDER BY entry_time DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df


def update_trade_exit(trade_id: int, exit_price: float, exit_time: str,
                      direction: str, entry_price: float, shares: int):
    mult  = 1 if direction == "LONG" else -1
    pnl   = mult * (exit_price - entry_price) * shares
    pnl_pct = mult * (exit_price - entry_price) / entry_price * 100
    outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
    conn = get_conn()
    conn.execute("""
        UPDATE trade_journal
        SET exit_price=?, exit_time=?, pnl=?, pnl_pct=?, outcome=?
        WHERE id=?
    """, (exit_price, exit_time, pnl, pnl_pct, outcome, trade_id))
    conn.commit()
    conn.close()


def delete_trade(trade_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM trade_journal WHERE id=?", (trade_id,))
    conn.commit()
    conn.close()


# ── STATS ─────────────────────────────────────────────────────────────
def get_daily_stats(ticker: str) -> dict:
    conn = get_conn()
    row = conn.execute("""
        SELECT COUNT(*) AS days_stored, MIN(low) AS all_time_low,
               MAX(high) AS all_time_high, AVG(volume) AS avg_volume,
               SUM(CASE WHEN pct_change > 0 THEN 1 ELSE 0 END) AS up_days,
               SUM(CASE WHEN pct_change < 0 THEN 1 ELSE 0 END) AS down_days
        FROM daily_prices WHERE ticker=?
    """, (ticker,)).fetchone()
    conn.close()
    keys = ["days_stored","all_time_low","all_time_high","avg_volume","up_days","down_days"]
    return dict(zip(keys, row)) if row else {}


def get_raw_sql(query: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


# ── EXTENDED PORTFOLIO (with target/stop/notes) ───────────────────────
def init_portfolio_extended():
    """Add extra columns to portfolio table if not present."""
    conn = get_conn()
    cur  = conn.cursor()
    existing = [r[1] for r in cur.execute("PRAGMA table_info(portfolio)").fetchall()]
    extras = {
        "target_price": "REAL",
        "stop_price":   "REAL",
        "notes":        "TEXT",
        "status":       "TEXT DEFAULT 'OPEN'",  # OPEN / CLOSED
        "entry_date":   "TEXT",
    }
    for col, typ in extras.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE portfolio ADD COLUMN {col} {typ}")
    conn.commit()
    conn.close()


def upsert_position_full(ticker: str, name: str, shares: int, avg_cost: float,
                          target_price=None, stop_price=None,
                          notes="", status="OPEN", entry_date=""):
    """Full upsert including extended fields. Inserts stock row if missing."""
    conn = get_conn()
    # Ensure stock exists
    conn.execute("""
        INSERT OR IGNORE INTO stocks (ticker, name, short_name, ipo_price)
        VALUES (?, ?, ?, NULL)
    """, (ticker, name, name[:12]))
    conn.execute("""
        INSERT INTO portfolio
          (ticker, shares, avg_cost, target_price, stop_price, notes, status, entry_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            shares=excluded.shares,
            avg_cost=excluded.avg_cost,
            target_price=excluded.target_price,
            stop_price=excluded.stop_price,
            notes=excluded.notes,
            status=excluded.status,
            entry_date=excluded.entry_date,
            updated_at=datetime('now')
    """, (ticker, shares, avg_cost, target_price, stop_price,
          notes, status, entry_date))
    conn.commit()
    conn.close()


def get_portfolio_full() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT p.id, p.ticker, s.name, p.shares, p.avg_cost,
               p.target_price, p.stop_price, p.notes,
               p.status, p.entry_date, p.updated_at
        FROM portfolio p
        LEFT JOIN stocks s ON p.ticker = s.ticker
        ORDER BY p.updated_at DESC
    """, conn)
    conn.close()
    return df


def delete_position(ticker: str):
    conn = get_conn()
    conn.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
    conn.commit()
    conn.close()


def close_position(ticker: str, exit_price: float, exit_date: str):
    """Mark position as CLOSED and auto-log to trade_journal."""
    conn = get_conn()
    row = conn.execute(
        "SELECT shares, avg_cost FROM portfolio WHERE ticker=?", (ticker,)
    ).fetchone()
    if row:
        shares, avg_cost = row
        pnl     = (exit_price - avg_cost) * shares
        pnl_pct = (exit_price - avg_cost) / avg_cost * 100 if avg_cost else 0
        outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
        conn.execute("""
            UPDATE portfolio SET status='CLOSED', updated_at=datetime('now')
            WHERE ticker=?
        """, (ticker,))
        # Auto-log to trade_journal
        conn.execute("""
            INSERT INTO trade_journal
              (ticker, direction, entry_price, exit_price, shares,
               entry_time, exit_time, pnl, pnl_pct, strategy, outcome)
            VALUES (?, 'LONG', ?, ?, ?, ?, ?, ?, ?, 'Portfolio close', ?)
        """, (ticker, avg_cost, exit_price, shares,
              "", exit_date, pnl, pnl_pct, outcome))
    conn.commit()
    conn.close()


# ── ACTIVITY LOG ─────────────────────────────────────────────────────
def init_activity_log():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            action      TEXT NOT NULL,
            ticker      TEXT,
            detail      TEXT,
            logged_at   TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def log_activity(action: str, ticker: str = "", detail: str = ""):
    conn = get_conn()
    conn.execute("""
        INSERT INTO activity_log (action, ticker, detail)
        VALUES (?, ?, ?)
    """, (action, ticker, detail))
    conn.commit()
    conn.close()


def get_activity_log(limit: int = 100) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT * FROM activity_log ORDER BY logged_at DESC LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df
