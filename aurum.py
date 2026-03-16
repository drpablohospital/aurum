import os, time, json, csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

CONFIG_FILE = "config.json"
STATE_FILE  = "state.json"
TRADES_FILE = "trades.csv"


# =========================
# UI helpers
# =========================
def now_utc_ts():
    return int(datetime.now(timezone.utc).timestamp())

def ts_hms():
    return datetime.now().strftime("%H:%M:%S")

def fmt(x, n=2):
    try:
        return round(float(x), n)
    except Exception:
        return x

def bar(title="", w=60, ch="═"):
    if title:
        title = f" {title} "
        left = max(0, (w - len(title)) // 2)
        right = max(0, w - len(title) - left)
        return ch * left + title + ch * right
    return ch * w

def pill(label, value):
    return f"{label}:{value}"

def ui(msg: str):
    print(f"[{ts_hms()}] {msg}")

def ui_panel(header: str, lines: list):
    ui(bar(header))
    for ln in lines:
        ui("  " + ln)
    ui(bar())


# =========================
# File helpers
# =========================
def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def ensure_trades_csv():
    if os.path.exists(TRADES_FILE):
        return
    with open(TRADES_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ts_utc","event","side","entry","mark","pnl_pct",
            "exit_reason","note"
        ])

def log_event(event, side, entry, mark, pnl_pct, exit_reason="", note=""):
    ensure_trades_csv()
    with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.utcnow().isoformat(),
            event, side, fmt(entry,2), fmt(mark,2), round(float(pnl_pct), 3),
            exit_reason, note
        ])


# =========================
# Indicators
# =========================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi_series(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close= df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


# =========================
# Exchange
# =========================
def get_exchange(testnet: bool):
    ex = ccxt.binanceusdm({
        "apiKey": os.getenv("BINANCE_KEY"),
        "secret": os.getenv("BINANCE_SECRET"),
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    if testnet:
        ex.set_sandbox_mode(True)
        ex.urls["api"] = {
            "public": "https://testnet.binancefuture.com",
            "private": "https://testnet.binancefuture.com",
        }
    return ex

def set_leverage(ex, market_symbol: str, lev: int):
    try:
        ex.set_leverage(lev, market_symbol)
    except Exception as e:
        ui(f"⚠️ WARN set_leverage: {repr(e)}")

def fetch_ohlc(ex, symbol: str, tf: str, limit=500):
    ohlc = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlc, columns=["ts","open","high","low","close","vol"])
    for c in ["open","high","low","close","vol"]:
        df[c] = df[c].astype(float)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def get_usdt_equity(ex) -> float:
    bal = ex.fetch_balance()
    total = bal.get("total", {})
    return float(total.get("USDT", 0.0) or 0.0)

def get_position(ex, symbol: str):
    try:
        positions = ex.fetch_positions([symbol])
    except Exception:
        positions = ex.fetch_positions()

    for p in positions:
        if p.get("symbol") == symbol:
            contracts = float(p.get("contracts") or 0.0)
            if abs(contracts) < 1e-12:
                return None
            side = p.get("side")  # 'long'/'short'
            entry = float(p.get("entryPrice") or 0.0)
            mark = float(p.get("markPrice") or 0.0) if p.get("markPrice") is not None else float(ex.fetch_ticker(symbol)["last"])
            side_norm = "LONG" if side and side.lower() == "long" else "SHORT"
            return {"contracts": contracts, "entry": entry, "mark": mark, "side": side_norm}
    return None

def cancel_if_exists(ex, symbol: str, order_id):
    if not order_id:
        return None
    try:
        ex.cancel_order(order_id, symbol)
    except Exception:
        pass
    return None

def order_filled(ex, symbol: str, order_id):
    if not order_id:
        return False
    try:
        o = ex.fetch_order(order_id, symbol)
        return o.get("status") == "closed"
    except Exception:
        return False


# =========================
# PnL math
# =========================
def pnl_percent(entry: float, mark: float, side: str, lev: float) -> float:
    if entry <= 0:
        return 0.0
    if side == "LONG":
        return ((mark - entry) / entry) * lev * 100
    else:
        return ((entry - mark) / entry) * lev * 100

def fee_equity_pct_from_slmove(risk_pct: float, fee_roundtrip: float, sl_move_pct: float) -> float:
    if sl_move_pct <= 1e-12:
        return 0.0
    return risk_pct * (fee_roundtrip / sl_move_pct)

def compute_notional_usdt(equity_usdt: float, risk_pct: float, sl_move_pct: float) -> float:
    # risk_usdt = equity * risk%
    # if stop distance (price move) is sl_move_pct, notional = risk / sl_move_pct
    risk_usdt = equity_usdt * (risk_pct / 100.0)
    if sl_move_pct <= 1e-12:
        return 0.0
    return float(risk_usdt / sl_move_pct)

def get_market_min_qty(ex, symbol: str) -> float | None:
    try:
        market = ex.market(symbol)
        mn = market.get("limits", {}).get("amount", {}).get("min", None)
        if mn is None:
            return None
        return float(mn)
    except Exception:
        return None

def adjust_notional_to_minimum(ex, symbol: str, target_notional: float, equity_usdt: float, lev: float, min_buffer: float = 1.01):
    ticker = ex.fetch_ticker(symbol)
    price = float(ticker["last"])

    min_qty = get_market_min_qty(ex, symbol)
    if min_qty is None:
        return target_notional, None, None, price

    min_notional = min_qty * price * min_buffer
    max_notional = equity_usdt * lev * 0.98

    if min_notional > max_notional:
        needed_equity = min_notional / lev
        raise ccxt.InvalidOrder(
            f"Notional mínimo ≈{min_notional:.2f} USDT (min_qty={min_qty}) excede tu máximo ≈{max_notional:.2f} USDT. "
            f"Con lev={lev}x necesitas equity futures ≥{needed_equity:.2f} USDT."
        )

    if target_notional < min_notional:
        return min_notional, min_qty, min_notional, price

    return min(target_notional, max_notional), min_qty, min_notional, price

def market_open_with_minimums(ex, symbol: str, side: str, target_notional_usdt: float, equity_usdt: float, lev: float):
    notional, min_qty, min_notional, price = adjust_notional_to_minimum(
        ex, symbol, target_notional_usdt, equity_usdt, lev, min_buffer=1.01
    )
    raw_qty = notional / price
    qty = float(ex.amount_to_precision(symbol, raw_qty))

    if min_qty is not None and qty < float(min_qty):
        if min_notional is None:
            raise ccxt.InvalidOrder(f"qty={qty} < min_qty={min_qty} (y no pude calcular min_notional)")
        raw_qty2 = (min_notional / price)
        qty2 = float(ex.amount_to_precision(symbol, raw_qty2))
        if qty2 < float(min_qty):
            raise ccxt.InvalidOrder(f"qty={qty2} sigue < min_qty={min_qty}. Sube equity o cambia símbolo.")
        qty = qty2
        notional = qty * price

    order_side = "buy" if side == "LONG" else "sell"
    order = ex.create_order(symbol, "market", order_side, qty)
    return order, qty, notional, price, min_qty, min_notional

def place_sl_tp_prices(ex, symbol: str, side: str, qty: float, sl_price: float, tp_price: float):
    qty = float(ex.amount_to_precision(symbol, qty))
    close_side = "sell" if side == "LONG" else "buy"

    sl = ex.create_order(
        symbol, "STOP_MARKET", close_side, qty, None,
        params={"stopPrice": float(f"{sl_price:.2f}"), "reduceOnly": True, "workingType": "MARK_PRICE"}
    )
    tp = ex.create_order(
        symbol, "TAKE_PROFIT_MARKET", close_side, qty, None,
        params={"stopPrice": float(f"{tp_price:.2f}"), "reduceOnly": True, "workingType": "MARK_PRICE"}
    )
    return sl["id"], tp["id"]

def replace_only_sl_price(ex, symbol: str, side: str, qty: float, new_sl_price: float, old_sl_id: str):
    cancel_if_exists(ex, symbol, old_sl_id)
    qty = float(ex.amount_to_precision(symbol, qty))
    close_side = "sell" if side == "LONG" else "buy"
    sl = ex.create_order(
        symbol, "STOP_MARKET", close_side, qty, None,
        params={"stopPrice": float(f"{new_sl_price:.2f}"), "reduceOnly": True, "workingType": "MARK_PRICE"}
    )
    return sl["id"]


# =========================
# Strategy helpers
# =========================
def floor_time(dt: pd.Timestamp, rule: str) -> pd.Timestamp:
    return dt.floor(rule)

def sl_move_pct(entry_px: float, init_sl_px: float, side: str) -> float:
    if side == "LONG":
        return max((entry_px - init_sl_px) / entry_px, 1e-9)
    else:
        return max((init_sl_px - entry_px) / entry_px, 1e-9)

def classify_stop_exit(side: str, entry_px: float, stop_px: float, eps: float = 1e-6) -> str:
    # for logging only
    if side == "LONG":
        if stop_px < entry_px - eps:
            return "SL"
        if abs(stop_px - entry_px) <= eps:
            return "BE"
        return "TRAIL"
    else:
        if stop_px > entry_px + eps:
            return "SL"
        if abs(stop_px - entry_px) <= eps:
            return "BE"
        return "TRAIL"


# =========================
# Main
# =========================
def main():
    load_dotenv()
    cfg = load_json(CONFIG_FILE)
    if not cfg:
        raise RuntimeError("Missing config.json")

    testnet = os.getenv("TESTNET", "true").lower() == "true"
    ex = get_exchange(testnet=testnet)

    symbol = cfg["symbol"]
    market_symbol = cfg.get("market_symbol", symbol)

    lev = float(cfg.get("leverage", 5))
    set_leverage(ex, market_symbol, int(lev))

    poll = int(cfg.get("poll_seconds", 10))
    heartbeat = int(cfg.get("status_heartbeat_seconds", 60))
    last_status_ts = 0

    # strategy params
    use_1h = bool(cfg.get("use_1h_trend_filter", True))
    ema_fast_1h = int(cfg.get("ema_fast_1h", 50))
    ema_slow_1h = int(cfg.get("ema_slow_1h", 200))

    ema_fast_15 = int(cfg.get("ema_fast_15m", 50))
    ema_slow_15 = int(cfg.get("ema_slow_15m", 200))
    rsi_len_15  = int(cfg.get("rsi_len_15m", 14))
    rsi_long    = float(cfg.get("rsi_long", 55))
    rsi_short   = float(cfg.get("rsi_short", 45))

    atr_len_15  = int(cfg.get("atr_len_15m", 14))
    atr_ma_win  = int(cfg.get("atr_ma_window_15m", 96))
    atr_min_mult= float(cfg.get("atr_min_mult", 1.25))

    don_w       = int(cfg.get("donchian_window_15m", 20))
    buf_k       = float(cfg.get("breakout_buffer_atr", 0.55))

    sl_k        = float(cfg.get("sl_atr_mult", 1.1))
    tp_k        = float(cfg.get("tp_atr_mult", 3.2))

    be_R        = float(cfg.get("breakeven_R", 1.0))
    trail_R     = float(cfg.get("trail_start_R", 1.5))
    trail_atr_k = float(cfg.get("trail_atr_mult", 1.0))

    armed_timeout_min = int(cfg.get("armed_timeout_minutes", 30))
    entry_window_5m_bars = int(cfg.get("entry_window_5m_bars", 6))
    max_hold_min = int(cfg.get("max_hold_minutes", 360))
    cooldown_min = int(cfg.get("cooldown_minutes", 30))

    risk_pct = float(cfg.get("risk_per_trade_pct", 1.0))
    fees_enabled = bool(cfg.get("fees_enabled", True))
    fee_rt = float(cfg.get("fee_rate_roundtrip", 0.001))
    slip_rt= float(cfg.get("slippage_roundtrip_pct", 0.0))

    tft_enabled = bool(cfg.get("tft_enabled", False))
    max_loss_streak = int(cfg.get("max_loss_streak", 5))

    # state
    state = load_json(STATE_FILE) or {}
    state.setdefault("paused", False)
    state.setdefault("loss_streak", 0)
    state.setdefault("cooldown_until", 0)

    # armed payload for breakout
    state.setdefault("armed", {
        "active": False,
        "side": None,
        "armed_ts": None,
        "expires_ts": None,
        "deadline_dt": None,
        "trigger_level": None,
        "atr15": None,
        "init_sl_px": None,
        "tp_px": None,
        "note": ""
    })
    state.setdefault("orders", {"sl_id": None, "tp_id": None})
    state.setdefault("trade", {
        "entry_time": None,
        "entry_px": None,
        "init_sl_px": None,
        "tp_px": None,
        "atr15": None,
        "max_fav_R": -999,
        "trail_active": False
    })
    save_json(STATE_FILE, state)

    ui_panel("AURUM — PITONISA LIVE", [
        pill("TESTNET", testnet),
        pill("SYMBOL", symbol),
        pill("LEV", lev),
        pill("Risk%", risk_pct),
        pill("15m EMA", f"{ema_fast_15}/{ema_slow_15}"),
        pill("1h filter", use_1h),
        pill("RSI long/short", f"{rsi_long}/{rsi_short}"),
        pill("ATR mult", atr_min_mult),
        pill("Donchian", don_w),
        pill("BUF_ATR", buf_k),
        pill("SL_ATR", sl_k),
        pill("TP_ATR", tp_k),
        pill("BE_R", be_R),
        pill("TRAIL_R", trail_R),
        pill("TRAIL_ATR", trail_atr_k),
    ])

    # sanity
    try:
        t = ex.fetch_time()
        eq = get_usdt_equity(ex)
        ui(f"Exchange time: {t} | USDT equity: {fmt(eq,2)}")
    except Exception as e:
        ui("FATAL: cannot fetch time/balance. Check keys & futures permission.")
        ui(repr(e))
        return

    last_15m_close_ts = None

    while True:
        try:
            state = load_json(STATE_FILE) or state

            if state.get("paused"):
                ui_panel("PAUSED", [
                    "🚫 Bot pausado por racha de pérdidas.",
                    "Para reanudar: edita state.json -> paused=false, loss_streak=0",
                ])
                time.sleep(10)
                continue

            nowts = now_utc_ts()
            if nowts < int(state.get("cooldown_until", 0)):
                # still cooldown, but manage any open pos anyway
                pass

            pos = get_position(ex, symbol)

            # ==================================
            # 1) Manage open position (BE + TRAIL + TIME)
            # ==================================
            if pos:
                entry = float(pos["entry"])
                mark  = float(pos["mark"])
                side  = pos["side"]

                # load trade context
                tr = state.get("trade", {})
                init_sl_px = float(tr.get("init_sl_px") or 0.0)
                atr15 = float(tr.get("atr15") or 0.0)
                tp_px = float(tr.get("tp_px") or 0.0)
                max_fav_R = float(tr.get("max_fav_R") or -999)

                # update max favorable R using mark as proxy (conservador)
                if init_sl_px > 0 and entry > 0:
                    init_risk = abs(entry - init_sl_px)
                    if init_risk < 1e-9:
                        cur_R = 0.0
                    else:
                        if side == "LONG":
                            cur_R = (mark - entry) / init_risk
                        else:
                            cur_R = (entry - mark) / init_risk
                    max_fav_R = max(max_fav_R, cur_R)
                    tr["max_fav_R"] = max_fav_R

                # Time exit
                entry_time_iso = tr.get("entry_time")
                if entry_time_iso:
                    entry_dt = pd.to_datetime(entry_time_iso, utc=True)
                    age_min = (pd.Timestamp.utcnow().tz_localize("UTC") - entry_dt).total_seconds() / 60.0
                else:
                    age_min = 0.0

                # trailing stop management: cancel+replace SL when needed
                pos_now = get_position(ex, symbol)
                if pos_now:
                    qty = abs(pos_now["contracts"])

                    # current SL price from state by refetch? we store last_sl_px
                    last_sl_px = float(tr.get("last_sl_px") or init_sl_px)

                    # 1) BE
                    if max_fav_R >= be_R:
                        be_px = entry
                        if side == "LONG":
                            new_sl_px = max(last_sl_px, be_px)
                        else:
                            new_sl_px = min(last_sl_px, be_px)
                    else:
                        new_sl_px = last_sl_px

                    # 2) TRAIL (ATR-based)
                    if max_fav_R >= trail_R and atr15 > 0:
                        # use mark as proxy for favorable extreme (conservador)
                        if side == "LONG":
                            trail_px = mark - atr15 * trail_atr_k
                            new_sl_px = max(new_sl_px, trail_px)
                        else:
                            trail_px = mark + atr15 * trail_atr_k
                            new_sl_px = min(new_sl_px, trail_px)

                    # if improved meaningfully, replace SL order
                    if abs(new_sl_px - last_sl_px) / max(1.0, entry) > 1e-4:
                        old_sl = state["orders"].get("sl_id")
                        new_sl_id = replace_only_sl_price(ex, symbol, side, qty, new_sl_px, old_sl)
                        state["orders"]["sl_id"] = new_sl_id
                        tr["last_sl_px"] = new_sl_px
                        state["trade"] = tr
                        save_json(STATE_FILE, state)
                        ui(f"🔁 SL updated -> {fmt(new_sl_px,2)}  (maxR={fmt(max_fav_R,2)})")

                # closure check
                tp_filled = order_filled(ex, symbol, state["orders"].get("tp_id"))
                sl_filled = order_filled(ex, symbol, state["orders"].get("sl_id"))
                pos_check = get_position(ex, symbol)

                if not pos_check:
                    pnl = pnl_percent(entry, mark, side, lev)
                    if tp_filled:
                        exit_reason = "TP"
                    elif sl_filled:
                        # infer stop type
                        sl_px = float(tr.get("last_sl_px") or init_sl_px)
                        exit_reason = classify_stop_exit(side, entry, sl_px)
                    else:
                        exit_reason = "UNKNOWN"

                    ui_panel("TRADE CLOSED", [
                        pill("Side", side),
                        pill("PnL%", fmt(pnl,2)),
                        pill("Exit", exit_reason),
                        pill("maxR", fmt(tr.get("max_fav_R", 0.0), 2)),
                    ])
                    log_event("CLOSED", side, entry, mark, pnl, exit_reason=exit_reason)

                    # loss streak handling (simple)
                    if exit_reason in ("SL",) and pnl < 0:
                        state["loss_streak"] = int(state.get("loss_streak", 0)) + 1
                        if state["loss_streak"] >= max_loss_streak:
                            state["paused"] = True
                    else:
                        state["loss_streak"] = 0

                    # cooldown
                    state["cooldown_until"] = now_utc_ts() + cooldown_min * 60

                    # reset orders/trade
                    state["orders"] = {"sl_id": None, "tp_id": None}
                    state["trade"] = {
                        "entry_time": None,
                        "entry_px": None,
                        "init_sl_px": None,
                        "tp_px": None,
                        "atr15": None,
                        "max_fav_R": -999,
                        "trail_active": False
                    }
                    state["armed"] = {
                        "active": False, "side": None, "armed_ts": None, "expires_ts": None,
                        "deadline_dt": None, "trigger_level": None, "atr15": None,
                        "init_sl_px": None, "tp_px": None, "note": ""
                    }
                    save_json(STATE_FILE, state)

                else:
                    if now_utc_ts() - last_status_ts >= heartbeat:
                        last_status_ts = now_utc_ts()
                        pnl = pnl_percent(entry, mark, side, lev)
                        ui_panel("POSITION STATUS", [
                            pill("Side", side),
                            pill("Entry", fmt(entry,2)),
                            pill("Mark", fmt(mark,2)),
                            pill("PnL%", fmt(pnl,2)),
                            pill("maxR", fmt(state.get("trade",{}).get("max_fav_R", 0.0), 2)),
                            pill("LossStreak", state.get("loss_streak", 0)),
                        ])

                # time-exit enforcement: if too old, close market
                if pos and state.get("trade", {}).get("entry_time"):
                    entry_dt = pd.to_datetime(state["trade"]["entry_time"], utc=True)
                    age_min = (pd.Timestamp.utcnow().tz_localize("UTC") - entry_dt).total_seconds() / 60.0
                    if age_min >= max_hold_min:
                        ui_panel("TIME EXIT", [f"⏱️ age_min={fmt(age_min,1)} >= {max_hold_min} -> closing market"])
                        # close full position market reduceOnly
                        qty = abs(pos["contracts"])
                        qty = float(ex.amount_to_precision(symbol, qty))
                        close_side = "sell" if side == "LONG" else "buy"
                        ex.create_order(symbol, "market", close_side, qty, params={"reduceOnly": True})

                time.sleep(poll)
                continue

            # ==================================
            # 2) No position: generate 15m signal (on new 15m candle)
            # ==================================
            df15 = fetch_ohlc(ex, symbol, "15m", limit=600)
            df5  = fetch_ohlc(ex, symbol, "5m",  limit=400)
            last15_ts = int(df15["ts"].iloc[-1])
            last15_close_dt = pd.to_datetime(last15_ts, unit="ms", utc=True)

            # detect new 15m
            if last_15m_close_ts is None:
                last_15m_close_ts = last15_ts
            is_new_15m = (last15_ts != last_15m_close_ts)

            # heartbeat (no position)
            if now_utc_ts() - last_status_ts >= heartbeat:
                last_status_ts = now_utc_ts()
                eq = get_usdt_equity(ex)
                armed = state.get("armed", {}).get("active", False)
                ui_panel("IDLE", [
                    pill("Equity", fmt(eq,2)),
                    pill("Cooldown", max(0, int(state.get("cooldown_until",0)) - now_utc_ts())),
                    pill("Armed", armed),
                    pill("LossStreak", state.get("loss_streak",0)),
                ])

            # do not arm during cooldown
            if now_utc_ts() < int(state.get("cooldown_until", 0)):
                time.sleep(poll)
                continue

            if is_new_15m:
                last_15m_close_ts = last15_ts

                c15 = df15["close"]
                df15["ema_fast"] = ema(c15, ema_fast_15)
                df15["ema_slow"] = ema(c15, ema_slow_15)
                df15["rsi15"] = rsi_series(c15, rsi_len_15)
                df15["atr15"] = atr(df15[["open","high","low","close"]], atr_len_15)
                df15["atr_ma"] = df15["atr15"].rolling(atr_ma_win).mean()

                # donchian hh/ll
                df15["hh"] = df15["high"].rolling(don_w).max()
                df15["ll"] = df15["low"].rolling(don_w).min()

                i = len(df15) - 1
                if i < max(ema_slow_15, don_w, atr_len_15, atr_ma_win) + 5:
                    time.sleep(poll)
                    continue

                row = df15.iloc[i]
                prev = df15.iloc[i-1]

                atr_ok = True
                if np.isfinite(row["atr_ma"]) and row["atr_ma"] > 0:
                    atr_ok = float(row["atr15"]) >= float(row["atr_ma"]) * atr_min_mult

                trend_long_15 = float(row["ema_fast"]) > float(row["ema_slow"])
                trend_short_15= float(row["ema_fast"]) < float(row["ema_slow"])

                # 1h filter
                trend_long_1h = True
                trend_short_1h= True
                if use_1h:
                    df1h = fetch_ohlc(ex, symbol, "1h", limit=400)
                    c1h = df1h["close"]
                    df1h["ema_fast"] = ema(c1h, ema_fast_1h)
                    df1h["ema_slow"] = ema(c1h, ema_slow_1h)
                    r1h = df1h.iloc[-1]
                    trend_long_1h = float(r1h["ema_fast"]) > float(r1h["ema_slow"])
                    trend_short_1h= float(r1h["ema_fast"]) < float(r1h["ema_slow"])

                hh_prev = float(prev["hh"])
                ll_prev = float(prev["ll"])
                atr_v = float(row["atr15"])
                buffer = atr_v * buf_k

                want_long  = atr_ok and trend_long_15 and trend_long_1h and (float(row["rsi15"]) >= rsi_long)
                want_short = atr_ok and trend_short_15 and trend_short_1h and (float(row["rsi15"]) <= rsi_short)

                signal_side = "LONG" if want_long else ("SHORT" if want_short else None)

                if signal_side is None or not np.isfinite(hh_prev) or not np.isfinite(ll_prev) or atr_v <= 0:
                    ui_panel("NO SETUP (15m)", [
                        pill("atr_ok", atr_ok),
                        pill("rsi15", fmt(row["rsi15"],1)),
                        pill("trend15", "L" if trend_long_15 else ("S" if trend_short_15 else "flat")),
                        pill("trend1h", "L" if trend_long_1h else ("S" if trend_short_1h else "flat")),
                    ])
                    time.sleep(poll)
                    continue

                # compute trigger + SL/TP prices like backtest
                if signal_side == "LONG":
                    trigger = hh_prev + buffer
                    init_sl_px = trigger - atr_v * sl_k
                    tp_px = trigger + atr_v * tp_k
                else:
                    trigger = ll_prev - buffer
                    init_sl_px = trigger + atr_v * sl_k
                    tp_px = trigger - atr_v * tp_k

                armed_ts = now_utc_ts()
                expires_ts = armed_ts + armed_timeout_min * 60

                # store a "deadline_dt" to enforce first N 5m bars
                # we arm on close of 15m; entry window is next N 5m bars
                deadline_dt = (last15_close_dt + pd.Timedelta(minutes=5*entry_window_5m_bars)).isoformat()

                state["armed"] = {
                    "active": True,
                    "side": signal_side,
                    "armed_ts": armed_ts,
                    "expires_ts": expires_ts,
                    "deadline_dt": deadline_dt,
                    "trigger_level": trigger,
                    "atr15": atr_v,
                    "init_sl_px": init_sl_px,
                    "tp_px": tp_px,
                    "note": f"hh_prev={fmt(hh_prev,2)} ll_prev={fmt(ll_prev,2)} atr15={fmt(atr_v,2)} buf={fmt(buffer,2)}"
                }
                save_json(STATE_FILE, state)

                ui_panel("ARMED (15m breakout)", [
                    pill("Side", signal_side),
                    pill("Trigger", fmt(trigger,2)),
                    pill("SL", fmt(init_sl_px,2)),
                    pill("TP", fmt(tp_px,2)),
                    pill("ATR15", fmt(atr_v,2)),
                    pill("Expira(s)", expires_ts - now_utc_ts()),
                    pill("WinBars5m", entry_window_5m_bars),
                    f"Note: {state['armed']['note']}"
                ])
                log_event("ARMED", signal_side, 0, float(df15["close"].iloc[-1]), 0.0, note=state["armed"]["note"])

            # ==================================
            # 3) If ARMED -> check 5m breakout trigger
            # ==================================
            state = load_json(STATE_FILE) or state
            armed = state.get("armed", {})
            if armed.get("active"):
                if now_utc_ts() > int(armed["expires_ts"]):
                    ui("⏱️ DISARM: timeout")
                    state["armed"]["active"] = False
                    save_json(STATE_FILE, state)
                    time.sleep(poll)
                    continue

                # enforce entry window in 5m bars
                deadline_dt = pd.to_datetime(armed.get("deadline_dt"), utc=True) if armed.get("deadline_dt") else None
                now_dt = pd.Timestamp.utcnow().tz_localize("UTC")
                if deadline_dt and now_dt > deadline_dt:
                    ui("⏱️ DISARM: entry window passed (5m bars)")
                    state["armed"]["active"] = False
                    save_json(STATE_FILE, state)
                    time.sleep(poll)
                    continue

                df5 = fetch_ohlc(ex, symbol, "5m", limit=50)
                hi = float(df5["high"].iloc[-1])
                lo = float(df5["low"].iloc[-1])
                last_price = float(df5["close"].iloc[-1])

                side = armed["side"]
                trigger = float(armed["trigger_level"])
                hit = (hi >= trigger) if side == "LONG" else (lo <= trigger)

                if hit:
                    equity = get_usdt_equity(ex)
                    init_sl_px = float(armed["init_sl_px"])
                    atr_v = float(armed["atr15"])
                    tp_px = float(armed["tp_px"])

                    # position sizing based on initial stop distance
                    sl_pct_move = sl_move_pct(trigger, init_sl_px, side)
                    target_notional = compute_notional_usdt(equity, risk_pct, sl_pct_move)

                    ui_panel("ENTER (market)", [
                        pill("Side", side),
                        pill("Trigger", fmt(trigger,2)),
                        pill("LastPx", fmt(last_price,2)),
                        pill("SL", fmt(init_sl_px,2)),
                        pill("TP", fmt(tp_px,2)),
                        pill("Notional", fmt(target_notional,2)),
                        pill("Equity", fmt(equity,2)),
                    ])

                    order, qty, used_notional, px, min_qty, min_notional = market_open_with_minimums(
                        ex, symbol, side, target_notional, equity, lev
                    )

                    time.sleep(2)
                    pos_new = get_position(ex, symbol)
                    if not pos_new:
                        ui("⚠️ No pude leer la posición tras entrar.")
                        time.sleep(poll)
                        continue

                    entry_px = float(pos_new["entry"])
                    qty_pos  = abs(pos_new["contracts"])
                    mark     = float(pos_new["mark"])

                    # place SL/TP using stored prices (based on trigger/atr)
                    sl_id, tp_id = place_sl_tp_prices(ex, symbol, side, qty_pos, init_sl_px, tp_px)

                    state["orders"]["sl_id"] = sl_id
                    state["orders"]["tp_id"] = tp_id
                    state["armed"]["active"] = False
                    state["trade"] = {
                        "entry_time": pd.Timestamp.utcnow().tz_localize("UTC").isoformat(),
                        "entry_px": entry_px,
                        "init_sl_px": init_sl_px,
                        "tp_px": tp_px,
                        "atr15": atr_v,
                        "max_fav_R": -999,
                        "trail_active": False,
                        "last_sl_px": init_sl_px
                    }
                    save_json(STATE_FILE, state)

                    # log entry
                    pnl0 = 0.0
                    if fees_enabled:
                        pnl0 -= fee_equity_pct_from_slmove(risk_pct, fee_rt, sl_pct_move)
                    if slip_rt > 0:
                        pnl0 -= fee_equity_pct_from_slmove(risk_pct, slip_rt, sl_pct_move)

                    log_event("ENTER", side, entry_px, mark, pnl0, note=f"trigger={fmt(trigger,2)} sl_move={fmt(sl_pct_move,6)} qty={fmt(qty_pos,6)}")
                    ui_panel("ORDERS PLACED", [
                        pill("Entry", fmt(entry_px,2)),
                        pill("Qty", fmt(qty_pos,6)),
                        pill("SL", fmt(init_sl_px,2)),
                        pill("TP", fmt(tp_px,2)),
                        pill("FeeSlipAdj(pnl%)", fmt(pnl0,3)),
                    ])

            time.sleep(poll)

        except Exception as e:
            ui("🧯 ERROR: " + repr(e))
            msg = str(e)
            if ("Notional mínimo" in msg) or ("minimum amount" in msg) or ("min_qty" in msg):
                st = load_json(STATE_FILE) or {}
                if st.get("armed", {}).get("active"):
                    ui("🚫 DISARM por mínimo / notional insuficiente.")
                    st["armed"]["active"] = False
                    save_json(STATE_FILE, st)
            time.sleep(5)


if __name__ == "__main__":
    main()
