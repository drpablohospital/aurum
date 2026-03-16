import time, json
import requests
import pandas as pd
import numpy as np

BASE = "https://fapi.binance.com"

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1500) -> pd.DataFrame:
    out = []
    cur = start_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": limit}
        r = requests.get(f"{BASE}/fapi/v1/klines", params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        last_open = data[-1][0]
        cur = last_open + 1
        if len(data) < limit:
            break
        time.sleep(0.08)

    df = pd.DataFrame(out, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","ntrades","tbbav","tbqav","ignore"
    ])
    df = df[["open_time","open","high","low","close","volume","close_time"]].copy()
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50.0)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.copy().set_index("open_time")
    o = d["open"].resample(rule).first()
    h = d["high"].resample(rule).max()
    l = d["low"].resample(rule).min()
    c = d["close"].resample(rule).last()
    v = d["volume"].resample(rule).sum()
    out = pd.DataFrame({"open_time": o.index, "open": o.values, "high": h.values, "low": l.values, "close": c.values, "volume": v.values})
    out = out.dropna().reset_index(drop=True)
    return out

def fee_equity_pct_from_slmove(risk_pct: float, fee_roundtrip: float, sl_move_pct: float) -> float:
    if sl_move_pct <= 1e-12:
        return 0.0
    return risk_pct * (fee_roundtrip / sl_move_pct)

def main():
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sym = cfg["market_symbol"]
    days = int(cfg.get("days", 3000))

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000

    print(f"Downloading BTCUSDT 5m Futures for last {days} days…")
    df5 = fetch_klines(sym, "5m", start_ms, now_ms)
    print("Bars:", len(df5))
    if df5.empty or len(df5) < 5000:
        print("Not enough data.")
        return

    df15 = resample_ohlcv(df5, "15min")
    df1h = resample_ohlcv(df5, "60min")

    # ---- 15m indicators ----
    df15["ema_fast"] = ema(df15["close"], int(cfg["ema_fast_15m"]))
    df15["ema_slow"] = ema(df15["close"], int(cfg["ema_slow_15m"]))
    df15["rsi15"] = rsi(df15["close"], int(cfg["rsi_len_15m"]))
    df15["atr15"] = atr(df15, int(cfg["atr_len_15m"]))
    df15["atr_ma"] = df15["atr15"].rolling(int(cfg["atr_ma_window_15m"])).mean()

    w = int(cfg["donchian_window_15m"])
    df15["hh"] = df15["high"].rolling(w).max()
    df15["ll"] = df15["low"].rolling(w).min()

    # ---- 1h trend filter ----
    use_1h = bool(cfg.get("use_1h_trend_filter", False))
    if use_1h:
        df1h["ema_fast"] = ema(df1h["close"], int(cfg["ema_fast_1h"]))
        df1h["ema_slow"] = ema(df1h["close"], int(cfg["ema_slow_1h"]))

    # time maps
    df5 = df5.copy().reset_index(drop=True)
    df5["t5"] = df5["open_time"].dt.floor("5min")
    df15["t15"] = df15["open_time"].dt.floor("15min")
    if use_1h:
        df1h["t1h"] = df1h["open_time"].dt.floor("60min")

    t5_to_i = {t: i for i, t in enumerate(df5["t5"].tolist())}
    if use_1h:
        t1h_to_row = {t: i for i, t in enumerate(df1h["t1h"].tolist())}

    # params
    rsi_long = float(cfg["rsi_long"])
    rsi_short = float(cfg["rsi_short"])
    atr_min_mult = float(cfg["atr_min_mult"])
    buf_k = float(cfg["breakout_buffer_atr"])
    sl_k = float(cfg["sl_atr_mult"])
    tp_k = float(cfg["tp_atr_mult"])

    be_R = float(cfg.get("breakeven_R", 1.0))
    trail_start_R = float(cfg.get("trail_start_R", 1.5))
    trail_atr_k = float(cfg.get("trail_atr_mult", 1.0))

    max_hold_min = int(cfg["max_hold_minutes"])
    cooldown_min = int(cfg["cooldown_minutes"])
    risk_pct = float(cfg["risk_per_trade_pct"])
    fees_enabled = bool(cfg.get("fees_enabled", True))
    fee_rt = float(cfg.get("fee_rate_roundtrip", 0.001))
    slip_rt = float(cfg.get("slippage_roundtrip_pct", 0.0))
    label_exits = bool(cfg.get("label_exits", True))

    trades = []
    cooldown_until_ts = None

    warmup = max(int(cfg["ema_slow_15m"]), w, int(cfg["atr_len_15m"]), int(cfg["atr_ma_window_15m"])) + 10
    if use_1h:
        warmup = max(warmup, int(cfg["ema_slow_1h"]) + 10)

    def sl_move_pct(entry_px, init_sl_px, side):
        if side == "LONG":
            return max((entry_px - init_sl_px) / entry_px, 1e-9)
        return max((init_sl_px - entry_px) / entry_px, 1e-9)

    def classify_stop_exit(side: str, entry_px: float, stop_px: float, eps: float = 1e-6) -> str:
        # stop executed; classify whether it was SL / BE / TRAIL
        if not label_exits:
            return "SL"
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

    for idx15 in range(warmup, len(df15)):
        row15 = df15.iloc[idx15]
        t15 = row15["t15"]
        exec_t = (t15 + pd.Timedelta(minutes=15)).floor("5min")
        start_i = t5_to_i.get(exec_t, None)
        if start_i is None or start_i >= len(df5)-2:
            continue

        if cooldown_until_ts is not None and exec_t <= cooldown_until_ts:
            continue

        if not np.isfinite(row15["atr_ma"]) or row15["atr_ma"] <= 0:
            continue
        atr_ok = row15["atr15"] >= row15["atr_ma"] * atr_min_mult

        trend_long_15m = row15["ema_fast"] > row15["ema_slow"]
        trend_short_15m = row15["ema_fast"] < row15["ema_slow"]

        if use_1h:
            t1h = exec_t.floor("60min")
            i1h = t1h_to_row.get(t1h, None)
            if i1h is None or i1h < int(cfg["ema_slow_1h"]):
                continue
            row1h = df1h.iloc[i1h]
            trend_long_1h = row1h["ema_fast"] > row1h["ema_slow"]
            trend_short_1h = row1h["ema_fast"] < row1h["ema_slow"]
        else:
            trend_long_1h = True
            trend_short_1h = True

        hh_prev = df15["hh"].iloc[idx15-1]
        ll_prev = df15["ll"].iloc[idx15-1]
        if not np.isfinite(hh_prev) or not np.isfinite(ll_prev) or not np.isfinite(row15["atr15"]):
            continue

        atr_v = float(row15["atr15"])
        buffer = atr_v * buf_k

        want_long = atr_ok and trend_long_15m and trend_long_1h and (row15["rsi15"] >= rsi_long)
        want_short = atr_ok and trend_short_15m and trend_short_1h and (row15["rsi15"] <= rsi_short)
        if not (want_long or want_short):
            continue

        entry_triggered = False
        side = None
        trigger_level = None
        entry_i = None

        end_i = min(start_i + 6, len(df5)-1)
        for i in range(start_i, end_i+1):
            hi = float(df5.at[i, "high"])
            lo = float(df5.at[i, "low"])

            if want_long:
                lvl = float(hh_prev) + buffer
                if hi >= lvl:
                    entry_triggered = True
                    side = "LONG"
                    trigger_level = lvl
                    entry_i = i
                    break

            if want_short:
                lvl = float(ll_prev) - buffer
                if lo <= lvl:
                    entry_triggered = True
                    side = "SHORT"
                    trigger_level = lvl
                    entry_i = i
                    break

        if not entry_triggered:
            continue

        entry_px = float(trigger_level)

        # initial SL/TP
        if side == "LONG":
            init_sl_px = entry_px - atr_v * sl_k
            sl_px = init_sl_px
            tp_px = entry_px + atr_v * tp_k
        else:
            init_sl_px = entry_px + atr_v * sl_k
            sl_px = init_sl_px
            tp_px = entry_px - atr_v * tp_k

        entry_time = df5.at[entry_i, "open_time"]

        exit_reason = None
        exit_time = None
        exit_px = None

        max_fav_R = -1e9
        max_hold_bars = int(max_hold_min / 5)

        for j in range(entry_i+1, min(entry_i+1+max_hold_bars, len(df5))):
            hi = float(df5.at[j, "high"])
            lo = float(df5.at[j, "low"])

            # current fav R (optimistic intrabar)
            if side == "LONG":
                cur_fav = (hi - entry_px) / (entry_px - init_sl_px + 1e-12)
            else:
                cur_fav = (entry_px - lo) / (init_sl_px - entry_px + 1e-12)
            max_fav_R = max(max_fav_R, cur_fav)

            # BE
            if max_fav_R >= be_R:
                if side == "LONG":
                    sl_px = max(sl_px, entry_px)
                else:
                    sl_px = min(sl_px, entry_px)

            # trailing
            if max_fav_R >= trail_start_R:
                if side == "LONG":
                    sl_px = max(sl_px, hi - atr_v * trail_atr_k)
                else:
                    sl_px = min(sl_px, lo + atr_v * trail_atr_k)

            # conservative: stop first
            if side == "LONG":
                if lo <= sl_px:
                    exit_reason = classify_stop_exit(side, entry_px, sl_px)
                    exit_px = sl_px
                elif hi >= tp_px:
                    exit_reason, exit_px = "TP", tp_px
            else:
                if hi >= sl_px:
                    exit_reason = classify_stop_exit(side, entry_px, sl_px)
                    exit_px = sl_px
                elif lo <= tp_px:
                    exit_reason, exit_px = "TP", tp_px

            if exit_reason is not None:
                exit_time = df5.at[j, "open_time"]
                break

        if exit_reason is None:
            j_end = min(entry_i+max_hold_bars, len(df5)-1)
            exit_time = df5.at[j_end, "open_time"]
            exit_px = float(df5.at[j_end, "close"])
            exit_reason = "TIME"

        # R based on initial risk
        init_risk = abs(entry_px - init_sl_px) / entry_px
        if init_risk <= 1e-12:
            R = 0.0
        else:
            if side == "LONG":
                move = (exit_px - entry_px) / entry_px
            else:
                move = (entry_px - exit_px) / entry_px
            R = move / init_risk

        pnl_equity_pct = R * risk_pct

        # fees + slippage (both approximated via initial stop distance)
        if fees_enabled:
            pnl_equity_pct -= fee_equity_pct_from_slmove(risk_pct, fee_rt, sl_move_pct(entry_px, init_sl_px, side))
        if slip_rt > 0:
            pnl_equity_pct -= fee_equity_pct_from_slmove(risk_pct, slip_rt, sl_move_pct(entry_px, init_sl_px, side))

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": side,
            "exit_reason": exit_reason,
            "R": R,
            "pnl_equity_pct": pnl_equity_pct
        })

        cooldown_until_ts = exit_time + pd.Timedelta(minutes=cooldown_min)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("\nNo trades.")
        return

    total = len(trades_df)
    winrate = (trades_df["pnl_equity_pct"] > 0).mean() * 100.0
    total_pnl = trades_df["pnl_equity_pct"].sum()
    avg_pnl = trades_df["pnl_equity_pct"].mean()

    curve = trades_df["pnl_equity_pct"].cumsum()
    dd = curve - curve.cummax()
    max_dd = dd.min()

    span_days = (trades_df["exit_time"].max() - trades_df["entry_time"].min()).total_seconds() / (24*3000)
    tpd = total / max(span_days, 1e-9)
    proj_month = 30.0 * tpd * avg_pnl

    print("\n=== BACKTEST SUMMARY ===")
    print(f"Trades: {total} | Trades/day: {tpd:.2f} | Winrate: {winrate:.1f}%")
    print(f"Total PnL (equity%): {total_pnl:.2f}% | Avg/trade: {avg_pnl:.3f}%")
    print(f"Max DD approx (equity%): {max_dd:.2f}%")
    print("\nExit reasons:")
    print(trades_df["exit_reason"].value_counts().to_string())
    print(f"\nProjected monthly (linear): {proj_month:.2f}% equity/month")

    trades_df.to_csv("pitonisa_trades.csv", index=False, encoding="utf-8")
    print("\nSaved: pitonisa_trades.csv")

if __name__ == "__main__":
    main()
