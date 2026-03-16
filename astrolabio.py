# pitonisa_grid.py
# Grid search for AURUM/Pitonisa strategy using Binance Futures public klines.
# Reads a base config.json and tests parameter grids.
#
# Notes:
# - Intrabar ambiguity handled conservatively: SL first if both touched in same candle.
# - Uses no overlapping positions (like your current backtest).
#
# Output:
# - grid_results.csv (sorted by total_pnl_equity_pct desc)
# - prints top 15 results

import json
import time
import itertools
import requests
import pandas as pd
import numpy as np

BASE = "https://fapi.binance.com"

# ---------------------------
# Data fetch / resample
# ---------------------------
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
        time.sleep(0.2)

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

def resample_15m_from_5m(df5: pd.DataFrame) -> pd.DataFrame:
    d = df5.copy().set_index("open_time")
    o = d["open"].resample("15min").first()
    h = d["high"].resample("15min").max()
    l = d["low"].resample("15min").min()
    c = d["close"].resample("15min").last()
    v = d["volume"].resample("15min").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna().reset_index()
    return out

# ---------------------------
# Indicators
# ---------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50.0)

def zscore(close: pd.Series, window: int) -> pd.Series:
    m = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    z = (close - m) / sd.replace(0, np.nan)
    return z.fillna(0.0)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def compute_z5(df5: pd.DataFrame, ema_len: int, z_w5: int) -> pd.Series:
    c = df5["close"]
    e = ema(c, ema_len)
    dev = c - e
    sd = dev.rolling(z_w5).std(ddof=0)
    z = dev / sd.replace(0, np.nan)
    return z.fillna(0.0)

# ---------------------------
# Rules
# ---------------------------
def price_for_target_pnl(entry: float, target_pnl: float, side: str, lev: float) -> float:
    delta_pct = (target_pnl / 100.0) / lev
    if side == "LONG":
        return entry * (1.0 + delta_pct)
    else:
        return entry * (1.0 - delta_pct)

def pnl_at(entry: float, px: float, side: str, lev: float) -> float:
    if side == "LONG":
        return ((px - entry) / entry) * lev * 100.0
    return ((entry - px) / entry) * lev * 100.0

def should_trigger_entry_A(armed_side: str, z5: float, z_th: float, buf: float) -> bool:
    if armed_side == "LONG":
        return z5 >= (-z_th + buf)
    return z5 <= (z_th - buf)

def build_signals(df15: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df15.copy()
    df["ema200"] = ema(df["close"], cfg["ema_len"])
    df["rsi15"] = rsi(df["close"], cfg["rsi_len"])
    df["z15"] = zscore(df["close"], cfg["z_window_15m"])

    # optional ATR filter
    df["atr15"] = atr(df, cfg["atr_len"])
    df["atr_ma"] = df["atr15"].rolling(cfg["atr_ma_window_15m"]).mean()
    atr_ok = pd.Series(True, index=df.index)
    if cfg["atr_filter_enabled"]:
        atr_ok = df["atr15"] >= (df["atr_ma"] * cfg["atr_min_mult"])

    # trend filter optional (default ON: mean reversion WITH trend filter)
    if cfg.get("trend_filter_enabled", True):
        long_trend_ok = df["close"] > df["ema200"]
        short_trend_ok = df["close"] < df["ema200"]
    else:
        long_trend_ok = pd.Series(True, index=df.index)
        short_trend_ok = pd.Series(True, index=df.index)

    long_ok  = (df["z15"] <= -cfg["z_threshold"]) & (df["rsi15"] <= cfg["rsi_low"])  & long_trend_ok & atr_ok
    short_ok = (df["z15"] >=  cfg["z_threshold"]) & (df["rsi15"] >= cfg["rsi_high"]) & short_trend_ok & atr_ok

    df["signal_raw"] = np.where(long_ok, "LONG", np.where(short_ok, "SHORT", ""))

    if cfg.get("entry_on_cross_only", False):
        prev = df["signal_raw"].shift(1).fillna("")
        df["signal"] = np.where((df["signal_raw"] != "") & (prev == ""), df["signal_raw"], "")
    else:
        df["signal"] = df["signal_raw"]

    return df

# ---------------------------
# Sim
# ---------------------------
def simulate_trade_5m(df5: pd.DataFrame, entry_i: int, side: str, cfg: dict) -> dict:
    lev = float(cfg["leverage"])
    entry = float(df5.at[entry_i, "close"])

    sl_pnl = float(cfg["initial_sl_pnl_pct"])
    tp_pnl = float(cfg["final_tp_pnl_pct"])

    sl_price = price_for_target_pnl(entry, sl_pnl, side, lev)
    tp_price = price_for_target_pnl(entry, tp_pnl, side, lev)

    fee_pnl = (cfg["fee_rate_roundtrip"] * 100.0) if cfg.get("fees_enabled", True) else 0.0

    # forced exits
    max_hold_bars = int((cfg["max_hold_minutes"]) / 5) if cfg.get("max_hold_minutes") else None
    force_revert = bool(cfg.get("force_exit_if_z15_reverted", False))
    z_revert_level = float(cfg.get("z_revert_level", 0.0))

    entry_t = df5.at[entry_i, "open_time"]

    # We need z15 on 15m timeline; approximate by mapping 5m bar to its 15m bucket and looking up z15 series passed in df5
    # We'll store z15_approx in df5 outside, if available.
    has_z15 = "z15_approx" in df5.columns

    for j in range(entry_i + 1, len(df5)):
        hi = float(df5.at[j, "high"])
        lo = float(df5.at[j, "low"])

        # hits
        if side == "LONG":
            sl_hit = lo <= sl_price
            tp_hit = hi >= tp_price
        else:
            sl_hit = hi >= sl_price
            tp_hit = lo <= tp_price

        # conservative: SL first
        if sl_hit:
            pnl = pnl_at(entry, sl_price, side, lev) - fee_pnl
            return {"exit_i": j, "exit_reason": "SL", "pnl_pos_pct": pnl}

        if tp_hit:
            pnl = pnl_at(entry, tp_price, side, lev) - fee_pnl
            return {"exit_i": j, "exit_reason": "TP", "pnl_pos_pct": pnl}

        # forced exit checks
        if max_hold_bars is not None:
            if (j - entry_i) >= max_hold_bars:
                px = float(df5.at[j, "close"])
                pnl = pnl_at(entry, px, side, lev) - fee_pnl
                return {"exit_i": j, "exit_reason": "TIME", "pnl_pos_pct": pnl}

        if force_revert and has_z15:
            z15_now = float(df5.at[j, "z15_approx"])
            if abs(z15_now) <= z_revert_level:
                px = float(df5.at[j, "close"])
                pnl = pnl_at(entry, px, side, lev) - fee_pnl
                return {"exit_i": j, "exit_reason": "REVERT", "pnl_pos_pct": pnl}

    # end of data
    px = float(df5.iloc[-1]["close"])
    pnl = pnl_at(entry, px, side, lev) - fee_pnl
    return {"exit_i": len(df5) - 1, "exit_reason": "EOD", "pnl_pos_pct": pnl}

def backtest(cfg: dict, df5: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    df15s = build_signals(df15, cfg)

    df5 = df5.copy().reset_index(drop=True)
    df5["t5"] = df5["open_time"].dt.floor("5min")

    df5["z5"] = compute_z5(df5, cfg["ema_len"], cfg["z_window_5m"]).values

    # map 15m z15 to 5m bars (for forced z-revert exits)
    z15_series = zscore(df15["close"], cfg["z_window_15m"]).rename("z15")
    df15_map = pd.DataFrame({"t15": df15["open_time"].dt.floor("15min"), "z15": z15_series.values})
    df5["t15"] = df5["open_time"].dt.floor("15min")
    z15_lookup = dict(zip(df15_map["t15"].tolist(), df15_map["z15"].tolist()))
    df5["z15_approx"] = df5["t15"].map(lambda t: z15_lookup.get(t, 0.0))

    # timestamp->index for 5m
    t_to_i = {t: i for i, t in enumerate(df5["t5"].tolist())}

    trades = []
    in_position_until_i = -1

    timeout_bars = int((cfg["armed_timeout_minutes"] * 60) / (5 * 60))

    for _, row in df15s.iterrows():
        sig = row["signal"]
        if sig == "":
            continue

        t15 = pd.Timestamp(row["open_time"]).floor("15min")
        armed_side = sig

        # When to start searching triggers: after the 15m candle closes
        start_t = (t15 + pd.Timedelta(minutes=15)).floor("5min")
        start_i = t_to_i.get(start_t, None)
        if start_i is None:
            continue
        if start_i <= in_position_until_i:
            continue

        end_i = min(start_i + timeout_bars, len(df5) - 2)

        entry_i = None
        for i in range(start_i, end_i + 1):
            z5 = float(df5.at[i, "z5"])
            if should_trigger_entry_A(armed_side, z5, cfg["z_threshold"], cfg["trigger_buffer"]):
                entry_i = i
                break

        if entry_i is None:
            continue

        sim = simulate_trade_5m(df5, entry_i, armed_side, cfg)
        in_position_until_i = sim["exit_i"]

        trades.append({
            "entry_time": df5.at[entry_i, "open_time"],
            "exit_time": df5.at[sim["exit_i"], "open_time"],
            "side": armed_side,
            "exit_reason": sim["exit_reason"],
            "pnl_pos_pct": sim["pnl_pos_pct"],
        })

    return pd.DataFrame(trades)

def summarize(trades: pd.DataFrame, cfg: dict) -> dict:
    if trades.empty:
        return {"trades": 0}

    # scale pos-pnl% -> equity% using your risk model
    scale = cfg["risk_per_trade_pct"] / abs(cfg["initial_sl_pnl_pct"])
    pnl_equity = trades["pnl_pos_pct"] * scale

    total_pnl = float(pnl_equity.sum())
    avg_trade = float(pnl_equity.mean())
    wins = int((pnl_equity > 0).sum())
    n = int(len(trades))
    winrate = 100.0 * wins / max(n, 1)

    curve = pnl_equity.cumsum()
    peak = curve.cummax()
    dd = curve - peak
    max_dd = float(dd.min())

    days = (trades["exit_time"].max() - trades["entry_time"].min()).total_seconds() / (24 * 3600)
    tpd = n / max(days, 1e-9)

    return {
        "trades": n,
        "winrate": winrate,
        "tpd": tpd,
        "total_pnl_equity_pct": total_pnl,
        "avg_pnl_equity_pct": avg_trade,
        "max_dd_equity_pct": max_dd,
    }

# ---------------------------
# Grid runner
# ---------------------------
def load_cfg(path="config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    base_cfg = load_cfg("config.json")

    days = int(base_cfg.get("days", 360))
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000

    print(f"Downloading BTCUSDT 5m Futures for last {days} days…")
    df5 = fetch_klines("BTCUSDT", "5m", start_ms, now_ms)
    df15 = resample_15m_from_5m(df5)
    print(f"Data ready: 5m={len(df5)} 15m={len(df15)}")

    # --- Parameter grids (edit as needed) ---
    grid = {
        "z_threshold": [0.18, 0.22, 0.25, 0.28, 0.32],
        "rsi_low": [44, 46, 48],
        "rsi_high": [52, 54, 56],
        "initial_sl_pnl_pct": [-6, -7, -8],
        "final_tp_pnl_pct": [6, 8, 9, 10, 12],
        "trigger_buffer": [0.00, 0.01, 0.02],
        "armed_timeout_minutes": [8, 10, 12, 15],
        "max_hold_minutes": [60, 90, 120, 180],
        "force_exit_if_z15_reverted": [True],
        "z_revert_level": [0.05, 0.10, 0.15],
        "atr_filter_enabled": [False, True],
        "atr_min_mult": [0.8, 0.9, 1.0],
        "trend_filter_enabled": [False, True],
        "entry_on_cross_only": [False, True],
    }

    # Basic sanity: ensure rsi_low < rsi_high always
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Total combos: {len(combos)}")

    results = []
    tested = 0

    for vals in combos:
        cfg = dict(base_cfg)
        for k, v in zip(keys, vals):
            cfg[k] = v

        if cfg["rsi_low"] >= cfg["rsi_high"]:
            continue

        trades = backtest(cfg, df5, df15)
        summ = summarize(trades, cfg)
        tested += 1

        if summ.get("trades", 0) == 0:
            continue

        results.append({
            **{k: cfg[k] for k in keys},
            **summ
        })

        if tested % 25 == 0:
            print(f"tested={tested} results={len(results)}")

    out = pd.DataFrame(results)
    if out.empty:
        print("No results.")
        return

    # Filter to avoid degenerate low-trade configs
    out = out[out["trades"] >= 80].copy()

    out = out.sort_values(
        by=["total_pnl_equity_pct", "max_dd_equity_pct", "tpd"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    out.to_csv("grid_results.csv", index=False, encoding="utf-8")
    print("\nSaved: grid_results.csv")
    print("\nTop 15:")
    cols = [
        "total_pnl_equity_pct","max_dd_equity_pct","winrate","tpd","trades",
        "z_threshold","rsi_low","rsi_high","initial_sl_pnl_pct","final_tp_pnl_pct",
        "max_hold_minutes","z_revert_level","atr_filter_enabled","trend_filter_enabled","entry_on_cross_only"
    ]
    print(out[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
