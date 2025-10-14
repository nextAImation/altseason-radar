# src/altseason/factors.py
from __future__ import annotations

import math
from datetime import datetime, UTC
from typing import Dict, Any, Tuple

import requests
import pandas as pd
import numpy as np

from altseason.config import (
    BINANCE_BASE,
    COINGECKO_BASE,
    THRESHOLDS,
    FACTOR_WEIGHTS,
)

# -------------------- تکنیکال --------------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _slope(series: pd.Series, lookback: int) -> float:
    if len(series) < lookback:
        return 0.0
    y = series[-lookback:].values.astype(float)
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x**2).sum()
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)

# -------------------- fetch --------------------
def _binance_klines(symbol: str, interval: str = "1d", limit: int = 400) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    cols = [
        "open_time","open","high","low","close","volume","close_time","qav",
        "trades","taker_base","taker_quote","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _coingecko_global() -> dict:
    url = f"{COINGECKO_BASE}/global"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json().get("data", {})

# -------------------- نمره‌دهی --------------------
def _score_trend_with_ema(close: pd.Series, short: int, long: int, weight: int) -> Tuple[float, bool, dict]:
    ema_s = _ema(close, short)
    ema_l = _ema(close, long)
    above = close.iloc[-1] > ema_s.iloc[-1] > ema_l.iloc[-1]
    if above:
        return float(weight), True, {"ema_s": float(ema_s.iloc[-1]), "ema_l": float(ema_l.iloc[-1])}
    elif close.iloc[-1] > ema_s.iloc[-1]:
        return float(weight) * 0.6, True, {"ema_s": float(ema_s.iloc[-1]), "ema_l": float(ema_l.iloc[-1])}
    else:
        return 0.0, False, {"ema_s": float(ema_s.iloc[-1]), "ema_l": float(ema_l.iloc[-1])}

# -------------------- محاسبه فاکتورها --------------------
class FactorCalculator:
    """
    خروجی هر فاکتور:
    {
      "score": float in [0..weight],
      "ok": bool,
      "explain": str
    }
    """
    def __init__(self):
        self.weights = FACTOR_WEIGHTS
        self.th = THRESHOLDS

    def _btc_dominance(self) -> Dict[str, Any]:
        # پروکسی: شیب روند بازده BTC (هرچه منفی‌تر → به نفع آلت‌ها)
        df = _binance_klines("BTCUSDT", "1d", 200)
        close = df["close"].dropna()
        slope = _slope(close.pct_change().fillna(0).cumsum(), lookback=50)
        dom_slope_min = float(self.th.get("DOM_SLOPE_MIN", -0.1))
        weight = int(self.weights.get("btc_dominance", 0))

        if slope >= 0:
            score, ok = 0.0, False
        elif slope <= dom_slope_min:
            score, ok = float(weight), True
        else:
            ratio = (0 - slope) / (0 - dom_slope_min)
            score, ok = float(weight) * ratio, True

        cg = {}
        try:
            cg = _coingecko_global()
        except Exception:
            pass
        dom_now = cg.get("market_cap_percentage", {}).get("btc")
        explain = f"slope:{slope:.4f}" + (f" | cg_btc_dominance:{dom_now:.2f}%" if dom_now else "")
        return {"score": score, "ok": ok, "explain": explain}

    def _eth_btc(self) -> Dict[str, Any]:
        df = _binance_klines("ETHBTC", "1d", 400)
        close = df["close"].dropna()
        rsi = float(_rsi(close, 14).iloc[-1])
        rsimin = float(self.th.get("ETHBTC_RSI_MIN", 55))
        weight = int(self.weights.get("eth_btc", 0))

        base, ok_trend, _ = _score_trend_with_ema(
            close,
            short=int(self.th.get("ETHBTC_EMA_SHORT", 50)),
            long=int(self.th.get("ETHBTC_EMA_LONG", 200)),
            weight=weight,
        )
        ok_rsi = rsi >= rsimin
        score = base * (1.0 if ok_rsi else 0.6)
        ok = ok_trend and ok_rsi
        explain = f"EMA({int(self.th.get('ETHBTC_EMA_SHORT', 50))}/{int(self.th.get('ETHBTC_EMA_LONG', 200))}), RSI:{rsi:.1f}"
        return {"score": score, "ok": ok, "explain": explain}

    def _total2(self) -> Dict[str, Any]:
        # پروکسی برای آلت‌ها: ETHUSDT
        df = _binance_klines("ETHUSDT", "1d", 400)
        close = df["close"].dropna()
        rsi = float(_rsi(close, 14).iloc[-1])
        rsimin = float(self.th.get("TOTAL_RSI_MIN", 55))
        weight = int(self.weights.get("total2", 0))

        base, ok_trend, _ = _score_trend_with_ema(close, short=int(self.th.get("TOTAL_EMA", 50)), long=200, weight=weight)
        ok_rsi = rsi >= rsimin
        score = base * (1.0 if ok_rsi else 0.7)
        ok = ok_trend and ok_rsi
        explain = f"EMA(50/200) proxy, RSI:{rsi:.1f}"
        return {"score": score, "ok": ok, "explain": explain}

    def _total3(self) -> Dict[str, Any]:
        # پروکسی آلت‌های کوچک‌تر: BNBUSDT (ساده‌سازی عملی)
        df = _binance_klines("BNBUSDT", "1d", 400)
        close = df["close"].dropna()
        rsi = float(_rsi(close, 14).iloc[-1])
        weight = int(self.weights.get("total3", 0))

        base, ok_trend, _ = _score_trend_with_ema(close, short=50, long=200, weight=weight)
        score = base * (1.0 if rsi >= 58 else 0.5)
        ok = ok_trend and (rsi >= 58)
        explain = f"EMA(50/200) proxy, RSI:{rsi:.1f}"
        return {"score": score, "ok": ok, "explain": explain}

    def _btc_regime(self) -> Dict[str, Any]:
        df = _binance_klines("BTCUSDT", "1d", 200)
        close = df["close"].dropna()
        sl = _slope(close.pct_change().fillna(0).rolling(2).mean(), lookback=50)
        weight = int(self.weights.get("btc_regime", 0))
        if sl <= 0:
            score, ok = float(weight), True
        elif sl >= 0.002:
            score, ok = 0.0, False
        else:
            ratio = (0.002 - sl) / 0.002
            score, ok = float(weight) * ratio, True
        explain = f"slope(returns,50):{sl:.5f}"
        return {"score": score, "ok": ok, "explain": explain}

    def _eth_trend(self) -> Dict[str, Any]:
        df = _binance_klines("ETHUSDT", "1d", 400)
        close = df["close"].dropna()
        weight = int(self.weights.get("eth_trend", 0))
        base, ok_trend, _ = _score_trend_with_ema(close, short=50, long=200, weight=weight)
        rsi = float(_rsi(close, 14).iloc[-1])
        boost = 1.1 if rsi >= 60 else 1.0
        score = min(float(weight), base * boost)
        ok = ok_trend and (rsi >= 52)
        explain = f"EMA(50/200), RSI:{rsi:.1f}"
        return {"score": score, "ok": ok, "explain": explain}

    def compute_factors(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, fn in [
            ("btc_dominance", self._btc_dominance),
            ("eth_btc",       self._eth_btc),
            ("total2",        self._total2),
            ("total3",        self._total3),
            ("btc_regime",    self._btc_regime),
            ("eth_trend",     self._eth_trend),
        ]:
            try:
                out[name] = fn()
            except Exception as e:
                out[name] = {"score": 0.0, "ok": False, "explain": f"error: {type(e).__name__}"}
        return out
