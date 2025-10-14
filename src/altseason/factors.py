# src/altseason/factors.py
from __future__ import annotations

import math
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
import numpy as np

from altseason.config import (
    BINANCE_BASE,
    COINGECKO_BASE,
    THRESHOLDS,
    FACTOR_WEIGHTS,
)

# ---------- HTTP session with headers ----------
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "altseason-radar/1.0 (+github-actions)",
        "Accept": "application/json",
        "Connection": "keep-alive",
    })
    return s

_SES = _session()

# ---------- Tech utils ----------
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

# ---------- Robust fetch (with host failover + retry) ----------
_BINANCE_HOSTS = [
    # primary from config:
    BINANCE_BASE.rstrip("/"),
    # fallbacks:
    "https://api1.binance.com/api/v3",
    "https://api2.binance.com/api/v3",
    "https://api3.binance.com/api/v3",
]

class FetchError(RuntimeError):
    pass

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.8, min=0.8, max=6),
    retry=retry_if_exception_type((requests.RequestException, FetchError)),
)
def _binance_klines(symbol: str, interval: str = "1d", limit: int = 400) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for base in _BINANCE_HOSTS:
        try:
            url = f"{base}/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            r = _SES.get(url, params=params, timeout=20)
            if r.status_code >= 500:
                # server error → retry
                raise FetchError(f"binance 5xx {r.status_code}")
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
            if df["close"].dropna().empty:
                raise FetchError("empty klines")
            return df
        except Exception as e:
            last_exc = e
            continue
    raise FetchError(f"binance failover exhausted: {last_exc}")

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.8, min=0.8, max=6),
    retry=retry_if_exception_type((requests.RequestException, FetchError)),
)
def _coingecko_global() -> dict:
    url = f"{COINGECKO_BASE}/global"
    r = _SES.get(url, timeout=20)
    if r.status_code >= 500:
        raise FetchError(f"coingecko 5xx {r.status_code}")
    r.raise_for_status()
    return r.json().get("data", {})

# Simple fallback when global fails (less rate-limited)
def _coingecko_simple_price(ids: List[str], vs: str = "usd") -> dict:
    try:
        url = f"{COINGECKO_BASE}/simple/price"
        r = _SES.get(url, params={"ids": ",".join(ids), "vs_currencies": vs}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

# ---------- Scoring helpers ----------
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

# ---------- Last-state fallback ----------
def _load_prev_state() -> dict:
    root = Path(__file__).resolve().parents[2]
    state_path = root / "reports" / "state.json"
    try:
        if state_path.exists():
            return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _fallback_factor(name: str, weight: int, reason: str) -> Dict[str, Any]:
    """
    اگر نت fetch شد: اول از state.json قبلی اگر موجود بود،
    وگرنه یک مقدار میانه (۰.5 * weight) می‌دهیم تا کل مدل صفر نشود.
    """
    prev = _load_prev_state()
    v = (prev.get("factors") or {}).get(name)
    if isinstance(v, dict) and "score" in v and "ok" in v:
        sc = float(v.get("score", 0.0))
        ok = bool(v.get("ok", False))
        ex = (v.get("explain") or "") + " | fallback: prev_state"
        return {"score": max(0.0, min(sc, float(weight))), "ok": ok, "explain": ex}
    # mid fallback
    return {"score": 0.5 * float(weight), "ok": False, "explain": f"fallback_mid ({reason})"}

# ---------- Factors ----------
class FactorCalculator:
    """
    Output per factor:
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
        weight = int(self.weights.get("btc_dominance", 0))
        try:
            df = _binance_klines("BTCUSDT", "1d", 220)
            close = df["close"].dropna()
            slope = _slope(close.pct_change().fillna(0).cumsum(), lookback=50)
            dom_slope_min = float(self.th.get("DOM_SLOPE_MIN", -0.1))

            if slope >= 0:
                score, ok = 0.0, False
            elif slope <= dom_slope_min:
                score, ok = float(weight), True
            else:
                ratio = (0 - slope) / (0 - dom_slope_min)
                score, ok = float(weight) * ratio, True

            dom_now = None
            try:
                cg = _coingecko_global()
                dom_now = cg.get("market_cap_percentage", {}).get("btc")
            except Exception:
                pass
            explain = f"slope:{slope:.4f}" + (f" | cg_btc_dominance:{dom_now:.2f}%" if dom_now else "")
            return {"score": score, "ok": ok, "explain": explain}
        except Exception as e:
            return _fallback_factor("btc_dominance", weight, f"fetch_error:{type(e).__name__}")

    def _eth_btc(self) -> Dict[str, Any]:
        weight = int(self.weights.get("eth_btc", 0))
        try:
            df = _binance_klines("ETHBTC", "1d", 400)
            close = df["close"].dropna()
            rsi = float(_rsi(close, 14).iloc[-1])
            rsimin = float(self.th.get("ETHBTC_RSI_MIN", 55))

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
        except Exception as e:
            return _fallback_factor("eth_btc", weight, f"fetch_error:{type(e).__name__}")

    def _total2(self) -> Dict[str, Any]:
        weight = int(self.weights.get("total2", 0))
        try:
            # proxy: ETHUSDT
            df = _binance_klines("ETHUSDT", "1d", 400)
            close = df["close"].dropna()
            rsi = float(_rsi(close, 14).iloc[-1])
            rsimin = float(self.th.get("TOTAL_RSI_MIN", 55))

            base, ok_trend, _ = _score_trend_with_ema(
                close, short=int(self.th.get("TOTAL_EMA", 50)), long=200, weight=weight
            )
            ok_rsi = rsi >= rsimin
            score = base * (1.0 if ok_rsi else 0.7)
            ok = ok_trend and ok_rsi
            explain = f"EMA(50/200) proxy, RSI:{rsi:.1f}"
            return {"score": score, "ok": ok, "explain": explain}
        except Exception as e:
            return _fallback_factor("total2", weight, f"fetch_error:{type(e).__name__}")

    def _total3(self) -> Dict[str, Any]:
        weight = int(self.weights.get("total3", 0))
        try:
            # proxy: BNBUSDT
            df = _binance_klines("BNBUSDT", "1d", 400)
            close = df["close"].dropna()
            rsi = float(_rsi(close, 14).iloc[-1])

            base, ok_trend, _ = _score_trend_with_ema(close, short=50, long=200, weight=weight)
            score = base * (1.0 if rsi >= 58 else 0.5)
            ok = ok_trend and (rsi >= 58)
            explain = f"EMA(50/200) proxy, RSI:{rsi:.1f}"
            return {"score": score, "ok": ok, "explain": explain}
        except Exception as e:
            return _fallback_factor("total3", weight, f"fetch_error:{type(e).__name__}")

    def _btc_regime(self) -> Dict[str, Any]:
        weight = int(self.weights.get("btc_regime", 0))
        try:
            df = _binance_klines("BTCUSDT", "1d", 220)
            close = df["close"].dropna()
            sl = _slope(close.pct_change().fillna(0).rolling(2).mean(), lookback=50)
            if sl <= 0:
                score, ok = float(weight), True
            elif sl >= 0.002:
                score, ok = 0.0, False
            else:
                ratio = (0.002 - sl) / 0.002
                score, ok = float(weight) * ratio, True
            explain = f"slope(returns,50):{sl:.5f}"
            return {"score": score, "ok": ok, "explain": explain}
        except Exception as e:
            return _fallback_factor("btc_regime", weight, f"fetch_error:{type(e).__name__}")

    def _eth_trend(self) -> Dict[str, Any]:
        weight = int(self.weights.get("eth_trend", 0))
        try:
            df = _binance_klines("ETHUSDT", "1d", 400)
            close = df["close"].dropna()
            base, ok_trend, _ = _score_trend_with_ema(close, short=50, long=200, weight=weight)
            rsi = float(_rsi(close, 14).iloc[-1])
            boost = 1.1 if rsi >= 60 else 1.0
            score = min(float(weight), base * boost)
            ok = ok_trend and (rsi >= 52)
            explain = f"EMA(50/200), RSI:{rsi:.1f}"
            return {"score": score, "ok": ok, "explain": explain}
        except Exception as e:
            return _fallback_factor("eth_trend", weight, f"fetch_error:{type(e).__name__}")

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
                # آخرین خط دفاع: اگر حتی فال‌بک هم خطا داد
                w = int(self.weights.get(name, 0))
                out[name] = {"score": 0.5 * float(w), "ok": False, "explain": f"hard-fallback:{type(e).__name__}"}
        return out
