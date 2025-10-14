# -*- coding: utf-8 -*-
"""
FactorCalculator with robust Binance -> CoinGecko fallback.
Returns factor dict: {name: {score: float, ok: bool, explain: str}}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import (
    BINANCE_BASE,
    COINGECKO_BASE,
    THRESHOLDS,
    FACTOR_WEIGHTS,
)

# --------------------------- Utilities ---------------------------

@dataclass
class SeriesWithSource:
    close: pd.Series
    source: str  # "binance" | "coingecko"

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _nznorm(x: float, lo: float, hi: float) -> float:
    """Normalize x to [0,1] between lo..hi with clipping."""
    if hi == lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n//2)).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _slope_pct(s: pd.Series, window: int = 50) -> float:
    """Approx daily slope (%) over last `window`: (last-first)/first * 100 / window."""
    if len(s) < window + 1:
        return 0.0
    seg = s.iloc[-window:]
    a, b = float(seg.iloc[0]), float(seg.iloc[-1])
    if a == 0:
        return 0.0
    return ((b - a) / a) * 100.0 / max(1, window)

# --------------------------- HTTP helpers ---------------------------

class _HttpErr(RuntimeError):
    pass

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_HttpErr),
)
def _get_json(url: str, params: Optional[dict] = None, timeout: int = 25) -> dict:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code >= 400:
            raise _HttpErr(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()
    except requests.RequestException as e:
        raise _HttpErr(str(e))

# --------------------------- Fetch: Binance & CoinGecko ---------------------------

def _binance_klines(symbol: str, interval: str = "1d", limit: int = 400) -> pd.Series:
    """Fetch close series (UTC index) from Binance; raises on failure."""
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    js = _get_json(url, params=params)
    # Kline: [ open_time, open, high, low, close, volume, close_time, ... ]
    if not isinstance(js, list) or not js:
        raise _HttpErr("Empty klines")
    df = pd.DataFrame(
        js,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore",
        ],
    )
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    ser = df.set_index("close_time")["close"].sort_index().dropna()
    if ser.empty:
        raise _HttpErr("No close data")
    return ser

def _cg_market_chart(coin_id: str, days: int = 400, vs: str = "usd") -> pd.Series:
    """Close series from CoinGecko market_chart (ts, price)."""
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": str(days), "interval": "daily", "precision": "full"}
    js = _get_json(url, params=params, timeout=30)
    prices = js.get("prices") or []
    if not prices:
        raise _HttpErr("No prices in market_chart")
    df = pd.DataFrame(prices, columns=["ts", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    ser = df.set_index("ts")["close"].sort_index().astype(float)
    if ser.empty:
        raise _HttpErr("Empty series from market_chart")
    return ser

_CG_ID = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    # نیاز شد اضافه کنین
}

def _series(symbol: str, limit: int = 400) -> SeriesWithSource:
    """Get close series with fallback & its source label."""
    # Primary: Binance
    try:
        if symbol.upper() == "ETHBTC":
            # Binance pair directly
            ser = _binance_klines("ETHBTC", limit=limit)
            return SeriesWithSource(ser, "binance")
        else:
            ser = _binance_klines(symbol.upper(), limit=limit)
            return SeriesWithSource(ser, "binance")
    except Exception as e:
        # Fallback: CoinGecko
        if symbol.upper() == "ETHBTC":
            eth = _cg_market_chart("ethereum", days=limit)
            btc = _cg_market_chart("bitcoin", days=limit)
            idx = eth.index.intersection(btc.index)
            ratio = (eth.reindex(idx) / btc.reindex(idx)).dropna()
            if ratio.empty:
                raise RuntimeError(f"CoinGecko fallback failed for ETHBTC: no ratio")
            return SeriesWithSource(ratio, "coingecko")
        coin_id = _CG_ID.get(symbol.upper())
        if not coin_id:
            raise RuntimeError(f"No CoinGecko mapping for {symbol}")
        ser = _cg_market_chart(coin_id, days=limit)
        return SeriesWithSource(ser, "coingecko")

# --------------------------- Factor scoring ---------------------------

def _trend_signals(ser: pd.Series, ema_short: int, ema_long: int) -> Tuple[float, float, float]:
    ema_s = _ema(ser, ema_short)
    ema_l = _ema(ser, ema_long)
    rsi = _rsi(ser, 14).iloc[-1]
    ratio = float((ema_s.iloc[-1] / ema_l.iloc[-1]) if ema_l.iloc[-1] else 1.0)
    slope = _slope_pct(ser, window=min(50, len(ser)//3))
    return ratio, rsi, slope

def _trend_score_ok(
    ser: pd.Series,
    ema_short: int,
    ema_long: int,
    rsi_min: float,
    slope_min: float = 0.0,
) -> Tuple[float, bool, Dict[str, float]]:
    ratio, rsi, slope = _trend_signals(ser, ema_short, ema_long)

    # normalize → raw in [0,1]
    ema_sig = _nznorm(ratio, 1.0, 1.2)        # 1.0 → 0, 1.2 → 1
    rsi_sig = _nznorm(rsi, 50.0, 70.0)        # 50 → 0, 70 → 1
    slp_sig = _nznorm(slope, slope_min, slope_min + 2.0)  # e.g. 0..2 %/day

    raw = float(np.mean([ema_sig, rsi_sig, slp_sig]))
    ok = (ratio > 1.0) and (rsi >= rsi_min) and (slope >= slope_min)

    return raw, ok, {
        "ratio": ratio,
        "rsi": rsi,
        "slope": slope,
    }

# --------------------------- Factors ---------------------------

class FactorCalculator:
    """
    Computes all factors using Binance with CoinGecko fallback.
    """

    def __init__(self, limit: int = 400):
        self.limit = limit

    def _btc_dominance(self) -> Dict[str, object]:
        """
        از ETHBTC به‌عنوان نیابتی از کاهش دامیننس BTC استفاده می‌کنیم:
        وقتی ETHBTC روند صعودی دارد (EMA50>EMA200, RSI>55, شیب>THRESHOLDS.DOM_SLOPE_MIN)،
        یعنی احتمال کاهش دامیننس و فاز آلت‌ها بیشتر است → امتیاز بالاتر.
        """
        sws = _series("ETHBTC", limit=self.limit)
        raw, ok, m = _trend_score_ok(
            sws.close,
            THRESHOLDS["ETHBTC_EMA_SHORT"],
            THRESHOLDS["ETHBTC_EMA_LONG"],
            THRESHOLDS["ETHBTC_RSI_MIN"],
            THRESHOLDS["DOM_SLOPE_MIN"],
        )
        score = FACTOR_WEIGHTS["btc_dominance"] * raw
        explain = (
            f"src={sws.source} | ETHBTC ratio={m['ratio']:.3f}, RSI={m['rsi']:.1f}, slope%/d={m['slope']:.3f}"
        )
        return {"score": round(score, 2), "ok": bool(ok), "explain": explain}

    def _eth_btc(self) -> Dict[str, object]:
        sws = _series("ETHBTC", limit=self.limit)
        raw, ok, m = _trend_score_ok(
            sws.close,
            THRESHOLDS["ETHBTC_EMA_SHORT"],
            THRESHOLDS["ETHBTC_EMA_LONG"],
            THRESHOLDS["ETHBTC_RSI_MIN"],
            0.0,
        )
        score = FACTOR_WEIGHTS["eth_btc"] * raw
        explain = (
            f"src={sws.source} | ETHBTC ratio={m['ratio']:.3f}, RSI={m['rsi']:.1f}, slope%/d={m['slope']:.3f}"
        )
        return {"score": round(score, 2), "ok": bool(ok), "explain": explain}

    def _total2(self) -> Dict[str, object]:
        """
        PROXY برای TOTAL2: روند ETHUSDT (نماینده آلتی با نقدینگی بالا).
        """
        sws = _series("ETHUSDT", limit=self.limit)
        raw, ok, m = _trend_score_ok(
            sws.close,
            THRESHOLDS["TOTAL_EMA"],
            THRESHOLDS["ETHBTC_EMA_LONG"],  # long=200
            THRESHOLDS["TOTAL_RSI_MIN"],
            0.0,
        )
        score = FACTOR_WEIGHTS["total2"] * raw
        explain = f"src={sws.source} | ETHUSD ratio={m['ratio']:.3f}, RSI={m['rsi']:.1f}, slope%/d={m['slope']:.3f}"
        return {"score": round(score, 2), "ok": bool(ok), "explain": explain}

    def _total3(self) -> Dict[str, object]:
        """
        PROXY برای TOTAL3: روند BNBUSDT (نماینده آلت‌های متوسط/کوین اکو‌سیستم).
        """
        sws = _series("BNBUSDT", limit=self.limit)
        raw, ok, m = _trend_score_ok(
            sws.close,
            THRESHOLDS["TOTAL_EMA"],
            THRESHOLDS["ETHBTC_EMA_LONG"],  # 200
            THRESHOLDS["TOTAL_RSI_MIN"],
            0.0,
        )
        # کمی سخت‌گیرتر: آلت‌های کوچک پرنوسان‌ترند → اندکی پنالتی روی raw
        raw_adj = max(0.0, min(1.0, raw * 0.95))
        score = FACTOR_WEIGHTS["total3"] * raw_adj
        explain = f"src={sws.source} | BNBUSD ratio={m['ratio']:.3f}, RSI={m['rsi']:.1f}, slope%/d={m['slope']:.3f}"
        return {"score": round(score, 2), "ok": bool(ok), "explain": explain}

    def _btc_regime(self) -> Dict[str, object]:
        """
        رژیم BTC: در آلت‌سیزن، معمولاً BTC رشد ملایم/خنثی دارد نه رکورددار مومنتوم.
        معیار: EMA50>EMA200 اما RSI نه خیلی داغ (<=65) و شیب مثبت اما ملایم.
        """
        sws = _series("BTCUSDT", limit=self.limit)
        ratio, rsi, slope = _trend_signals(sws.close, 50, 200)

        ok = (ratio > 1.0) and (50.0 <= rsi <= 65.0) and (slope >= 0.0)
        # نمره: اگر خیلی داغ باشد، امتیاز کم می‌کنیم؛ اگر معتدل باشد، امتیاز خوب.
        # raw از سه قطعه:
        ema_sig = _nznorm(ratio, 1.0, 1.15)            # مثبت اما نه شدید
        rsi_sig = 1.0 - _nznorm(rsi, 65.0, 75.0)       # هرچه RSI به 65 نزدیک‌تر بهتر (بالا بدتر)
        slp_sig = _nznorm(slope, 0.0, 1.5)
        raw = float(np.mean([ema_sig, rsi_sig, slp_sig]))
        score = FACTOR_WEIGHTS["btc_regime"] * raw

        explain = f"src={sws.source} | BTCUSD ratio={ratio:.3f}, RSI={rsi:.1f}, slope%/d={slope:.3f}"
        return {"score": round(score, 2), "ok": bool(ok), "explain": explain}

    def _eth_trend(self) -> Dict[str, object]:
        sws = _series("ETHUSDT", limit=self.limit)
        raw, ok, m = _trend_score_ok(
            sws.close,
            THRESHOLDS["TOTAL_EMA"],     # 50
            THRESHOLDS["ETHBTC_EMA_LONG"],  # 200
            THRESHOLDS["TOTAL_RSI_MIN"], # 55
            0.0,
        )
        score = FACTOR_WEIGHTS["eth_trend"] * raw
        explain = f"src={sws.source} | ETHUSD ratio={m['ratio']:.3f}, RSI={m['rsi']:.1f}, slope%/d={m['slope']:.3f}"
        return {"score": round(score, 2), "ok": bool(ok), "explain": explain}

    # --------------------------- public ---------------------------

    def compute_factors(self) -> Dict[str, Dict[str, object]]:
        """
        محاسبه همه فاکتورها با فول‌بک واقعی.
        """
        out: Dict[str, Dict[str, object]] = {}
        out["btc_dominance"] = self._btc_dominance()
        out["eth_btc"] = self._eth_btc()
        out["total2"] = self._total2()
        out["total3"] = self._total3()
        out["btc_regime"] = self._btc_regime()
        out["eth_trend"] = self._eth_trend()
        return out
