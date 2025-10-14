# File: src/altseason/factors.py
from __future__ import annotations

import os
import math
from typing import Dict, Any, Optional, NamedTuple, List, Tuple

import requests
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import (
    BINANCE_BASE,
    COINGECKO_BASE,
    THRESHOLDS,
    FACTOR_WEIGHTS,
)

# =========================
# Errors & small containers
# =========================

class _HttpErr(RuntimeError):
    pass

class SeriesWithSource(NamedTuple):
    close: pd.Series   # datetime-indexed close series
    source: str        # "binance" or "coingecko"

# =========================
# HTTP helper (with retry)
# =========================

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_HttpErr),
)
def _get_json(url: str, params: Optional[dict] = None, timeout: int = 25, headers: Optional[dict] = None) -> dict:
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=headers)
        if r.status_code >= 400:
            raise _HttpErr(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()
    except requests.RequestException as e:
        raise _HttpErr(str(e))

# =========================
# CoinGecko fallback
# =========================

# Symbol→CoinGecko id map برای fallback
_CG_ID = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    # اگر بعداً لازم شد اضافه کنید:
    # "SOLUSDT": "solana",
}

def _cg_market_chart(coin_id: str, days: int = 365, vs: str = "usd") -> pd.Series:
    """
    Close series from CoinGecko market_chart (ts, price) with:
    - adaptive days fallback (365→180→90→30)
    - auto-swap to public base on 401/limited-range
    - optional Pro API key via COINGECKO_API_KEY
    """
    base_env = os.getenv("COINGECKO_BASE", COINGECKO_BASE).rstrip("/")
    api_key = os.getenv("COINGECKO_API_KEY", "").strip()
    headers = {"x-cg-pro-api-key": api_key} if api_key else None

    def _try_fetch(base: str, d: int) -> pd.Series:
        url = f"{base}/coins/{coin_id}/market_chart"
        params = {"vs_currency": vs, "days": str(d), "interval": "daily", "precision": "full"}
        js = _get_json(url, params=params, timeout=30, headers=headers)
        prices = js.get("prices") or []
        if not prices:
            raise _HttpErr("No prices in market_chart")
        df = pd.DataFrame(prices, columns=["ts", "close"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        ser = df.set_index("ts")["close"].sort_index().astype(float)
        if ser.empty:
            raise _HttpErr("Empty series from market_chart")
        return ser

    # 1) تلاش روی base تنظیم‌شده
    last_err_msg = ""
    for d in (days, 365, 180, 90, 30):
        try:
            return _try_fetch(base_env, d)
        except _HttpErr as e:
            last_err_msg = str(e)
            # اگر خطا مربوط به محدودیت/401 باشد، بعداً public را امتحان می‌کنیم
            continue

    # 2) تلاش روی پابلیک CoinGecko
    public_base = "https://api.coingecko.com/api/v3"
    for d in (365, 180, 90, 30):
        try:
            return _try_fetch(public_base, d)
        except _HttpErr as e:
            last_err_msg = str(e)
            continue

    raise _HttpErr(f"CoinGecko fallback failed for {coin_id}: {last_err_msg[:200]}")

# =========================
# Binance primary + fallback
# =========================

def _binance_klines(symbol: str, interval: str = "1d", limit: int = 365) -> pd.DataFrame:
    """
    Fetch OHLC klines from Binance; fallback to CoinGecko close series when needed.
    Returns a DataFrame with at least 'close' and datetime index (UTC).
    """
    try:
        url = f"{BINANCE_BASE}/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        js = _get_json(url, params=params, timeout=25)
        cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
        df = pd.DataFrame(js, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time").sort_index()
        for c in ("open","high","low","close","volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # فقط close برای ما مهم است، اما بقیه هم موجود است
        return df[["close"]]
    except Exception as e:
        # Fallback
        if symbol == "ETHBTC":
            # ETHBTC = ETH/USD / BTC/USD
            eth = _cg_market_chart("ethereum", days=limit)
            btc = _cg_market_chart("bitcoin", days=limit)
            common = eth.index.intersection(btc.index)
            close = (eth.reindex(common) / btc.reindex(common)).dropna()
            return pd.DataFrame({"close": close})
        # USDT pairs via coin id
        coin_id = _CG_ID.get(symbol)
        if not coin_id:
            raise _HttpErr(f"No CoinGecko mapping for symbol: {symbol} (fallback needed)")
        ser = _cg_market_chart(coin_id, days=limit)
        return pd.DataFrame({"close": ser})

def _series(symbol: str, limit: int = 365) -> SeriesWithSource:
    """
    Unified accessor for 1D close series with fallback.
    """
    try:
        df = _binance_klines(symbol, "1d", limit)
        ser = df["close"].astype(float)
        ser.index = pd.to_datetime(ser.index, utc=True)
        return SeriesWithSource(ser.sort_index(), "binance")
    except Exception:
        # already handled fallback inside _binance_klines; just label as coingecko
        df = _binance_klines(symbol, "1d", limit)
        ser = df["close"].astype(float)
        ser.index = pd.to_datetime(ser.index, utc=True)
        return SeriesWithSource(ser.sort_index(), "coingecko")

# =========================
# Indicators
# =========================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n//3)).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def daily_slope(s: pd.Series, window: int = 30) -> float:
    """
    Slope of last `window` closes using simple linear regression (index as 0..n).
    Returns slope per step (unit: price per day). For relative, we scale by last price later.
    """
    if len(s) < max(10, window):
        return 0.0
    y = s.tail(window).values.astype(float)
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x**2).sum()
    if denom == 0:
        return 0.0
    m = (x*y).sum() / denom
    return float(m)

# =========================
# Scoring helpers
# =========================

def _bounded01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _score_bool(ok: bool, weight: float, boost: float = 1.0) -> float:
    """Return weight if ok else 0, then apply optional boost (<=1.0 typical)."""
    return weight*boost if ok else 0.0

def _score_trend(cross: bool, rsi_ok: bool, weight: float) -> float:
    # 60% از weight برای EMA cross، 40% برای RSI
    p = 0.6*(1.0 if cross else 0.0) + 0.4*(1.0 if rsi_ok else 0.0)
    return weight * p

def _score_slope(slope_rel: float, slope_min: float, weight: float) -> float:
    """
    Map relative slope (e.g., %/day) to [0..1] using threshold.
    If slope_rel <= slope_min (e.g., dominance is falling faster than threshold) → full weight.
    """
    # اگر min منفی است، هرچه کوچک‌تر (منفی‌تر) بهتر
    if slope_min < 0:
        # slope_rel <= slope_min → 1.0
        # slope_rel >= 0        → 0.0
        if slope_rel <= slope_min:
            frac = 1.0
        elif slope_rel >= 0:
            frac = 0.0
        else:
            frac = (0 - slope_rel) / (0 - slope_min)
    else:
        # حالت عمومی
        # slope_rel >= slope_min → 1.0
        # slope_rel <= 0         → 0.0
        if slope_rel >= slope_min:
            frac = 1.0
        elif slope_rel <= 0:
            frac = 0.0
        else:
            frac = slope_rel / slope_min
    return weight * _bounded01(frac)

# =========================
# FactorCalculator
# =========================

class FactorCalculator:
    """
    محاسبه فاکتورهای شش‌گانه با امتیازدهی وزن‌دار و fallback پایدار.
    خروجی: dict با کلیدهای:
      - score (float, در مقیاس وزن همان فاکتور)
      - ok (bool)
      - explain (str)
    """

    def __init__(self, lookback_days: int = 365):
        self.lookback = lookback_days

    def _eth_btc(self) -> Dict[str, Any]:
        weight = FACTOR_WEIGHTS.get("eth_btc", 20)
        ser, src = _series("ETHBTC", self.lookback)
        e50 = ema(ser, THRESHOLDS.get("ETHBTC_EMA_SHORT", 50))
        e200 = ema(ser, THRESHOLDS.get("ETHBTC_EMA_LONG", 200))
        cross = bool(e50.iloc[-1] > e200.iloc[-1])
        r = rsi(ser, 14).iloc[-1]
        rsi_ok = bool(r >= THRESHOLDS.get("ETHBTC_RSI_MIN", 55))
        score = _score_trend(cross, rsi_ok, weight)
        ok = cross and rsi_ok
        explain = f"EMA50>{'YES' if cross else 'NO'} EMA200 | RSI={r:.1f} (src:{src})"
        return {"score": round(score, 2), "ok": ok, "explain": explain}

    def _btc_regime(self) -> Dict[str, Any]:
        weight = FACTOR_WEIGHTS.get("btc_regime", 15)
        ser, src = _series("BTCUSDT", self.lookback)
        e50 = ema(ser, 50)
        e200 = ema(ser, 200)
        cross = bool(e50.iloc[-1] > e200.iloc[-1])
        r = rsi(ser, 14).iloc[-1]
        rsi_ok = bool(r >= 50)
        score = _score_trend(cross, rsi_ok, weight)
        ok = cross and rsi_ok
        explain = f"BTC EMA50>{'YES' if cross else 'NO'} EMA200 | RSI={r:.1f} (src:{src})"
        return {"score": round(score, 2), "ok": ok, "explain": explain}

    def _eth_trend(self) -> Dict[str, Any]:
        weight = FACTOR_WEIGHTS.get("eth_trend", 15)
        ser, src = _series("ETHUSDT", self.lookback)
        e50 = ema(ser, 50)
        e200 = ema(ser, 200)
        cross = bool(e50.iloc[-1] > e200.iloc[-1])
        r = rsi(ser, 14).iloc[-1]
        rsi_ok = bool(r >= 55)
        score = _score_trend(cross, rsi_ok, weight)
        ok = cross and rsi_ok
        explain = f"ETH EMA50>{'YES' if cross else 'NO'} EMA200 | RSI={r:.1f} (src:{src})"
        return {"score": round(score, 2), "ok": ok, "explain": explain}

    def _total2(self) -> Dict[str, Any]:
        """
        Proxy TOTAL2 با ETHUSDT: ترند مثبت آلت‌های بزرگ.
        """
        weight = FACTOR_WEIGHTS.get("total2", 15)
        ser, src = _series("ETHUSDT", self.lookback)
        e50 = ema(ser, 50)
        e200 = ema(ser, 200)
        cross = bool(e50.iloc[-1] > e200.iloc[-1])
        r = rsi(ser, 14).iloc[-1]
        rsi_ok = bool(r >= THRESHOLDS.get("TOTAL_RSI_MIN", 55))
        score = _score_trend(cross, rsi_ok, weight)
        ok = cross and rsi_ok
        explain = f"TOTAL2~ETH EMA50>{'YES' if cross else 'NO'} EMA200 | RSI={r:.1f} (src:{src})"
        return {"score": round(score, 2), "ok": ok, "explain": explain}

    def _total3(self) -> Dict[str, Any]:
        """
        Proxy TOTAL3 با BNBUSDT: ترند مثبت آلت‌های کوچک‌تر (تقریبی/محافظه‌کارانه).
        """
        weight = FACTOR_WEIGHTS.get("total3", 15)
        ser, src = _series("BNBUSDT", self.lookback)
        e50 = ema(ser, 50)
        e200 = ema(ser, 200)
        cross = bool(e50.iloc[-1] > e200.iloc[-1])
        r = rsi(ser, 14).iloc[-1]
        rsi_ok = bool(r >= THRESHOLDS.get("TOTAL_RSI_MIN", 55))
        score = _score_trend(cross, rsi_ok, weight)
        ok = cross and rsi_ok
        explain = f"TOTAL3~BNB EMA50>{'YES' if cross else 'NO'} EMA200 | RSI={r:.1f} (src:{src})"
        return {"score": round(score, 2), "ok": ok, "explain": explain}

    def _btc_dominance(self) -> Dict[str, Any]:
        """
        چون سری Dominance مستقیم نداریم، از ETHBTC به عنوان آینه استفاده می‌کنیم:
        - اگر ETHBTC رو به بالا باشد (slope مثبت)، یعنی تمایل به کاهش dominance بیت‌کوین → مثبت برای آلت‌سیزن.
        آستانه DOM_SLOPE_MIN منفی بود، اینجا slope_rel (درصد/روز) را استفاده می‌کنیم.
        """
        weight = FACTOR_WEIGHTS.get("btc_dominance", 20)
        ser, src = _series("ETHBTC", self.lookback)

        # slope نسبی (%/روز) روی 60 روز
        window = 60
        m = daily_slope(ser, window=window)
        last = float(ser.iloc[-1])
        slope_rel = 0.0 if last == 0 else (m / last) * 100.0  # درصد در روز
        # برای dominance کاهش مطلوب است، پس اگر ETHBTC بالا می‌رود (slope_rel>0) امتیاز خوب
        # آستانه را از THRESHOLDS می‌گیریم ولی چون min آن برای dominance منفی تعریف شده بود،
        # اینجا از یک نگاشت ساده استفاده می‌کنیم: هرچه slope_rel بزرگ‌تر، بهتر.
        # اگر بخواهیم دقیقاً از DOM_SLOPE_MIN استفاده کنیم، آن برای سری dominance بود؛
        # بنابراین اینجا یک نگاشت خطی ساده:
        # slope_rel >= 0.2%/day → full weight,  slope_rel <= 0 → zero.
        full = 0.2  # %/day
        if slope_rel <= 0:
            frac = 0.0
        elif slope_rel >= full:
            frac = 1.0
        else:
            frac = slope_rel / full

        score = weight * _bounded01(frac)
        ok = frac >= 0.6  # اگر حداقل 60% از full برسد، ok
        explain = f"ETHBTC slope≈{slope_rel:.3f}%/day over {window}d (src:{src})"
        return {"score": round(score, 2), "ok": ok, "explain": explain}

    def compute_factors(self) -> Dict[str, Dict[str, Any]]:
        """
        محاسبه همه فاکتورها با handling خطا برای هر کدام.
        """
        out: Dict[str, Dict[str, Any]] = {}

        def _safe(name: str, fn) -> None:
            try:
                out[name] = fn()
            except Exception as e:
                # fail-soft: امتیاز 0 اما توضیح مفید بده
                out[name] = {"score": 0.0, "ok": False, "explain": f"error: {e}"}

        _safe("btc_dominance", self._btc_dominance)
        _safe("eth_btc", self._eth_btc)
        _safe("total2", self._total2)
        _safe("total3", self._total3)
        _safe("btc_regime", self._btc_regime)
        _safe("eth_trend", self._eth_trend)

        return out
