# src/altseason/config.py
from __future__ import annotations
import os
from typing import Dict

# ---------- Telegram ----------
def get_telegram_token() -> str:
    """Read bot token from env. Do NOT hardcode here."""
    return os.getenv("TELEGRAM_BOT_TOKEN", "")

def get_telegram_chat_id() -> str:
    """Read chat ID from env. Do NOT hardcode here."""
    return os.getenv("TELEGRAM_CHAT_ID", "")

# ---------- External APIs ----------
BINANCE_BASE   = os.getenv("BINANCE_BASE", "https://api.binance.com/api/v3")
COINGECKO_BASE = os.getenv("COINGECKO_BASE", "https://api.coingecko.com/api/v3")

# ---------- Display / Timezone ----------
TZ_DISPLAY = os.getenv("TZ_DISPLAY", "Europe/Stockholm")

# ---------- Scoring ----------
FACTOR_WEIGHTS: Dict[str, int] = {
    "btc_dominance": 20,
    "eth_btc":       20,
    "total2":        15,
    "total3":        15,
    "btc_regime":    15,
    "eth_trend":     15,
}

THRESHOLDS: Dict[str, float] = {
    "DOM_MA_SHORT":       50,
    "DOM_MA_LONG":        200,
    "DOM_SLOPE_MIN":      -0.1,
    "ETHBTC_EMA_SHORT":   50,
    "ETHBTC_EMA_LONG":    200,
    "ETHBTC_RSI_MIN":     55,
    "TOTAL_EMA":          50,
    "TOTAL_RSI_MIN":      55,
    "ALTSEASON_MIN_SCORE": 75,
    "ALTSEASON_MIN_FACTORS": 4,
    "FORMING_MIN":        60,
    "NEUTRAL_MIN":        45,
}

__all__ = [
    "get_telegram_token",
    "get_telegram_chat_id",
    "BINANCE_BASE",
    "COINGECKO_BASE",
    "TZ_DISPLAY",
    "FACTOR_WEIGHTS",
    "THRESHOLDS",
]
