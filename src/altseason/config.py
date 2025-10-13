import os
from typing import Dict, Any

TELEGRAM_BOT_TOKEN_HARDCODED = "7512369490:AAHiQqOzjLxh5zjcx3gmUT-hEj1196tIHfI"
TELEGRAM_CHAT_ID_HARDCODED = "-1002925489017"

def get_telegram_token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN") or TELEGRAM_BOT_TOKEN_HARDCODED

def get_telegram_chat_id() -> str:
    return os.getenv("TELEGRAM_CHAT_ID") or TELEGRAM_CHAT_ID_HARDCODED

BINANCE_BASE = os.getenv("BINANCE_BASE", "https://api.binance.com/api/v3")
COINGECKO_BASE = os.getenv("COINGECKO_BASE", "https://api.coingecko.com/api/v3")
TZ_DISPLAY = os.getenv("TZ_DISPLAY", "Europe/Stockholm")

FACTOR_WEIGHTS = {
    "btc_dominance": 20, 
    "eth_btc": 20, 
    "total2": 15, 
    "total3": 15, 
    "btc_regime": 15, 
    "eth_trend": 15,
}

THRESHOLDS = {
    "DOM_MA_SHORT": 50,
    "DOM_MA_LONG": 200,
    "DOM_SLOPE_MIN": -0.1,
    "ETHBTC_EMA_SHORT": 50,
    "ETHBTC_EMA_LONG": 200,
    "ETHBTC_RSI_MIN": 55,
    "TOTAL_EMA": 50,
    "TOTAL_RSI_MIN": 55,
    "ALTSEASON_MIN_SCORE": 75,
    "ALTSEASON_MIN_FACTORS": 4,
    "FORMING_MIN": 60,
    "NEUTRAL_MIN": 45,
}