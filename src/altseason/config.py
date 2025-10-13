# src/altseason/config.py
import os

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return (v.lower() in ("1", "true", "yes", "on")) if v is not None else default

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# اگر مقدار ندادیم، ولی سکرت‌ها هست، به‌صورت خودکار فعال باشد
TELEGRAM_ENABLED = _env_bool(
    "TELEGRAM_ENABLED",
    default=bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
)

BINANCE_BASE   = os.getenv("BINANCE_BASE", "https://api.binance.com/api/v3")
COINGECKO_BASE = os.getenv("COINGECKO_BASE", "https://api.coingecko.com/api/v3")
TZ_DISPLAY     = os.getenv("TZ_DISPLAY", "Europe/Stockholm")

FACTOR_WEIGHTS = {
    "btc_dominance": 20, "eth_btc": 20, "total2": 15,
    "total3": 15, "btc_regime": 15, "eth_trend": 15,
}

THRESHOLDS = {
    "DOM_MA_SHORT": 50, "DOM_MA_LONG": 200, "DOM_SLOPE_MIN": -0.1,
    "ETHBTC_EMA_SHORT": 50, "ETHBTC_EMA_LONG": 200, "ETHBTC_RSI_MIN": 55,
    "TOTAL_EMA": 50, "TOTAL_RSI_MIN": 55,
    "ALTSEASON_MIN_SCORE": 75, "ALTSEASON_MIN_FACTORS": 4,
    "FORMING_MIN": 60, "NEUTRAL_MIN": 45,
}
