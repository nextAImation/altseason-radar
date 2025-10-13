import json, os, urllib.request, urllib.parse
from .config import TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram(text: str) -> bool:
    """ارسال پیام تلگرام با بررسی خطاها"""
    if not TELEGRAM_ENABLED:
        print("[telegram] disabled -> skip")
        return False
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[telegram] missing token/chat_id -> skip")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode()

    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data)) as r:
            res = json.loads(r.read().decode())
            if not res.get("ok"):
                print("[telegram] api not ok:", res)
            else:
                print("[telegram] message sent successfully ✅")
            return bool(res.get("ok"))
    except Exception as e:
        print("[telegram] exception:", repr(e))
        return False
