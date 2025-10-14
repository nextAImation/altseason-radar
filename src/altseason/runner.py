# src/altseason/runner.py
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, Tuple

from altseason.config import FACTOR_WEIGHTS, THRESHOLDS

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
STATE_PATH = REPORTS_DIR / "state.json"

log = logging.getLogger("altseason")


# ========== تنظیمات قابل‌تغییر با ENV ==========
def _penalty_factor() -> float:
    """ضریب پنالتی برای فاکتورهای ok=False در جمع وزنی."""
    try:
        return float(os.getenv("ALT_PENALTY", "0.85"))
    except Exception:
        return 0.85

def _use_penalty() -> bool:
    return str(os.getenv("ALT_PENALTY_ENABLE", "1")).lower() in ("1", "true", "yes", "y")

# ==============================================


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _normalize_to_weight(key: str, fval: Dict[str, Any]) -> float:
    """
    نرمال‌سازی امتیاز هر فاکتور به بازه [0, weight]:
      - اگر score_raw و score_max موجود باشد: score = (raw / max) * weight
      - در غیر اینصورت اگر score موجود باشد:
          * اگر score <= weight → همان score
          * اگر score > weight → min(score, weight) برای جلوگیری از تورم
      - اگر هیچ‌کدام نبود: 0
    """
    w = float(FACTOR_WEIGHTS.get(key, 0))
    if w <= 0:
        return 0.0

    raw = fval.get("score_raw", None)
    raw_max = fval.get("score_max", None)
    if raw is not None and raw_max:
        try:
            raw = float(raw)
            raw_max = float(raw_max)
            if raw_max > 0:
                return max(0.0, min((raw / raw_max) * w, w))
        except Exception:
            pass

    sc = fval.get("score", None)
    if sc is None:
        return 0.0

    try:
        sc = float(sc)
    except Exception:
        return 0.0

    if sc <= w:
        return max(0.0, sc)
    # اگر مقیاس قدیمی بوده و بزرگ‌تر از وزن است، برای بی‌خطر بودن کَپ می‌کنیم
    return w


def _weighted_total_and_okcount(factors: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    """
    جمع وزنیِ نرمال‌شده با پنالتی ملایم برای ok=False (اختیاری).
    """
    total_weight = sum(float(v) for v in FACTOR_WEIGHTS.values())
    if total_weight <= 0:
        return 0, 0

    penalty = _penalty_factor() if _use_penalty() else 1.0

    total = 0.0
    ok_count = 0
    for key, fval in factors.items():
        w = float(FACTOR_WEIGHTS.get(key, 0))
        if w <= 0:
            continue

        base = _normalize_to_weight(key, fval)  # [0..w]
        if not base:
            # اگر امتیاز این فاکتور نداشتیم، صرفاً 0
            continue

        is_ok = bool(fval.get("ok"))
        if is_ok:
            ok_count += 1
            total += base
        else:
            total += base * penalty

    # به نزدیک‌ترین عدد صحیح
    return int(round(total)), ok_count


def _classify(total_score: int, ok_count: int) -> Tuple[str, bool]:
    """
    وضعیت نهایی با استفاده از آستانه‌ها + حداقل تعداد فاکتورهای OK.
    """
    min_factors = _safe_int(THRESHOLDS.get("ALTSEASON_MIN_FACTORS", 4), 4)
    forming_min = _safe_int(THRESHOLDS.get("FORMING_MIN", 60), 60)
    neutral_min = _safe_int(THRESHOLDS.get("NEUTRAL_MIN", 45), 45)
    altseason_min = _safe_int(THRESHOLDS.get("ALTSEASON_MIN_SCORE", 75), 75)

    # Altseason Likely: هم امتیاز بالا هم حداقل تعداد فاکتور OK
    if total_score >= altseason_min and ok_count >= min_factors:
        return "Altseason Likely", True

    # Forming / Watch: امتیاز بالا ولی به حداقل OKها نرسیده
    if total_score >= forming_min:
        return "Forming / Watch", True

    if total_score >= neutral_min:
        return "Neutral", False

    return "Risk-Off", False


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown_report(report_dir: Path, state: Dict[str, Any]) -> Path:
    ts = datetime.now(UTC)
    name = ts.strftime("%Y-%m-%d") + ".md"
    path = report_dir / name

    lines = []
    lines.append(f"# Altseason Radar — Daily Report ({ts.date()})")
    lines.append("")
    lines.append(f"- **Total Score:** {state.get('total_score', 'N/A')}/100")
    lines.append(f"- **Status:** {state.get('status', 'Unknown')}")
    lines.append(f"- **Generated:** {ts.isoformat()}")
    lines.append("")

    facs = state.get("factors") or {}
    if facs:
        lines.append("## Factors")
        for k, v in facs.items():
            w = FACTOR_WEIGHTS.get(k, 0)
            base = _normalize_to_weight(k, v)
            is_ok = "✅" if v.get("ok") else "❌"
            explain = v.get("explain") or ""
            lines.append(
                f"- **{k}** (w={w}): {base:.0f} {is_ok}"
                + (f" — {explain}" if explain else "")
            )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class AltseasonRunner:
    """
    Runner جدید با:
      - جمع وزنی نرمال‌شده + پنالتی ملایم
      - enforce حداقل فاکتورهای OK در کلاس‌بندی
      - تحمل دادهٔ ناقص
      - سازگاری کامل با خروجی state.json + گزارش md
    """

    def __init__(self, reports_dir: Path | None = None):
        self.logger = logging.getLogger("altseason")
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR

    # اگر کلاس/پایپ‌لاین واقعی داری، همین امضا را نگه دار و خروجی را در همین قالب بده.
    def _compute_factors(self) -> Dict[str, Dict[str, Any]]:
        """
        در تولید، این متد باید با دادهٔ واقعی پر شود.
        ساختار هر فاکتور:
          {
            "score": int | float             # (اختیاری) اگر از قدیم در مقیاس وزن بود
            "score_raw": float               # (اختیاری) مقدار خام
            "score_max": float               # (اختیاری) بیشینهٔ مقدار خام
            "ok": bool,
            "explain": str
          }
        - اگر score_raw/score_max بدهی، به صورت خودکار به weight آن فاکتور نرمال می‌شود.
        - اگر فقط score بدهی و <= weight باشد، همان استفاده می‌شود (سازگار با کد فعلی).
        """
        # نمونهٔ فعلی شما (سازگار با قبل):
        return {
            "btc_dominance": {"score": 15, "ok": True,  "explain": "Dominance زیر EMA-ها"},
            "eth_btc":       {"score": 18, "ok": True,  "explain": "ETH/BTC بالای EMA50"},
            "total2":        {"score": 12, "ok": True,  "explain": "TOTAL2 روند مثبت"},
            "total3":        {"score": 10, "ok": False, "explain": "TOTAL3 ضعف آلت‌های کوچک"},
            "btc_regime":    {"score": 8,  "ok": False, "explain": "ریسک رِژیم BTC"},
            "eth_trend":     {"score": 11, "ok": True,  "explain": "ETH روند صعودی سبک"},
        }

    def run_daily_analysis(self) -> bool:
        print("🔄 Starting altseason analysis...")
        print("📡 Fetching market data...")
        print("📈 Calculating market factors...")

        # 1) محاسبهٔ فاکتورها
        factors = self._compute_factors()

        # 2) جمع وزنی نرمال‌شده + پنالتی
        total_score, ok_count = _weighted_total_and_okcount(factors)

        # 3) وضعیت نهایی با درنظرگرفتن حداقل فاکتورهای OK
        status, forming = _classify(total_score, ok_count)

        # 4) گزارش کنسولی
        badge = "🟢" if forming and status.startswith("Altseason") else \
                "🟡" if forming else \
                "⚪️" if status.startswith("Neutral") else "🔴"
        print(f"📊 Total Score: {total_score}/100")
        print(f"🎯 Status: {status} {badge}")
        print("✅ Analysis completed successfully!")

        # 5) ذخیرهٔ state.json
        state = {
            "total_score": total_score,
            "status": status,
            "forming": bool(forming),
            "as_of": datetime.now(UTC).isoformat(),
            "factors": factors,
            "ok_count": ok_count,
        }
        _write_json(STATE_PATH, state)

        # 6) ذخیرهٔ گزارش مارک‌داون روز
        _ = _write_markdown_report(self.reports_dir, state)

        return True
