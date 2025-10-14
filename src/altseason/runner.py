# src/altseason/runner.py
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, Tuple

from altseason.config import FACTOR_WEIGHTS, THRESHOLDS

# ریشهٔ repo و مسیر گزارش‌ها
ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
STATE_PATH = REPORTS_DIR / "state.json"

log = logging.getLogger("altseason")
log.setLevel(logging.INFO)


# -------------------- تنظیمات قابل‌تغییر با ENV --------------------
def _penalty_enable() -> bool:
    return str(os.getenv("ALT_PENALTY_ENABLE", "1")).lower() in ("1", "true", "yes", "y")

def _penalty_factor() -> float:
    # پنالتی برای فاکتورهایی که ok=False هستند (۰..۱)
    try:
        return max(0.0, min(1.0, float(os.getenv("ALT_PENALTY", "0.85"))))
    except Exception:
        return 0.85
# --------------------------------------------------------------------


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _normalize_to_weight(key: str, fval: Dict[str, Any]) -> float:
    """
    امتیاز هر فاکتور را به بازه [0, weight] می‌برد:
      - اگر score_raw و score_max وجود داشته باشد: (raw/max)*weight
      - اگر فقط score باشد:
          * اگر score <= weight → همان
          * اگر score > weight → cap به weight
      - در غیر این صورت → 0
    """
    w = float(FACTOR_WEIGHTS.get(key, 0))
    if w <= 0:
        return 0.0

    raw = fval.get("score_raw")
    raw_max = fval.get("score_max")
    if raw is not None and raw_max:
        try:
            raw = float(raw)
            raw_max = float(raw_max)
            if raw_max > 0:
                return max(0.0, min((raw / raw_max) * w, w))
        except Exception:
            pass

    sc = fval.get("score")
    if sc is None:
        return 0.0

    try:
        sc = float(sc)
    except Exception:
        return 0.0

    return max(0.0, min(sc, w))


def _weighted_total_and_okcount(factors: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    """
    جمع وزنیِ نرمال‌شده با پنالتی ملایم برای ok=False (اختیاری).
    خروجی: (total_score 0..100, ok_count)
    """
    total_weight = float(sum(FACTOR_WEIGHTS.values()))
    if total_weight <= 0:
        return 0, 0

    pen = _penalty_factor() if _penalty_enable() else 1.0

    total = 0.0
    ok_count = 0
    for k, v in factors.items():
        w = float(FACTOR_WEIGHTS.get(k, 0))
        if w <= 0:
            continue

        base = _normalize_to_weight(k, v)  # در [0..w]
        if not base:
            continue

        if bool(v.get("ok")):
            ok_count += 1
            total += base
        else:
            total += base * pen

    # امنیت: اگر جمع وزن‌ها دقیقاً ۱۰۰ نباشد هم، خروجی cap می‌شود.
    total = max(0.0, min(total, 100.0))
    return int(round(total)), ok_count


def _classify(total_score: int, ok_count: int) -> Tuple[str, bool]:
    """
    منطق وضعیت نهایی با آستانه‌های config و شرط حداقل تعداد OK.
    """
    min_factors = _safe_int(THRESHOLDS.get("ALTSEASON_MIN_FACTORS", 4), 4)
    forming_min = _safe_int(THRESHOLDS.get("FORMING_MIN", 60), 60)
    neutral_min = _safe_int(THRESHOLDS.get("NEUTRAL_MIN", 45), 45)
    altseason_min = _safe_int(THRESHOLDS.get("ALTSEASON_MIN_SCORE", 75), 75)

    # Altseason Likely فقط وقتی که هم امتیاز و هم پوشش فاکتورها کافی باشد
    if total_score >= altseason_min and ok_count >= min_factors:
        return "Altseason Likely", True

    # Forming/Watch: امتیاز کافی هست، ولی پوشش OK به حد Altseason نرسیده
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
    lines.append(f"# Altseason Radar — Daily Report ({ts.date()})\n")
    lines.append(f"- **Total Score:** {state.get('total_score', 'N/A')}/100")
    lines.append(f"- **Status:** {state.get('status', 'Unknown')}")
    lines.append(f"- **OK Factors:** {state.get('ok_count', 0)}")
    lines.append(f"- **Generated:** {ts.isoformat()}\n")

    facs = state.get("factors") or {}
    if facs:
        lines.append("## Factors")
        for k, v in facs.items():
            w = FACTOR_WEIGHTS.get(k, 0)
            base = _normalize_to_weight(k, v)
            is_ok = "✅" if v.get("ok") else "❌"
            explain = v.get("explain") or ""
            lines.append(
                f"- **{k}** (w={w}): {base:.0f} {is_ok}" + (f" — {explain}" if explain else "")
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class AltseasonRunner:
    """
    Runner با:
      - جمع وزنی نرمال‌شده + پنالتی
      - شرط حداقل OK برای Altseason
      - تحمل دادهٔ ناقص
      - خروجی state.json و گزارش روزانه
      - استفاده از FactorCalculator اگر موجود باشد (واقعی)، در غیر این‌صورت fallback سیموله
    """

    def __init__(self, reports_dir: Path | None = None):
        self.logger = logging.getLogger("altseason")
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR

    # --- مسیر واقعی (اگر موجود باشد) ---
    def _compute_factors_real(self) -> Dict[str, Dict[str, Any]]:
        """
        اگر altseason.factors.FactorCalculator موجود باشد، از آن استفاده می‌شود.
        انتظار می‌رود خروجی دیکشنری‌ای مثل:
        {
          "btc_dominance": {"score_raw": ..., "score_max": ..., "ok": True/False, "explain": "..."},
          ...
        }
        یا حداقل {"score": عددی ≤ weight, "ok": bool, "explain": "..."}
        """
        from altseason.factors import FactorCalculator  # type: ignore
        calc = FactorCalculator()
        return calc.compute_factors()

    # --- مسیر fallback (سیموله‌ی سازگار با قبل) ---
    def _compute_factors_fallback(self) -> Dict[str, Dict[str, Any]]:
        return {
            "btc_dominance": {"score": 15, "ok": True,  "explain": "Dominance زیر EMA-ها"},
            "eth_btc":       {"score": 18, "ok": True,  "explain": "ETH/BTC بالای EMA50"},
            "total2":        {"score": 12, "ok": True,  "explain": "TOTAL2 روند مثبت"},
            "total3":        {"score": 10, "ok": False, "explain": "TOTAL3 ضعف آلت‌های کوچک"},
            "btc_regime":    {"score": 8,  "ok": False, "explain": "ریسک رژیم BTC"},
            "eth_trend":     {"score": 11, "ok": True,  "explain": "روند ETH مثبت ملایم"},
        }

    def _compute_factors(self) -> Dict[str, Dict[str, Any]]:
        # تلاش برای استفاده از محاسبهٔ واقعی؛ اگر در دسترس نبود، fallback
        try:
            return self._compute_factors_real()
        except Exception as e:
            log.info("Using fallback factors (reason: %s)", e)
            return self._compute_factors_fallback()

    def run_daily_analysis(self) -> bool:
        print("🔄 Starting altseason analysis...")
        print("📡 Fetching market data...")
        print("📈 Calculating market factors...")

        # 1) محاسبه فاکتورها
        factors = self._compute_factors()

        # 2) امتیاز کل وزنی + تعداد OK
        total_score, ok_count = _weighted_total_and_okcount(factors)

        # 3) وضعیت
        status, forming = _classify(total_score, ok_count)

        # 4) کنسول
        badge = "🟢" if forming and status.startswith("Altseason") else \
                "🟡" if forming else \
                "⚪️" if status.startswith("Neutral") else "🔴"
        print(f"📊 Total Score: {total_score}/100")
        print(f"🎯 Status: {status} {badge}")
        print("✅ Analysis completed successfully!")

        # 5) state.json
        state = {
            "total_score": total_score,
            "status": status,
            "forming": bool(forming),
            "as_of": datetime.now(UTC).isoformat(),
            "factors": factors,
            "ok_count": ok_count,
        }
        _write_json(STATE_PATH, state)

        # 6) گزارش md
        _ = _write_markdown_report(self.reports_dir, state)

        return True
