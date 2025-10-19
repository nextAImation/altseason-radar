# src/altseason/runner.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime, UTC

from altseason.config import FACTOR_WEIGHTS, THRESHOLDS
from altseason.factors import FactorCalculator

_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[2]
REPORTS_DIR = ROOT / "reports"
STATE_PATH = REPORTS_DIR / "state.json"

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _write_markdown_report(reports_dir: Path, state: Dict[str, Any]) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).date().isoformat()
    md_path = reports_dir / f"{date_str}.md"

    score   = state.get("total_score", "N/A")
    status  = state.get("status", "Unknown")
    forming = "Yes" if state.get("forming") else "No"
    as_of   = state.get("as_of", "")
    okc     = state.get("ok_count")
    conf    = state.get("confidence")

    lines = [
        f"# Altseason Radar — {date_str}",
        "",
        f"**Score:** {score}/100",
        f"**Status:** {status}",
        f"**Forming:** {forming}",
        f"**Generated:** {as_of}",
    ]
    if okc is not None or conf is not None:
        diag = []
        if okc is not None:  diag.append(f"**OK Factors:** {okc}")
        if conf is not None: diag.append(f"**Confidence:** {conf:.3f}")
        lines.append("")
        lines.append(" | ".join(diag))

    lines += ["", "## Factors"]
    factors: Dict[str, Any] = state.get("factors", {})
    for k, v in factors.items():
        sc = v.get("score", 0)
        ok = "✅" if v.get("ok") else "❌"
        ex = v.get("explain", "")
        lines.append(f"- **{k}**: {sc:.2f} {ok} — {ex}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path

def _penalty_enable() -> bool:
    return os.getenv("ALT_PENALTY_ENABLE", "1").strip() not in ("0", "false", "False")

def _penalty_factor() -> float:
    try:
        return float(os.getenv("ALT_PENALTY", "0.85"))
    except Exception:
        return 0.85

def _normalize_to_weight(name: str, v: Dict[str, Any]) -> float:
    w = float(FACTOR_WEIGHTS.get(name, 0))
    if w <= 0:
        return 0.0
    if "score_raw" in v and "score_max" in v:
        try:
            raw = float(v["score_raw"])
            mx = float(v["score_max"]) or 1.0
            return max(0.0, min((raw / mx) * w, w))
        except Exception:
            return 0.0
    try:
        sc = float(v.get("score", 0.0))
        return max(0.0, min(sc, w))
    except Exception:
        return 0.0

def _weighted_total_and_okcount(factors: Dict[str, Dict[str, Any]]) -> Tuple[int, int, float]:
    sum_w = float(sum(FACTOR_WEIGHTS.values()))
    if sum_w <= 0:
        return 0, 0, 0.0

    pen = _penalty_factor() if _penalty_enable() else 1.0

    total_raw = 0.0
    ok_count = 0
    for k, v in factors.items():
        base = _normalize_to_weight(k, v)
        if base <= 0:
            continue
        if bool(v.get("ok")):
            ok_count += 1
            total_raw += base
        else:
            total_raw += base * pen

    total_score = int(round((total_raw / sum_w) * 100.0))
    confidence = max(0.0, min(total_raw / sum_w, 1.0))
    return total_score, ok_count, confidence

def _classify(total_score: int, ok_count: int) -> Tuple[str, bool]:
    min_ok = int(THRESHOLDS.get("ALTSEASON_MIN_FACTORS", 4))
    if total_score >= int(THRESHOLDS.get("ALTSEASON_MIN_SCORE", 75)) and ok_count >= min_ok:
        return "Altseason Likely", True
    if total_score >= int(THRESHOLDS.get("FORMING_MIN", 60)) and ok_count >= max(2, min_ok - 1):
        return "Forming / Watch", True
    if total_score >= int(THRESHOLDS.get("NEUTRAL_MIN", 45)):
        return "Neutral", False
    return "Risk-Off", False

class AltseasonRunner:
    def __init__(self, reports_dir: str | Path | None = None, state_path: str | Path | None = None):
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
        self.state_path  = Path(state_path)  if state_path  else STATE_PATH
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _compute_factors(self) -> Dict[str, Dict[str, Any]]:
        calc = FactorCalculator()
        return calc.compute_factors()

    def run_daily_analysis(self) -> bool:
        print("🔄 Starting altseason analysis...")
        print("📡 Fetching market data...")
        print("📈 Calculating market factors...")

        factors = self._compute_factors()
        total_score, ok_count, confidence = _weighted_total_and_okcount(factors)
        status, forming = _classify(total_score, ok_count)

        badge = "🟢" if forming and status.startswith("Altseason") else \
                "🟡" if forming else \
                "⚪️" if status.startswith("Neutral") else "🔴"

        print(f"📊 Total Score: {total_score}/100")
        print(f"🎯 Status: {status} {badge}")
        print("✅ Analysis completed successfully!")

        state = {
            "total_score": total_score,
            "status": status,
            "forming": bool(forming),
            "as_of": datetime.now(UTC).isoformat(),
            "factors": factors,
            "ok_count": ok_count,
            "confidence": round(confidence, 3),
        }
        _write_json(self.state_path, state)
        _ = _write_markdown_report(self.reports_dir, state)
        return True
