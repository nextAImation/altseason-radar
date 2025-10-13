# src/altseason/runner.py
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class FactorResult:
    score: int
    ok: bool
    explain: Optional[str] = None


class AltseasonRunner:
    """
    Runs the daily Altseason analysis and persists results for downstream consumers.

    Responsibilities:
      - simulate/compute factors (you can later swap _compute_factors with real logic)
      - compute total score and status
      - write a machine-readable state.json for scripts/run_daily.py
      - write a human-readable daily markdown report under reports/
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("altseason")
        self.logger.setLevel(logging.INFO)

        # allow override via env (useful in CI)
        reports_dir_env = os.getenv("REPORTS_DIR", "").strip()
        self.reports_dir = Path(reports_dir_env) if reports_dir_env else Path.cwd() / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- public API --------------------

    def run_daily_analysis(self) -> bool:
        """Main entrypoint used by scripts/run_daily.py. Returns True on success."""
        print("🔄 Starting altseason analysis...")

        try:
            print("📡 Fetching market data...")
            # TODO: plug real providers here

            print("📈 Calculating market factors...")
            factors = self._compute_factors()

            total_score = sum(v["score"] for v in factors.values())
            status, forming = self._status_from_score(total_score)

            print(f"📊 Total Score: {total_score}/100")
            print(f"🎯 Status: {status}")
            print("✅ Analysis completed successfully!")

            # persist artifacts for other steps (telegram etc.)
            self._write_state_json(
                total_score=total_score,
                status=status,
                forming=forming,
                factors=factors,
            )
            self._write_daily_markdown(
                total_score=total_score,
                status=status,
                forming=forming,
                factors=factors,
            )

            return True

        except Exception as e:
            self.logger.exception("Daily analysis failed: %s", e)
            print("❌ Analysis failed")
            return False

    # -------------------- internals --------------------

    def _compute_factors(self) -> Dict[str, Dict[str, Any]]:
        """
        Simulated factors. Replace with real logic and real scoring later.
        Keep keys & structure stable (used in reports/state.json and Markdown).
        """
        simulated: Dict[str, FactorResult] = {
            "btc_dominance": FactorResult(score=15, ok=True,  explain="Dominance below falling MA"),
            "eth_btc":       FactorResult(score=18, ok=True,  explain="ETH/BTC trending up"),
            "total2":        FactorResult(score=12, ok=True,  explain="TOTAL2 above EMA"),
            "total3":        FactorResult(score=10, ok=False, explain="TOTAL3 momentum mixed"),
            "btc_regime":    FactorResult(score= 8, ok=False, explain="BTC in range-bound regime"),
            "eth_trend":     FactorResult(score=11, ok=True,  explain="ETH in constructive trend"),
        }
        # convert dataclasses to dicts
        return {k: {"score": v.score, "ok": v.ok, "explain": v.explain} for k, v in simulated.items()}

    @staticmethod
    def _status_from_score(total_score: int) -> tuple[str, bool]:
        """
        Business rules for mapping total score → status.
        Returns (status_label, forming_bool)
        """
        if total_score >= 75:
            return "Altseason Likely 🟢", True
        if total_score >= 60:
            return "Forming / Watch 🟡", True
        if total_score >= 45:
            return "Neutral ⚪️", False
        return "Risk-Off 🔴", False

    def _write_state_json(self, *, total_score: int, status: str, forming: bool, factors: Dict[str, Any]) -> None:
        """
        Writes a compact machine-readable file for the notifier/script.
        Path: reports/state.json
        Schema:
          {
            "total_score": int,
            "status": str,
            "forming": bool,
            "as_of": ISO-8601 UTC,
            "factors": { <factor>: {"score": int, "ok": bool, "explain": str|None}, ... }
          }
        """
        payload = {
            "total_score": int(total_score),
            "status": str(status),
            "forming": bool(forming),
            "as_of": datetime.now(timezone.utc).isoformat(),
            "factors": factors,
        }
        out = self.reports_dir / "state.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Wrote %s", out)

    def _write_daily_markdown(self, *, total_score: int, status: str, forming: bool, factors: Dict[str, Any]) -> None:
        """
        Writes a human-friendly markdown report the same day.
        Path: reports/YYYY-MM-DD.md  (UTC date)
        Includes 'Score:' و 'Status:' خطوطی که پارسر تلگرام به‌راحتی می‌خوانَد.
        """
        now_utc = datetime.now(timezone.utc)
        date_str = now_utc.date().isoformat()
        md_path = self.reports_dir / f"{date_str}.md"

        # simple table of factors
        lines = [
            f"# Altseason Radar — {date_str}",
            "",
            f"**Score:** {total_score}/100",
            f"**Status:** {status}",
            f"**Generated (UTC):** {now_utc.isoformat()}",
            "",
            "## Factors",
            "",
            "| Factor | Score | OK | Explain |",
            "|-------:|------:|:--:|:--------|",
        ]
        for name, d in factors.items():
            ok_emoji = "✅" if d.get("ok") else "❌"
            lines.append(f"| `{name}` | {d.get('score', '')} | {ok_emoji} | {d.get('explain') or ''} |")

        # tiny TL;DR
        tldr = "Forming" if forming else "Not forming"
        lines += [
            "",
            f"**TL;DR:** {tldr}.",
            "",
        ]

        md_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info("Wrote %s", md_path)
