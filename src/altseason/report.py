# src/altseason/report.py
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

class ReportGenerator:
    """Generates daily markdown + JSON reports for Altseason analysis."""
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

    def generate_daily_report(self, factors: Dict[str, Any], total_score: int, status: str, cache_used: bool = False):
        """Create daily .md and .json reports"""
        d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._write_md(d, factors, total_score, status, cache_used)
        self._write_state(d, factors, total_score, status)

    def _write_md(self, d, f, s, st, cache):
        """Write markdown summary report"""
        p = self.reports_dir / f"{d}.md"
        emo = {
            "Altseason Likely": "🚀",
            "Forming / Watch": "🟡",
            "Neutral": "⚪",
            "Risk-Off": "🔴",
        }.get(st, "❔")

        content = f"""# Altseason Radar Report — {d}

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**Status:** {emo} {st}  
**Total Score:** {s}/100  
{'⚠️ _Note: Cached data used_' if cache else ''}

| Factor | Score | OK | Explanation |
|--------|-------|----|-------------|
"""
        for name, info in f.items():
            if name in ("total_score", "status"):
                continue
            sc = info.get("score", 0)
            ok = "✅" if info.get("ok", False) else "❌"
            ex = info.get("explain", "N/A")
            content += f"| {name} | {sc} | {ok} | {ex} |\n"

        content += f"\n**Final Verdict:** {emo} {st} — {s}/100\n"
        p.write_text(content, encoding="utf-8")

    def _write_state(self, d, f, s, st):
        """Write JSON state file"""
        state = {
            "date": d,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_score": s,
            "status": st,
            "factors": f,
        }
        (self.reports_dir / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
