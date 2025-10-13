#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Altseason Radar — daily runner with Telegram notify and robust logging.

Usage:
  python -m scripts.run_daily
  python -m scripts.run_daily --no-telegram
  python -m scripts.run_daily --state ./reports/state.json --reports ./reports
"""

from __future__ import annotations

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime
from dateutil import tz

# Allow "python -m scripts.run_daily"
CURR = Path(__file__).resolve()
ROOT = CURR.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from altseason.runner import AltseasonRunner  # noqa: E402
from altseason.config import (                 # noqa: E402
    get_telegram_token,
    get_telegram_chat_id,
    TZ_DISPLAY,
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except Exception:
    # Minimal fallback if 'rich' is unavailable
    class _DummyConsole:
        def print(self, *a, **k):  # type: ignore
            print(*a)
        def rule(self, *a, **k):   # type: ignore
            print("-" * 60)
    Console = _DummyConsole  # type: ignore
    Panel = Text = Table = object  # type: ignore

import requests  # After requirements install

console: Console = Console()  # type: ignore

# ----------------------------- helpers: files & time -----------------------------

def ensure_state_file(path: Path) -> None:
    """Create a minimal state.json if it doesn't exist (prevents hard failure)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    minimal = {
        "total_score": None,
        "status": "",
        "forming": False,
        "as_of": datetime.utcnow().isoformat() + "Z",
        "factors": {},
    }
    path.write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")


def read_state(state_path: Path) -> Dict[str, Any]:
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_local(dt_iso: str, tz_name: str) -> str:
    """Format ISO datetime string to display timezone for human-friendly message."""
    try:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        to_zone = tz.gettz(tz_name or "UTC")
        return dt.astimezone(to_zone).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return dt_iso

# ----------------------------- fallback: parse latest md -----------------------------

MD_SCORE_RE = re.compile(r"(?i)Total\s*Score\s*:\s*(\d+)\s*/\s*100")
MD_STATUS_RE = re.compile(r"(?i)Status\s*:\s*([^\n]+)")

def parse_latest_md(reports_dir: Path) -> Dict[str, Any]:
    """Parse the most recent Markdown report to extract score/status as a fallback."""
    md_files: List[Path] = sorted(
        reports_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not md_files:
        return {}

    text = md_files[0].read_text(encoding="utf-8", errors="ignore")
    score: Optional[int] = None
    status: Optional[str] = None

    m = MD_SCORE_RE.search(text)
    if m:
        try:
            score = int(m.group(1))
        except Exception:
            pass

    m = MD_STATUS_RE.search(text)
    if m:
        # Example line: "Forming / Watch 🟡"
        status = m.group(1).strip()

    return {
        "total_score": score,
        "status": status,
        "forming": bool(status and ("Form" in status or "Watch" in status)),
        "as_of": datetime.utcnow().isoformat() + "Z",
        "factors": {},
    }


def load_summary(state_path: Path, reports_dir: Path) -> Dict[str, Any]:
    """
    Load daily summary from state.json; if missing/incomplete,
    parse the newest Markdown report in `reports_dir`.
    """
    def _has_values(d: Dict[str, Any]) -> bool:
        return (
            isinstance(d, dict)
            and d.get("total_score") not in (None, "", "N/A")
            and isinstance(d.get("total_score"), (int, float))
            and str(d.get("status", "")).strip() != ""
        )

    state: Dict[str, Any] = {}
    # 1) سعی در خواندن state.json
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    # 2) اگر ناقص بود از Markdown پرش کن
    needs_fallback = not _has_values(state)

    if needs_fallback:
        # reports/*.md را پیدا کن
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_files = sorted(reports_dir.glob("*.md"))
        if md_files:
            latest_md = md_files[-1]
            text = latest_md.read_text(encoding="utf-8")

            import re
            score = None
            status = None

            # امتیاز — چند الگو برای پوشش حالت‌های مختلف
            for pat in [
                r"Total\s*Score:\s*(\d{1,3})\s*/\s*100",
                r"\*\*Total\s*Score:\*\*\s*(\d{1,3})\s*/\s*100",
                r"Final\s*Verdict:.*?(\d{1,3})\s*/\s*100",
            ]:
                m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    score = int(m.group(1))
                    break

            # وضعیت — هم با عنوان Status هم با Final Verdict
            for pat in [
                r"Status:\s*([^\n\r]+)",
                r"\*\*Status:\*\*\s*([^\n\r]+)",
                r"Final\s*Verdict:\s*([^\n\r]+)",
            ]:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m:
                    status_raw = m.group(1).strip()
                    # تمیز کردن ایموجی/بولت‌ها
                    status = (
                        status_raw.replace("🟢", "")
                                  .replace("🟡", "")
                                  .replace("🟠", "")
                                  .replace("🔴", "")
                                  .replace("⚪", "")
                                  .replace("⚪️", "")
                                  .strip(" -:|")
                                  .strip()
                    )
                    # بعضی گزارش‌ها به شکل: "Neutral — 60/100"
                    status = status.split("—")[0].strip()
                    break

            if score is not None:
                state["total_score"] = score
            if status:
                state["status"] = status

            # forming را از status استنتاج کن (اختیاری)
            st_low = str(state.get("status", "")).lower()
            if "form" in st_low or "watch" in st_low:
                state["forming"] = True
            elif "neutral" in st_low:
                state["forming"] = False

            # زمان
            if "as_of" not in state or not state["as_of"]:
                # Generated یا تاریخ سربرگ را بخوان
                ts = None
                m = re.search(r"\*\*Generated:\*\*\s*([^\n\r]+)", text)
                if m:
                    ts = m.group(1).strip()
                if ts:
                    state["as_of"] = ts
                else:
                    from datetime import datetime, UTC
                    state["as_of"] = datetime.now(UTC).isoformat()

    # تضمین وجود کلیدها
    state.setdefault("forming", False)
    if "as_of" not in state or not state["as_of"]:
        from datetime import datetime, UTC
        state["as_of"] = datetime.now(UTC).isoformat()

    return state


# ----------------------------- message & telegram -----------------------------

def build_message(state: Dict[str, Any]) -> str:
    score = state.get("total_score")
    status = state.get("status") or "Unknown"
    forming = bool(state.get("forming"))
    dt_raw = state.get("as_of") or state.get("date") or datetime.utcnow().isoformat() + "Z"
    when = fmt_local(dt_raw, TZ_DISPLAY)

    parts = [
        "📡 *Altseason Radar — Daily*",
        f"📊 *Score:* `{score}/100`" if score is not None else "📊 *Score:* `N/A`",
        f"🎯 *Status:* {status} {'🟢' if forming else '🟡' if 'Form' in status or 'Watch' in status else '⚪️'}",
        f"🕒 *As of:* {when}",
    ]

    # Optional factor bullets (kept short for Telegram)
    facs: Dict[str, Any] = state.get("factors") or {}
    if facs:
        top = []
        for key, val in facs.items():
            try:
                sc = val.get("score")
                ok = "✅" if val.get("ok") else "❌"
                top.append(f"• {key}: {sc} {ok}")
            except Exception:
                pass
        if top:
            parts.append("—\n*Factors*\n" + "\n".join(top[:8]))  # cap to 8 lines

    return "\n".join(parts)


def send_telegram(text: str) -> Optional[Dict[str, Any]]:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or get_telegram_token()
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or get_telegram_chat_id()

    if not token or not chat_id:
        console.print("[yellow]Telegram not configured (missing token/chat_id). Skipping.[/yellow]")
        return None

    api = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    r = requests.post(api, json=payload, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram API error: {r.status_code} {r.text}")
    return r.json()

# ----------------------------- runner -----------------------------

def run_analysis() -> bool:
    console.rule("[bold cyan]Altseason Radar — Daily Run")
    console.print("🔄 Starting altseason analysis...")
    runner = AltseasonRunner()
    ok = runner.run_daily_analysis()
    if ok:
        console.print("✅ Analysis completed successfully!")
    else:
        console.print("[red]❌ Analysis failed[/red]")
    return ok


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run daily analysis and optionally notify Telegram.")
    p.add_argument(
        "--no-telegram",
        action="store_true",
        help="Do not send Telegram notification even if analysis succeeds.",
    )
    p.add_argument(
        "--state",
        type=str,
        default=str(ROOT / "reports" / "state.json"),
        help="Path to state.json generated by analysis.",
    )
    p.add_argument(
        "--reports",
        type=str,
        default=str(ROOT / "reports"),
        help="Reports directory (for Markdown fallback parsing).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = make_parser().parse_args(argv)

    state_path = Path(args.state)
    reports_dir = Path(args.reports)

    try:
        # Make sure state file exists so the next steps never hard-fail
        ensure_state_file(state_path)

        ok = run_analysis()
        if not ok:
            return 1

        # Load summary with Markdown fallback when state.json is empty/partial
        state = load_summary(state_path, reports_dir)
        msg = build_message(state)

        # Console table (best-effort; if rich not present, it will just print text)
        try:
            table = Table(title="Daily Summary")
            table.add_column("Field", style="bold")
            table.add_column("Value")
            table.add_row("Score", f"{state.get('total_score', 'N/A')}")
            table.add_row("Status", state.get("status", "Unknown") or "Unknown")
            table.add_row("Forming", "Yes" if state.get("forming") else "No")
            when = fmt_local(state.get("as_of") or "", TZ_DISPLAY)
            table.add_row("As of", when or "-")
            console.print(table)
        except Exception:
            console.print(msg)

        if args.no_telegram:
            console.print("[yellow]Skipping Telegram by flag.[/yellow]")
            return 0

        console.print("📨 Sending Telegram notification…")
        _res = send_telegram(msg)
        console.print("✅ Telegram notification sent.")

        console.print("[green]Daily analysis completed successfully![/green]")
        return 0

    except requests.RequestException as e:
        console.print(f"[red]Network/Telegram error:[/red] {e}")
        return 3
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        return 9


if __name__ == "__main__":
    sys.exit(main())
